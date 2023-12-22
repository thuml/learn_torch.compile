
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


# kernel path: /tmp/torchinductor_youkaichao/nn/cnn7o3dr6y6mexdn76xsqxszncezqgfrjpwwqqb2yk5mo2rujlwy.py
# Source Nodes: [x], Original ATen: [aten.constant_pad_nd]
# x => constant_pad_nd
triton_poi_fused_constant_pad_nd_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 893976
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 193) % 193
    x0 = xindex % 193
    x2 = (xindex // 37249)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 192, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (192*x1) + (36864*x2)), tmp5 & xmask, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/iu/ciurfys7u7jfiydchqiifwccaerwfbfhvxlvvbn42xd3wfmoalfx.py
# Source Nodes: [batch_norm, weight], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm => add, mul_1, mul_2, rsqrt, squeeze_1, sub, var_mean
# weight => view_2
triton_per_fused__native_batch_norm_legit_view_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 32],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 27
    RBLOCK: tl.constexpr = 32
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (27*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 27, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 27.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = 0.19245008972987526
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (27*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oq/coqzjfwo4vrcmrtyehaapuaifx256amgqzfabsrwtr5two7gmdhd.py
# Source Nodes: [conv2d, gelu, mul_], Original ATen: [aten.convolution, aten.gelu, aten.mul]
# conv2d => convolution
# gelu => add_1, erf, mul_3, mul_4, mul_5
# mul_ => mul_6
triton_poi_fused_convolution_gelu_mul_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_mul_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 9216) % 16
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = 1.7015043497085571
    tmp12 = tmp10 * tmp11
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6l/c6lknqhe36xy3vuzc24r7riyp3hjjpeeu3dvlixmlwgsvglkhcco.py
# Source Nodes: [batch_norm_1, weight_1], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_1 => add_2, mul_8, mul_9, rsqrt_1, squeeze_3, sub_1, var_mean_1
# weight_1 => view_5
triton_per_fused__native_batch_norm_legit_view_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (144*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 144, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 144.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = 0.08333333333333333
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (144*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/em/cemtcm73754a7o5uh4eba5sxkdcgc6i7fdvir6z7p7j3uytvikgz.py
# Source Nodes: [conv2d_1, gelu_1, mul__1], Original ATen: [aten.convolution, aten.gelu, aten.mul]
# conv2d_1 => convolution_1
# gelu_1 => add_3, erf_1, mul_10, mul_11, mul_12
# mul__1 => mul_13
triton_poi_fused_convolution_gelu_mul_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_mul_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 9216) % 32
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = 1.7015043497085571
    tmp12 = tmp10 * tmp11
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qe/cqe2chons4agiaq2f3nhmlfzpghdfafvlfg2nqn7xoq6ie3nhbuu.py
# Source Nodes: [batch_norm_2, weight_2], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_2 => add_4, mul_15, mul_16, rsqrt_2, squeeze_5, sub_2, var_mean_2
# weight_2 => view_8
triton_per_fused__native_batch_norm_legit_view_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 64
    XBLOCK: tl.constexpr = 1
    rnumel = 288
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (288*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 288, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp0 - tmp10
    tmp18 = 288.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = 0.05892556509887896
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (288*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nn/cnn3dia44vigdqvlz2edyuevloholdlnvar7szwqgwbyjeyx7v5b.py
# Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
# conv2d_2 => convolution_2
triton_poi_fused_convolution_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4718592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 9216) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/st/cstehspj362gac2apffzivzpe3toybp22323yximvjsdmmhhx2ar.py
# Source Nodes: [gelu_2, mul__2, x_2], Original ATen: [aten.constant_pad_nd, aten.gelu, aten.mul]
# gelu_2 => add_5, erf_2, mul_17, mul_18, mul_19
# mul__2 => mul_20
# x_2 => constant_pad_nd_1
triton_poi_fused_constant_pad_nd_gelu_mul_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_gelu_mul_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4817408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 97) % 97
    x0 = xindex % 97
    x2 = (xindex // 9409)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 96, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (96*x1) + (9216*x2)), tmp5 & xmask, other=0.0)
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = 0.7071067811865476
    tmp10 = tmp6 * tmp9
    tmp11 = tl.math.erf(tmp10)
    tmp12 = 1.0
    tmp13 = tmp11 + tmp12
    tmp14 = tmp8 * tmp13
    tmp15 = 1.7015043497085571
    tmp16 = tmp14 * tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp5, tmp16, tmp17)
    tl.store(out_ptr0 + (x3), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vf/cvfnjgjx3q5zhqvvwyyr25eeomvw343hfsijznz3vrn2eknqi4cf.py
# Source Nodes: [batch_norm_3, weight_3], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_3 => add_6, mul_22, mul_23, rsqrt_3, squeeze_7, sub_3, var_mean_3
# weight_3 => view_11
triton_per_fused__native_batch_norm_legit_view_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 128
    XBLOCK: tl.constexpr = 1
    rnumel = 576
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (576*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 576, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp0 - tmp10
    tmp18 = 576.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = 0.041666666666666664
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (576*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nc/cncjycp2ngsu3rqaxifdzdzvw3hdhzzfhzttmy7uqalgrzies5h4.py
# Source Nodes: [gelu_3, mul__3, out, shortcut], Original ATen: [aten.convolution, aten.gelu, aten.mul]
# gelu_3 => add_7, erf_3, mul_24, mul_25, mul_26
# mul__3 => mul_27
# out => mul_28
# shortcut => convolution_3
triton_poi_fused_convolution_gelu_mul_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_mul_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 2304) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = 1.7015043497085571
    tmp12 = tmp10 * tmp11
    tmp13 = tmp12 * tmp8
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ie/cie7dzdx6uwoaezrmsrlelsnbaiox3e5jtzb6q7h3zq6gvywjjlg.py
# Source Nodes: [batch_norm_4, weight_4], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_4 => add_8, mul_30, mul_31, rsqrt_4, squeeze_9, sub_4, var_mean_4
# weight_4 => view_14
triton_per_fused__native_batch_norm_legit_view_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 128.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = 0.08838834764831845
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dw/cdwpgnqlr4ojxn7zdqnast2mafffpdgp7fh5cp3m225j5a33clfr.py
# Source Nodes: [batch_norm_5, weight_5], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_5 => add_9, mul_33, mul_34, rsqrt_5, squeeze_11, sub_5, var_mean_5
# weight_5 => view_17
triton_per_fused__native_batch_norm_legit_view_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
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
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 128, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 128.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = 0.08838834764831845
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dx/cdxo27y7tisghdnggttkhhe44wvc64k52wddwynwn7cbz5xsuvvz.py
# Source Nodes: [gelu_4, mul__4, out_1], Original ATen: [aten.convolution, aten.gelu, aten.mul]
# gelu_4 => add_10, erf_4, mul_35, mul_36, mul_37
# mul__4 => mul_38
# out_1 => convolution_5
triton_poi_fused_convolution_gelu_mul_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_mul_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 2304) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = 1.7015043497085571
    tmp12 = tmp10 * tmp11
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/t5/ct5bc6ymlhxrwafsv2nkfevhssu3o5jiwzm4j7uccxpnooorhh7k.py
# Source Nodes: [batch_norm_6, weight_6], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_6 => add_11, mul_40, mul_41, rsqrt_6, squeeze_13, sub_6, var_mean_6
# weight_6 => view_20
triton_red_fused__native_batch_norm_legit_view_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[128, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_view_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp13 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr0 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tmp5 - tmp2
        tmp7 = 1152.0
        tmp8 = tmp3 / tmp7
        tmp9 = 1e-05
        tmp10 = tmp8 + tmp9
        tmp11 = tl.math.rsqrt(tmp10)
        tmp12 = tmp6 * tmp11
        tmp14 = 0.02946278254943948
        tmp15 = tmp13 * tmp14
        tmp16 = tmp12 * tmp15
        tl.store(out_ptr2 + (r1 + (1152*x0)), tmp16, rmask & xmask)
    tmp17 = 1152.0
    tmp18 = tmp3 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tl.store(out_ptr3 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/on/concrnkbuno6ga4efi7lxfqqaw6b5nky3eb7bgmeuky5777uacb2.py
# Source Nodes: [out_4, x_se], Original ATen: [aten.convolution, aten.mean]
# out_4 => convolution_8
# x_se => mean
triton_red_fused_convolution_mean_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_mean_14', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 2304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 256
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r2 + (2304*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
        tl.store(in_out_ptr0 + (r2 + (2304*x3)), tmp2, rmask)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp6 = 2304.0
    tmp7 = tmp4 / tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cz/cczhsrryuy2ggmfhwzv3fgix3njq25z2y6dwea6gunvj6xe4t5t3.py
# Source Nodes: [x_se_1, x_se_2], Original ATen: [aten.convolution, aten.relu]
# x_se_1 => convolution_9
# x_se_2 => relu
triton_poi_fused_convolution_relu_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vb/cvbwh2onqgpxjhc7s7zrhohx62mlbpszf4bevnqhf2zgbrsf5nww.py
# Source Nodes: [x_se_3], Original ATen: [aten.convolution]
# x_se_3 => convolution_10
triton_poi_fused_convolution_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/u5/cu5pmsiqmynaydaodxnhj5byyrdckj2rbyj35jg6r4a77j3ie6ck.py
# Source Nodes: [gelu_7, mul_10, mul_12, mul__7, mul__8, out_5, out_8, shortcut_1, shortcut_2, sigmoid], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.sigmoid]
# gelu_7 => add_17, erf_7, mul_60, mul_61, mul_62
# mul_10 => mul_56
# mul_12 => mul_59
# mul__7 => mul_58
# mul__8 => mul_63
# out_5 => mul_57
# out_8 => mul_64
# shortcut_1 => convolution_4
# shortcut_2 => add_16
# sigmoid => sigmoid
triton_poi_fused_add_convolution_gelu_mul_sigmoid_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mul_sigmoid_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4718592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 2304) % 256
    x4 = (xindex // 2304)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp2 = tmp0 + tmp1
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tmp7 = 2.0
    tmp8 = tmp6 * tmp7
    tmp11 = tmp8 * tmp10
    tmp12 = 0.2
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13 + tmp2
    tmp15 = 0.5
    tmp16 = tmp14 * tmp15
    tmp17 = 0.7071067811865476
    tmp18 = tmp14 * tmp17
    tmp19 = tl.math.erf(tmp18)
    tmp20 = 1.0
    tmp21 = tmp19 + tmp20
    tmp22 = tmp16 * tmp21
    tmp23 = 1.7015043497085571
    tmp24 = tmp22 * tmp23
    tmp25 = 0.9805806756909201
    tmp26 = tmp24 * tmp25
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp26, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3c/c3clukpgeqiq5mor4v6fmhsmztbhn7z5zcrouozz36morux26tvt.py
# Source Nodes: [getattr_getattr_l__mod___stages___1_____0___downsample_pool], Original ATen: [aten.avg_pool2d]
# getattr_getattr_l__mod___stages___1_____0___downsample_pool => avg_pool2d
triton_poi_fused_avg_pool2d_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 24
    x1 = (xindex // 24)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (96*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (96*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (48 + (2*x0) + (96*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (49 + (2*x0) + (96*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5k/c5kt3kh3prjbtvp3po26nohig2g3nevhkqpiwrhwdegnexinalvd.py
# Source Nodes: [batch_norm_9, weight_9], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_9 => add_18, mul_66, mul_67, rsqrt_9, squeeze_19, sub_9, var_mean_9
# weight_9 => view_29
triton_per_fused__native_batch_norm_legit_view_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 512
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
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 256, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp0 - tmp10
    tmp18 = 256.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = 0.0625
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wl/cwlgdpr7kr2w4ehdriffu4cdqgqu5bu44ely3wxamy6i6rft6qik.py
# Source Nodes: [batch_norm_10, weight_10], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_10 => add_19, mul_69, mul_70, rsqrt_10, squeeze_21, sub_10, var_mean_10
# weight_10 => view_32
triton_per_fused__native_batch_norm_legit_view_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_20', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 256
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
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 256, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp0 - tmp10
    tmp18 = 256.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = 0.0625
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ow/cowog63vljpvmy6ed4unrqcz3cxil5m67bn27xo2zbm4z6zw7mv2.py
# Source Nodes: [out_9], Original ATen: [aten.convolution]
# out_9 => convolution_12
triton_poi_fused_convolution_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4718592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 2304) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ke/ckede2u4map6z3jaixcaifr2uoittpag3tvbediv7mdsg7dyzhsj.py
# Source Nodes: [gelu_8, mul__9, x_5], Original ATen: [aten.constant_pad_nd, aten.gelu, aten.mul]
# gelu_8 => add_20, erf_8, mul_71, mul_72, mul_73
# mul__9 => mul_74
# x_5 => constant_pad_nd_2
triton_poi_fused_constant_pad_nd_gelu_mul_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_gelu_mul_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4917248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 49) % 49
    x0 = xindex % 49
    x2 = (xindex // 2401)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 48, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (48*x1) + (2304*x2)), tmp5, other=0.0)
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = 0.7071067811865476
    tmp10 = tmp6 * tmp9
    tmp11 = tl.math.erf(tmp10)
    tmp12 = 1.0
    tmp13 = tmp11 + tmp12
    tmp14 = tmp8 * tmp13
    tmp15 = 1.7015043497085571
    tmp16 = tmp14 * tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp5, tmp16, tmp17)
    tl.store(out_ptr0 + (x3), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rg/crguhvydo2xwll7qv4vxdupilaxut6tkypxvgi4khore7i4s3cqi.py
# Source Nodes: [batch_norm_11, weight_11], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_11 => add_21, mul_76, mul_77, rsqrt_11, squeeze_23, sub_11, var_mean_11
# weight_11 => view_35
triton_red_fused__native_batch_norm_legit_view_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_view_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp13 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr0 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tmp5 - tmp2
        tmp7 = 1152.0
        tmp8 = tmp3 / tmp7
        tmp9 = 1e-05
        tmp10 = tmp8 + tmp9
        tmp11 = tl.math.rsqrt(tmp10)
        tmp12 = tmp6 * tmp11
        tmp14 = 0.02946278254943948
        tmp15 = tmp13 * tmp14
        tmp16 = tmp12 * tmp15
        tl.store(out_ptr2 + (r1 + (1152*x0)), tmp16, rmask & xmask)
    tmp17 = 1152.0
    tmp18 = tmp3 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tl.store(out_ptr3 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k5/ck53idwxscv7mtbsosxne3d7fg5kghlp6pl4ip53swqacrcwugwc.py
# Source Nodes: [gelu_9, mul__10, out_10], Original ATen: [aten.convolution, aten.gelu, aten.mul]
# gelu_9 => add_22, erf_9, mul_78, mul_79, mul_80
# mul__10 => mul_81
# out_10 => convolution_13
triton_poi_fused_convolution_gelu_mul_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_mul_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 576) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = 1.7015043497085571
    tmp12 = tmp10 * tmp11
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/52/c52bobeyr37owuhfqy3mbngtom54k3zbk7mfcesoumqi6x22iapo.py
# Source Nodes: [out_12, x_se_4], Original ATen: [aten.convolution, aten.mean]
# out_12 => convolution_15
# x_se_4 => mean_1
triton_per_fused_convolution_mean_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_mean_25', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 576
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (r2 + (576*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp7 = 576.0
    tmp8 = tmp6 / tmp7
    tl.store(in_out_ptr0 + (r2 + (576*x3)), tmp2, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/in/cinyhe6kzjn4u43zoq7tsqyc4mls3p5qcrcpk2ojiziqrldfgv6f.py
# Source Nodes: [x_se_5, x_se_6], Original ATen: [aten.convolution, aten.relu]
# x_se_5 => convolution_16
# x_se_6 => relu_1
triton_poi_fused_convolution_relu_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_26', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gi/cgi7whum7o74bcvqytx6cgkjvjl2w3up5ysglhnvy2o4ac55sssn.py
# Source Nodes: [x_se_7], Original ATen: [aten.convolution]
# x_se_7 => convolution_17
triton_poi_fused_convolution_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_27', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hr/chrq6h4cslar2rl3pirn2gr56q5c26ouxeolfy47364csrgglsbn.py
# Source Nodes: [gelu_11, mul_19, mul_21, mul__12, mul__13, out_13, out_16, shortcut_3, shortcut_4, sigmoid_1], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.sigmoid]
# gelu_11 => add_27, erf_11, mul_96, mul_97, mul_98
# mul_19 => mul_92
# mul_21 => mul_95
# mul__12 => mul_94
# mul__13 => mul_99
# out_13 => mul_93
# out_16 => mul_100
# shortcut_3 => convolution_11
# shortcut_4 => add_26
# sigmoid_1 => sigmoid_1
triton_poi_fused_add_convolution_gelu_mul_sigmoid_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mul_sigmoid_28', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 576) % 512
    x4 = (xindex // 576)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp2 = tmp0 + tmp1
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tmp7 = 2.0
    tmp8 = tmp6 * tmp7
    tmp11 = tmp8 * tmp10
    tmp12 = 0.2
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13 + tmp2
    tmp15 = 0.5
    tmp16 = tmp14 * tmp15
    tmp17 = 0.7071067811865476
    tmp18 = tmp14 * tmp17
    tmp19 = tl.math.erf(tmp18)
    tmp20 = 1.0
    tmp21 = tmp19 + tmp20
    tmp22 = tmp16 * tmp21
    tmp23 = 1.7015043497085571
    tmp24 = tmp22 * tmp23
    tmp25 = 0.9805806756909201
    tmp26 = tmp24 * tmp25
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp26, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kx/ckxua6pwsohhzgzeecl2hzcdgd2wurfx6qghhlho3zyy56u4ss3h.py
# Source Nodes: [batch_norm_14, weight_14], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_14 => add_28, mul_102, mul_103, rsqrt_14, squeeze_29, sub_14, var_mean_14
# weight_14 => view_44
triton_per_fused__native_batch_norm_legit_view_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_29', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 256
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
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 512, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp0 - tmp10
    tmp18 = 512.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = 0.04419417382415922
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vw/cvwdui3qy65srm72phz6pzwqwsvwnpt3sd3gpxmydwg644qzm6jl.py
# Source Nodes: [gelu_15, mul_19, mul_21, mul_27, mul_29, mul__12, mul__17, mul__18, out_13, out_21, out_24, shortcut_4, shortcut_5, sigmoid_1, sigmoid_2], Original ATen: [aten.add, aten.gelu, aten.mul, aten.sigmoid]
# gelu_15 => add_36, erf_15, mul_129, mul_130, mul_131
# mul_19 => mul_92
# mul_21 => mul_95
# mul_27 => mul_125
# mul_29 => mul_128
# mul__12 => mul_94
# mul__17 => mul_127
# mul__18 => mul_132
# out_13 => mul_93
# out_21 => mul_126
# out_24 => mul_133
# shortcut_4 => add_26
# shortcut_5 => add_35
# sigmoid_1 => sigmoid_1
# sigmoid_2 => sigmoid_2
triton_poi_fused_add_gelu_mul_sigmoid_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_mul_sigmoid_30', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 576)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp11 = tl.load(in_ptr3 + (x2), None)
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (0))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp20 = tl.load(in_ptr6 + (x2), None)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp8 = tmp5 * tmp7
    tmp9 = 0.2
    tmp10 = tmp8 * tmp9
    tmp13 = tl.sigmoid(tmp12)
    tmp14 = tmp11 * tmp13
    tmp15 = tmp14 * tmp4
    tmp18 = tmp15 * tmp17
    tmp19 = tmp18 * tmp9
    tmp21 = tmp19 + tmp20
    tmp22 = tmp10 + tmp21
    tmp23 = 0.5
    tmp24 = tmp22 * tmp23
    tmp25 = 0.7071067811865476
    tmp26 = tmp22 * tmp25
    tmp27 = tl.math.erf(tmp26)
    tmp28 = 1.0
    tmp29 = tmp27 + tmp28
    tmp30 = tmp24 * tmp29
    tmp31 = 1.7015043497085571
    tmp32 = tmp30 * tmp31
    tmp33 = 0.9622504486493761
    tmp34 = tmp32 * tmp33
    tl.store(in_out_ptr0 + (x2), tmp34, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ie/ciejx32n3c37mbqh562qmzf4rkvcspo64jd3otnaikx7qlmtlmud.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____0___downsample_pool], Original ATen: [aten.avg_pool2d]
# getattr_getattr_l__mod___stages___2_____0___downsample_pool => avg_pool2d_1
triton_poi_fused_avg_pool2d_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 589824
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 12
    x1 = (xindex // 12)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (48*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (48*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (24 + (2*x0) + (48*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (25 + (2*x0) + (48*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/r3/cr3mndbqergaj6i5agsuysqylqdvakvqjeqji2gfs3tcaadgocnu.py
# Source Nodes: [batch_norm_18, weight_18], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_18 => add_37, mul_135, mul_136, rsqrt_18, squeeze_37, sub_18, var_mean_18
# weight_18 => view_56
triton_per_fused__native_batch_norm_legit_view_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1536
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
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 512, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp0 - tmp10
    tmp18 = 512.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = 0.04419417382415922
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ym/cymlmo5zf5rkde4dpjret63zwrct3lip76xsvkw5tbi32qrznnwi.py
# Source Nodes: [batch_norm_19, weight_19], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_19 => add_38, mul_138, mul_139, rsqrt_19, squeeze_39, sub_19, var_mean_19
# weight_19 => view_59
triton_per_fused__native_batch_norm_legit_view_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (r1 + (512*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 512, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp0 - tmp10
    tmp18 = 512.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = 0.04419417382415922
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7x/c7x2627ueuhkhpv6o2zta742ox454upnyqrikqi4aejgienksgho.py
# Source Nodes: [out_25], Original ATen: [aten.convolution]
# out_25 => convolution_25
triton_poi_fused_convolution_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_34', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3538944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 576) % 768
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wp/cwpbrvfs55t3h7g32fh6wxe7gmmp7tlju5cvb5vjtfwzdtx6dwvs.py
# Source Nodes: [gelu_16, mul__19, x_7], Original ATen: [aten.constant_pad_nd, aten.gelu, aten.mul]
# gelu_16 => add_39, erf_16, mul_140, mul_141, mul_142
# mul__19 => mul_143
# x_7 => constant_pad_nd_3
triton_poi_fused_constant_pad_nd_gelu_mul_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_gelu_mul_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3840000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 25) % 25
    x0 = xindex % 25
    x2 = (xindex // 625)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 24, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (24*x1) + (576*x2)), tmp5, other=0.0)
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = 0.7071067811865476
    tmp10 = tmp6 * tmp9
    tmp11 = tl.math.erf(tmp10)
    tmp12 = 1.0
    tmp13 = tmp11 + tmp12
    tmp14 = tmp8 * tmp13
    tmp15 = 1.7015043497085571
    tmp16 = tmp14 * tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp5, tmp16, tmp17)
    tl.store(out_ptr0 + (x3), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pj/cpjj4irm3rfdwri5tk5vti54ebp6ugzsqpmeln2lsehurp6jegcw.py
# Source Nodes: [batch_norm_20, weight_20], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_20 => add_40, mul_145, mul_146, rsqrt_20, squeeze_41, sub_20, var_mean_20
# weight_20 => view_62
triton_red_fused__native_batch_norm_legit_view_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_view_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp13 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr0 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tmp5 - tmp2
        tmp7 = 1152.0
        tmp8 = tmp3 / tmp7
        tmp9 = 1e-05
        tmp10 = tmp8 + tmp9
        tmp11 = tl.math.rsqrt(tmp10)
        tmp12 = tmp6 * tmp11
        tmp14 = 0.02946278254943948
        tmp15 = tmp13 * tmp14
        tmp16 = tmp12 * tmp15
        tl.store(out_ptr2 + (r1 + (1152*x0)), tmp16, rmask & xmask)
    tmp17 = 1152.0
    tmp18 = tmp3 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tl.store(out_ptr3 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bt/cbtm5j5ta4absyf5d5uz4h7douolqjbymuipyalzglfsblhrs7vx.py
# Source Nodes: [gelu_17, mul__20, out_26], Original ATen: [aten.convolution, aten.gelu, aten.mul]
# gelu_17 => add_41, erf_17, mul_147, mul_148, mul_149
# mul__20 => mul_150
# out_26 => convolution_26
triton_poi_fused_convolution_gelu_mul_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_mul_37', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 884736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 144) % 768
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = 1.7015043497085571
    tmp12 = tmp10 * tmp11
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2m/c2mibnyldzkjsgxx4tzsmxofnltkhujnpnhkr6jz4uo2zn6hmwfe.py
# Source Nodes: [batch_norm_22, weight_22], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_22 => add_44, mul_159, mul_160, rsqrt_22, squeeze_45, sub_22, var_mean_22
# weight_22 => view_68
triton_per_fused__native_batch_norm_legit_view_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_38', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
    xnumel = 1536
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
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 768, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp0 - tmp10
    tmp18 = 768.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = 0.03608439182435161
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (768*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/io/ciouevqax6htlxxutctmkak5vvoc3pk32qp7qyxpj65caeioxuv5.py
# Source Nodes: [out_28, x_se_12], Original ATen: [aten.convolution, aten.mean]
# out_28 => convolution_28
# x_se_12 => mean_3
triton_per_fused_convolution_mean_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_mean_39', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1536
    tmp0 = tl.load(in_out_ptr0 + (r2 + (144*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 144.0
    tmp8 = tmp6 / tmp7
    tl.store(in_out_ptr0 + (r2 + (144*x3)), tmp2, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vj/cvj6jc5uvye55ga7txwbvxyzu3aofsi6vz2ygzrq5k2kutr4xvsw.py
# Source Nodes: [x_se_13, x_se_14], Original ATen: [aten.convolution, aten.relu]
# x_se_13 => convolution_29
# x_se_14 => relu_3
triton_poi_fused_convolution_relu_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_40', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/p7/cp7zkunbjupozyqtp3vzodvmklgn4ycngqaogpcvh7yypwi7lbc2.py
# Source Nodes: [x_se_15], Original ATen: [aten.convolution]
# x_se_15 => convolution_30
triton_poi_fused_convolution_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_41', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1536
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xp/cxp6272hcw7nphxpnpfclglls5jq7rbkdyyfgbu4yidfuyyvgrsb.py
# Source Nodes: [gelu_19, mul_36, mul_38, mul__22, mul__23, out_29, out_32, shortcut_6, shortcut_7, sigmoid_3], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.sigmoid]
# gelu_19 => add_46, erf_19, mul_165, mul_166, mul_167
# mul_36 => mul_161
# mul_38 => mul_164
# mul__22 => mul_163
# mul__23 => mul_168
# out_29 => mul_162
# out_32 => mul_169
# shortcut_6 => convolution_24
# shortcut_7 => add_45
# sigmoid_3 => sigmoid_3
triton_poi_fused_add_convolution_gelu_mul_sigmoid_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mul_sigmoid_42', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1769472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 144) % 1536
    x4 = (xindex // 144)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp2 = tmp0 + tmp1
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tmp7 = 2.0
    tmp8 = tmp6 * tmp7
    tmp11 = tmp8 * tmp10
    tmp12 = 0.2
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13 + tmp2
    tmp15 = 0.5
    tmp16 = tmp14 * tmp15
    tmp17 = 0.7071067811865476
    tmp18 = tmp14 * tmp17
    tmp19 = tl.math.erf(tmp18)
    tmp20 = 1.0
    tmp21 = tmp19 + tmp20
    tmp22 = tmp16 * tmp21
    tmp23 = 1.7015043497085571
    tmp24 = tmp22 * tmp23
    tmp25 = 0.9805806756909201
    tmp26 = tmp24 * tmp25
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp26, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4q/c4qnwqqiwtybypiag2dnrihajfomvxuqlcqm4ubh74bvwphbit4o.py
# Source Nodes: [batch_norm_23, weight_23], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_23 => add_47, mul_171, mul_172, rsqrt_23, squeeze_47, sub_23, var_mean_23
# weight_23 => view_71
triton_red_fused__native_batch_norm_legit_view_43 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_view_43', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp13 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tmp5 - tmp2
        tmp7 = 1536.0
        tmp8 = tmp3 / tmp7
        tmp9 = 1e-05
        tmp10 = tmp8 + tmp9
        tmp11 = tl.math.rsqrt(tmp10)
        tmp12 = tmp6 * tmp11
        tmp14 = 0.02551551815399144
        tmp15 = tmp13 * tmp14
        tmp16 = tmp12 * tmp15
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp16, rmask & xmask)
    tmp17 = 1536.0
    tmp18 = tmp3 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tl.store(out_ptr3 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xk/cxkbt5ie62vgsx37ergbp3xqox6qt3iu7dowhd72pet3ayu2djzi.py
# Source Nodes: [gelu_23, mul_36, mul_38, mul_44, mul_46, mul__22, mul__27, mul__28, out_29, out_37, out_40, shortcut_7, shortcut_8, sigmoid_3, sigmoid_4], Original ATen: [aten.add, aten.gelu, aten.mul, aten.sigmoid]
# gelu_23 => add_55, erf_23, mul_198, mul_199, mul_200
# mul_36 => mul_161
# mul_38 => mul_164
# mul_44 => mul_194
# mul_46 => mul_197
# mul__22 => mul_163
# mul__27 => mul_196
# mul__28 => mul_201
# out_29 => mul_162
# out_37 => mul_195
# out_40 => mul_202
# shortcut_7 => add_45
# shortcut_8 => add_54
# sigmoid_3 => sigmoid_3
# sigmoid_4 => sigmoid_4
triton_poi_fused_add_gelu_mul_sigmoid_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_mul_sigmoid_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1769472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 144)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp11 = tl.load(in_ptr3 + (x2), None)
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (0))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp20 = tl.load(in_ptr6 + (x2), None)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp8 = tmp5 * tmp7
    tmp9 = 0.2
    tmp10 = tmp8 * tmp9
    tmp13 = tl.sigmoid(tmp12)
    tmp14 = tmp11 * tmp13
    tmp15 = tmp14 * tmp4
    tmp18 = tmp15 * tmp17
    tmp19 = tmp18 * tmp9
    tmp21 = tmp19 + tmp20
    tmp22 = tmp10 + tmp21
    tmp23 = 0.5
    tmp24 = tmp22 * tmp23
    tmp25 = 0.7071067811865476
    tmp26 = tmp22 * tmp25
    tmp27 = tl.math.erf(tmp26)
    tmp28 = 1.0
    tmp29 = tmp27 + tmp28
    tmp30 = tmp24 * tmp29
    tmp31 = 1.7015043497085571
    tmp32 = tmp30 * tmp31
    tmp33 = 0.9622504486493761
    tmp34 = tmp32 * tmp33
    tl.store(out_ptr0 + (x2), tmp22, None)
    tl.store(out_ptr1 + (x2), tmp34, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/m5/cm5562eyrxqqe4rmvbyf3bqak7tbhzbqzb5a74dn67zdicilgtxh.py
# Source Nodes: [gelu_27, mul_52, mul_54, mul__32, mul__33, out_45, out_48, shortcut_9, sigmoid_5], Original ATen: [aten.add, aten.gelu, aten.mul, aten.sigmoid]
# gelu_27 => add_64, erf_27, mul_231, mul_232, mul_233
# mul_52 => mul_227
# mul_54 => mul_230
# mul__32 => mul_229
# mul__33 => mul_234
# out_45 => mul_228
# out_48 => mul_235
# shortcut_9 => add_63
# sigmoid_5 => sigmoid_5
triton_poi_fused_add_gelu_mul_sigmoid_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_mul_sigmoid_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1769472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 144)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp11 = tl.load(in_ptr3 + (x2), None)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp8 = tmp5 * tmp7
    tmp9 = 0.2
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = 0.5
    tmp14 = tmp12 * tmp13
    tmp15 = 0.7071067811865476
    tmp16 = tmp12 * tmp15
    tmp17 = tl.math.erf(tmp16)
    tmp18 = 1.0
    tmp19 = tmp17 + tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = 1.7015043497085571
    tmp22 = tmp20 * tmp21
    tmp23 = 0.9449111825230679
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x2), tmp24, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/v7/cv7z326k6t4nrju24y6sylntvb37p24scc3l7gd7scrwb3cgipns.py
# Source Nodes: [gelu_31, mul_52, mul_54, mul_60, mul_62, mul__32, mul__37, mul__38, out_45, out_53, out_56, shortcut_10, shortcut_9, sigmoid_5, sigmoid_6], Original ATen: [aten.add, aten.gelu, aten.mul, aten.sigmoid]
# gelu_31 => add_73, erf_31, mul_264, mul_265, mul_266
# mul_52 => mul_227
# mul_54 => mul_230
# mul_60 => mul_260
# mul_62 => mul_263
# mul__32 => mul_229
# mul__37 => mul_262
# mul__38 => mul_267
# out_45 => mul_228
# out_53 => mul_261
# out_56 => mul_268
# shortcut_10 => add_72
# shortcut_9 => add_63
# sigmoid_5 => sigmoid_5
# sigmoid_6 => sigmoid_6
triton_poi_fused_add_gelu_mul_sigmoid_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_mul_sigmoid_46', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1769472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 144)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp11 = tl.load(in_ptr3 + (x2), None)
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (0))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp20 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp8 = tmp5 * tmp7
    tmp9 = 0.2
    tmp10 = tmp8 * tmp9
    tmp13 = tl.sigmoid(tmp12)
    tmp14 = tmp11 * tmp13
    tmp15 = tmp14 * tmp4
    tmp18 = tmp15 * tmp17
    tmp19 = tmp18 * tmp9
    tmp21 = tmp19 + tmp20
    tmp22 = tmp10 + tmp21
    tmp23 = 0.5
    tmp24 = tmp22 * tmp23
    tmp25 = 0.7071067811865476
    tmp26 = tmp22 * tmp25
    tmp27 = tl.math.erf(tmp26)
    tmp28 = 1.0
    tmp29 = tmp27 + tmp28
    tmp30 = tmp24 * tmp29
    tmp31 = 1.7015043497085571
    tmp32 = tmp30 * tmp31
    tmp33 = 0.9284766908852592
    tmp34 = tmp32 * tmp33
    tl.store(in_out_ptr0 + (x2), tmp22, None)
    tl.store(out_ptr0 + (x2), tmp34, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xf/cxft52yao4v6mwoo2paaphsffvokqazuxaensix4ef5wk2v5v3jq.py
# Source Nodes: [gelu_35, mul_68, mul_70, mul__42, mul__43, out_61, out_64, shortcut_11, sigmoid_7], Original ATen: [aten.add, aten.gelu, aten.mul, aten.sigmoid]
# gelu_35 => add_82, erf_35, mul_297, mul_298, mul_299
# mul_68 => mul_293
# mul_70 => mul_296
# mul__42 => mul_295
# mul__43 => mul_300
# out_61 => mul_294
# out_64 => mul_301
# shortcut_11 => add_81
# sigmoid_7 => sigmoid_7
triton_poi_fused_add_gelu_mul_sigmoid_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_mul_sigmoid_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1769472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 144)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp11 = tl.load(in_ptr3 + (x2), None)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp8 = tmp5 * tmp7
    tmp9 = 0.2
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = 0.5
    tmp14 = tmp12 * tmp13
    tmp15 = 0.7071067811865476
    tmp16 = tmp12 * tmp15
    tmp17 = tl.math.erf(tmp16)
    tmp18 = 1.0
    tmp19 = tmp17 + tmp18
    tmp20 = tmp14 * tmp19
    tmp21 = 1.7015043497085571
    tmp22 = tmp20 * tmp21
    tmp23 = 0.9128709291752768
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr0 + (x2), tmp24, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/aq/caqu4f5qyniyaa2xayjjlslz66yhuomvqbsiekfzhld3lhw3v5xi.py
# Source Nodes: [gelu_39, mul_68, mul_70, mul_76, mul_78, mul__42, mul__47, mul__48, out_61, out_69, out_72, shortcut_11, shortcut_12, sigmoid_7, sigmoid_8], Original ATen: [aten.add, aten.gelu, aten.mul, aten.sigmoid]
# gelu_39 => add_91, erf_39, mul_330, mul_331, mul_332
# mul_68 => mul_293
# mul_70 => mul_296
# mul_76 => mul_326
# mul_78 => mul_329
# mul__42 => mul_295
# mul__47 => mul_328
# mul__48 => mul_333
# out_61 => mul_294
# out_69 => mul_327
# out_72 => mul_334
# shortcut_11 => add_81
# shortcut_12 => add_90
# sigmoid_7 => sigmoid_7
# sigmoid_8 => sigmoid_8
triton_poi_fused_add_gelu_mul_sigmoid_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_mul_sigmoid_48', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1769472
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 144)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp11 = tl.load(in_ptr3 + (x2), None)
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (0))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp20 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp8 = tmp5 * tmp7
    tmp9 = 0.2
    tmp10 = tmp8 * tmp9
    tmp13 = tl.sigmoid(tmp12)
    tmp14 = tmp11 * tmp13
    tmp15 = tmp14 * tmp4
    tmp18 = tmp15 * tmp17
    tmp19 = tmp18 * tmp9
    tmp21 = tmp19 + tmp20
    tmp22 = tmp10 + tmp21
    tmp23 = 0.5
    tmp24 = tmp22 * tmp23
    tmp25 = 0.7071067811865476
    tmp26 = tmp22 * tmp25
    tmp27 = tl.math.erf(tmp26)
    tmp28 = 1.0
    tmp29 = tmp27 + tmp28
    tmp30 = tmp24 * tmp29
    tmp31 = 1.7015043497085571
    tmp32 = tmp30 * tmp31
    tmp33 = 0.8980265101338745
    tmp34 = tmp32 * tmp33
    tl.store(in_out_ptr0 + (x2), tmp34, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/to/ctoigfbxf5mta2x34a2dcjnjl5p5enjmye7jladq3fzefivwsyb5.py
# Source Nodes: [getattr_getattr_l__mod___stages___3_____0___downsample_pool], Original ATen: [aten.avg_pool2d]
# getattr_getattr_l__mod___stages___3_____0___downsample_pool => avg_pool2d_2
triton_poi_fused_avg_pool2d_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 442368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6
    x1 = (xindex // 6)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (24*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (24*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (12 + (2*x0) + (24*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (13 + (2*x0) + (24*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3v/c3vkvxzl7pnhxtlohfuef65j7iniwsy6y6kiecfgbezre4hri55y.py
# Source Nodes: [batch_norm_43, weight_43], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_43 => add_92, mul_336, mul_337, rsqrt_43, squeeze_87, sub_43, var_mean_43
# weight_43 => view_131
triton_red_fused__native_batch_norm_legit_view_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_view_50', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp13 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tmp5 - tmp2
        tmp7 = 1536.0
        tmp8 = tmp3 / tmp7
        tmp9 = 1e-05
        tmp10 = tmp8 + tmp9
        tmp11 = tl.math.rsqrt(tmp10)
        tmp12 = tmp6 * tmp11
        tmp14 = 0.02551551815399144
        tmp15 = tmp13 * tmp14
        tmp16 = tmp12 * tmp15
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp16, rmask & xmask)
    tmp17 = 1536.0
    tmp18 = tmp3 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tl.store(out_ptr3 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lw/clwskuhl67aq5ym2cwp7ba5gzsznqhqbfvbov66kxgiuqd5matns.py
# Source Nodes: [out_73], Original ATen: [aten.convolution]
# out_73 => convolution_62
triton_poi_fused_convolution_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_51', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 884736
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 144) % 768
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7b/c7b6egw44ogmpwhnnmsgolpgh5kaeypenla5ce3zu3lm42rsc4h7.py
# Source Nodes: [gelu_40, mul__49, x_9], Original ATen: [aten.constant_pad_nd, aten.gelu, aten.mul]
# gelu_40 => add_94, erf_40, mul_341, mul_342, mul_343
# mul__49 => mul_344
# x_9 => constant_pad_nd_4
triton_poi_fused_constant_pad_nd_gelu_mul_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_gelu_mul_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1038336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 13) % 13
    x0 = xindex % 13
    x2 = (xindex // 169)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 12, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (12*x1) + (144*x2)), tmp5, other=0.0)
    tmp7 = 0.5
    tmp8 = tmp6 * tmp7
    tmp9 = 0.7071067811865476
    tmp10 = tmp6 * tmp9
    tmp11 = tl.math.erf(tmp10)
    tmp12 = 1.0
    tmp13 = tmp11 + tmp12
    tmp14 = tmp8 * tmp13
    tmp15 = 1.7015043497085571
    tmp16 = tmp14 * tmp15
    tmp17 = tl.full(tmp16.shape, 0.0, tmp16.dtype)
    tmp18 = tl.where(tmp5, tmp16, tmp17)
    tl.store(out_ptr0 + (x3), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uc/cucnozdecef55sjt6nh3isb32wwuuiwgwpvqiggiklgh5zxquxo4.py
# Source Nodes: [gelu_41, mul__50, out_74], Original ATen: [aten.convolution, aten.gelu, aten.mul]
# gelu_41 => add_96, erf_41, mul_348, mul_349, mul_350
# mul__50 => mul_351
# out_74 => convolution_63
triton_poi_fused_convolution_gelu_mul_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_mul_53', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 221184
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 36) % 768
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = 1.7015043497085571
    tmp12 = tmp10 * tmp11
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/u5/cu5kp4bealqepaf3i7l3jf23hrcoycqqxn45jtm7lgtxnwszmddn.py
# Source Nodes: [out_76, x_se_36], Original ATen: [aten.convolution, aten.mean]
# out_76 => convolution_65
# x_se_36 => mean_9
triton_per_fused_convolution_mean_54 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_mean_54', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 36
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1536
    tmp0 = tl.load(in_out_ptr0 + (r2 + (36*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 36.0
    tmp8 = tmp6 / tmp7
    tl.store(in_out_ptr0 + (r2 + (36*x3)), tmp2, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3s/c3sjwinfsvlkgq7eyyeg65oerdsbfirrerg5jf2dmzkzq67smaxw.py
# Source Nodes: [gelu_43, mul_85, mul_87, mul__52, mul__53, out_77, out_80, shortcut_13, shortcut_14, sigmoid_9], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.sigmoid]
# gelu_43 => add_101, erf_43, mul_366, mul_367, mul_368
# mul_85 => mul_362
# mul_87 => mul_365
# mul__52 => mul_364
# mul__53 => mul_369
# out_77 => mul_363
# out_80 => mul_370
# shortcut_13 => convolution_61
# shortcut_14 => add_100
# sigmoid_9 => sigmoid_9
triton_poi_fused_add_convolution_gelu_mul_sigmoid_55 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mul_sigmoid_55', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 442368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 36) % 1536
    x4 = (xindex // 36)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x3), None)
    tmp4 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr3 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK])
    tmp2 = tmp0 + tmp1
    tmp5 = tl.sigmoid(tmp4)
    tmp6 = tmp3 * tmp5
    tmp7 = 2.0
    tmp8 = tmp6 * tmp7
    tmp11 = tmp8 * tmp10
    tmp12 = 0.2
    tmp13 = tmp11 * tmp12
    tmp14 = tmp13 + tmp2
    tmp15 = 0.5
    tmp16 = tmp14 * tmp15
    tmp17 = 0.7071067811865476
    tmp18 = tmp14 * tmp17
    tmp19 = tl.math.erf(tmp18)
    tmp20 = 1.0
    tmp21 = tmp19 + tmp20
    tmp22 = tmp16 * tmp21
    tmp23 = 1.7015043497085571
    tmp24 = tmp22 * tmp23
    tmp25 = 0.9805806756909201
    tmp26 = tmp24 * tmp25
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp26, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pq/cpqj4fgkgxndhaowufqkylap2l36kwluba3wmzljgj4idneaa3pt.py
# Source Nodes: [gelu_47, mul_85, mul_87, mul_93, mul_95, mul__52, mul__57, mul__58, out_77, out_85, out_88, shortcut_14, shortcut_15, sigmoid_10, sigmoid_9], Original ATen: [aten.add, aten.gelu, aten.mul, aten.sigmoid]
# gelu_47 => add_110, erf_47, mul_399, mul_400, mul_401
# mul_85 => mul_362
# mul_87 => mul_365
# mul_93 => mul_395
# mul_95 => mul_398
# mul__52 => mul_364
# mul__57 => mul_397
# mul__58 => mul_402
# out_77 => mul_363
# out_85 => mul_396
# out_88 => mul_403
# shortcut_14 => add_100
# shortcut_15 => add_109
# sigmoid_10 => sigmoid_10
# sigmoid_9 => sigmoid_9
triton_poi_fused_add_gelu_mul_sigmoid_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_gelu_mul_sigmoid_56', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 442368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 36)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp11 = tl.load(in_ptr3 + (x2), None)
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr5 + (0))
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK])
    tmp20 = tl.load(in_ptr6 + (x2), None)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp8 = tmp5 * tmp7
    tmp9 = 0.2
    tmp10 = tmp8 * tmp9
    tmp13 = tl.sigmoid(tmp12)
    tmp14 = tmp11 * tmp13
    tmp15 = tmp14 * tmp4
    tmp18 = tmp15 * tmp17
    tmp19 = tmp18 * tmp9
    tmp21 = tmp19 + tmp20
    tmp22 = tmp10 + tmp21
    tmp23 = 0.5
    tmp24 = tmp22 * tmp23
    tmp25 = 0.7071067811865476
    tmp26 = tmp22 * tmp25
    tmp27 = tl.math.erf(tmp26)
    tmp28 = 1.0
    tmp29 = tmp27 + tmp28
    tmp30 = tmp24 * tmp29
    tmp31 = 1.7015043497085571
    tmp32 = tmp30 * tmp31
    tmp33 = 0.9622504486493761
    tmp34 = tmp32 * tmp33
    tl.store(out_ptr0 + (x2), tmp22, None)
    tl.store(out_ptr1 + (x2), tmp34, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6x/c6xlckilrklcttqtvikbmwm4wgm5qvfs7rv2qjs3zhiowlioozvv.py
# Source Nodes: [mul_101, mul_103, mul__62, out_93, sigmoid_11, x_10], Original ATen: [aten.add, aten.mul, aten.sigmoid]
# mul_101 => mul_428
# mul_103 => mul_431
# mul__62 => mul_430
# out_93 => mul_429
# sigmoid_11 => sigmoid_11
# x_10 => add_118
triton_poi_fused_add_mul_sigmoid_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_57', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 442368
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 36)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr2 + (0))
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK])
    tmp11 = tl.load(in_out_ptr0 + (x2), None)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp8 = tmp5 * tmp7
    tmp9 = 0.2
    tmp10 = tmp8 * tmp9
    tmp12 = tmp10 + tmp11
    tl.store(in_out_ptr0 + (x2), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6l/c6lmnu22vfykqytnzfkuh55ugtqcrikxaodvta6bzauvozhjkq3v.py
# Source Nodes: [batch_norm_56, weight_56], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_56 => add_119, mul_433, mul_434, rsqrt_56, squeeze_113, sub_56, var_mean_56
# weight_56 => view_170
triton_red_fused__native_batch_norm_legit_view_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4096, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_view_58', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 1536
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask & xmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask & xmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask & xmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x0), tmp2, xmask)
    tmp13 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr0 + (r1 + (1536*x0)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp6 = tmp5 - tmp2
        tmp7 = 1536.0
        tmp8 = tmp3 / tmp7
        tmp9 = 1e-05
        tmp10 = tmp8 + tmp9
        tmp11 = tl.math.rsqrt(tmp10)
        tmp12 = tmp6 * tmp11
        tmp14 = 0.02551551815399144
        tmp15 = tmp13 * tmp14
        tmp16 = tmp12 * tmp15
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp16, rmask & xmask)
    tmp17 = 1536.0
    tmp18 = tmp3 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tl.store(out_ptr3 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i2/ci2655pjspnxfzumuctf3233s35t2627igivd5e6zaxz4qcit23d.py
# Source Nodes: [gelu_51, x_11, x_13, x_14, x_16], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul, aten.view]
# gelu_51 => add_120, erf_51, mul_435, mul_436, mul_437
# x_11 => convolution_80
# x_13 => mul_438
# x_14 => mean_12
# x_16 => view_171
triton_per_fused_convolution_gelu_mean_mul_view_59 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_gelu_mean_mul_view_59', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 36
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 3072
    tmp0 = tl.load(in_out_ptr0 + (r2 + (36*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 0.5
    tmp4 = tmp2 * tmp3
    tmp5 = 0.7071067811865476
    tmp6 = tmp2 * tmp5
    tmp7 = tl.math.erf(tmp6)
    tmp8 = 1.0
    tmp9 = tmp7 + tmp8
    tmp10 = tmp4 * tmp9
    tmp11 = 1.7015043497085571
    tmp12 = tmp10 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 36.0
    tmp18 = tmp16 / tmp17
    tl.store(in_out_ptr0 + (r2 + (36*x3)), tmp2, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp18, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234 = args
    args.clear()
    assert_size_stride(primals_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (16, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_3, (16, ), (1, ))
    assert_size_stride(primals_4, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_5, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_6, (32, ), (1, ))
    assert_size_stride(primals_7, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_8, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_9, (64, ), (1, ))
    assert_size_stride(primals_10, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_11, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_14, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_16, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_17, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_18, (128, ), (1, ))
    assert_size_stride(primals_19, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_20, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_22, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_23, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_24, (128, ), (1, ))
    assert_size_stride(primals_25, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_26, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_27, (256, ), (1, ))
    assert_size_stride(primals_28, (), ())
    assert_size_stride(primals_29, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_30, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_31, (512, ), (1, ))
    assert_size_stride(primals_32, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_33, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_34, (256, ), (1, ))
    assert_size_stride(primals_35, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_36, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_37, (256, ), (1, ))
    assert_size_stride(primals_38, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_39, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_40, (256, ), (1, ))
    assert_size_stride(primals_41, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_42, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_43, (512, ), (1, ))
    assert_size_stride(primals_44, (), ())
    assert_size_stride(primals_45, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_46, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_47, (256, ), (1, ))
    assert_size_stride(primals_48, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_49, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_50, (256, ), (1, ))
    assert_size_stride(primals_51, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_52, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_53, (256, ), (1, ))
    assert_size_stride(primals_54, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_55, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_56, (512, ), (1, ))
    assert_size_stride(primals_57, (), ())
    assert_size_stride(primals_58, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_59, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_60, (1536, ), (1, ))
    assert_size_stride(primals_61, (768, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_62, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_63, (768, ), (1, ))
    assert_size_stride(primals_64, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_65, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_66, (768, ), (1, ))
    assert_size_stride(primals_67, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_68, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_69, (768, ), (1, ))
    assert_size_stride(primals_70, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_71, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_72, (1536, ), (1, ))
    assert_size_stride(primals_73, (), ())
    assert_size_stride(primals_74, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_75, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_76, (768, ), (1, ))
    assert_size_stride(primals_77, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_78, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_79, (768, ), (1, ))
    assert_size_stride(primals_80, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_81, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_82, (768, ), (1, ))
    assert_size_stride(primals_83, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_84, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_85, (1536, ), (1, ))
    assert_size_stride(primals_86, (), ())
    assert_size_stride(primals_87, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_88, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_89, (768, ), (1, ))
    assert_size_stride(primals_90, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_91, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_92, (768, ), (1, ))
    assert_size_stride(primals_93, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_94, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_95, (768, ), (1, ))
    assert_size_stride(primals_96, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_97, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_98, (1536, ), (1, ))
    assert_size_stride(primals_99, (), ())
    assert_size_stride(primals_100, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_101, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_102, (768, ), (1, ))
    assert_size_stride(primals_103, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_104, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_105, (768, ), (1, ))
    assert_size_stride(primals_106, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_107, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_108, (768, ), (1, ))
    assert_size_stride(primals_109, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_110, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_111, (1536, ), (1, ))
    assert_size_stride(primals_112, (), ())
    assert_size_stride(primals_113, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_114, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_115, (768, ), (1, ))
    assert_size_stride(primals_116, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_117, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_118, (768, ), (1, ))
    assert_size_stride(primals_119, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_120, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_121, (768, ), (1, ))
    assert_size_stride(primals_122, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_123, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_124, (1536, ), (1, ))
    assert_size_stride(primals_125, (), ())
    assert_size_stride(primals_126, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_127, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_128, (768, ), (1, ))
    assert_size_stride(primals_129, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_130, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_131, (768, ), (1, ))
    assert_size_stride(primals_132, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_133, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_134, (768, ), (1, ))
    assert_size_stride(primals_135, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_136, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_137, (1536, ), (1, ))
    assert_size_stride(primals_138, (), ())
    assert_size_stride(primals_139, (1536, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_140, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_141, (1536, ), (1, ))
    assert_size_stride(primals_142, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_143, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_144, (768, ), (1, ))
    assert_size_stride(primals_145, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_146, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_147, (768, ), (1, ))
    assert_size_stride(primals_148, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_149, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_150, (768, ), (1, ))
    assert_size_stride(primals_151, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_152, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_153, (1536, ), (1, ))
    assert_size_stride(primals_154, (), ())
    assert_size_stride(primals_155, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_156, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_157, (768, ), (1, ))
    assert_size_stride(primals_158, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_159, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_160, (768, ), (1, ))
    assert_size_stride(primals_161, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_162, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_163, (768, ), (1, ))
    assert_size_stride(primals_164, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_165, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_166, (1536, ), (1, ))
    assert_size_stride(primals_167, (), ())
    assert_size_stride(primals_168, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_169, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_170, (768, ), (1, ))
    assert_size_stride(primals_171, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_172, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_173, (768, ), (1, ))
    assert_size_stride(primals_174, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_175, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_176, (768, ), (1, ))
    assert_size_stride(primals_177, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_178, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_179, (1536, ), (1, ))
    assert_size_stride(primals_180, (), ())
    assert_size_stride(primals_181, (3072, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_182, (3072, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_183, (3072, ), (1, ))
    assert_size_stride(primals_184, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_185, (128, ), (1, ))
    assert_size_stride(primals_186, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_187, (256, ), (1, ))
    assert_size_stride(primals_188, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_189, (256, ), (1, ))
    assert_size_stride(primals_190, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_191, (512, ), (1, ))
    assert_size_stride(primals_192, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_193, (256, ), (1, ))
    assert_size_stride(primals_194, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_195, (512, ), (1, ))
    assert_size_stride(primals_196, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_197, (768, ), (1, ))
    assert_size_stride(primals_198, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_199, (1536, ), (1, ))
    assert_size_stride(primals_200, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_201, (768, ), (1, ))
    assert_size_stride(primals_202, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_203, (1536, ), (1, ))
    assert_size_stride(primals_204, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_205, (768, ), (1, ))
    assert_size_stride(primals_206, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_207, (1536, ), (1, ))
    assert_size_stride(primals_208, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_209, (768, ), (1, ))
    assert_size_stride(primals_210, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_211, (1536, ), (1, ))
    assert_size_stride(primals_212, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_213, (768, ), (1, ))
    assert_size_stride(primals_214, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_215, (1536, ), (1, ))
    assert_size_stride(primals_216, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_217, (768, ), (1, ))
    assert_size_stride(primals_218, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_219, (1536, ), (1, ))
    assert_size_stride(primals_220, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_221, (768, ), (1, ))
    assert_size_stride(primals_222, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_223, (1536, ), (1, ))
    assert_size_stride(primals_224, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_225, (768, ), (1, ))
    assert_size_stride(primals_226, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_227, (1536, ), (1, ))
    assert_size_stride(primals_228, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_229, (768, ), (1, ))
    assert_size_stride(primals_230, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_231, (1536, ), (1, ))
    assert_size_stride(primals_232, (1000, 3072), (3072, 1))
    assert_size_stride(primals_233, (1000, ), (1, ))
    assert_size_stride(primals_234, (8, 3, 192, 192), (110592, 36864, 192, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty((8, 3, 193, 193), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.constant_pad_nd]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_constant_pad_nd_0.run(primals_234, buf0, 893976, grid=grid(893976), stream=stream0)
        del primals_234
        buf1 = empty_strided((1, 16, 1), (16, 1, 16), device='cuda', dtype=torch.float32)
        buf5 = empty((16, 3, 3, 3), device='cuda', dtype=torch.float32)
        buf4 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm, weight], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_1.run(primals_1, primals_2, buf1, buf5, buf4, 16, 27, grid=grid(16), stream=stream0)
        # Source Nodes: [conv2d], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf0, buf5, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 16, 96, 96), (147456, 9216, 96, 1))
        buf7 = buf6; del buf6  # reuse
        buf8 = empty((8, 16, 96, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv2d, gelu, mul_], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_2.run(buf7, primals_3, buf8, 1179648, grid=grid(1179648), stream=stream0)
        del primals_3
        buf9 = empty_strided((1, 32, 1), (32, 1, 32), device='cuda', dtype=torch.float32)
        buf13 = empty((32, 16, 3, 3), device='cuda', dtype=torch.float32)
        buf12 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_1, weight_1], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_3.run(primals_4, primals_5, buf9, buf13, buf12, 32, 144, grid=grid(32), stream=stream0)
        # Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf8, buf13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (8, 32, 96, 96), (294912, 9216, 96, 1))
        buf15 = buf14; del buf14  # reuse
        buf16 = empty((8, 32, 96, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv2d_1, gelu_1, mul__1], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_4.run(buf15, primals_6, buf16, 2359296, grid=grid(2359296), stream=stream0)
        del primals_6
        buf17 = empty_strided((1, 64, 1), (64, 1, 64), device='cuda', dtype=torch.float32)
        buf21 = empty((64, 32, 3, 3), device='cuda', dtype=torch.float32)
        buf20 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_2, weight_2], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_5.run(primals_7, primals_8, buf17, buf21, buf20, 64, 288, grid=grid(64), stream=stream0)
        # Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf16, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (8, 64, 96, 96), (589824, 9216, 96, 1))
        buf23 = buf22; del buf22  # reuse
        # Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_6.run(buf23, primals_9, 4718592, grid=grid(4718592), stream=stream0)
        del primals_9
        buf24 = empty((8, 64, 97, 97), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_2, mul__2, x_2], Original ATen: [aten.constant_pad_nd, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_gelu_mul_7.run(buf23, buf24, 4817408, grid=grid(4817408), stream=stream0)
        buf25 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf29 = empty((128, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf28 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_3, weight_3], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_8.run(primals_10, primals_11, buf25, buf29, buf28, 128, 576, grid=grid(128), stream=stream0)
        # Source Nodes: [shortcut], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf24, buf29, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 128, 48, 48), (294912, 2304, 48, 1))
        buf31 = buf30; del buf30  # reuse
        buf32 = empty((8, 128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_3, mul__3, out, shortcut], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_9.run(buf31, primals_12, buf32, 2359296, grid=grid(2359296), stream=stream0)
        del primals_12
        buf33 = empty_strided((1, 256, 1), (256, 1, 256), device='cuda', dtype=torch.float32)
        buf37 = empty((256, 128, 1, 1), device='cuda', dtype=torch.float32)
        buf36 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_4, weight_4], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_10.run(primals_13, primals_14, buf33, buf37, buf36, 256, 128, grid=grid(256), stream=stream0)
        # Source Nodes: [shortcut_1], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf32, buf37, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 256, 48, 48), (589824, 2304, 48, 1))
        buf40 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf44 = empty((128, 128, 1, 1), device='cuda', dtype=torch.float32)
        buf43 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_5, weight_5], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_11.run(primals_16, primals_17, buf40, buf44, buf43, 128, 128, grid=grid(128), stream=stream0)
        # Source Nodes: [out_1], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf32, buf44, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (8, 128, 48, 48), (294912, 2304, 48, 1))
        buf46 = buf45; del buf45  # reuse
        buf47 = empty((8, 128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_4, mul__4, out_1], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_12.run(buf46, primals_18, buf47, 2359296, grid=grid(2359296), stream=stream0)
        del primals_18
        buf48 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf52 = empty((128, 128, 3, 3), device='cuda', dtype=torch.float32)
        buf51 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_6, weight_6], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_13.run(primals_19, primals_20, buf48, buf52, buf51, 128, 1152, grid=grid(128), stream=stream0)
        # Source Nodes: [out_2], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf47, buf52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (8, 128, 48, 48), (294912, 2304, 48, 1))
        buf54 = buf53; del buf53  # reuse
        buf55 = empty((8, 128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_5, mul__5, out_2], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_12.run(buf54, primals_21, buf55, 2359296, grid=grid(2359296), stream=stream0)
        del primals_21
        buf56 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf60 = empty((128, 128, 3, 3), device='cuda', dtype=torch.float32)
        buf59 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_7, weight_7], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_13.run(primals_22, primals_23, buf56, buf60, buf59, 128, 1152, grid=grid(128), stream=stream0)
        # Source Nodes: [out_3], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf55, buf60, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (8, 128, 48, 48), (294912, 2304, 48, 1))
        buf62 = buf61; del buf61  # reuse
        buf63 = empty((8, 128, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_6, mul__6, out_3], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_12.run(buf62, primals_24, buf63, 2359296, grid=grid(2359296), stream=stream0)
        del primals_24
        buf64 = empty_strided((1, 256, 1), (256, 1, 256), device='cuda', dtype=torch.float32)
        buf68 = empty((256, 128, 1, 1), device='cuda', dtype=torch.float32)
        buf67 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_8, weight_8], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_10.run(primals_25, primals_26, buf64, buf68, buf67, 256, 128, grid=grid(256), stream=stream0)
        # Source Nodes: [out_4], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf63, buf68, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (8, 256, 48, 48), (589824, 2304, 48, 1))
        buf70 = buf69; del buf69  # reuse
        buf71 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf72 = reinterpret_tensor(buf71, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf71  # reuse
        # Source Nodes: [out_4, x_se], Original ATen: [aten.convolution, aten.mean]
        triton_red_fused_convolution_mean_14.run(buf70, buf72, primals_27, 2048, 2304, grid=grid(2048), stream=stream0)
        del primals_27
        # Source Nodes: [x_se_1], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, primals_184, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (8, 128, 1, 1), (128, 1, 1, 1))
        buf74 = buf73; del buf73  # reuse
        # Source Nodes: [x_se_1, x_se_2], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_15.run(buf74, primals_185, 1024, grid=grid(1024), stream=stream0)
        del primals_185
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, primals_186, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (8, 256, 1, 1), (256, 1, 1, 1))
        buf76 = buf75; del buf75  # reuse
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_16.run(buf76, primals_187, 2048, grid=grid(2048), stream=stream0)
        del primals_187
        buf39 = buf38; del buf38  # reuse
        buf77 = empty((8, 256, 48, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_7, mul_10, mul_12, mul__7, mul__8, out_5, out_8, shortcut_1, shortcut_2, sigmoid], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.sigmoid]
        triton_poi_fused_add_convolution_gelu_mul_sigmoid_17.run(buf39, primals_15, buf70, buf76, primals_28, buf77, 4718592, grid=grid(4718592), stream=stream0)
        del primals_15
        buf78 = empty((8, 256, 24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___downsample_pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_18.run(buf77, buf78, 1179648, grid=grid(1179648), stream=stream0)
        buf79 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf83 = empty((512, 256, 1, 1), device='cuda', dtype=torch.float32)
        buf82 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_9, weight_9], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_19.run(primals_29, primals_30, buf79, buf83, buf82, 512, 256, grid=grid(512), stream=stream0)
        # Source Nodes: [shortcut_3], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf78, buf83, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 512, 24, 24), (294912, 576, 24, 1))
        buf86 = empty_strided((1, 256, 1), (256, 1, 256), device='cuda', dtype=torch.float32)
        buf90 = empty((256, 256, 1, 1), device='cuda', dtype=torch.float32)
        buf89 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_10, weight_10], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_20.run(primals_32, primals_33, buf86, buf90, buf89, 256, 256, grid=grid(256), stream=stream0)
        # Source Nodes: [out_9], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf77, buf90, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (8, 256, 48, 48), (589824, 2304, 48, 1))
        buf92 = buf91; del buf91  # reuse
        # Source Nodes: [out_9], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_21.run(buf92, primals_34, 4718592, grid=grid(4718592), stream=stream0)
        del primals_34
        buf93 = empty((8, 256, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_8, mul__9, x_5], Original ATen: [aten.constant_pad_nd, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_gelu_mul_22.run(buf92, buf93, 4917248, grid=grid(4917248), stream=stream0)
        buf94 = empty_strided((1, 256, 1), (256, 1, 256), device='cuda', dtype=torch.float32)
        buf98 = empty((256, 128, 3, 3), device='cuda', dtype=torch.float32)
        buf97 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_11, weight_11], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_23.run(primals_35, primals_36, buf94, buf98, buf97, 256, 1152, grid=grid(256), stream=stream0)
        # Source Nodes: [out_10], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf93, buf98, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf99, (8, 256, 24, 24), (147456, 576, 24, 1))
        buf100 = buf99; del buf99  # reuse
        buf101 = empty((8, 256, 24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_9, mul__10, out_10], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_24.run(buf100, primals_37, buf101, 1179648, grid=grid(1179648), stream=stream0)
        del primals_37
        buf102 = empty_strided((1, 256, 1), (256, 1, 256), device='cuda', dtype=torch.float32)
        buf106 = empty((256, 128, 3, 3), device='cuda', dtype=torch.float32)
        buf105 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_12, weight_12], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_23.run(primals_38, primals_39, buf102, buf106, buf105, 256, 1152, grid=grid(256), stream=stream0)
        # Source Nodes: [out_11], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf101, buf106, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf107, (8, 256, 24, 24), (147456, 576, 24, 1))
        buf108 = buf107; del buf107  # reuse
        buf109 = empty((8, 256, 24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_10, mul__11, out_11], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_24.run(buf108, primals_40, buf109, 1179648, grid=grid(1179648), stream=stream0)
        del primals_40
        buf110 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf114 = empty((512, 256, 1, 1), device='cuda', dtype=torch.float32)
        buf113 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_13, weight_13], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_19.run(primals_41, primals_42, buf110, buf114, buf113, 512, 256, grid=grid(512), stream=stream0)
        # Source Nodes: [out_12], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf109, buf114, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (8, 512, 24, 24), (294912, 576, 24, 1))
        buf116 = buf115; del buf115  # reuse
        buf117 = empty_strided((8, 512, 1, 1), (512, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf118 = reinterpret_tensor(buf117, (8, 512, 1, 1), (512, 1, 1, 1), 0); del buf117  # reuse
        # Source Nodes: [out_12, x_se_4], Original ATen: [aten.convolution, aten.mean]
        triton_per_fused_convolution_mean_25.run(buf116, buf118, primals_43, 4096, 576, grid=grid(4096), stream=stream0)
        del primals_43
        # Source Nodes: [x_se_5], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf118, primals_188, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (8, 256, 1, 1), (256, 1, 1, 1))
        buf120 = buf119; del buf119  # reuse
        # Source Nodes: [x_se_5, x_se_6], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_26.run(buf120, primals_189, 2048, grid=grid(2048), stream=stream0)
        del primals_189
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf120, primals_190, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (8, 512, 1, 1), (512, 1, 1, 1))
        buf122 = buf121; del buf121  # reuse
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(buf122, primals_191, 4096, grid=grid(4096), stream=stream0)
        del primals_191
        buf85 = buf84; del buf84  # reuse
        buf123 = empty((8, 512, 24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_11, mul_19, mul_21, mul__12, mul__13, out_13, out_16, shortcut_3, shortcut_4, sigmoid_1], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.sigmoid]
        triton_poi_fused_add_convolution_gelu_mul_sigmoid_28.run(buf85, primals_31, buf116, buf122, primals_44, buf123, 2359296, grid=grid(2359296), stream=stream0)
        del primals_31
        buf124 = empty_strided((1, 256, 1), (256, 1, 256), device='cuda', dtype=torch.float32)
        buf128 = empty((256, 512, 1, 1), device='cuda', dtype=torch.float32)
        buf127 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_14, weight_14], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_29.run(primals_45, primals_46, buf124, buf128, buf127, 256, 512, grid=grid(256), stream=stream0)
        # Source Nodes: [out_17], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf123, buf128, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (8, 256, 24, 24), (147456, 576, 24, 1))
        buf130 = buf129; del buf129  # reuse
        buf131 = empty((8, 256, 24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_12, mul__14, out_17], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_24.run(buf130, primals_47, buf131, 1179648, grid=grid(1179648), stream=stream0)
        del primals_47
        buf132 = empty_strided((1, 256, 1), (256, 1, 256), device='cuda', dtype=torch.float32)
        buf136 = empty((256, 128, 3, 3), device='cuda', dtype=torch.float32)
        buf135 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_15, weight_15], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_23.run(primals_48, primals_49, buf132, buf136, buf135, 256, 1152, grid=grid(256), stream=stream0)
        # Source Nodes: [out_18], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf131, buf136, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf137, (8, 256, 24, 24), (147456, 576, 24, 1))
        buf138 = buf137; del buf137  # reuse
        buf139 = empty((8, 256, 24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_13, mul__15, out_18], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_24.run(buf138, primals_50, buf139, 1179648, grid=grid(1179648), stream=stream0)
        del primals_50
        buf140 = empty_strided((1, 256, 1), (256, 1, 256), device='cuda', dtype=torch.float32)
        buf144 = empty((256, 128, 3, 3), device='cuda', dtype=torch.float32)
        buf143 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_16, weight_16], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_23.run(primals_51, primals_52, buf140, buf144, buf143, 256, 1152, grid=grid(256), stream=stream0)
        # Source Nodes: [out_19], Original ATen: [aten.convolution]
        buf145 = extern_kernels.convolution(buf139, buf144, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf145, (8, 256, 24, 24), (147456, 576, 24, 1))
        buf146 = buf145; del buf145  # reuse
        buf147 = empty((8, 256, 24, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_14, mul__16, out_19], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_24.run(buf146, primals_53, buf147, 1179648, grid=grid(1179648), stream=stream0)
        del primals_53
        buf148 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf152 = empty((512, 256, 1, 1), device='cuda', dtype=torch.float32)
        buf151 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_17, weight_17], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_19.run(primals_54, primals_55, buf148, buf152, buf151, 512, 256, grid=grid(512), stream=stream0)
        # Source Nodes: [out_20], Original ATen: [aten.convolution]
        buf153 = extern_kernels.convolution(buf147, buf152, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (8, 512, 24, 24), (294912, 576, 24, 1))
        buf154 = buf153; del buf153  # reuse
        buf155 = empty_strided((8, 512, 1, 1), (512, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf156 = reinterpret_tensor(buf155, (8, 512, 1, 1), (512, 1, 1, 1), 0); del buf155  # reuse
        # Source Nodes: [out_20, x_se_8], Original ATen: [aten.convolution, aten.mean]
        triton_per_fused_convolution_mean_25.run(buf154, buf156, primals_56, 4096, 576, grid=grid(4096), stream=stream0)
        del primals_56
        # Source Nodes: [x_se_9], Original ATen: [aten.convolution]
        buf157 = extern_kernels.convolution(buf156, primals_192, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (8, 256, 1, 1), (256, 1, 1, 1))
        buf158 = buf157; del buf157  # reuse
        # Source Nodes: [x_se_10, x_se_9], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_26.run(buf158, primals_193, 2048, grid=grid(2048), stream=stream0)
        del primals_193
        # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf158, primals_194, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (8, 512, 1, 1), (512, 1, 1, 1))
        buf160 = buf159; del buf159  # reuse
        # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_27.run(buf160, primals_195, 4096, grid=grid(4096), stream=stream0)
        del primals_195
        buf161 = empty((8, 512, 24, 24), device='cuda', dtype=torch.float32)
        buf162 = buf161; del buf161  # reuse
        # Source Nodes: [gelu_15, mul_19, mul_21, mul_27, mul_29, mul__12, mul__17, mul__18, out_13, out_21, out_24, shortcut_4, shortcut_5, sigmoid_1, sigmoid_2], Original ATen: [aten.add, aten.gelu, aten.mul, aten.sigmoid]
        triton_poi_fused_add_gelu_mul_sigmoid_30.run(buf162, buf154, buf160, primals_57, buf116, buf122, primals_44, buf85, 2359296, grid=grid(2359296), stream=stream0)
        buf163 = empty((8, 512, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___downsample_pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_31.run(buf162, buf163, 589824, grid=grid(589824), stream=stream0)
        buf164 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cuda', dtype=torch.float32)
        buf168 = empty((1536, 512, 1, 1), device='cuda', dtype=torch.float32)
        buf167 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_18, weight_18], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_32.run(primals_58, primals_59, buf164, buf168, buf167, 1536, 512, grid=grid(1536), stream=stream0)
        # Source Nodes: [shortcut_6], Original ATen: [aten.convolution]
        buf169 = extern_kernels.convolution(buf163, buf168, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (8, 1536, 12, 12), (221184, 144, 12, 1))
        buf171 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf175 = empty((768, 512, 1, 1), device='cuda', dtype=torch.float32)
        buf174 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_19, weight_19], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_33.run(primals_61, primals_62, buf171, buf175, buf174, 768, 512, grid=grid(768), stream=stream0)
        # Source Nodes: [out_25], Original ATen: [aten.convolution]
        buf176 = extern_kernels.convolution(buf162, buf175, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (8, 768, 24, 24), (442368, 576, 24, 1))
        buf177 = buf176; del buf176  # reuse
        # Source Nodes: [out_25], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_34.run(buf177, primals_63, 3538944, grid=grid(3538944), stream=stream0)
        del primals_63
        buf178 = empty((8, 768, 25, 25), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_16, mul__19, x_7], Original ATen: [aten.constant_pad_nd, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_gelu_mul_35.run(buf177, buf178, 3840000, grid=grid(3840000), stream=stream0)
        buf179 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf183 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        buf182 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_20, weight_20], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_36.run(primals_64, primals_65, buf179, buf183, buf182, 768, 1152, grid=grid(768), stream=stream0)
        # Source Nodes: [out_26], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf178, buf183, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf184, (8, 768, 12, 12), (110592, 144, 12, 1))
        buf185 = buf184; del buf184  # reuse
        buf186 = empty((8, 768, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_17, mul__20, out_26], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_37.run(buf185, primals_66, buf186, 884736, grid=grid(884736), stream=stream0)
        del primals_66
        buf187 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf191 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        buf190 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_21, weight_21], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_36.run(primals_67, primals_68, buf187, buf191, buf190, 768, 1152, grid=grid(768), stream=stream0)
        # Source Nodes: [out_27], Original ATen: [aten.convolution]
        buf192 = extern_kernels.convolution(buf186, buf191, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf192, (8, 768, 12, 12), (110592, 144, 12, 1))
        buf193 = buf192; del buf192  # reuse
        buf194 = empty((8, 768, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_18, mul__21, out_27], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_37.run(buf193, primals_69, buf194, 884736, grid=grid(884736), stream=stream0)
        del primals_69
        buf195 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cuda', dtype=torch.float32)
        buf199 = empty((1536, 768, 1, 1), device='cuda', dtype=torch.float32)
        buf198 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_22, weight_22], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_38.run(primals_70, primals_71, buf195, buf199, buf198, 1536, 768, grid=grid(1536), stream=stream0)
        # Source Nodes: [out_28], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf194, buf199, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (8, 1536, 12, 12), (221184, 144, 12, 1))
        buf201 = buf200; del buf200  # reuse
        buf202 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cuda', dtype=torch.float32)
        buf203 = reinterpret_tensor(buf202, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf202  # reuse
        # Source Nodes: [out_28, x_se_12], Original ATen: [aten.convolution, aten.mean]
        triton_per_fused_convolution_mean_39.run(buf201, buf203, primals_72, 12288, 144, grid=grid(12288), stream=stream0)
        del primals_72
        # Source Nodes: [x_se_13], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf203, primals_196, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (8, 768, 1, 1), (768, 1, 1, 1))
        buf205 = buf204; del buf204  # reuse
        # Source Nodes: [x_se_13, x_se_14], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_40.run(buf205, primals_197, 6144, grid=grid(6144), stream=stream0)
        del primals_197
        # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf205, primals_198, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (8, 1536, 1, 1), (1536, 1, 1, 1))
        buf207 = buf206; del buf206  # reuse
        # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf207, primals_199, 12288, grid=grid(12288), stream=stream0)
        del primals_199
        buf170 = buf169; del buf169  # reuse
        buf208 = empty((8, 1536, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_19, mul_36, mul_38, mul__22, mul__23, out_29, out_32, shortcut_6, shortcut_7, sigmoid_3], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.sigmoid]
        triton_poi_fused_add_convolution_gelu_mul_sigmoid_42.run(buf170, primals_60, buf201, buf207, primals_73, buf208, 1769472, grid=grid(1769472), stream=stream0)
        del primals_60
        buf209 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf213 = empty((768, 1536, 1, 1), device='cuda', dtype=torch.float32)
        buf212 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_23, weight_23], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_43.run(primals_74, primals_75, buf209, buf213, buf212, 768, 1536, grid=grid(768), stream=stream0)
        # Source Nodes: [out_33], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf208, buf213, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (8, 768, 12, 12), (110592, 144, 12, 1))
        buf215 = buf214; del buf214  # reuse
        buf216 = empty((8, 768, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_20, mul__24, out_33], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_37.run(buf215, primals_76, buf216, 884736, grid=grid(884736), stream=stream0)
        del primals_76
        buf217 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf221 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        buf220 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_24, weight_24], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_36.run(primals_77, primals_78, buf217, buf221, buf220, 768, 1152, grid=grid(768), stream=stream0)
        # Source Nodes: [out_34], Original ATen: [aten.convolution]
        buf222 = extern_kernels.convolution(buf216, buf221, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf222, (8, 768, 12, 12), (110592, 144, 12, 1))
        buf223 = buf222; del buf222  # reuse
        buf224 = empty((8, 768, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_21, mul__25, out_34], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_37.run(buf223, primals_79, buf224, 884736, grid=grid(884736), stream=stream0)
        del primals_79
        buf225 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf229 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        buf228 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_25, weight_25], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_36.run(primals_80, primals_81, buf225, buf229, buf228, 768, 1152, grid=grid(768), stream=stream0)
        # Source Nodes: [out_35], Original ATen: [aten.convolution]
        buf230 = extern_kernels.convolution(buf224, buf229, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf230, (8, 768, 12, 12), (110592, 144, 12, 1))
        buf231 = buf230; del buf230  # reuse
        buf232 = empty((8, 768, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_22, mul__26, out_35], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_37.run(buf231, primals_82, buf232, 884736, grid=grid(884736), stream=stream0)
        del primals_82
        buf233 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cuda', dtype=torch.float32)
        buf237 = empty((1536, 768, 1, 1), device='cuda', dtype=torch.float32)
        buf236 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_26, weight_26], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_38.run(primals_83, primals_84, buf233, buf237, buf236, 1536, 768, grid=grid(1536), stream=stream0)
        # Source Nodes: [out_36], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf232, buf237, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (8, 1536, 12, 12), (221184, 144, 12, 1))
        buf239 = buf238; del buf238  # reuse
        buf240 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cuda', dtype=torch.float32)
        buf241 = reinterpret_tensor(buf240, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf240  # reuse
        # Source Nodes: [out_36, x_se_16], Original ATen: [aten.convolution, aten.mean]
        triton_per_fused_convolution_mean_39.run(buf239, buf241, primals_85, 12288, 144, grid=grid(12288), stream=stream0)
        del primals_85
        # Source Nodes: [x_se_17], Original ATen: [aten.convolution]
        buf242 = extern_kernels.convolution(buf241, primals_200, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf242, (8, 768, 1, 1), (768, 1, 1, 1))
        buf243 = buf242; del buf242  # reuse
        # Source Nodes: [x_se_17, x_se_18], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_40.run(buf243, primals_201, 6144, grid=grid(6144), stream=stream0)
        del primals_201
        # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(buf243, primals_202, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf244, (8, 1536, 1, 1), (1536, 1, 1, 1))
        buf245 = buf244; del buf244  # reuse
        # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf245, primals_203, 12288, grid=grid(12288), stream=stream0)
        del primals_203
        buf246 = empty((8, 1536, 12, 12), device='cuda', dtype=torch.float32)
        buf247 = empty((8, 1536, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_23, mul_36, mul_38, mul_44, mul_46, mul__22, mul__27, mul__28, out_29, out_37, out_40, shortcut_7, shortcut_8, sigmoid_3, sigmoid_4], Original ATen: [aten.add, aten.gelu, aten.mul, aten.sigmoid]
        triton_poi_fused_add_gelu_mul_sigmoid_44.run(buf239, buf245, primals_86, buf201, buf207, primals_73, buf170, buf246, buf247, 1769472, grid=grid(1769472), stream=stream0)
        buf248 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf252 = empty((768, 1536, 1, 1), device='cuda', dtype=torch.float32)
        buf251 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_27, weight_27], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_43.run(primals_87, primals_88, buf248, buf252, buf251, 768, 1536, grid=grid(768), stream=stream0)
        # Source Nodes: [out_41], Original ATen: [aten.convolution]
        buf253 = extern_kernels.convolution(buf247, buf252, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (8, 768, 12, 12), (110592, 144, 12, 1))
        buf254 = buf253; del buf253  # reuse
        buf255 = empty((8, 768, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_24, mul__29, out_41], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_37.run(buf254, primals_89, buf255, 884736, grid=grid(884736), stream=stream0)
        del primals_89
        buf256 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf260 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        buf259 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_28, weight_28], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_36.run(primals_90, primals_91, buf256, buf260, buf259, 768, 1152, grid=grid(768), stream=stream0)
        # Source Nodes: [out_42], Original ATen: [aten.convolution]
        buf261 = extern_kernels.convolution(buf255, buf260, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf261, (8, 768, 12, 12), (110592, 144, 12, 1))
        buf262 = buf261; del buf261  # reuse
        buf263 = empty((8, 768, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_25, mul__30, out_42], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_37.run(buf262, primals_92, buf263, 884736, grid=grid(884736), stream=stream0)
        del primals_92
        buf264 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf268 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        buf267 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_29, weight_29], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_36.run(primals_93, primals_94, buf264, buf268, buf267, 768, 1152, grid=grid(768), stream=stream0)
        # Source Nodes: [out_43], Original ATen: [aten.convolution]
        buf269 = extern_kernels.convolution(buf263, buf268, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf269, (8, 768, 12, 12), (110592, 144, 12, 1))
        buf270 = buf269; del buf269  # reuse
        buf271 = empty((8, 768, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_26, mul__31, out_43], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_37.run(buf270, primals_95, buf271, 884736, grid=grid(884736), stream=stream0)
        del primals_95
        buf272 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cuda', dtype=torch.float32)
        buf276 = empty((1536, 768, 1, 1), device='cuda', dtype=torch.float32)
        buf275 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_30, weight_30], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_38.run(primals_96, primals_97, buf272, buf276, buf275, 1536, 768, grid=grid(1536), stream=stream0)
        # Source Nodes: [out_44], Original ATen: [aten.convolution]
        buf277 = extern_kernels.convolution(buf271, buf276, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf277, (8, 1536, 12, 12), (221184, 144, 12, 1))
        buf278 = buf277; del buf277  # reuse
        buf279 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cuda', dtype=torch.float32)
        buf280 = reinterpret_tensor(buf279, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf279  # reuse
        # Source Nodes: [out_44, x_se_20], Original ATen: [aten.convolution, aten.mean]
        triton_per_fused_convolution_mean_39.run(buf278, buf280, primals_98, 12288, 144, grid=grid(12288), stream=stream0)
        del primals_98
        # Source Nodes: [x_se_21], Original ATen: [aten.convolution]
        buf281 = extern_kernels.convolution(buf280, primals_204, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf281, (8, 768, 1, 1), (768, 1, 1, 1))
        buf282 = buf281; del buf281  # reuse
        # Source Nodes: [x_se_21, x_se_22], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_40.run(buf282, primals_205, 6144, grid=grid(6144), stream=stream0)
        del primals_205
        # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
        buf283 = extern_kernels.convolution(buf282, primals_206, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (8, 1536, 1, 1), (1536, 1, 1, 1))
        buf284 = buf283; del buf283  # reuse
        # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf284, primals_207, 12288, grid=grid(12288), stream=stream0)
        del primals_207
        buf285 = empty((8, 1536, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_27, mul_52, mul_54, mul__32, mul__33, out_45, out_48, shortcut_9, sigmoid_5], Original ATen: [aten.add, aten.gelu, aten.mul, aten.sigmoid]
        triton_poi_fused_add_gelu_mul_sigmoid_45.run(buf278, buf284, primals_99, buf246, buf285, 1769472, grid=grid(1769472), stream=stream0)
        buf286 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf290 = empty((768, 1536, 1, 1), device='cuda', dtype=torch.float32)
        buf289 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_31, weight_31], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_43.run(primals_100, primals_101, buf286, buf290, buf289, 768, 1536, grid=grid(768), stream=stream0)
        # Source Nodes: [out_49], Original ATen: [aten.convolution]
        buf291 = extern_kernels.convolution(buf285, buf290, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (8, 768, 12, 12), (110592, 144, 12, 1))
        buf292 = buf291; del buf291  # reuse
        buf293 = empty((8, 768, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_28, mul__34, out_49], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_37.run(buf292, primals_102, buf293, 884736, grid=grid(884736), stream=stream0)
        del primals_102
        buf294 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf298 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        buf297 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_32, weight_32], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_36.run(primals_103, primals_104, buf294, buf298, buf297, 768, 1152, grid=grid(768), stream=stream0)
        # Source Nodes: [out_50], Original ATen: [aten.convolution]
        buf299 = extern_kernels.convolution(buf293, buf298, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf299, (8, 768, 12, 12), (110592, 144, 12, 1))
        buf300 = buf299; del buf299  # reuse
        buf301 = empty((8, 768, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_29, mul__35, out_50], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_37.run(buf300, primals_105, buf301, 884736, grid=grid(884736), stream=stream0)
        del primals_105
        buf302 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf306 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        buf305 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_33, weight_33], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_36.run(primals_106, primals_107, buf302, buf306, buf305, 768, 1152, grid=grid(768), stream=stream0)
        # Source Nodes: [out_51], Original ATen: [aten.convolution]
        buf307 = extern_kernels.convolution(buf301, buf306, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf307, (8, 768, 12, 12), (110592, 144, 12, 1))
        buf308 = buf307; del buf307  # reuse
        buf309 = empty((8, 768, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_30, mul__36, out_51], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_37.run(buf308, primals_108, buf309, 884736, grid=grid(884736), stream=stream0)
        del primals_108
        buf310 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cuda', dtype=torch.float32)
        buf314 = empty((1536, 768, 1, 1), device='cuda', dtype=torch.float32)
        buf313 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_34, weight_34], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_38.run(primals_109, primals_110, buf310, buf314, buf313, 1536, 768, grid=grid(1536), stream=stream0)
        # Source Nodes: [out_52], Original ATen: [aten.convolution]
        buf315 = extern_kernels.convolution(buf309, buf314, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf315, (8, 1536, 12, 12), (221184, 144, 12, 1))
        buf316 = buf315; del buf315  # reuse
        buf317 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cuda', dtype=torch.float32)
        buf318 = reinterpret_tensor(buf317, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf317  # reuse
        # Source Nodes: [out_52, x_se_24], Original ATen: [aten.convolution, aten.mean]
        triton_per_fused_convolution_mean_39.run(buf316, buf318, primals_111, 12288, 144, grid=grid(12288), stream=stream0)
        del primals_111
        # Source Nodes: [x_se_25], Original ATen: [aten.convolution]
        buf319 = extern_kernels.convolution(buf318, primals_208, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf319, (8, 768, 1, 1), (768, 1, 1, 1))
        buf320 = buf319; del buf319  # reuse
        # Source Nodes: [x_se_25, x_se_26], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_40.run(buf320, primals_209, 6144, grid=grid(6144), stream=stream0)
        del primals_209
        # Source Nodes: [x_se_27], Original ATen: [aten.convolution]
        buf321 = extern_kernels.convolution(buf320, primals_210, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf321, (8, 1536, 1, 1), (1536, 1, 1, 1))
        buf322 = buf321; del buf321  # reuse
        # Source Nodes: [x_se_27], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf322, primals_211, 12288, grid=grid(12288), stream=stream0)
        del primals_211
        buf323 = buf246; del buf246  # reuse
        buf324 = empty((8, 1536, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_31, mul_52, mul_54, mul_60, mul_62, mul__32, mul__37, mul__38, out_45, out_53, out_56, shortcut_10, shortcut_9, sigmoid_5, sigmoid_6], Original ATen: [aten.add, aten.gelu, aten.mul, aten.sigmoid]
        triton_poi_fused_add_gelu_mul_sigmoid_46.run(buf323, buf316, buf322, primals_112, buf278, buf284, primals_99, buf324, 1769472, grid=grid(1769472), stream=stream0)
        buf325 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf329 = empty((768, 1536, 1, 1), device='cuda', dtype=torch.float32)
        buf328 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_35, weight_35], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_43.run(primals_113, primals_114, buf325, buf329, buf328, 768, 1536, grid=grid(768), stream=stream0)
        # Source Nodes: [out_57], Original ATen: [aten.convolution]
        buf330 = extern_kernels.convolution(buf324, buf329, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf330, (8, 768, 12, 12), (110592, 144, 12, 1))
        buf331 = buf330; del buf330  # reuse
        buf332 = empty((8, 768, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_32, mul__39, out_57], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_37.run(buf331, primals_115, buf332, 884736, grid=grid(884736), stream=stream0)
        del primals_115
        buf333 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf337 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        buf336 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_36, weight_36], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_36.run(primals_116, primals_117, buf333, buf337, buf336, 768, 1152, grid=grid(768), stream=stream0)
        # Source Nodes: [out_58], Original ATen: [aten.convolution]
        buf338 = extern_kernels.convolution(buf332, buf337, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf338, (8, 768, 12, 12), (110592, 144, 12, 1))
        buf339 = buf338; del buf338  # reuse
        buf340 = empty((8, 768, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_33, mul__40, out_58], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_37.run(buf339, primals_118, buf340, 884736, grid=grid(884736), stream=stream0)
        del primals_118
        buf341 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf345 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        buf344 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_37, weight_37], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_36.run(primals_119, primals_120, buf341, buf345, buf344, 768, 1152, grid=grid(768), stream=stream0)
        # Source Nodes: [out_59], Original ATen: [aten.convolution]
        buf346 = extern_kernels.convolution(buf340, buf345, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf346, (8, 768, 12, 12), (110592, 144, 12, 1))
        buf347 = buf346; del buf346  # reuse
        buf348 = empty((8, 768, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_34, mul__41, out_59], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_37.run(buf347, primals_121, buf348, 884736, grid=grid(884736), stream=stream0)
        del primals_121
        buf349 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cuda', dtype=torch.float32)
        buf353 = empty((1536, 768, 1, 1), device='cuda', dtype=torch.float32)
        buf352 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_38, weight_38], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_38.run(primals_122, primals_123, buf349, buf353, buf352, 1536, 768, grid=grid(1536), stream=stream0)
        # Source Nodes: [out_60], Original ATen: [aten.convolution]
        buf354 = extern_kernels.convolution(buf348, buf353, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf354, (8, 1536, 12, 12), (221184, 144, 12, 1))
        buf355 = buf354; del buf354  # reuse
        buf356 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cuda', dtype=torch.float32)
        buf357 = reinterpret_tensor(buf356, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf356  # reuse
        # Source Nodes: [out_60, x_se_28], Original ATen: [aten.convolution, aten.mean]
        triton_per_fused_convolution_mean_39.run(buf355, buf357, primals_124, 12288, 144, grid=grid(12288), stream=stream0)
        del primals_124
        # Source Nodes: [x_se_29], Original ATen: [aten.convolution]
        buf358 = extern_kernels.convolution(buf357, primals_212, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf358, (8, 768, 1, 1), (768, 1, 1, 1))
        buf359 = buf358; del buf358  # reuse
        # Source Nodes: [x_se_29, x_se_30], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_40.run(buf359, primals_213, 6144, grid=grid(6144), stream=stream0)
        del primals_213
        # Source Nodes: [x_se_31], Original ATen: [aten.convolution]
        buf360 = extern_kernels.convolution(buf359, primals_214, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf360, (8, 1536, 1, 1), (1536, 1, 1, 1))
        buf361 = buf360; del buf360  # reuse
        # Source Nodes: [x_se_31], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf361, primals_215, 12288, grid=grid(12288), stream=stream0)
        del primals_215
        buf362 = empty((8, 1536, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_35, mul_68, mul_70, mul__42, mul__43, out_61, out_64, shortcut_11, sigmoid_7], Original ATen: [aten.add, aten.gelu, aten.mul, aten.sigmoid]
        triton_poi_fused_add_gelu_mul_sigmoid_47.run(buf355, buf361, primals_125, buf323, buf362, 1769472, grid=grid(1769472), stream=stream0)
        buf363 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf367 = empty((768, 1536, 1, 1), device='cuda', dtype=torch.float32)
        buf366 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_39, weight_39], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_43.run(primals_126, primals_127, buf363, buf367, buf366, 768, 1536, grid=grid(768), stream=stream0)
        # Source Nodes: [out_65], Original ATen: [aten.convolution]
        buf368 = extern_kernels.convolution(buf362, buf367, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf368, (8, 768, 12, 12), (110592, 144, 12, 1))
        buf369 = buf368; del buf368  # reuse
        buf370 = empty((8, 768, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_36, mul__44, out_65], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_37.run(buf369, primals_128, buf370, 884736, grid=grid(884736), stream=stream0)
        del primals_128
        buf371 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf375 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        buf374 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_40, weight_40], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_36.run(primals_129, primals_130, buf371, buf375, buf374, 768, 1152, grid=grid(768), stream=stream0)
        # Source Nodes: [out_66], Original ATen: [aten.convolution]
        buf376 = extern_kernels.convolution(buf370, buf375, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf376, (8, 768, 12, 12), (110592, 144, 12, 1))
        buf377 = buf376; del buf376  # reuse
        buf378 = empty((8, 768, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_37, mul__45, out_66], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_37.run(buf377, primals_131, buf378, 884736, grid=grid(884736), stream=stream0)
        del primals_131
        buf379 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf383 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        buf382 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_41, weight_41], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_36.run(primals_132, primals_133, buf379, buf383, buf382, 768, 1152, grid=grid(768), stream=stream0)
        # Source Nodes: [out_67], Original ATen: [aten.convolution]
        buf384 = extern_kernels.convolution(buf378, buf383, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf384, (8, 768, 12, 12), (110592, 144, 12, 1))
        buf385 = buf384; del buf384  # reuse
        buf386 = empty((8, 768, 12, 12), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_38, mul__46, out_67], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_37.run(buf385, primals_134, buf386, 884736, grid=grid(884736), stream=stream0)
        del primals_134
        buf387 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cuda', dtype=torch.float32)
        buf391 = empty((1536, 768, 1, 1), device='cuda', dtype=torch.float32)
        buf390 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_42, weight_42], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_38.run(primals_135, primals_136, buf387, buf391, buf390, 1536, 768, grid=grid(1536), stream=stream0)
        # Source Nodes: [out_68], Original ATen: [aten.convolution]
        buf392 = extern_kernels.convolution(buf386, buf391, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf392, (8, 1536, 12, 12), (221184, 144, 12, 1))
        buf393 = buf392; del buf392  # reuse
        buf394 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cuda', dtype=torch.float32)
        buf395 = reinterpret_tensor(buf394, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf394  # reuse
        # Source Nodes: [out_68, x_se_32], Original ATen: [aten.convolution, aten.mean]
        triton_per_fused_convolution_mean_39.run(buf393, buf395, primals_137, 12288, 144, grid=grid(12288), stream=stream0)
        del primals_137
        # Source Nodes: [x_se_33], Original ATen: [aten.convolution]
        buf396 = extern_kernels.convolution(buf395, primals_216, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf396, (8, 768, 1, 1), (768, 1, 1, 1))
        buf397 = buf396; del buf396  # reuse
        # Source Nodes: [x_se_33, x_se_34], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_40.run(buf397, primals_217, 6144, grid=grid(6144), stream=stream0)
        del primals_217
        # Source Nodes: [x_se_35], Original ATen: [aten.convolution]
        buf398 = extern_kernels.convolution(buf397, primals_218, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf398, (8, 1536, 1, 1), (1536, 1, 1, 1))
        buf399 = buf398; del buf398  # reuse
        # Source Nodes: [x_se_35], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf399, primals_219, 12288, grid=grid(12288), stream=stream0)
        del primals_219
        buf400 = buf323; del buf323  # reuse
        buf401 = buf400; del buf400  # reuse
        # Source Nodes: [gelu_39, mul_68, mul_70, mul_76, mul_78, mul__42, mul__47, mul__48, out_61, out_69, out_72, shortcut_11, shortcut_12, sigmoid_7, sigmoid_8], Original ATen: [aten.add, aten.gelu, aten.mul, aten.sigmoid]
        triton_poi_fused_add_gelu_mul_sigmoid_48.run(buf401, buf393, buf399, primals_138, buf355, buf361, primals_125, 1769472, grid=grid(1769472), stream=stream0)
        buf402 = empty((8, 1536, 6, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____0___downsample_pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_49.run(buf401, buf402, 442368, grid=grid(442368), stream=stream0)
        buf403 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cuda', dtype=torch.float32)
        buf407 = empty((1536, 1536, 1, 1), device='cuda', dtype=torch.float32)
        buf406 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_43, weight_43], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_50.run(primals_139, primals_140, buf403, buf407, buf406, 1536, 1536, grid=grid(1536), stream=stream0)
        # Source Nodes: [shortcut_13], Original ATen: [aten.convolution]
        buf408 = extern_kernels.convolution(buf402, buf407, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf408, (8, 1536, 6, 6), (55296, 36, 6, 1))
        buf410 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf414 = empty((768, 1536, 1, 1), device='cuda', dtype=torch.float32)
        buf413 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_44, weight_44], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_43.run(primals_142, primals_143, buf410, buf414, buf413, 768, 1536, grid=grid(768), stream=stream0)
        # Source Nodes: [out_73], Original ATen: [aten.convolution]
        buf415 = extern_kernels.convolution(buf401, buf414, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf415, (8, 768, 12, 12), (110592, 144, 12, 1))
        buf416 = buf415; del buf415  # reuse
        # Source Nodes: [out_73], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_51.run(buf416, primals_144, 884736, grid=grid(884736), stream=stream0)
        del primals_144
        buf417 = empty((8, 768, 13, 13), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_40, mul__49, x_9], Original ATen: [aten.constant_pad_nd, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_gelu_mul_52.run(buf416, buf417, 1038336, grid=grid(1038336), stream=stream0)
        buf418 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf422 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        buf421 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_45, weight_45], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_36.run(primals_145, primals_146, buf418, buf422, buf421, 768, 1152, grid=grid(768), stream=stream0)
        # Source Nodes: [out_74], Original ATen: [aten.convolution]
        buf423 = extern_kernels.convolution(buf417, buf422, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf423, (8, 768, 6, 6), (27648, 36, 6, 1))
        buf424 = buf423; del buf423  # reuse
        buf425 = empty((8, 768, 6, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_41, mul__50, out_74], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_53.run(buf424, primals_147, buf425, 221184, grid=grid(221184), stream=stream0)
        del primals_147
        buf426 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf430 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        buf429 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_46, weight_46], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_36.run(primals_148, primals_149, buf426, buf430, buf429, 768, 1152, grid=grid(768), stream=stream0)
        # Source Nodes: [out_75], Original ATen: [aten.convolution]
        buf431 = extern_kernels.convolution(buf425, buf430, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf431, (8, 768, 6, 6), (27648, 36, 6, 1))
        buf432 = buf431; del buf431  # reuse
        buf433 = empty((8, 768, 6, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_42, mul__51, out_75], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_53.run(buf432, primals_150, buf433, 221184, grid=grid(221184), stream=stream0)
        del primals_150
        buf434 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cuda', dtype=torch.float32)
        buf438 = empty((1536, 768, 1, 1), device='cuda', dtype=torch.float32)
        buf437 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_47, weight_47], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_38.run(primals_151, primals_152, buf434, buf438, buf437, 1536, 768, grid=grid(1536), stream=stream0)
        # Source Nodes: [out_76], Original ATen: [aten.convolution]
        buf439 = extern_kernels.convolution(buf433, buf438, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf439, (8, 1536, 6, 6), (55296, 36, 6, 1))
        buf440 = buf439; del buf439  # reuse
        buf441 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cuda', dtype=torch.float32)
        buf442 = reinterpret_tensor(buf441, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf441  # reuse
        # Source Nodes: [out_76, x_se_36], Original ATen: [aten.convolution, aten.mean]
        triton_per_fused_convolution_mean_54.run(buf440, buf442, primals_153, 12288, 36, grid=grid(12288), stream=stream0)
        del primals_153
        # Source Nodes: [x_se_37], Original ATen: [aten.convolution]
        buf443 = extern_kernels.convolution(buf442, primals_220, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf443, (8, 768, 1, 1), (768, 1, 1, 1))
        buf444 = buf443; del buf443  # reuse
        # Source Nodes: [x_se_37, x_se_38], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_40.run(buf444, primals_221, 6144, grid=grid(6144), stream=stream0)
        del primals_221
        # Source Nodes: [x_se_39], Original ATen: [aten.convolution]
        buf445 = extern_kernels.convolution(buf444, primals_222, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf445, (8, 1536, 1, 1), (1536, 1, 1, 1))
        buf446 = buf445; del buf445  # reuse
        # Source Nodes: [x_se_39], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf446, primals_223, 12288, grid=grid(12288), stream=stream0)
        del primals_223
        buf409 = buf408; del buf408  # reuse
        buf447 = empty((8, 1536, 6, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_43, mul_85, mul_87, mul__52, mul__53, out_77, out_80, shortcut_13, shortcut_14, sigmoid_9], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mul, aten.sigmoid]
        triton_poi_fused_add_convolution_gelu_mul_sigmoid_55.run(buf409, primals_141, buf440, buf446, primals_154, buf447, 442368, grid=grid(442368), stream=stream0)
        del primals_141
        buf448 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf452 = empty((768, 1536, 1, 1), device='cuda', dtype=torch.float32)
        buf451 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_48, weight_48], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_43.run(primals_155, primals_156, buf448, buf452, buf451, 768, 1536, grid=grid(768), stream=stream0)
        # Source Nodes: [out_81], Original ATen: [aten.convolution]
        buf453 = extern_kernels.convolution(buf447, buf452, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf453, (8, 768, 6, 6), (27648, 36, 6, 1))
        buf454 = buf453; del buf453  # reuse
        buf455 = empty((8, 768, 6, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_44, mul__54, out_81], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_53.run(buf454, primals_157, buf455, 221184, grid=grid(221184), stream=stream0)
        del primals_157
        buf456 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf460 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        buf459 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_49, weight_49], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_36.run(primals_158, primals_159, buf456, buf460, buf459, 768, 1152, grid=grid(768), stream=stream0)
        # Source Nodes: [out_82], Original ATen: [aten.convolution]
        buf461 = extern_kernels.convolution(buf455, buf460, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf461, (8, 768, 6, 6), (27648, 36, 6, 1))
        buf462 = buf461; del buf461  # reuse
        buf463 = empty((8, 768, 6, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_45, mul__55, out_82], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_53.run(buf462, primals_160, buf463, 221184, grid=grid(221184), stream=stream0)
        del primals_160
        buf464 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf468 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        buf467 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_50, weight_50], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_36.run(primals_161, primals_162, buf464, buf468, buf467, 768, 1152, grid=grid(768), stream=stream0)
        # Source Nodes: [out_83], Original ATen: [aten.convolution]
        buf469 = extern_kernels.convolution(buf463, buf468, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf469, (8, 768, 6, 6), (27648, 36, 6, 1))
        buf470 = buf469; del buf469  # reuse
        buf471 = empty((8, 768, 6, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_46, mul__56, out_83], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_53.run(buf470, primals_163, buf471, 221184, grid=grid(221184), stream=stream0)
        del primals_163
        buf472 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cuda', dtype=torch.float32)
        buf476 = empty((1536, 768, 1, 1), device='cuda', dtype=torch.float32)
        buf475 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_51, weight_51], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_38.run(primals_164, primals_165, buf472, buf476, buf475, 1536, 768, grid=grid(1536), stream=stream0)
        # Source Nodes: [out_84], Original ATen: [aten.convolution]
        buf477 = extern_kernels.convolution(buf471, buf476, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf477, (8, 1536, 6, 6), (55296, 36, 6, 1))
        buf478 = buf477; del buf477  # reuse
        buf479 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cuda', dtype=torch.float32)
        buf480 = reinterpret_tensor(buf479, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf479  # reuse
        # Source Nodes: [out_84, x_se_40], Original ATen: [aten.convolution, aten.mean]
        triton_per_fused_convolution_mean_54.run(buf478, buf480, primals_166, 12288, 36, grid=grid(12288), stream=stream0)
        del primals_166
        # Source Nodes: [x_se_41], Original ATen: [aten.convolution]
        buf481 = extern_kernels.convolution(buf480, primals_224, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf481, (8, 768, 1, 1), (768, 1, 1, 1))
        buf482 = buf481; del buf481  # reuse
        # Source Nodes: [x_se_41, x_se_42], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_40.run(buf482, primals_225, 6144, grid=grid(6144), stream=stream0)
        del primals_225
        # Source Nodes: [x_se_43], Original ATen: [aten.convolution]
        buf483 = extern_kernels.convolution(buf482, primals_226, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf483, (8, 1536, 1, 1), (1536, 1, 1, 1))
        buf484 = buf483; del buf483  # reuse
        # Source Nodes: [x_se_43], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf484, primals_227, 12288, grid=grid(12288), stream=stream0)
        del primals_227
        buf485 = empty((8, 1536, 6, 6), device='cuda', dtype=torch.float32)
        buf486 = empty((8, 1536, 6, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_47, mul_85, mul_87, mul_93, mul_95, mul__52, mul__57, mul__58, out_77, out_85, out_88, shortcut_14, shortcut_15, sigmoid_10, sigmoid_9], Original ATen: [aten.add, aten.gelu, aten.mul, aten.sigmoid]
        triton_poi_fused_add_gelu_mul_sigmoid_56.run(buf478, buf484, primals_167, buf440, buf446, primals_154, buf409, buf485, buf486, 442368, grid=grid(442368), stream=stream0)
        buf487 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf491 = empty((768, 1536, 1, 1), device='cuda', dtype=torch.float32)
        buf490 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_52, weight_52], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_43.run(primals_168, primals_169, buf487, buf491, buf490, 768, 1536, grid=grid(768), stream=stream0)
        # Source Nodes: [out_89], Original ATen: [aten.convolution]
        buf492 = extern_kernels.convolution(buf486, buf491, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf492, (8, 768, 6, 6), (27648, 36, 6, 1))
        buf493 = buf492; del buf492  # reuse
        buf494 = empty((8, 768, 6, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_48, mul__59, out_89], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_53.run(buf493, primals_170, buf494, 221184, grid=grid(221184), stream=stream0)
        del primals_170
        buf495 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf499 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        buf498 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_53, weight_53], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_36.run(primals_171, primals_172, buf495, buf499, buf498, 768, 1152, grid=grid(768), stream=stream0)
        # Source Nodes: [out_90], Original ATen: [aten.convolution]
        buf500 = extern_kernels.convolution(buf494, buf499, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf500, (8, 768, 6, 6), (27648, 36, 6, 1))
        buf501 = buf500; del buf500  # reuse
        buf502 = empty((8, 768, 6, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_49, mul__60, out_90], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_53.run(buf501, primals_173, buf502, 221184, grid=grid(221184), stream=stream0)
        del primals_173
        buf503 = empty_strided((1, 768, 1), (768, 1, 768), device='cuda', dtype=torch.float32)
        buf507 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        buf506 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_54, weight_54], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_36.run(primals_174, primals_175, buf503, buf507, buf506, 768, 1152, grid=grid(768), stream=stream0)
        # Source Nodes: [out_91], Original ATen: [aten.convolution]
        buf508 = extern_kernels.convolution(buf502, buf507, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf508, (8, 768, 6, 6), (27648, 36, 6, 1))
        buf509 = buf508; del buf508  # reuse
        buf510 = empty((8, 768, 6, 6), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_50, mul__61, out_91], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_53.run(buf509, primals_176, buf510, 221184, grid=grid(221184), stream=stream0)
        del primals_176
        buf511 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cuda', dtype=torch.float32)
        buf515 = empty((1536, 768, 1, 1), device='cuda', dtype=torch.float32)
        buf514 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_55, weight_55], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_38.run(primals_177, primals_178, buf511, buf515, buf514, 1536, 768, grid=grid(1536), stream=stream0)
        # Source Nodes: [out_92], Original ATen: [aten.convolution]
        buf516 = extern_kernels.convolution(buf510, buf515, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf516, (8, 1536, 6, 6), (55296, 36, 6, 1))
        buf517 = buf516; del buf516  # reuse
        buf518 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cuda', dtype=torch.float32)
        buf519 = reinterpret_tensor(buf518, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf518  # reuse
        # Source Nodes: [out_92, x_se_44], Original ATen: [aten.convolution, aten.mean]
        triton_per_fused_convolution_mean_54.run(buf517, buf519, primals_179, 12288, 36, grid=grid(12288), stream=stream0)
        del primals_179
        # Source Nodes: [x_se_45], Original ATen: [aten.convolution]
        buf520 = extern_kernels.convolution(buf519, primals_228, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf520, (8, 768, 1, 1), (768, 1, 1, 1))
        buf521 = buf520; del buf520  # reuse
        # Source Nodes: [x_se_45, x_se_46], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_40.run(buf521, primals_229, 6144, grid=grid(6144), stream=stream0)
        del primals_229
        # Source Nodes: [x_se_47], Original ATen: [aten.convolution]
        buf522 = extern_kernels.convolution(buf521, primals_230, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf522, (8, 1536, 1, 1), (1536, 1, 1, 1))
        buf523 = buf522; del buf522  # reuse
        # Source Nodes: [x_se_47], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_41.run(buf523, primals_231, 12288, grid=grid(12288), stream=stream0)
        del primals_231
        buf524 = buf485; del buf485  # reuse
        # Source Nodes: [mul_101, mul_103, mul__62, out_93, sigmoid_11, x_10], Original ATen: [aten.add, aten.mul, aten.sigmoid]
        triton_poi_fused_add_mul_sigmoid_57.run(buf524, buf517, buf523, primals_180, 442368, grid=grid(442368), stream=stream0)
        buf525 = empty_strided((1, 3072, 1), (3072, 1, 3072), device='cuda', dtype=torch.float32)
        buf529 = empty((3072, 1536, 1, 1), device='cuda', dtype=torch.float32)
        buf528 = empty((3072, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_56, weight_56], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_58.run(primals_181, primals_182, buf525, buf529, buf528, 3072, 1536, grid=grid(3072), stream=stream0)
        # Source Nodes: [x_11], Original ATen: [aten.convolution]
        buf530 = extern_kernels.convolution(buf524, buf529, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf530, (8, 3072, 6, 6), (110592, 36, 6, 1))
        buf531 = buf530; del buf530  # reuse
        buf532 = empty_strided((8, 3072, 1, 1), (3072, 1, 24576, 24576), device='cuda', dtype=torch.float32)
        buf533 = reinterpret_tensor(buf532, (8, 3072), (3072, 1), 0); del buf532  # reuse
        # Source Nodes: [gelu_51, x_11, x_13, x_14, x_16], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul, aten.view]
        triton_per_fused_convolution_gelu_mean_mul_view_59.run(buf531, buf533, primals_183, 24576, 36, grid=grid(24576), stream=stream0)
        del primals_183
        buf534 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_18], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_233, buf533, reinterpret_tensor(primals_232, (3072, 1000), (1, 3072), 0), alpha=1, beta=1, out=buf534)
        del primals_233
        return (buf534, primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_30, primals_32, primals_33, primals_35, primals_36, primals_38, primals_39, primals_41, primals_42, primals_44, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_75, primals_77, primals_78, primals_80, primals_81, primals_83, primals_84, primals_86, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_114, primals_116, primals_117, primals_119, primals_120, primals_122, primals_123, primals_125, primals_126, primals_127, primals_129, primals_130, primals_132, primals_133, primals_135, primals_136, primals_138, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_156, primals_158, primals_159, primals_161, primals_162, primals_164, primals_165, primals_167, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_182, primals_184, primals_186, primals_188, primals_190, primals_192, primals_194, primals_196, primals_198, primals_200, primals_202, primals_204, primals_206, primals_208, primals_210, primals_212, primals_214, primals_216, primals_218, primals_220, primals_222, primals_224, primals_226, primals_228, primals_230, buf0, buf4, buf5, buf7, buf8, buf12, buf13, buf15, buf16, buf20, buf21, buf23, buf24, buf28, buf29, buf31, buf32, buf36, buf37, buf39, buf43, buf44, buf46, buf47, buf51, buf52, buf54, buf55, buf59, buf60, buf62, buf63, buf67, buf68, buf70, buf72, buf74, buf76, buf77, buf78, buf82, buf83, buf85, buf89, buf90, buf92, buf93, buf97, buf98, buf100, buf101, buf105, buf106, buf108, buf109, buf113, buf114, buf116, buf118, buf120, buf122, buf123, buf127, buf128, buf130, buf131, buf135, buf136, buf138, buf139, buf143, buf144, buf146, buf147, buf151, buf152, buf154, buf156, buf158, buf160, buf162, buf163, buf167, buf168, buf170, buf174, buf175, buf177, buf178, buf182, buf183, buf185, buf186, buf190, buf191, buf193, buf194, buf198, buf199, buf201, buf203, buf205, buf207, buf208, buf212, buf213, buf215, buf216, buf220, buf221, buf223, buf224, buf228, buf229, buf231, buf232, buf236, buf237, buf239, buf241, buf243, buf245, buf247, buf251, buf252, buf254, buf255, buf259, buf260, buf262, buf263, buf267, buf268, buf270, buf271, buf275, buf276, buf278, buf280, buf282, buf284, buf285, buf289, buf290, buf292, buf293, buf297, buf298, buf300, buf301, buf305, buf306, buf308, buf309, buf313, buf314, buf316, buf318, buf320, buf322, buf324, buf328, buf329, buf331, buf332, buf336, buf337, buf339, buf340, buf344, buf345, buf347, buf348, buf352, buf353, buf355, buf357, buf359, buf361, buf362, buf366, buf367, buf369, buf370, buf374, buf375, buf377, buf378, buf382, buf383, buf385, buf386, buf390, buf391, buf393, buf395, buf397, buf399, buf401, buf402, buf406, buf407, buf409, buf413, buf414, buf416, buf417, buf421, buf422, buf424, buf425, buf429, buf430, buf432, buf433, buf437, buf438, buf440, buf442, buf444, buf446, buf447, buf451, buf452, buf454, buf455, buf459, buf460, buf462, buf463, buf467, buf468, buf470, buf471, buf475, buf476, buf478, buf480, buf482, buf484, buf486, buf490, buf491, buf493, buf494, buf498, buf499, buf501, buf502, buf506, buf507, buf509, buf510, buf514, buf515, buf517, buf519, buf521, buf523, buf524, buf528, buf529, buf531, buf533, reinterpret_tensor(primals_232, (1000, 3072), (3072, 1), 0), reinterpret_tensor(buf525, (1, 3072, 1), (3072, 1, 1), 0), reinterpret_tensor(buf511, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf503, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf495, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf487, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf472, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf464, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf456, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf448, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf434, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf426, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf418, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf410, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf403, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf387, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf379, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf371, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf363, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf349, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf341, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf333, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf325, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf310, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf302, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf294, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf286, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf272, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf264, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf256, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf248, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf233, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf225, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf217, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf209, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf195, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf187, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf179, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf171, (1, 768, 1), (768, 1, 1), 0), reinterpret_tensor(buf164, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf148, (1, 512, 1), (512, 1, 1), 0), reinterpret_tensor(buf140, (1, 256, 1), (256, 1, 1), 0), reinterpret_tensor(buf132, (1, 256, 1), (256, 1, 1), 0), reinterpret_tensor(buf124, (1, 256, 1), (256, 1, 1), 0), reinterpret_tensor(buf110, (1, 512, 1), (512, 1, 1), 0), reinterpret_tensor(buf102, (1, 256, 1), (256, 1, 1), 0), reinterpret_tensor(buf94, (1, 256, 1), (256, 1, 1), 0), reinterpret_tensor(buf86, (1, 256, 1), (256, 1, 1), 0), reinterpret_tensor(buf79, (1, 512, 1), (512, 1, 1), 0), reinterpret_tensor(buf64, (1, 256, 1), (256, 1, 1), 0), reinterpret_tensor(buf56, (1, 128, 1), (128, 1, 1), 0), reinterpret_tensor(buf48, (1, 128, 1), (128, 1, 1), 0), reinterpret_tensor(buf40, (1, 128, 1), (128, 1, 1), 0), reinterpret_tensor(buf33, (1, 256, 1), (256, 1, 1), 0), reinterpret_tensor(buf25, (1, 128, 1), (128, 1, 1), 0), reinterpret_tensor(buf17, (1, 64, 1), (64, 1, 1), 0), reinterpret_tensor(buf9, (1, 32, 1), (32, 1, 1), 0), reinterpret_tensor(buf1, (1, 16, 1), (16, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((768, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((1536, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((3072, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((3072, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((1000, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((8, 3, 192, 192), (110592, 36864, 192, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('dm_nfnet_f0', benchmark_compiled_module)
