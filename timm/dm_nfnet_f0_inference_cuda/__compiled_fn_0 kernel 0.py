
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


# kernel path: /tmp/torchinductor_youkaichao/cb/ccbnkoatdzl23qeiobjdsroful2ddcp3cxpfhzohogc6oar4qqxe.py
# Source Nodes: [batch_norm, conv2d, x], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution]
# batch_norm => var_mean
# conv2d => convolution
# x => constant_pad_nd
triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/56/c56tph4slh74aq5d6q5j5cvqdwzcz3yequevptvwho4jhvyrike2.py
# Source Nodes: [batch_norm_1, conv2d, conv2d_1, gelu, mul_, x], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
# batch_norm_1 => var_mean_1
# conv2d => convolution
# conv2d_1 => convolution_1
# gelu => add_1, erf, mul_3, mul_4, mul_5
# mul_ => mul_6
# x => constant_pad_nd
triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_1', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
''')


# kernel path: /tmp/torchinductor_youkaichao/e3/ce3pmr6omt6zp3ollmjd4wvbngii27s5o7t4rdlwkvwn3dx2ynk4.py
# Source Nodes: [batch_norm_2, conv2d, conv2d_1, conv2d_2, gelu, gelu_1, mul_, mul__1, x], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
# batch_norm_2 => var_mean_2
# conv2d => convolution
# conv2d_1 => convolution_1
# conv2d_2 => convolution_2
# gelu => add_1, erf, mul_3, mul_4, mul_5
# gelu_1 => add_3, erf_1, mul_10, mul_11, mul_12
# mul_ => mul_6
# mul__1 => mul_13
# x => constant_pad_nd
triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_2', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
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
''')


# kernel path: /tmp/torchinductor_youkaichao/ob/cob3wpfrg36fij52hfz7enffiv3a3egvz2dsfaw447upsa6km7ru.py
# Source Nodes: [batch_norm_3, conv2d, conv2d_1, conv2d_2, gelu, gelu_1, gelu_2, mul_, mul__1, mul__2, shortcut, x, x_2], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
# batch_norm_3 => var_mean_3
# conv2d => convolution
# conv2d_1 => convolution_1
# conv2d_2 => convolution_2
# gelu => add_1, erf, mul_3, mul_4, mul_5
# gelu_1 => add_3, erf_1, mul_10, mul_11, mul_12
# gelu_2 => add_5, erf_2, mul_17, mul_18, mul_19
# mul_ => mul_6
# mul__1 => mul_13
# mul__2 => mul_20
# shortcut => convolution_3
# x => constant_pad_nd
# x_2 => constant_pad_nd_1
triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
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
''')


# kernel path: /tmp/torchinductor_youkaichao/tz/ctzqonm4orctz5tp7vayuquolvd2qgmq5pbhipwwitwrlnfgqdtg.py
# Source Nodes: [batch_norm_4, shortcut_1], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
# batch_norm_4 => var_mean_4
# shortcut_1 => convolution_4
triton_per_fused__native_batch_norm_legit_convolution_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
''')


# kernel path: /tmp/torchinductor_youkaichao/wj/cwjlh7utvsjufl3jbob73k2uw3jniivnwlg3l2thatvrhhbfa6ci.py
# Source Nodes: [batch_norm_5, out_1], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
# batch_norm_5 => var_mean_5
# out_1 => convolution_5
triton_per_fused__native_batch_norm_legit_convolution_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
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
''')


# kernel path: /tmp/torchinductor_youkaichao/j6/cj6p5eaoi7mcll6rvh7dtiuhk6kqzyqbycjubuo4mo6tai4fmtgv.py
# Source Nodes: [batch_norm_6, gelu_4, mul__4, out_1, out_2], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
# batch_norm_6 => var_mean_6
# gelu_4 => add_10, erf_4, mul_35, mul_36, mul_37
# mul__4 => mul_38
# out_1 => convolution_5
# out_2 => convolution_6
triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
''')


# kernel path: /tmp/torchinductor_youkaichao/3n/c3nmeevgtq66a6t3ig45t3ywgxffagaxpe26lomo5sutj65q7jea.py
# Source Nodes: [batch_norm_9, getattr_getattr_l__mod___stages___1_____0___downsample_pool, shortcut_3], Original ATen: [aten._native_batch_norm_legit, aten.avg_pool2d, aten.convolution]
# batch_norm_9 => var_mean_9
# getattr_getattr_l__mod___stages___1_____0___downsample_pool => avg_pool2d
# shortcut_3 => convolution_11
triton_per_fused__native_batch_norm_legit_avg_pool2d_convolution_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_avg_pool2d_convolution_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
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
''')


# kernel path: /tmp/torchinductor_youkaichao/ag/cagyogdo4q5xoqyrfhowq6qjopznb2ei6oxkcsfd54mecospc2sn.py
# Source Nodes: [batch_norm_10, out_9], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
# batch_norm_10 => var_mean_10
# out_9 => convolution_12
triton_per_fused__native_batch_norm_legit_convolution_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_8', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
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
''')


# kernel path: /tmp/torchinductor_youkaichao/v2/cv2mpc2wyka3yixhghtyy7ice6d337goj3razt5liu3q3xtauwkw.py
# Source Nodes: [batch_norm_11, gelu_8, mul__9, out_10, out_9, x_5], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
# batch_norm_11 => var_mean_11
# gelu_8 => add_20, erf_8, mul_71, mul_72, mul_73
# mul__9 => mul_74
# out_10 => convolution_13
# out_9 => convolution_12
# x_5 => constant_pad_nd_2
triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
''')


# kernel path: /tmp/torchinductor_youkaichao/tz/ctzi5k37qh45hiuilya2w546qcvbrlbluzg76o7e4iaqf7s73334.py
# Source Nodes: [batch_norm_14, gelu_11, mul__13, out_16, out_17], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
# batch_norm_14 => var_mean_14
# gelu_11 => add_27, erf_11, mul_96, mul_97, mul_98
# mul__13 => mul_99
# out_16 => mul_100
# out_17 => convolution_18
triton_per_fused__native_batch_norm_legit_convolution_gelu_mul_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_gelu_mul_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
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
''')


# kernel path: /tmp/torchinductor_youkaichao/zd/czdk47mpnn3gehicw32gnaysshsq3vzqtv22pjiqqhwhjaab526i.py
# Source Nodes: [batch_norm_18, getattr_getattr_l__mod___stages___2_____0___downsample_pool, shortcut_6], Original ATen: [aten._native_batch_norm_legit, aten.avg_pool2d, aten.convolution]
# batch_norm_18 => var_mean_18
# getattr_getattr_l__mod___stages___2_____0___downsample_pool => avg_pool2d_1
# shortcut_6 => convolution_24
triton_per_fused__native_batch_norm_legit_avg_pool2d_convolution_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_avg_pool2d_convolution_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
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
''')


# kernel path: /tmp/torchinductor_youkaichao/st/cstuphtklv2v4fucojvpbcswfe4zvqbqxxndjzyhf2lv2zkyab4b.py
# Source Nodes: [batch_norm_19, out_25], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
# batch_norm_19 => var_mean_19
# out_25 => convolution_25
triton_per_fused__native_batch_norm_legit_convolution_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
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
''')


# kernel path: /tmp/torchinductor_youkaichao/6p/c6pn36sm6vdsn3ibbwj3465ryopehwao5zlancs4av5onakjtoaz.py
# Source Nodes: [batch_norm_20, gelu_16, mul__19, out_25, out_26, x_7], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
# batch_norm_20 => var_mean_20
# gelu_16 => add_39, erf_16, mul_140, mul_141, mul_142
# mul__19 => mul_143
# out_25 => convolution_25
# out_26 => convolution_26
# x_7 => constant_pad_nd_3
triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
''')


# kernel path: /tmp/torchinductor_youkaichao/22/c22fk6wdewukurur46iqjizyuirfgpx6yz4we266sow4k6clsxtz.py
# Source Nodes: [batch_norm_22, gelu_16, gelu_17, gelu_18, mul__19, mul__20, mul__21, out_25, out_26, out_27, out_28, x_7], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
# batch_norm_22 => var_mean_22
# gelu_16 => add_39, erf_16, mul_140, mul_141, mul_142
# gelu_17 => add_41, erf_17, mul_147, mul_148, mul_149
# gelu_18 => add_43, erf_18, mul_154, mul_155, mul_156
# mul__19 => mul_143
# mul__20 => mul_150
# mul__21 => mul_157
# out_25 => convolution_25
# out_26 => convolution_26
# out_27 => convolution_27
# out_28 => convolution_28
# x_7 => constant_pad_nd_3
triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
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
''')


# kernel path: /tmp/torchinductor_youkaichao/kk/ckkwt6h2xigyckzjmliasaz45khzvdh4jpsd77arzq5on33hnheh.py
# Source Nodes: [batch_norm_23, gelu_19, mul__23, out_32, out_33], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
# batch_norm_23 => var_mean_23
# gelu_19 => add_46, erf_19, mul_165, mul_166, mul_167
# mul__23 => mul_168
# out_32 => mul_169
# out_33 => convolution_31
triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
''')


# kernel path: /tmp/torchinductor_youkaichao/43/c43f3kxzilqn7k5hjh6r7xghsngo4kbfm6o7r4dfvwln443yd6zh.py
# Source Nodes: [batch_norm_43, getattr_getattr_l__mod___stages___3_____0___downsample_pool, shortcut_13], Original ATen: [aten._native_batch_norm_legit, aten.avg_pool2d, aten.convolution]
# batch_norm_43 => var_mean_43
# getattr_getattr_l__mod___stages___3_____0___downsample_pool => avg_pool2d_2
# shortcut_13 => convolution_61
triton_red_fused__native_batch_norm_legit_avg_pool2d_convolution_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_avg_pool2d_convolution_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
''')


# kernel path: /tmp/torchinductor_youkaichao/x6/cx66bvdl6g4cjy4z2wj4nsjwzikolx27f4bsom56zgw6e3k6oeso.py
# Source Nodes: [batch_norm_56, gelu_47, gelu_48, gelu_49, gelu_50, mul_101, mul_103, mul__58, mul__59, mul__60, mul__61, mul__62, out_88, out_89, out_90, out_91, out_92, out_93, sigmoid_11, x_10, x_11, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten._native_batch_norm_legit, aten.add, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# batch_norm_56 => var_mean_56
# gelu_47 => add_110, erf_47, mul_399, mul_400, mul_401
# gelu_48 => add_112, erf_48, mul_407, mul_408, mul_409
# gelu_49 => add_114, erf_49, mul_414, mul_415, mul_416
# gelu_50 => add_116, erf_50, mul_421, mul_422, mul_423
# mul_101 => mul_428
# mul_103 => mul_431
# mul__58 => mul_402
# mul__59 => mul_410
# mul__60 => mul_417
# mul__61 => mul_424
# mul__62 => mul_430
# out_88 => mul_403
# out_89 => convolution_74
# out_90 => convolution_75
# out_91 => convolution_76
# out_92 => convolution_77
# out_93 => mul_429
# sigmoid_11 => sigmoid_11
# x_10 => add_118
# x_11 => convolution_80
# x_se_44 => mean_11
# x_se_45 => convolution_78
# x_se_46 => relu_11
# x_se_47 => convolution_79
triton_red_fused__native_batch_norm_legit_add_convolution_gelu_mean_mul_relu_sigmoid_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_add_convolution_gelu_mean_mul_relu_sigmoid_17', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
''')


# kernel path: /tmp/torchinductor_youkaichao/x3/cx3mbwnoou72ltdfrjeaiwjgywbrhkm6axxh5q2sjnhowlypophi.py
# Source Nodes: [conv2d, x], Original ATen: [aten.constant_pad_nd, aten.convolution]
# conv2d => convolution
# x => constant_pad_nd
triton_poi_fused_constant_pad_nd_convolution_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1585176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 257) % 257
    x0 = xindex % 257
    x2 = (xindex // 66049)
    x3 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 256, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (256*x1) + (65536*x2)), tmp5 & xmask, other=0.0)
    tmp7 = tl.full(tmp6.shape, 0.0, tmp6.dtype)
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tl.store(out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oi/coiprealeoftmfw4kwingmtbg7xqxxb53jsxrvrnqnpohhlg337s.py
# Source Nodes: [conv2d, conv2d_1, gelu, mul_, x], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
# conv2d => convolution
# conv2d_1 => convolution_1
# gelu => add_1, erf, mul_3, mul_4, mul_5
# mul_ => mul_6
# x => constant_pad_nd
triton_poi_fused_constant_pad_nd_convolution_gelu_mul_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_gelu_mul_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 16
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
    tl.store(in_out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nm/cnm3wqxijmdmv6gfuyu4mckf4lgwejmomx34vj4ify3y7iwxp2o5.py
# Source Nodes: [conv2d, conv2d_1, conv2d_2, gelu, gelu_1, mul_, mul__1, x], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
# conv2d => convolution
# conv2d_1 => convolution_1
# conv2d_2 => convolution_2
# gelu => add_1, erf, mul_3, mul_4, mul_5
# gelu_1 => add_3, erf_1, mul_10, mul_11, mul_12
# mul_ => mul_6
# mul__1 => mul_13
# x => constant_pad_nd
triton_poi_fused_constant_pad_nd_convolution_gelu_mul_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_gelu_mul_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 32
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
    tl.store(in_out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qp/cqpxe3ey5bagcxganwq5fop2ne5vr4iej7bbdloiu5z4gxq4gnzg.py
# Source Nodes: [conv2d, conv2d_1, conv2d_2, gelu, gelu_1, gelu_2, mul_, mul__1, mul__2, shortcut, x, x_2], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
# conv2d => convolution
# conv2d_1 => convolution_1
# conv2d_2 => convolution_2
# gelu => add_1, erf, mul_3, mul_4, mul_5
# gelu_1 => add_3, erf_1, mul_10, mul_11, mul_12
# gelu_2 => add_5, erf_2, mul_17, mul_18, mul_19
# mul_ => mul_6
# mul__1 => mul_13
# mul__2 => mul_20
# shortcut => convolution_3
# x => constant_pad_nd
# x_2 => constant_pad_nd_1
triton_poi_fused_constant_pad_nd_convolution_gelu_mul_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_gelu_mul_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8520192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 129) % 129
    x0 = xindex % 129
    x4 = (xindex // 16641)
    x2 = (xindex // 16641) % 64
    x5 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 128, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (128*x1) + (16384*x4)), tmp5 & xmask, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x2), tmp5 & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = 0.7071067811865476
    tmp12 = tmp8 * tmp11
    tmp13 = tl.math.erf(tmp12)
    tmp14 = 1.0
    tmp15 = tmp13 + tmp14
    tmp16 = tmp10 * tmp15
    tmp17 = 1.7015043497085571
    tmp18 = tmp16 * tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp5, tmp18, tmp19)
    tl.store(out_ptr0 + (x5), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4q/c4qoyh26ii5riiczilanpdjhj4ptjuvbtr6si4pqu5tm422esncx.py
# Source Nodes: [conv2d, conv2d_1, conv2d_2, gelu, gelu_1, gelu_2, gelu_3, mul_, mul__1, mul__2, mul__3, out, shortcut, x, x_2], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
# conv2d => convolution
# conv2d_1 => convolution_1
# conv2d_2 => convolution_2
# gelu => add_1, erf, mul_3, mul_4, mul_5
# gelu_1 => add_3, erf_1, mul_10, mul_11, mul_12
# gelu_2 => add_5, erf_2, mul_17, mul_18, mul_19
# gelu_3 => add_7, erf_3, mul_24, mul_25, mul_26
# mul_ => mul_6
# mul__1 => mul_13
# mul__2 => mul_20
# mul__3 => mul_27
# out => mul_28
# shortcut => convolution_3
# x => constant_pad_nd
# x_2 => constant_pad_nd_1
triton_poi_fused_constant_pad_nd_convolution_gelu_mul_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_gelu_mul_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 128
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
    tl.store(in_out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zt/czttpq7wabk2cvklzf7gqcvfr572ludv5hpopyszxbx7ivjmvpd2.py
# Source Nodes: [gelu_4, mul__4, out_1, out_2], Original ATen: [aten.convolution, aten.gelu, aten.mul]
# gelu_4 => add_10, erf_4, mul_35, mul_36, mul_37
# mul__4 => mul_38
# out_1 => convolution_5
# out_2 => convolution_6
triton_poi_fused_convolution_gelu_mul_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_mul_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 128
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
    tl.store(in_out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jl/cjlc672qecaik7j5zskwjzn3sutzbq6gdl5qp6fo4br2u4q2dvhs.py
# Source Nodes: [gelu_4, gelu_5, gelu_6, mul__4, mul__5, mul__6, out_1, out_2, out_3, out_4, x_se, x_se_1], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul]
# gelu_4 => add_10, erf_4, mul_35, mul_36, mul_37
# gelu_5 => add_12, erf_5, mul_42, mul_43, mul_44
# gelu_6 => add_14, erf_6, mul_49, mul_50, mul_51
# mul__4 => mul_38
# mul__5 => mul_45
# mul__6 => mul_52
# out_1 => convolution_5
# out_2 => convolution_6
# out_3 => convolution_7
# out_4 => convolution_8
# x_se => mean
# x_se_1 => convolution_9
triton_red_fused_convolution_gelu_mean_mul_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_gelu_mean_mul_24', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 256
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (4096*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp6 = 4096.0
    tmp7 = tmp4 / tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ld/clduqbp4jgrfpld5yanndbbu3cng4if6qqymaxg2mo5fme7yqopo.py
# Source Nodes: [gelu_4, gelu_5, gelu_6, mul__4, mul__5, mul__6, out_1, out_2, out_3, out_4, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
# gelu_4 => add_10, erf_4, mul_35, mul_36, mul_37
# gelu_5 => add_12, erf_5, mul_42, mul_43, mul_44
# gelu_6 => add_14, erf_6, mul_49, mul_50, mul_51
# mul__4 => mul_38
# mul__5 => mul_45
# mul__6 => mul_52
# out_1 => convolution_5
# out_2 => convolution_6
# out_3 => convolution_7
# out_4 => convolution_8
# x_se => mean
# x_se_1 => convolution_9
# x_se_2 => relu
# x_se_3 => convolution_10
triton_poi_fused_convolution_gelu_mean_mul_relu_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_gelu_mean_mul_relu_25', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/r4/cr4i236k5hcwqnkfbnhexn5i7o7mtyfwv5glrmhqormrl5q55qx3.py
# Source Nodes: [gelu_4, gelu_5, gelu_6, gelu_7, mul_10, mul_12, mul__4, mul__5, mul__6, mul__7, mul__8, out_1, out_2, out_3, out_4, out_5, out_8, shortcut_1, shortcut_2, sigmoid, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# gelu_4 => add_10, erf_4, mul_35, mul_36, mul_37
# gelu_5 => add_12, erf_5, mul_42, mul_43, mul_44
# gelu_6 => add_14, erf_6, mul_49, mul_50, mul_51
# gelu_7 => add_17, erf_7, mul_60, mul_61, mul_62
# mul_10 => mul_56
# mul_12 => mul_59
# mul__4 => mul_38
# mul__5 => mul_45
# mul__6 => mul_52
# mul__7 => mul_58
# mul__8 => mul_63
# out_1 => convolution_5
# out_2 => convolution_6
# out_3 => convolution_7
# out_4 => convolution_8
# out_5 => mul_57
# out_8 => mul_64
# shortcut_1 => convolution_4
# shortcut_2 => add_16
# sigmoid => sigmoid
# x_se => mean
# x_se_1 => convolution_9
# x_se_2 => relu
# x_se_3 => convolution_10
triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_26', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 256
    x4 = (xindex // 4096)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp12 = tmp9 * tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp17 = tmp15 + tmp16
    tmp18 = tmp14 + tmp17
    tmp19 = 0.5
    tmp20 = tmp18 * tmp19
    tmp21 = 0.7071067811865476
    tmp22 = tmp18 * tmp21
    tmp23 = tl.math.erf(tmp22)
    tmp24 = 1.0
    tmp25 = tmp23 + tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = 1.7015043497085571
    tmp28 = tmp26 * tmp27
    tmp29 = 0.9805806756909201
    tmp30 = tmp28 * tmp29
    tl.store(in_out_ptr0 + (x3), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/63/c63gdvu3rlden5b4fquqwnryyd2bnr56uxcirhcresuidblfyoe4.py
# Source Nodes: [gelu_8, mul__9, out_10, out_9, x_5], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
# gelu_8 => add_20, erf_8, mul_71, mul_72, mul_73
# mul__9 => mul_74
# out_10 => convolution_13
# out_9 => convolution_12
# x_5 => constant_pad_nd_2
triton_poi_fused_constant_pad_nd_convolution_gelu_mul_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_gelu_mul_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8652800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 65) % 65
    x0 = xindex % 65
    x4 = (xindex // 4225)
    x2 = (xindex // 4225) % 256
    x5 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 64, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (64*x1) + (4096*x4)), tmp5, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x2), tmp5, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = 0.7071067811865476
    tmp12 = tmp8 * tmp11
    tmp13 = tl.math.erf(tmp12)
    tmp14 = 1.0
    tmp15 = tmp13 + tmp14
    tmp16 = tmp10 * tmp15
    tmp17 = 1.7015043497085571
    tmp18 = tmp16 * tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp5, tmp18, tmp19)
    tl.store(out_ptr0 + (x5), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ua/cua4euwu3ksfok447lr3dnagcq65rtbsx7dkargm5o24ytsqut4f.py
# Source Nodes: [gelu_8, gelu_9, mul__10, mul__9, out_10, out_11, out_9, x_5], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
# gelu_8 => add_20, erf_8, mul_71, mul_72, mul_73
# gelu_9 => add_22, erf_9, mul_78, mul_79, mul_80
# mul__10 => mul_81
# mul__9 => mul_74
# out_10 => convolution_13
# out_11 => convolution_14
# out_9 => convolution_12
# x_5 => constant_pad_nd_2
triton_poi_fused_constant_pad_nd_convolution_gelu_mul_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_gelu_mul_28', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 256
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
    tl.store(in_out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fm/cfmfpmbjsyst3ea7kibwmwxyjelvnbp5snu6s3hu725yfyayr773.py
# Source Nodes: [gelu_10, gelu_8, gelu_9, mul__10, mul__11, mul__9, out_10, out_11, out_12, out_9, x_5, x_se_4, x_se_5], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mean, aten.mul]
# gelu_10 => add_24, erf_10, mul_85, mul_86, mul_87
# gelu_8 => add_20, erf_8, mul_71, mul_72, mul_73
# gelu_9 => add_22, erf_9, mul_78, mul_79, mul_80
# mul__10 => mul_81
# mul__11 => mul_88
# mul__9 => mul_74
# out_10 => convolution_13
# out_11 => convolution_14
# out_12 => convolution_15
# out_9 => convolution_12
# x_5 => constant_pad_nd_2
# x_se_4 => mean_1
# x_se_5 => convolution_16
triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_29', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (r2 + (1024*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp7 = 1024.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/he/chea75kipmjoivtkxepua3etdgluacgp6oaoezvhuflsz3fa5nmp.py
# Source Nodes: [gelu_10, gelu_8, gelu_9, mul__10, mul__11, mul__9, out_10, out_11, out_12, out_9, x_5, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
# gelu_10 => add_24, erf_10, mul_85, mul_86, mul_87
# gelu_8 => add_20, erf_8, mul_71, mul_72, mul_73
# gelu_9 => add_22, erf_9, mul_78, mul_79, mul_80
# mul__10 => mul_81
# mul__11 => mul_88
# mul__9 => mul_74
# out_10 => convolution_13
# out_11 => convolution_14
# out_12 => convolution_15
# out_9 => convolution_12
# x_5 => constant_pad_nd_2
# x_se_4 => mean_1
# x_se_5 => convolution_16
# x_se_6 => relu_1
# x_se_7 => convolution_17
triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_30', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/a3/ca3xhv7slrd7n4uue3bthdnawfjne744q74g37p5rjjuh7mkvagq.py
# Source Nodes: [getattr_getattr_l__mod___stages___1_____0___downsample_pool, shortcut_3], Original ATen: [aten.avg_pool2d, aten.convolution]
# getattr_getattr_l__mod___stages___1_____0___downsample_pool => avg_pool2d
# shortcut_3 => convolution_11
triton_poi_fused_avg_pool2d_convolution_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_convolution_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (128*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (128*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (64 + (2*x0) + (128*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (65 + (2*x0) + (128*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hb/chbkzo57bjt77x7wqxfayml3ouiocu2ut6balxn6hr4wvdw6zpvv.py
# Source Nodes: [gelu_10, gelu_11, gelu_8, gelu_9, getattr_getattr_l__mod___stages___1_____0___downsample_pool, mul_19, mul_21, mul__10, mul__11, mul__12, mul__13, mul__9, out_10, out_11, out_12, out_13, out_16, out_17, out_9, shortcut_3, shortcut_4, sigmoid_1, x_5, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.add, aten.avg_pool2d, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# gelu_10 => add_24, erf_10, mul_85, mul_86, mul_87
# gelu_11 => add_27, erf_11, mul_96, mul_97, mul_98
# gelu_8 => add_20, erf_8, mul_71, mul_72, mul_73
# gelu_9 => add_22, erf_9, mul_78, mul_79, mul_80
# getattr_getattr_l__mod___stages___1_____0___downsample_pool => avg_pool2d
# mul_19 => mul_92
# mul_21 => mul_95
# mul__10 => mul_81
# mul__11 => mul_88
# mul__12 => mul_94
# mul__13 => mul_99
# mul__9 => mul_74
# out_10 => convolution_13
# out_11 => convolution_14
# out_12 => convolution_15
# out_13 => mul_93
# out_16 => mul_100
# out_17 => convolution_18
# out_9 => convolution_12
# shortcut_3 => convolution_11
# shortcut_4 => add_26
# sigmoid_1 => sigmoid_1
# x_5 => constant_pad_nd_2
# x_se_4 => mean_1
# x_se_5 => convolution_16
# x_se_6 => relu_1
# x_se_7 => convolution_17
triton_poi_fused_add_avg_pool2d_constant_pad_nd_convolution_gelu_mean_mul_relu_sigmoid_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_constant_pad_nd_convolution_gelu_mean_mul_relu_sigmoid_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 512
    x4 = (xindex // 1024)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp12 = tmp9 * tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp17 = tmp15 + tmp16
    tmp18 = tmp14 + tmp17
    tmp19 = 0.5
    tmp20 = tmp18 * tmp19
    tmp21 = 0.7071067811865476
    tmp22 = tmp18 * tmp21
    tmp23 = tl.math.erf(tmp22)
    tmp24 = 1.0
    tmp25 = tmp23 + tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = 1.7015043497085571
    tmp28 = tmp26 * tmp27
    tmp29 = 0.9805806756909201
    tmp30 = tmp28 * tmp29
    tl.store(in_out_ptr0 + (x3), tmp18, None)
    tl.store(out_ptr0 + (x3), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tc/ctcic3o7o3nqtvhlpq6ynhepmvajnxb5jasx7vlbovpb6wrq6tvs.py
# Source Nodes: [gelu_11, gelu_12, gelu_13, gelu_14, gelu_15, mul_27, mul_29, mul__13, mul__14, mul__15, mul__16, mul__17, mul__18, out_16, out_17, out_18, out_19, out_20, out_21, out_24, shortcut_5, sigmoid_2, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# gelu_11 => add_27, erf_11, mul_96, mul_97, mul_98
# gelu_12 => add_29, erf_12, mul_104, mul_105, mul_106
# gelu_13 => add_31, erf_13, mul_111, mul_112, mul_113
# gelu_14 => add_33, erf_14, mul_118, mul_119, mul_120
# gelu_15 => add_36, erf_15, mul_129, mul_130, mul_131
# mul_27 => mul_125
# mul_29 => mul_128
# mul__13 => mul_99
# mul__14 => mul_107
# mul__15 => mul_114
# mul__16 => mul_121
# mul__17 => mul_127
# mul__18 => mul_132
# out_16 => mul_100
# out_17 => convolution_18
# out_18 => convolution_19
# out_19 => convolution_20
# out_20 => convolution_21
# out_21 => mul_126
# out_24 => mul_133
# shortcut_5 => add_35
# sigmoid_2 => sigmoid_2
# x_se_10 => relu_2
# x_se_11 => convolution_23
# x_se_8 => mean_2
# x_se_9 => convolution_22
triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_33', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 512
    x4 = (xindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp12 = tmp9 * tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = 0.5
    tmp18 = tmp16 * tmp17
    tmp19 = 0.7071067811865476
    tmp20 = tmp16 * tmp19
    tmp21 = tl.math.erf(tmp20)
    tmp22 = 1.0
    tmp23 = tmp21 + tmp22
    tmp24 = tmp18 * tmp23
    tmp25 = 1.7015043497085571
    tmp26 = tmp24 * tmp25
    tmp27 = 0.9622504486493761
    tmp28 = tmp26 * tmp27
    tl.store(in_out_ptr0 + (x3), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xs/cxs7dhitkjccongpcoiwahewhtzm5or3ku4jrazrk6ter5bmafgb.py
# Source Nodes: [gelu_16, mul__19, out_25, out_26, x_7], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
# gelu_16 => add_39, erf_16, mul_140, mul_141, mul_142
# mul__19 => mul_143
# out_25 => convolution_25
# out_26 => convolution_26
# x_7 => constant_pad_nd_3
triton_poi_fused_constant_pad_nd_convolution_gelu_mul_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_gelu_mul_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6690816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 33) % 33
    x0 = xindex % 33
    x4 = (xindex // 1089)
    x2 = (xindex // 1089) % 768
    x5 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 32, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (32*x1) + (1024*x4)), tmp5, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x2), tmp5, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = 0.7071067811865476
    tmp12 = tmp8 * tmp11
    tmp13 = tl.math.erf(tmp12)
    tmp14 = 1.0
    tmp15 = tmp13 + tmp14
    tmp16 = tmp10 * tmp15
    tmp17 = 1.7015043497085571
    tmp18 = tmp16 * tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp5, tmp18, tmp19)
    tl.store(out_ptr0 + (x5), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pf/cpfry6kpae3t6x7vkmckztgzxgmswgrvbjsgayspujexbcdmnvp6.py
# Source Nodes: [gelu_16, gelu_17, mul__19, mul__20, out_25, out_26, out_27, x_7], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
# gelu_16 => add_39, erf_16, mul_140, mul_141, mul_142
# gelu_17 => add_41, erf_17, mul_147, mul_148, mul_149
# mul__19 => mul_143
# mul__20 => mul_150
# out_25 => convolution_25
# out_26 => convolution_26
# out_27 => convolution_27
# x_7 => constant_pad_nd_3
triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 768
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
    tl.store(in_out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qq/cqqzn2i7lkzhz5hbgmemiegcq7fjiouqnx6o4xo4zb6sl4t4dugp.py
# Source Nodes: [gelu_16, gelu_17, gelu_18, mul__19, mul__20, mul__21, out_25, out_26, out_27, out_28, x_7, x_se_12, x_se_13], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mean, aten.mul]
# gelu_16 => add_39, erf_16, mul_140, mul_141, mul_142
# gelu_17 => add_41, erf_17, mul_147, mul_148, mul_149
# gelu_18 => add_43, erf_18, mul_154, mul_155, mul_156
# mul__19 => mul_143
# mul__20 => mul_150
# mul__21 => mul_157
# out_25 => convolution_25
# out_26 => convolution_26
# out_27 => convolution_27
# out_28 => convolution_28
# x_7 => constant_pad_nd_3
# x_se_12 => mean_3
# x_se_13 => convolution_29
triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_36', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel):
    xnumel = 12288
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1536
    tmp0 = tl.load(in_ptr0 + (r2 + (256*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp7 = 256.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gp/cgp5eoxvtlwvavf5nrgnpcl6wgnwl27obkcflbagzef4ke73fzav.py
# Source Nodes: [gelu_16, gelu_17, gelu_18, mul__19, mul__20, mul__21, out_25, out_26, out_27, out_28, x_7, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
# gelu_16 => add_39, erf_16, mul_140, mul_141, mul_142
# gelu_17 => add_41, erf_17, mul_147, mul_148, mul_149
# gelu_18 => add_43, erf_18, mul_154, mul_155, mul_156
# mul__19 => mul_143
# mul__20 => mul_150
# mul__21 => mul_157
# out_25 => convolution_25
# out_26 => convolution_26
# out_27 => convolution_27
# out_28 => convolution_28
# x_7 => constant_pad_nd_3
# x_se_12 => mean_3
# x_se_13 => convolution_29
# x_se_14 => relu_3
# x_se_15 => convolution_30
triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_37', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/tr/ctrkfgxxnjq5orxtzd26w6sj27syvovzjtl3isvbzdixdmprp5de.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____0___downsample_pool, shortcut_6], Original ATen: [aten.avg_pool2d, aten.convolution]
# getattr_getattr_l__mod___stages___2_____0___downsample_pool => avg_pool2d_1
# shortcut_6 => convolution_24
triton_poi_fused_avg_pool2d_convolution_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_convolution_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (64*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (64*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (32 + (2*x0) + (64*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (33 + (2*x0) + (64*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3z/c3zfdzrsq7z3tugtxfanalo3jlngnp2fsecivsg3y36avhpkx7ic.py
# Source Nodes: [gelu_16, gelu_17, gelu_18, gelu_19, getattr_getattr_l__mod___stages___2_____0___downsample_pool, mul_36, mul_38, mul__19, mul__20, mul__21, mul__22, mul__23, out_25, out_26, out_27, out_28, out_29, out_32, out_33, shortcut_6, shortcut_7, sigmoid_3, x_7, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.add, aten.avg_pool2d, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# gelu_16 => add_39, erf_16, mul_140, mul_141, mul_142
# gelu_17 => add_41, erf_17, mul_147, mul_148, mul_149
# gelu_18 => add_43, erf_18, mul_154, mul_155, mul_156
# gelu_19 => add_46, erf_19, mul_165, mul_166, mul_167
# getattr_getattr_l__mod___stages___2_____0___downsample_pool => avg_pool2d_1
# mul_36 => mul_161
# mul_38 => mul_164
# mul__19 => mul_143
# mul__20 => mul_150
# mul__21 => mul_157
# mul__22 => mul_163
# mul__23 => mul_168
# out_25 => convolution_25
# out_26 => convolution_26
# out_27 => convolution_27
# out_28 => convolution_28
# out_29 => mul_162
# out_32 => mul_169
# out_33 => convolution_31
# shortcut_6 => convolution_24
# shortcut_7 => add_45
# sigmoid_3 => sigmoid_3
# x_7 => constant_pad_nd_3
# x_se_12 => mean_3
# x_se_13 => convolution_29
# x_se_14 => relu_3
# x_se_15 => convolution_30
triton_poi_fused_add_avg_pool2d_constant_pad_nd_convolution_gelu_mean_mul_relu_sigmoid_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_constant_pad_nd_convolution_gelu_mean_mul_relu_sigmoid_39', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 1536
    x4 = (xindex // 256)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp12 = tmp9 * tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp17 = tmp15 + tmp16
    tmp18 = tmp14 + tmp17
    tmp19 = 0.5
    tmp20 = tmp18 * tmp19
    tmp21 = 0.7071067811865476
    tmp22 = tmp18 * tmp21
    tmp23 = tl.math.erf(tmp22)
    tmp24 = 1.0
    tmp25 = tmp23 + tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = 1.7015043497085571
    tmp28 = tmp26 * tmp27
    tmp29 = 0.9805806756909201
    tmp30 = tmp28 * tmp29
    tl.store(in_out_ptr0 + (x3), tmp18, None)
    tl.store(out_ptr0 + (x3), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/u7/cu7wq6cgc4ftgugm4mteyamxkcgyueasylct7q2y6sj6j7tctqcl.py
# Source Nodes: [gelu_19, gelu_20, gelu_21, gelu_22, gelu_23, mul_44, mul_46, mul__23, mul__24, mul__25, mul__26, mul__27, mul__28, out_32, out_33, out_34, out_35, out_36, out_37, out_40, out_41, shortcut_8, sigmoid_4, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# gelu_19 => add_46, erf_19, mul_165, mul_166, mul_167
# gelu_20 => add_48, erf_20, mul_173, mul_174, mul_175
# gelu_21 => add_50, erf_21, mul_180, mul_181, mul_182
# gelu_22 => add_52, erf_22, mul_187, mul_188, mul_189
# gelu_23 => add_55, erf_23, mul_198, mul_199, mul_200
# mul_44 => mul_194
# mul_46 => mul_197
# mul__23 => mul_168
# mul__24 => mul_176
# mul__25 => mul_183
# mul__26 => mul_190
# mul__27 => mul_196
# mul__28 => mul_201
# out_32 => mul_169
# out_33 => convolution_31
# out_34 => convolution_32
# out_35 => convolution_33
# out_36 => convolution_34
# out_37 => mul_195
# out_40 => mul_202
# out_41 => convolution_37
# shortcut_8 => add_54
# sigmoid_4 => sigmoid_4
# x_se_16 => mean_4
# x_se_17 => convolution_35
# x_se_18 => relu_4
# x_se_19 => convolution_36
triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_40', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 1536
    x4 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp12 = tmp9 * tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = 0.5
    tmp18 = tmp16 * tmp17
    tmp19 = 0.7071067811865476
    tmp20 = tmp16 * tmp19
    tmp21 = tl.math.erf(tmp20)
    tmp22 = 1.0
    tmp23 = tmp21 + tmp22
    tmp24 = tmp18 * tmp23
    tmp25 = 1.7015043497085571
    tmp26 = tmp24 * tmp25
    tmp27 = 0.9622504486493761
    tmp28 = tmp26 * tmp27
    tl.store(in_out_ptr0 + (x3), tmp16, None)
    tl.store(out_ptr0 + (x3), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/il/cillyi6ziriflvci2q5wglopxxgcyr2hdbbbeuzqbyarlcpiyo3p.py
# Source Nodes: [gelu_23, gelu_24, gelu_25, gelu_26, gelu_27, mul_52, mul_54, mul__28, mul__29, mul__30, mul__31, mul__32, mul__33, out_40, out_41, out_42, out_43, out_44, out_45, out_48, out_49, shortcut_9, sigmoid_5, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# gelu_23 => add_55, erf_23, mul_198, mul_199, mul_200
# gelu_24 => add_57, erf_24, mul_206, mul_207, mul_208
# gelu_25 => add_59, erf_25, mul_213, mul_214, mul_215
# gelu_26 => add_61, erf_26, mul_220, mul_221, mul_222
# gelu_27 => add_64, erf_27, mul_231, mul_232, mul_233
# mul_52 => mul_227
# mul_54 => mul_230
# mul__28 => mul_201
# mul__29 => mul_209
# mul__30 => mul_216
# mul__31 => mul_223
# mul__32 => mul_229
# mul__33 => mul_234
# out_40 => mul_202
# out_41 => convolution_37
# out_42 => convolution_38
# out_43 => convolution_39
# out_44 => convolution_40
# out_45 => mul_228
# out_48 => mul_235
# out_49 => convolution_43
# shortcut_9 => add_63
# sigmoid_5 => sigmoid_5
# x_se_20 => mean_5
# x_se_21 => convolution_41
# x_se_22 => relu_5
# x_se_23 => convolution_42
triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_41', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 1536
    x4 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp12 = tmp9 * tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = 0.5
    tmp18 = tmp16 * tmp17
    tmp19 = 0.7071067811865476
    tmp20 = tmp16 * tmp19
    tmp21 = tl.math.erf(tmp20)
    tmp22 = 1.0
    tmp23 = tmp21 + tmp22
    tmp24 = tmp18 * tmp23
    tmp25 = 1.7015043497085571
    tmp26 = tmp24 * tmp25
    tmp27 = 0.9449111825230679
    tmp28 = tmp26 * tmp27
    tl.store(in_out_ptr0 + (x3), tmp16, None)
    tl.store(out_ptr0 + (x3), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6e/c6esyt2lbcub35gcli3qbmx5ks2umbpm56iqpmeerhpjn2khgjel.py
# Source Nodes: [gelu_27, gelu_28, gelu_29, gelu_30, gelu_31, mul_60, mul_62, mul__33, mul__34, mul__35, mul__36, mul__37, mul__38, out_48, out_49, out_50, out_51, out_52, out_53, out_56, out_57, shortcut_10, sigmoid_6, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# gelu_27 => add_64, erf_27, mul_231, mul_232, mul_233
# gelu_28 => add_66, erf_28, mul_239, mul_240, mul_241
# gelu_29 => add_68, erf_29, mul_246, mul_247, mul_248
# gelu_30 => add_70, erf_30, mul_253, mul_254, mul_255
# gelu_31 => add_73, erf_31, mul_264, mul_265, mul_266
# mul_60 => mul_260
# mul_62 => mul_263
# mul__33 => mul_234
# mul__34 => mul_242
# mul__35 => mul_249
# mul__36 => mul_256
# mul__37 => mul_262
# mul__38 => mul_267
# out_48 => mul_235
# out_49 => convolution_43
# out_50 => convolution_44
# out_51 => convolution_45
# out_52 => convolution_46
# out_53 => mul_261
# out_56 => mul_268
# out_57 => convolution_49
# shortcut_10 => add_72
# sigmoid_6 => sigmoid_6
# x_se_24 => mean_6
# x_se_25 => convolution_47
# x_se_26 => relu_6
# x_se_27 => convolution_48
triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_42', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 1536
    x4 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp12 = tmp9 * tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = 0.5
    tmp18 = tmp16 * tmp17
    tmp19 = 0.7071067811865476
    tmp20 = tmp16 * tmp19
    tmp21 = tl.math.erf(tmp20)
    tmp22 = 1.0
    tmp23 = tmp21 + tmp22
    tmp24 = tmp18 * tmp23
    tmp25 = 1.7015043497085571
    tmp26 = tmp24 * tmp25
    tmp27 = 0.9284766908852592
    tmp28 = tmp26 * tmp27
    tl.store(in_out_ptr0 + (x3), tmp16, None)
    tl.store(out_ptr0 + (x3), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cu/ccuptxytk6764bwqgk3pz57qehtyxu45peya6jxdvktjddh3qidj.py
# Source Nodes: [gelu_31, gelu_32, gelu_33, gelu_34, gelu_35, mul_68, mul_70, mul__38, mul__39, mul__40, mul__41, mul__42, mul__43, out_56, out_57, out_58, out_59, out_60, out_61, out_64, out_65, shortcut_11, sigmoid_7, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# gelu_31 => add_73, erf_31, mul_264, mul_265, mul_266
# gelu_32 => add_75, erf_32, mul_272, mul_273, mul_274
# gelu_33 => add_77, erf_33, mul_279, mul_280, mul_281
# gelu_34 => add_79, erf_34, mul_286, mul_287, mul_288
# gelu_35 => add_82, erf_35, mul_297, mul_298, mul_299
# mul_68 => mul_293
# mul_70 => mul_296
# mul__38 => mul_267
# mul__39 => mul_275
# mul__40 => mul_282
# mul__41 => mul_289
# mul__42 => mul_295
# mul__43 => mul_300
# out_56 => mul_268
# out_57 => convolution_49
# out_58 => convolution_50
# out_59 => convolution_51
# out_60 => convolution_52
# out_61 => mul_294
# out_64 => mul_301
# out_65 => convolution_55
# shortcut_11 => add_81
# sigmoid_7 => sigmoid_7
# x_se_28 => mean_7
# x_se_29 => convolution_53
# x_se_30 => relu_7
# x_se_31 => convolution_54
triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_43', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 1536
    x4 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp12 = tmp9 * tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = 0.5
    tmp18 = tmp16 * tmp17
    tmp19 = 0.7071067811865476
    tmp20 = tmp16 * tmp19
    tmp21 = tl.math.erf(tmp20)
    tmp22 = 1.0
    tmp23 = tmp21 + tmp22
    tmp24 = tmp18 * tmp23
    tmp25 = 1.7015043497085571
    tmp26 = tmp24 * tmp25
    tmp27 = 0.9128709291752768
    tmp28 = tmp26 * tmp27
    tl.store(in_out_ptr0 + (x3), tmp16, None)
    tl.store(out_ptr0 + (x3), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ix/cixpmxu2qqya5hxluenqoifnoh6pjrbo4k3vszrdfb3rm5qon3tm.py
# Source Nodes: [gelu_35, gelu_36, gelu_37, gelu_38, gelu_39, mul_76, mul_78, mul__43, mul__44, mul__45, mul__46, mul__47, mul__48, out_64, out_65, out_66, out_67, out_68, out_69, out_72, shortcut_12, sigmoid_8, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# gelu_35 => add_82, erf_35, mul_297, mul_298, mul_299
# gelu_36 => add_84, erf_36, mul_305, mul_306, mul_307
# gelu_37 => add_86, erf_37, mul_312, mul_313, mul_314
# gelu_38 => add_88, erf_38, mul_319, mul_320, mul_321
# gelu_39 => add_91, erf_39, mul_330, mul_331, mul_332
# mul_76 => mul_326
# mul_78 => mul_329
# mul__43 => mul_300
# mul__44 => mul_308
# mul__45 => mul_315
# mul__46 => mul_322
# mul__47 => mul_328
# mul__48 => mul_333
# out_64 => mul_301
# out_65 => convolution_55
# out_66 => convolution_56
# out_67 => convolution_57
# out_68 => convolution_58
# out_69 => mul_327
# out_72 => mul_334
# shortcut_12 => add_90
# sigmoid_8 => sigmoid_8
# x_se_32 => mean_8
# x_se_33 => convolution_59
# x_se_34 => relu_8
# x_se_35 => convolution_60
triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_44', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 1536
    x4 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp12 = tmp9 * tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = 0.5
    tmp18 = tmp16 * tmp17
    tmp19 = 0.7071067811865476
    tmp20 = tmp16 * tmp19
    tmp21 = tl.math.erf(tmp20)
    tmp22 = 1.0
    tmp23 = tmp21 + tmp22
    tmp24 = tmp18 * tmp23
    tmp25 = 1.7015043497085571
    tmp26 = tmp24 * tmp25
    tmp27 = 0.8980265101338745
    tmp28 = tmp26 * tmp27
    tl.store(in_out_ptr0 + (x3), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/aq/caqpt7gwu3hona7ipexlrwj5d6gh6dcumpvch4mpza5xjoh2xlzo.py
# Source Nodes: [gelu_40, mul__49, out_73, out_74, x_9], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
# gelu_40 => add_94, erf_40, mul_341, mul_342, mul_343
# mul__49 => mul_344
# out_73 => convolution_62
# out_74 => convolution_63
# x_9 => constant_pad_nd_4
triton_poi_fused_constant_pad_nd_convolution_gelu_mul_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_gelu_mul_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1775616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 17) % 17
    x0 = xindex % 17
    x4 = (xindex // 289)
    x2 = (xindex // 289) % 768
    x5 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 16, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = x0
    tmp4 = tmp3 < tmp1
    tmp5 = tmp2 & tmp4
    tmp6 = tl.load(in_ptr0 + (x0 + (16*x1) + (256*x4)), tmp5, other=0.0)
    tmp7 = tl.load(in_ptr1 + (x2), tmp5, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = 0.7071067811865476
    tmp12 = tmp8 * tmp11
    tmp13 = tl.math.erf(tmp12)
    tmp14 = 1.0
    tmp15 = tmp13 + tmp14
    tmp16 = tmp10 * tmp15
    tmp17 = 1.7015043497085571
    tmp18 = tmp16 * tmp17
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp5, tmp18, tmp19)
    tl.store(out_ptr0 + (x5), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4n/c4n5xvkwuljmxk6m2xofptdeb3titofs6b6kksl2ntsnzm4b4rn3.py
# Source Nodes: [gelu_40, gelu_41, mul__49, mul__50, out_73, out_74, out_75, x_9], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
# gelu_40 => add_94, erf_40, mul_341, mul_342, mul_343
# gelu_41 => add_96, erf_41, mul_348, mul_349, mul_350
# mul__49 => mul_344
# mul__50 => mul_351
# out_73 => convolution_62
# out_74 => convolution_63
# out_75 => convolution_64
# x_9 => constant_pad_nd_4
triton_poi_fused_constant_pad_nd_convolution_gelu_mul_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_convolution_gelu_mul_46', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 768
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
    tl.store(in_out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yf/cyfwqizjvqrncwgtq52ei5adl56zikr4lzt4ox52uej3jgiyw6fn.py
# Source Nodes: [gelu_40, gelu_41, gelu_42, mul__49, mul__50, mul__51, out_73, out_74, out_75, out_76, x_9, x_se_36, x_se_37], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mean, aten.mul]
# gelu_40 => add_94, erf_40, mul_341, mul_342, mul_343
# gelu_41 => add_96, erf_41, mul_348, mul_349, mul_350
# gelu_42 => add_98, erf_42, mul_355, mul_356, mul_357
# mul__49 => mul_344
# mul__50 => mul_351
# mul__51 => mul_358
# out_73 => convolution_62
# out_74 => convolution_63
# out_75 => convolution_64
# out_76 => convolution_65
# x_9 => constant_pad_nd_4
# x_se_36 => mean_9
# x_se_37 => convolution_66
triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_47', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1536
    tmp0 = tl.load(in_ptr0 + (r2 + (64*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 64.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zs/czssfeziubobpzr5ylhpoexkmbr5csy3ginljasmzieqaa5ovvxt.py
# Source Nodes: [getattr_getattr_l__mod___stages___3_____0___downsample_pool, shortcut_13], Original ATen: [aten.avg_pool2d, aten.convolution]
# getattr_getattr_l__mod___stages___3_____0___downsample_pool => avg_pool2d_2
# shortcut_13 => convolution_61
triton_poi_fused_avg_pool2d_convolution_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_convolution_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 8
    x1 = (xindex // 8)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (32*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (32*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (16 + (2*x0) + (32*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (17 + (2*x0) + (32*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ix/cixioe4jzvkmmpsn2ftwe5se6hf6rryolfnqtp6zgjwasr6agf6w.py
# Source Nodes: [gelu_40, gelu_41, gelu_42, gelu_43, getattr_getattr_l__mod___stages___3_____0___downsample_pool, mul_85, mul_87, mul__49, mul__50, mul__51, mul__52, mul__53, out_73, out_74, out_75, out_76, out_77, out_80, out_81, shortcut_13, shortcut_14, sigmoid_9, x_9, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.add, aten.avg_pool2d, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# gelu_40 => add_94, erf_40, mul_341, mul_342, mul_343
# gelu_41 => add_96, erf_41, mul_348, mul_349, mul_350
# gelu_42 => add_98, erf_42, mul_355, mul_356, mul_357
# gelu_43 => add_101, erf_43, mul_366, mul_367, mul_368
# getattr_getattr_l__mod___stages___3_____0___downsample_pool => avg_pool2d_2
# mul_85 => mul_362
# mul_87 => mul_365
# mul__49 => mul_344
# mul__50 => mul_351
# mul__51 => mul_358
# mul__52 => mul_364
# mul__53 => mul_369
# out_73 => convolution_62
# out_74 => convolution_63
# out_75 => convolution_64
# out_76 => convolution_65
# out_77 => mul_363
# out_80 => mul_370
# out_81 => convolution_68
# shortcut_13 => convolution_61
# shortcut_14 => add_100
# sigmoid_9 => sigmoid_9
# x_9 => constant_pad_nd_4
# x_se_36 => mean_9
# x_se_37 => convolution_66
# x_se_38 => relu_9
# x_se_39 => convolution_67
triton_poi_fused_add_avg_pool2d_constant_pad_nd_convolution_gelu_mean_mul_relu_sigmoid_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_constant_pad_nd_convolution_gelu_mean_mul_relu_sigmoid_49', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 1536
    x4 = (xindex // 64)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp12 = tmp9 * tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp17 = tmp15 + tmp16
    tmp18 = tmp14 + tmp17
    tmp19 = 0.5
    tmp20 = tmp18 * tmp19
    tmp21 = 0.7071067811865476
    tmp22 = tmp18 * tmp21
    tmp23 = tl.math.erf(tmp22)
    tmp24 = 1.0
    tmp25 = tmp23 + tmp24
    tmp26 = tmp20 * tmp25
    tmp27 = 1.7015043497085571
    tmp28 = tmp26 * tmp27
    tmp29 = 0.9805806756909201
    tmp30 = tmp28 * tmp29
    tl.store(in_out_ptr0 + (x3), tmp18, None)
    tl.store(out_ptr0 + (x3), tmp30, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lv/clvdadhdnsrrqn54sx3ww4vqnsrj7zrudvb3st2xv6aw6wrcp5u4.py
# Source Nodes: [gelu_43, gelu_44, gelu_45, gelu_46, gelu_47, mul_93, mul_95, mul__53, mul__54, mul__55, mul__56, mul__57, mul__58, out_80, out_81, out_82, out_83, out_84, out_85, out_88, out_89, shortcut_15, sigmoid_10, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# gelu_43 => add_101, erf_43, mul_366, mul_367, mul_368
# gelu_44 => add_103, erf_44, mul_374, mul_375, mul_376
# gelu_45 => add_105, erf_45, mul_381, mul_382, mul_383
# gelu_46 => add_107, erf_46, mul_388, mul_389, mul_390
# gelu_47 => add_110, erf_47, mul_399, mul_400, mul_401
# mul_93 => mul_395
# mul_95 => mul_398
# mul__53 => mul_369
# mul__54 => mul_377
# mul__55 => mul_384
# mul__56 => mul_391
# mul__57 => mul_397
# mul__58 => mul_402
# out_80 => mul_370
# out_81 => convolution_68
# out_82 => convolution_69
# out_83 => convolution_70
# out_84 => convolution_71
# out_85 => mul_396
# out_88 => mul_403
# out_89 => convolution_74
# shortcut_15 => add_109
# sigmoid_10 => sigmoid_10
# x_se_40 => mean_10
# x_se_41 => convolution_72
# x_se_42 => relu_10
# x_se_43 => convolution_73
triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_50 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_50', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 1536
    x4 = (xindex // 64)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp12 = tmp9 * tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = 0.5
    tmp18 = tmp16 * tmp17
    tmp19 = 0.7071067811865476
    tmp20 = tmp16 * tmp19
    tmp21 = tl.math.erf(tmp20)
    tmp22 = 1.0
    tmp23 = tmp21 + tmp22
    tmp24 = tmp18 * tmp23
    tmp25 = 1.7015043497085571
    tmp26 = tmp24 * tmp25
    tmp27 = 0.9622504486493761
    tmp28 = tmp26 * tmp27
    tl.store(in_out_ptr0 + (x3), tmp16, None)
    tl.store(out_ptr0 + (x3), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ih/cih3o6ioxzgps5ixz2e35sf533dlc6nz56mead6mvc6t4dpxk67y.py
# Source Nodes: [gelu_47, gelu_48, gelu_49, gelu_50, mul_101, mul_103, mul__58, mul__59, mul__60, mul__61, mul__62, out_88, out_89, out_90, out_91, out_92, out_93, sigmoid_11, x_10, x_11, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# gelu_47 => add_110, erf_47, mul_399, mul_400, mul_401
# gelu_48 => add_112, erf_48, mul_407, mul_408, mul_409
# gelu_49 => add_114, erf_49, mul_414, mul_415, mul_416
# gelu_50 => add_116, erf_50, mul_421, mul_422, mul_423
# mul_101 => mul_428
# mul_103 => mul_431
# mul__58 => mul_402
# mul__59 => mul_410
# mul__60 => mul_417
# mul__61 => mul_424
# mul__62 => mul_430
# out_88 => mul_403
# out_89 => convolution_74
# out_90 => convolution_75
# out_91 => convolution_76
# out_92 => convolution_77
# out_93 => mul_429
# sigmoid_11 => sigmoid_11
# x_10 => add_118
# x_11 => convolution_80
# x_se_44 => mean_11
# x_se_45 => convolution_78
# x_se_46 => relu_11
# x_se_47 => convolution_79
triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_51', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 1536
    x4 = (xindex // 64)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr4 + (0))
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK])
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp12 = tmp9 * tmp11
    tmp13 = 0.2
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/k6/ck6ojpguvco64eugtnpm3d4fdrfjk7nfdrat3i24lxzbgyqg45bd.py
# Source Nodes: [gelu_47, gelu_48, gelu_49, gelu_50, gelu_51, mul_101, mul_103, mul__58, mul__59, mul__60, mul__61, mul__62, out_88, out_89, out_90, out_91, out_92, out_93, sigmoid_11, x_10, x_11, x_13, x_14, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
# gelu_47 => add_110, erf_47, mul_399, mul_400, mul_401
# gelu_48 => add_112, erf_48, mul_407, mul_408, mul_409
# gelu_49 => add_114, erf_49, mul_414, mul_415, mul_416
# gelu_50 => add_116, erf_50, mul_421, mul_422, mul_423
# gelu_51 => add_120, erf_51, mul_435, mul_436, mul_437
# mul_101 => mul_428
# mul_103 => mul_431
# mul__58 => mul_402
# mul__59 => mul_410
# mul__60 => mul_417
# mul__61 => mul_424
# mul__62 => mul_430
# out_88 => mul_403
# out_89 => convolution_74
# out_90 => convolution_75
# out_91 => convolution_76
# out_92 => convolution_77
# out_93 => mul_429
# sigmoid_11 => sigmoid_11
# x_10 => add_118
# x_11 => convolution_80
# x_13 => mul_438
# x_14 => mean_12
# x_se_44 => mean_11
# x_se_45 => convolution_78
# x_se_46 => relu_11
# x_se_47 => convolution_79
triton_per_fused_add_convolution_gelu_mean_mul_relu_sigmoid_52 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_gelu_mean_mul_relu_sigmoid_52', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 3072
    tmp0 = tl.load(in_ptr0 + (r2 + (64*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
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
    tmp17 = 64.0
    tmp18 = tmp16 / tmp17
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp18, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (16, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg2_1, (16, ), (1, ))
    assert_size_stride(arg3_1, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg4_1, (32, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg5_1, (32, ), (1, ))
    assert_size_stride(arg6_1, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg7_1, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg10_1, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg11_1, (128, ), (1, ))
    assert_size_stride(arg12_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg13_1, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg14_1, (256, ), (1, ))
    assert_size_stride(arg15_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg16_1, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg17_1, (128, ), (1, ))
    assert_size_stride(arg18_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg19_1, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg20_1, (128, ), (1, ))
    assert_size_stride(arg21_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg22_1, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg23_1, (128, ), (1, ))
    assert_size_stride(arg24_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg25_1, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg26_1, (256, ), (1, ))
    assert_size_stride(arg27_1, (), ())
    assert_size_stride(arg28_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg29_1, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg30_1, (512, ), (1, ))
    assert_size_stride(arg31_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg32_1, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg33_1, (256, ), (1, ))
    assert_size_stride(arg34_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg35_1, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg36_1, (256, ), (1, ))
    assert_size_stride(arg37_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg38_1, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg39_1, (256, ), (1, ))
    assert_size_stride(arg40_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg41_1, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg42_1, (512, ), (1, ))
    assert_size_stride(arg43_1, (), ())
    assert_size_stride(arg44_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg45_1, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg46_1, (256, ), (1, ))
    assert_size_stride(arg47_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg48_1, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg49_1, (256, ), (1, ))
    assert_size_stride(arg50_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg51_1, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg53_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg54_1, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg55_1, (512, ), (1, ))
    assert_size_stride(arg56_1, (), ())
    assert_size_stride(arg57_1, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg58_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg59_1, (1536, ), (1, ))
    assert_size_stride(arg60_1, (768, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg61_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg62_1, (768, ), (1, ))
    assert_size_stride(arg63_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg64_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg65_1, (768, ), (1, ))
    assert_size_stride(arg66_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg67_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg68_1, (768, ), (1, ))
    assert_size_stride(arg69_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg70_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg71_1, (1536, ), (1, ))
    assert_size_stride(arg72_1, (), ())
    assert_size_stride(arg73_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg74_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg75_1, (768, ), (1, ))
    assert_size_stride(arg76_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg77_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg78_1, (768, ), (1, ))
    assert_size_stride(arg79_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg80_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg81_1, (768, ), (1, ))
    assert_size_stride(arg82_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg83_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg84_1, (1536, ), (1, ))
    assert_size_stride(arg85_1, (), ())
    assert_size_stride(arg86_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg87_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg88_1, (768, ), (1, ))
    assert_size_stride(arg89_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg90_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg91_1, (768, ), (1, ))
    assert_size_stride(arg92_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg93_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg94_1, (768, ), (1, ))
    assert_size_stride(arg95_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg96_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg97_1, (1536, ), (1, ))
    assert_size_stride(arg98_1, (), ())
    assert_size_stride(arg99_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg100_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg101_1, (768, ), (1, ))
    assert_size_stride(arg102_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg103_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg104_1, (768, ), (1, ))
    assert_size_stride(arg105_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg106_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg107_1, (768, ), (1, ))
    assert_size_stride(arg108_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg109_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg110_1, (1536, ), (1, ))
    assert_size_stride(arg111_1, (), ())
    assert_size_stride(arg112_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg113_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg114_1, (768, ), (1, ))
    assert_size_stride(arg115_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg116_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg117_1, (768, ), (1, ))
    assert_size_stride(arg118_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg119_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg120_1, (768, ), (1, ))
    assert_size_stride(arg121_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg122_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg123_1, (1536, ), (1, ))
    assert_size_stride(arg124_1, (), ())
    assert_size_stride(arg125_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg126_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg127_1, (768, ), (1, ))
    assert_size_stride(arg128_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg129_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg130_1, (768, ), (1, ))
    assert_size_stride(arg131_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg132_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg133_1, (768, ), (1, ))
    assert_size_stride(arg134_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg135_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg136_1, (1536, ), (1, ))
    assert_size_stride(arg137_1, (), ())
    assert_size_stride(arg138_1, (1536, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg139_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg140_1, (1536, ), (1, ))
    assert_size_stride(arg141_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg142_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg143_1, (768, ), (1, ))
    assert_size_stride(arg144_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg145_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg146_1, (768, ), (1, ))
    assert_size_stride(arg147_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg148_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg149_1, (768, ), (1, ))
    assert_size_stride(arg150_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg151_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg152_1, (1536, ), (1, ))
    assert_size_stride(arg153_1, (), ())
    assert_size_stride(arg154_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg155_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg156_1, (768, ), (1, ))
    assert_size_stride(arg157_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg158_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg159_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg161_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg162_1, (768, ), (1, ))
    assert_size_stride(arg163_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg164_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg165_1, (1536, ), (1, ))
    assert_size_stride(arg166_1, (), ())
    assert_size_stride(arg167_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg168_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg169_1, (768, ), (1, ))
    assert_size_stride(arg170_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg171_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg172_1, (768, ), (1, ))
    assert_size_stride(arg173_1, (768, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg174_1, (768, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg175_1, (768, ), (1, ))
    assert_size_stride(arg176_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg177_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg178_1, (1536, ), (1, ))
    assert_size_stride(arg179_1, (), ())
    assert_size_stride(arg180_1, (3072, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg181_1, (3072, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg182_1, (3072, ), (1, ))
    assert_size_stride(arg183_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg184_1, (128, ), (1, ))
    assert_size_stride(arg185_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg186_1, (256, ), (1, ))
    assert_size_stride(arg187_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg188_1, (256, ), (1, ))
    assert_size_stride(arg189_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg190_1, (512, ), (1, ))
    assert_size_stride(arg191_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg192_1, (256, ), (1, ))
    assert_size_stride(arg193_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg194_1, (512, ), (1, ))
    assert_size_stride(arg195_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg196_1, (768, ), (1, ))
    assert_size_stride(arg197_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg198_1, (1536, ), (1, ))
    assert_size_stride(arg199_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg200_1, (768, ), (1, ))
    assert_size_stride(arg201_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg202_1, (1536, ), (1, ))
    assert_size_stride(arg203_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg204_1, (768, ), (1, ))
    assert_size_stride(arg205_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg206_1, (1536, ), (1, ))
    assert_size_stride(arg207_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg208_1, (768, ), (1, ))
    assert_size_stride(arg209_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg210_1, (1536, ), (1, ))
    assert_size_stride(arg211_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg212_1, (768, ), (1, ))
    assert_size_stride(arg213_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg214_1, (1536, ), (1, ))
    assert_size_stride(arg215_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg216_1, (768, ), (1, ))
    assert_size_stride(arg217_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg218_1, (1536, ), (1, ))
    assert_size_stride(arg219_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg220_1, (768, ), (1, ))
    assert_size_stride(arg221_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg222_1, (1536, ), (1, ))
    assert_size_stride(arg223_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg224_1, (768, ), (1, ))
    assert_size_stride(arg225_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg226_1, (1536, ), (1, ))
    assert_size_stride(arg227_1, (768, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg228_1, (768, ), (1, ))
    assert_size_stride(arg229_1, (1536, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg230_1, (1536, ), (1, ))
    assert_size_stride(arg231_1, (1000, 3072), (3072, 1))
    assert_size_stride(arg232_1, (1000, ), (1, ))
    assert_size_stride(arg233_1, (8, 3, 256, 256), (196608, 65536, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf172 = empty((16, 3, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm, conv2d, x], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_0.run(arg0_1, arg1_1, buf172, 16, 27, grid=grid(16), stream=stream0)
        del arg0_1
        del arg1_1
        buf175 = empty((32, 16, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_1, conv2d, conv2d_1, gelu, mul_, x], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_1.run(arg3_1, arg4_1, buf175, 32, 144, grid=grid(32), stream=stream0)
        del arg3_1
        del arg4_1
        buf178 = empty((64, 32, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_2, conv2d, conv2d_1, conv2d_2, gelu, gelu_1, mul_, mul__1, x], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_2.run(arg6_1, arg7_1, buf178, 64, 288, grid=grid(64), stream=stream0)
        del arg6_1
        del arg7_1
        buf181 = empty((128, 64, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_3, conv2d, conv2d_1, conv2d_2, gelu, gelu_1, gelu_2, mul_, mul__1, mul__2, shortcut, x, x_2], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_3.run(arg9_1, arg10_1, buf181, 128, 576, grid=grid(128), stream=stream0)
        del arg10_1
        del arg9_1
        buf200 = empty((256, 128, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_4, shortcut_1], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
        triton_per_fused__native_batch_norm_legit_convolution_4.run(arg12_1, arg13_1, buf200, 256, 128, grid=grid(256), stream=stream0)
        del arg12_1
        del arg13_1
        buf184 = empty((128, 128, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_5, out_1], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
        triton_per_fused__native_batch_norm_legit_convolution_5.run(arg15_1, arg16_1, buf184, 128, 128, grid=grid(128), stream=stream0)
        del arg15_1
        del arg16_1
        buf187 = empty((128, 128, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_6, gelu_4, mul__4, out_1, out_2], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_6.run(arg18_1, arg19_1, buf187, 128, 1152, grid=grid(128), stream=stream0)
        del arg18_1
        del arg19_1
        buf190 = empty((128, 128, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_7, gelu_4, gelu_5, mul__4, mul__5, out_1, out_2, out_3], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_6.run(arg21_1, arg22_1, buf190, 128, 1152, grid=grid(128), stream=stream0)
        del arg21_1
        del arg22_1
        buf193 = empty((256, 128, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_8, gelu_4, gelu_5, gelu_6, mul__4, mul__5, mul__6, out_1, out_2, out_3, out_4], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_per_fused__native_batch_norm_legit_convolution_4.run(arg24_1, arg25_1, buf193, 256, 128, grid=grid(256), stream=stream0)
        del arg24_1
        del arg25_1
        buf221 = empty((512, 256, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_9, getattr_getattr_l__mod___stages___1_____0___downsample_pool, shortcut_3], Original ATen: [aten._native_batch_norm_legit, aten.avg_pool2d, aten.convolution]
        triton_per_fused__native_batch_norm_legit_avg_pool2d_convolution_7.run(arg28_1, arg29_1, buf221, 512, 256, grid=grid(512), stream=stream0)
        del arg28_1
        del arg29_1
        buf204 = empty((256, 256, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_10, out_9], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
        triton_per_fused__native_batch_norm_legit_convolution_8.run(arg31_1, arg32_1, buf204, 256, 256, grid=grid(256), stream=stream0)
        del arg31_1
        del arg32_1
        buf207 = empty((256, 128, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_11, gelu_8, mul__9, out_10, out_9, x_5], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_9.run(arg34_1, arg35_1, buf207, 256, 1152, grid=grid(256), stream=stream0)
        del arg34_1
        del arg35_1
        buf210 = empty((256, 128, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_12, gelu_8, gelu_9, mul__10, mul__9, out_10, out_11, out_9, x_5], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_9.run(arg37_1, arg38_1, buf210, 256, 1152, grid=grid(256), stream=stream0)
        del arg37_1
        del arg38_1
        buf213 = empty((512, 256, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_13, gelu_10, gelu_8, gelu_9, mul__10, mul__11, mul__9, out_10, out_11, out_12, out_9, x_5], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_per_fused__native_batch_norm_legit_avg_pool2d_convolution_7.run(arg40_1, arg41_1, buf213, 512, 256, grid=grid(512), stream=stream0)
        del arg40_1
        del arg41_1
        buf225 = empty((256, 512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_14, gelu_11, mul__13, out_16, out_17], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_per_fused__native_batch_norm_legit_convolution_gelu_mul_10.run(arg44_1, arg45_1, buf225, 256, 512, grid=grid(256), stream=stream0)
        del arg44_1
        del arg45_1
        buf228 = empty((256, 128, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_15, gelu_11, gelu_12, mul__13, mul__14, out_16, out_17, out_18], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_9.run(arg47_1, arg48_1, buf228, 256, 1152, grid=grid(256), stream=stream0)
        del arg47_1
        del arg48_1
        buf231 = empty((256, 128, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_16, gelu_11, gelu_12, gelu_13, mul__13, mul__14, mul__15, out_16, out_17, out_18, out_19], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_9.run(arg50_1, arg51_1, buf231, 256, 1152, grid=grid(256), stream=stream0)
        del arg50_1
        del arg51_1
        buf234 = empty((512, 256, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_17, gelu_11, gelu_12, gelu_13, gelu_14, mul__13, mul__14, mul__15, mul__16, out_16, out_17, out_18, out_19, out_20], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_per_fused__native_batch_norm_legit_avg_pool2d_convolution_7.run(arg53_1, arg54_1, buf234, 512, 256, grid=grid(512), stream=stream0)
        del arg53_1
        del arg54_1
        buf260 = empty((1536, 512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_18, getattr_getattr_l__mod___stages___2_____0___downsample_pool, shortcut_6], Original ATen: [aten._native_batch_norm_legit, aten.avg_pool2d, aten.convolution]
        triton_per_fused__native_batch_norm_legit_avg_pool2d_convolution_11.run(arg57_1, arg58_1, buf260, 1536, 512, grid=grid(1536), stream=stream0)
        del arg57_1
        del arg58_1
        buf243 = empty((768, 512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_19, out_25], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
        triton_per_fused__native_batch_norm_legit_convolution_12.run(arg60_1, arg61_1, buf243, 768, 512, grid=grid(768), stream=stream0)
        del arg60_1
        del arg61_1
        buf246 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_20, gelu_16, mul__19, out_25, out_26, x_7], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg63_1, arg64_1, buf246, 768, 1152, grid=grid(768), stream=stream0)
        del arg63_1
        del arg64_1
        buf249 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_21, gelu_16, gelu_17, mul__19, mul__20, out_25, out_26, out_27, x_7], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg66_1, arg67_1, buf249, 768, 1152, grid=grid(768), stream=stream0)
        del arg66_1
        del arg67_1
        buf252 = empty((1536, 768, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_22, gelu_16, gelu_17, gelu_18, mul__19, mul__20, mul__21, out_25, out_26, out_27, out_28, x_7], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_14.run(arg69_1, arg70_1, buf252, 1536, 768, grid=grid(1536), stream=stream0)
        del arg69_1
        del arg70_1
        buf264 = empty((768, 1536, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_23, gelu_19, mul__23, out_32, out_33], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_15.run(arg73_1, arg74_1, buf264, 768, 1536, grid=grid(768), stream=stream0)
        del arg73_1
        del arg74_1
        buf267 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_24, gelu_19, gelu_20, mul__23, mul__24, out_32, out_33, out_34], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg76_1, arg77_1, buf267, 768, 1152, grid=grid(768), stream=stream0)
        del arg76_1
        del arg77_1
        buf270 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_25, gelu_19, gelu_20, gelu_21, mul__23, mul__24, mul__25, out_32, out_33, out_34, out_35], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg79_1, arg80_1, buf270, 768, 1152, grid=grid(768), stream=stream0)
        del arg79_1
        del arg80_1
        buf273 = empty((1536, 768, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_26, gelu_19, gelu_20, gelu_21, gelu_22, mul__23, mul__24, mul__25, mul__26, out_32, out_33, out_34, out_35, out_36], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_14.run(arg82_1, arg83_1, buf273, 1536, 768, grid=grid(1536), stream=stream0)
        del arg82_1
        del arg83_1
        buf282 = empty((768, 1536, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_27, gelu_23, mul__28, out_40, out_41], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_15.run(arg86_1, arg87_1, buf282, 768, 1536, grid=grid(768), stream=stream0)
        del arg86_1
        del arg87_1
        buf285 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_28, gelu_23, gelu_24, mul__28, mul__29, out_40, out_41, out_42], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg89_1, arg90_1, buf285, 768, 1152, grid=grid(768), stream=stream0)
        del arg89_1
        del arg90_1
        buf288 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_29, gelu_23, gelu_24, gelu_25, mul__28, mul__29, mul__30, out_40, out_41, out_42, out_43], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg92_1, arg93_1, buf288, 768, 1152, grid=grid(768), stream=stream0)
        del arg92_1
        del arg93_1
        buf291 = empty((1536, 768, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_30, gelu_23, gelu_24, gelu_25, gelu_26, mul__28, mul__29, mul__30, mul__31, out_40, out_41, out_42, out_43, out_44], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_14.run(arg95_1, arg96_1, buf291, 1536, 768, grid=grid(1536), stream=stream0)
        del arg95_1
        del arg96_1
        buf300 = empty((768, 1536, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_31, gelu_27, mul__33, out_48, out_49], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_15.run(arg99_1, arg100_1, buf300, 768, 1536, grid=grid(768), stream=stream0)
        del arg100_1
        del arg99_1
        buf303 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_32, gelu_27, gelu_28, mul__33, mul__34, out_48, out_49, out_50], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg102_1, arg103_1, buf303, 768, 1152, grid=grid(768), stream=stream0)
        del arg102_1
        del arg103_1
        buf306 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_33, gelu_27, gelu_28, gelu_29, mul__33, mul__34, mul__35, out_48, out_49, out_50, out_51], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg105_1, arg106_1, buf306, 768, 1152, grid=grid(768), stream=stream0)
        del arg105_1
        del arg106_1
        buf309 = empty((1536, 768, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_34, gelu_27, gelu_28, gelu_29, gelu_30, mul__33, mul__34, mul__35, mul__36, out_48, out_49, out_50, out_51, out_52], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_14.run(arg108_1, arg109_1, buf309, 1536, 768, grid=grid(1536), stream=stream0)
        del arg108_1
        del arg109_1
        buf318 = empty((768, 1536, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_35, gelu_31, mul__38, out_56, out_57], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_15.run(arg112_1, arg113_1, buf318, 768, 1536, grid=grid(768), stream=stream0)
        del arg112_1
        del arg113_1
        buf321 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_36, gelu_31, gelu_32, mul__38, mul__39, out_56, out_57, out_58], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg115_1, arg116_1, buf321, 768, 1152, grid=grid(768), stream=stream0)
        del arg115_1
        del arg116_1
        buf324 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_37, gelu_31, gelu_32, gelu_33, mul__38, mul__39, mul__40, out_56, out_57, out_58, out_59], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg118_1, arg119_1, buf324, 768, 1152, grid=grid(768), stream=stream0)
        del arg118_1
        del arg119_1
        buf327 = empty((1536, 768, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_38, gelu_31, gelu_32, gelu_33, gelu_34, mul__38, mul__39, mul__40, mul__41, out_56, out_57, out_58, out_59, out_60], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_14.run(arg121_1, arg122_1, buf327, 1536, 768, grid=grid(1536), stream=stream0)
        del arg121_1
        del arg122_1
        buf336 = empty((768, 1536, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_39, gelu_35, mul__43, out_64, out_65], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_15.run(arg125_1, arg126_1, buf336, 768, 1536, grid=grid(768), stream=stream0)
        del arg125_1
        del arg126_1
        buf339 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_40, gelu_35, gelu_36, mul__43, mul__44, out_64, out_65, out_66], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg128_1, arg129_1, buf339, 768, 1152, grid=grid(768), stream=stream0)
        del arg128_1
        del arg129_1
        buf342 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_41, gelu_35, gelu_36, gelu_37, mul__43, mul__44, mul__45, out_64, out_65, out_66, out_67], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg131_1, arg132_1, buf342, 768, 1152, grid=grid(768), stream=stream0)
        del arg131_1
        del arg132_1
        buf345 = empty((1536, 768, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_42, gelu_35, gelu_36, gelu_37, gelu_38, mul__43, mul__44, mul__45, mul__46, out_64, out_65, out_66, out_67, out_68], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_14.run(arg134_1, arg135_1, buf345, 1536, 768, grid=grid(1536), stream=stream0)
        del arg134_1
        del arg135_1
        buf371 = empty((1536, 1536, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_43, getattr_getattr_l__mod___stages___3_____0___downsample_pool, shortcut_13], Original ATen: [aten._native_batch_norm_legit, aten.avg_pool2d, aten.convolution]
        triton_red_fused__native_batch_norm_legit_avg_pool2d_convolution_16.run(arg138_1, arg139_1, buf371, 1536, 1536, grid=grid(1536), stream=stream0)
        del arg138_1
        del arg139_1
        buf354 = empty((768, 1536, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_44, out_73], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
        triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_15.run(arg141_1, arg142_1, buf354, 768, 1536, grid=grid(768), stream=stream0)
        del arg141_1
        del arg142_1
        buf357 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_45, gelu_40, mul__49, out_73, out_74, x_9], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg144_1, arg145_1, buf357, 768, 1152, grid=grid(768), stream=stream0)
        del arg144_1
        del arg145_1
        buf360 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_46, gelu_40, gelu_41, mul__49, mul__50, out_73, out_74, out_75, x_9], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg147_1, arg148_1, buf360, 768, 1152, grid=grid(768), stream=stream0)
        del arg147_1
        del arg148_1
        buf363 = empty((1536, 768, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_47, gelu_40, gelu_41, gelu_42, mul__49, mul__50, mul__51, out_73, out_74, out_75, out_76, x_9], Original ATen: [aten._native_batch_norm_legit, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_14.run(arg150_1, arg151_1, buf363, 1536, 768, grid=grid(1536), stream=stream0)
        del arg150_1
        del arg151_1
        buf375 = empty((768, 1536, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_48, gelu_43, mul__53, out_80, out_81], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_15.run(arg154_1, arg155_1, buf375, 768, 1536, grid=grid(768), stream=stream0)
        del arg154_1
        del arg155_1
        buf378 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_49, gelu_43, gelu_44, mul__53, mul__54, out_80, out_81, out_82], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg157_1, arg158_1, buf378, 768, 1152, grid=grid(768), stream=stream0)
        del arg157_1
        del arg158_1
        buf381 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_50, gelu_43, gelu_44, gelu_45, mul__53, mul__54, mul__55, out_80, out_81, out_82, out_83], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg160_1, arg161_1, buf381, 768, 1152, grid=grid(768), stream=stream0)
        del arg160_1
        del arg161_1
        buf384 = empty((1536, 768, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_51, gelu_43, gelu_44, gelu_45, gelu_46, mul__53, mul__54, mul__55, mul__56, out_80, out_81, out_82, out_83, out_84], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_14.run(arg163_1, arg164_1, buf384, 1536, 768, grid=grid(1536), stream=stream0)
        del arg163_1
        del arg164_1
        buf393 = empty((768, 1536, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_52, gelu_47, mul__58, out_88, out_89], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_convolution_gelu_mul_15.run(arg167_1, arg168_1, buf393, 768, 1536, grid=grid(768), stream=stream0)
        del arg167_1
        del arg168_1
        buf396 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_53, gelu_47, gelu_48, mul__58, mul__59, out_88, out_89, out_90], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg170_1, arg171_1, buf396, 768, 1152, grid=grid(768), stream=stream0)
        del arg170_1
        del arg171_1
        buf399 = empty((768, 128, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_54, gelu_47, gelu_48, gelu_49, mul__58, mul__59, mul__60, out_88, out_89, out_90, out_91], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_red_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_13.run(arg173_1, arg174_1, buf399, 768, 1152, grid=grid(768), stream=stream0)
        del arg173_1
        del arg174_1
        buf402 = empty((1536, 768, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_55, gelu_47, gelu_48, gelu_49, gelu_50, mul__58, mul__59, mul__60, mul__61, out_88, out_89, out_90, out_91, out_92], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.gelu, aten.mul]
        triton_per_fused__native_batch_norm_legit_constant_pad_nd_convolution_gelu_mul_14.run(arg176_1, arg177_1, buf402, 1536, 768, grid=grid(1536), stream=stream0)
        del arg176_1
        del arg177_1
        buf410 = empty((3072, 1536, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_56, gelu_47, gelu_48, gelu_49, gelu_50, mul_101, mul_103, mul__58, mul__59, mul__60, mul__61, mul__62, out_88, out_89, out_90, out_91, out_92, out_93, sigmoid_11, x_10, x_11, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten._native_batch_norm_legit, aten.add, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_red_fused__native_batch_norm_legit_add_convolution_gelu_mean_mul_relu_sigmoid_17.run(arg180_1, arg181_1, buf410, 3072, 1536, grid=grid(3072), stream=stream0)
        del arg180_1
        del arg181_1
        buf171 = empty((8, 3, 257, 257), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv2d, x], Original ATen: [aten.constant_pad_nd, aten.convolution]
        triton_poi_fused_constant_pad_nd_convolution_18.run(arg233_1, buf171, 1585176, grid=grid(1585176), stream=stream0)
        del arg233_1
        # Source Nodes: [conv2d, x], Original ATen: [aten.constant_pad_nd, aten.convolution]
        buf173 = extern_kernels.convolution(buf171, buf172, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (8, 16, 128, 128), (262144, 16384, 128, 1))
        del buf171
        del buf172
        buf174 = buf173; del buf173  # reuse
        # Source Nodes: [conv2d, conv2d_1, gelu, mul_, x], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_19.run(buf174, arg2_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg2_1
        # Source Nodes: [conv2d, conv2d_1, gelu, mul_, x], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        buf176 = extern_kernels.convolution(buf174, buf175, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf176, (8, 32, 128, 128), (524288, 16384, 128, 1))
        del buf174
        del buf175
        buf177 = buf176; del buf176  # reuse
        # Source Nodes: [conv2d, conv2d_1, conv2d_2, gelu, gelu_1, mul_, mul__1, x], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_20.run(buf177, arg5_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg5_1
        # Source Nodes: [conv2d, conv2d_1, conv2d_2, gelu, gelu_1, mul_, mul__1, x], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        buf179 = extern_kernels.convolution(buf177, buf178, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (8, 64, 128, 128), (1048576, 16384, 128, 1))
        del buf177
        del buf178
        buf180 = empty((8, 64, 129, 129), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv2d, conv2d_1, conv2d_2, gelu, gelu_1, gelu_2, mul_, mul__1, mul__2, shortcut, x, x_2], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_21.run(buf179, arg8_1, buf180, 8520192, grid=grid(8520192), stream=stream0)
        del arg8_1
        del buf179
        # Source Nodes: [conv2d, conv2d_1, conv2d_2, gelu, gelu_1, gelu_2, mul_, mul__1, mul__2, shortcut, x, x_2], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        buf182 = extern_kernels.convolution(buf180, buf181, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (8, 128, 64, 64), (524288, 4096, 64, 1))
        del buf180
        del buf181
        buf183 = buf182; del buf182  # reuse
        # Source Nodes: [conv2d, conv2d_1, conv2d_2, gelu, gelu_1, gelu_2, gelu_3, mul_, mul__1, mul__2, mul__3, out, shortcut, x, x_2], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_22.run(buf183, arg11_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg11_1
        # Source Nodes: [out_1], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf183, buf184, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (8, 128, 64, 64), (524288, 4096, 64, 1))
        del buf184
        buf186 = buf185; del buf185  # reuse
        # Source Nodes: [gelu_4, mul__4, out_1, out_2], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_23.run(buf186, arg17_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg17_1
        # Source Nodes: [gelu_4, mul__4, out_1, out_2], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf188 = extern_kernels.convolution(buf186, buf187, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (8, 128, 64, 64), (524288, 4096, 64, 1))
        del buf186
        del buf187
        buf189 = buf188; del buf188  # reuse
        # Source Nodes: [gelu_4, gelu_5, mul__4, mul__5, out_1, out_2, out_3], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_23.run(buf189, arg20_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg20_1
        # Source Nodes: [gelu_4, gelu_5, mul__4, mul__5, out_1, out_2, out_3], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf191 = extern_kernels.convolution(buf189, buf190, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (8, 128, 64, 64), (524288, 4096, 64, 1))
        del buf189
        del buf190
        buf192 = buf191; del buf191  # reuse
        # Source Nodes: [gelu_4, gelu_5, gelu_6, mul__4, mul__5, mul__6, out_1, out_2, out_3, out_4], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_convolution_gelu_mul_23.run(buf192, arg23_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg23_1
        # Source Nodes: [gelu_4, gelu_5, gelu_6, mul__4, mul__5, mul__6, out_1, out_2, out_3, out_4], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf194 = extern_kernels.convolution(buf192, buf193, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del buf192
        del buf193
        buf195 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf196 = reinterpret_tensor(buf195, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf195  # reuse
        # Source Nodes: [gelu_4, gelu_5, gelu_6, mul__4, mul__5, mul__6, out_1, out_2, out_3, out_4, x_se, x_se_1], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul]
        triton_red_fused_convolution_gelu_mean_mul_24.run(buf196, buf194, arg26_1, 2048, 4096, grid=grid(2048), stream=stream0)
        # Source Nodes: [gelu_4, gelu_5, gelu_6, mul__4, mul__5, mul__6, out_1, out_2, out_3, out_4, x_se, x_se_1], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul]
        buf197 = extern_kernels.convolution(buf196, arg183_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg183_1
        del buf196
        buf198 = buf197; del buf197  # reuse
        # Source Nodes: [gelu_4, gelu_5, gelu_6, mul__4, mul__5, mul__6, out_1, out_2, out_3, out_4, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_convolution_gelu_mean_mul_relu_25.run(buf198, arg184_1, 1024, grid=grid(1024), stream=stream0)
        del arg184_1
        # Source Nodes: [gelu_4, gelu_5, gelu_6, mul__4, mul__5, mul__6, out_1, out_2, out_3, out_4, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
        buf199 = extern_kernels.convolution(buf198, arg185_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg185_1
        del buf198
        # Source Nodes: [shortcut_1], Original ATen: [aten.convolution]
        buf201 = extern_kernels.convolution(buf183, buf200, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del buf200
        buf202 = buf194; del buf194  # reuse
        buf203 = buf202; del buf202  # reuse
        # Source Nodes: [gelu_4, gelu_5, gelu_6, gelu_7, mul_10, mul_12, mul__4, mul__5, mul__6, mul__7, mul__8, out_1, out_2, out_3, out_4, out_5, out_8, shortcut_1, shortcut_2, sigmoid, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_26.run(buf203, arg26_1, buf199, arg186_1, arg27_1, buf201, arg14_1, 8388608, grid=grid(8388608), stream=stream0)
        del arg14_1
        del arg186_1
        del arg26_1
        del arg27_1
        del buf199
        del buf201
        # Source Nodes: [out_9], Original ATen: [aten.convolution]
        buf205 = extern_kernels.convolution(buf203, buf204, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf205, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        del buf204
        buf206 = empty((8, 256, 65, 65), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_8, mul__9, out_10, out_9, x_5], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_27.run(buf205, arg33_1, buf206, 8652800, grid=grid(8652800), stream=stream0)
        del arg33_1
        del buf205
        # Source Nodes: [gelu_8, mul__9, out_10, out_9, x_5], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        buf208 = extern_kernels.convolution(buf206, buf207, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf208, (8, 256, 32, 32), (262144, 1024, 32, 1))
        del buf206
        del buf207
        buf209 = buf208; del buf208  # reuse
        # Source Nodes: [gelu_8, gelu_9, mul__10, mul__9, out_10, out_11, out_9, x_5], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_28.run(buf209, arg36_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg36_1
        # Source Nodes: [gelu_8, gelu_9, mul__10, mul__9, out_10, out_11, out_9, x_5], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        buf211 = extern_kernels.convolution(buf209, buf210, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf211, (8, 256, 32, 32), (262144, 1024, 32, 1))
        del buf209
        del buf210
        buf212 = buf211; del buf211  # reuse
        # Source Nodes: [gelu_10, gelu_8, gelu_9, mul__10, mul__11, mul__9, out_10, out_11, out_12, out_9, x_5], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_28.run(buf212, arg39_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg39_1
        # Source Nodes: [gelu_10, gelu_8, gelu_9, mul__10, mul__11, mul__9, out_10, out_11, out_12, out_9, x_5], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        buf214 = extern_kernels.convolution(buf212, buf213, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (8, 512, 32, 32), (524288, 1024, 32, 1))
        del buf213
        buf215 = empty_strided((8, 512, 1, 1), (512, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf216 = reinterpret_tensor(buf215, (8, 512, 1, 1), (512, 1, 1, 1), 0); del buf215  # reuse
        # Source Nodes: [gelu_10, gelu_8, gelu_9, mul__10, mul__11, mul__9, out_10, out_11, out_12, out_9, x_5, x_se_4, x_se_5], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mean, aten.mul]
        triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_29.run(buf216, buf214, arg42_1, 4096, 1024, grid=grid(4096), stream=stream0)
        # Source Nodes: [gelu_10, gelu_8, gelu_9, mul__10, mul__11, mul__9, out_10, out_11, out_12, out_9, x_5, x_se_4, x_se_5], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mean, aten.mul]
        buf217 = extern_kernels.convolution(buf216, arg187_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg187_1
        del buf216
        buf218 = buf217; del buf217  # reuse
        # Source Nodes: [gelu_10, gelu_8, gelu_9, mul__10, mul__11, mul__9, out_10, out_11, out_12, out_9, x_5, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_30.run(buf218, arg188_1, 2048, grid=grid(2048), stream=stream0)
        del arg188_1
        # Source Nodes: [gelu_10, gelu_8, gelu_9, mul__10, mul__11, mul__9, out_10, out_11, out_12, out_9, x_5, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
        buf219 = extern_kernels.convolution(buf218, arg189_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg189_1
        del buf218
        buf220 = buf212; del buf212  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___downsample_pool, shortcut_3], Original ATen: [aten.avg_pool2d, aten.convolution]
        triton_poi_fused_avg_pool2d_convolution_31.run(buf203, buf220, 2097152, grid=grid(2097152), stream=stream0)
        del buf203
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___downsample_pool, shortcut_3], Original ATen: [aten.avg_pool2d, aten.convolution]
        buf222 = extern_kernels.convolution(buf220, buf221, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (8, 512, 32, 32), (524288, 1024, 32, 1))
        del buf220
        del buf221
        buf223 = buf214; del buf214  # reuse
        buf224 = reinterpret_tensor(buf183, (8, 512, 32, 32), (524288, 1024, 32, 1), 0); del buf183  # reuse
        # Source Nodes: [gelu_10, gelu_11, gelu_8, gelu_9, getattr_getattr_l__mod___stages___1_____0___downsample_pool, mul_19, mul_21, mul__10, mul__11, mul__12, mul__13, mul__9, out_10, out_11, out_12, out_13, out_16, out_17, out_9, shortcut_3, shortcut_4, sigmoid_1, x_5, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.add, aten.avg_pool2d, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_add_avg_pool2d_constant_pad_nd_convolution_gelu_mean_mul_relu_sigmoid_32.run(buf223, arg42_1, buf219, arg190_1, arg43_1, buf222, arg30_1, buf224, 4194304, grid=grid(4194304), stream=stream0)
        del arg190_1
        del arg30_1
        del arg42_1
        del arg43_1
        del buf222
        # Source Nodes: [gelu_11, mul__13, out_16, out_17], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf226 = extern_kernels.convolution(buf224, buf225, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (8, 256, 32, 32), (262144, 1024, 32, 1))
        del buf224
        del buf225
        buf227 = buf226; del buf226  # reuse
        # Source Nodes: [gelu_11, gelu_12, mul__13, mul__14, out_16, out_17, out_18], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_28.run(buf227, arg46_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg46_1
        # Source Nodes: [gelu_11, gelu_12, mul__13, mul__14, out_16, out_17, out_18], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf229 = extern_kernels.convolution(buf227, buf228, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf229, (8, 256, 32, 32), (262144, 1024, 32, 1))
        del buf227
        del buf228
        buf230 = buf229; del buf229  # reuse
        # Source Nodes: [gelu_11, gelu_12, gelu_13, mul__13, mul__14, mul__15, out_16, out_17, out_18, out_19], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_28.run(buf230, arg49_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg49_1
        # Source Nodes: [gelu_11, gelu_12, gelu_13, mul__13, mul__14, mul__15, out_16, out_17, out_18, out_19], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf232 = extern_kernels.convolution(buf230, buf231, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf232, (8, 256, 32, 32), (262144, 1024, 32, 1))
        del buf230
        del buf231
        buf233 = buf232; del buf232  # reuse
        # Source Nodes: [gelu_11, gelu_12, gelu_13, gelu_14, mul__13, mul__14, mul__15, mul__16, out_16, out_17, out_18, out_19, out_20], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_28.run(buf233, arg52_1, 2097152, grid=grid(2097152), stream=stream0)
        del arg52_1
        # Source Nodes: [gelu_11, gelu_12, gelu_13, gelu_14, mul__13, mul__14, mul__15, mul__16, out_16, out_17, out_18, out_19, out_20], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf235 = extern_kernels.convolution(buf233, buf234, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (8, 512, 32, 32), (524288, 1024, 32, 1))
        del buf233
        del buf234
        buf236 = reinterpret_tensor(buf219, (8, 512, 1, 1), (512, 1, 4096, 4096), 0); del buf219  # reuse
        buf237 = reinterpret_tensor(buf236, (8, 512, 1, 1), (512, 1, 1, 1), 0); del buf236  # reuse
        # Source Nodes: [gelu_11, gelu_12, gelu_13, gelu_14, mul__13, mul__14, mul__15, mul__16, out_16, out_17, out_18, out_19, out_20, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul]
        triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_29.run(buf237, buf235, arg55_1, 4096, 1024, grid=grid(4096), stream=stream0)
        # Source Nodes: [gelu_11, gelu_12, gelu_13, gelu_14, mul__13, mul__14, mul__15, mul__16, out_16, out_17, out_18, out_19, out_20, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul]
        buf238 = extern_kernels.convolution(buf237, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg191_1
        del buf237
        buf239 = buf238; del buf238  # reuse
        # Source Nodes: [gelu_11, gelu_12, gelu_13, gelu_14, mul__13, mul__14, mul__15, mul__16, out_16, out_17, out_18, out_19, out_20, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_30.run(buf239, arg192_1, 2048, grid=grid(2048), stream=stream0)
        del arg192_1
        # Source Nodes: [gelu_11, gelu_12, gelu_13, gelu_14, mul__13, mul__14, mul__15, mul__16, out_16, out_17, out_18, out_19, out_20, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
        buf240 = extern_kernels.convolution(buf239, arg193_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf240, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg193_1
        del buf239
        buf241 = buf223; del buf223  # reuse
        buf242 = buf241; del buf241  # reuse
        # Source Nodes: [gelu_11, gelu_12, gelu_13, gelu_14, gelu_15, mul_27, mul_29, mul__13, mul__14, mul__15, mul__16, mul__17, mul__18, out_16, out_17, out_18, out_19, out_20, out_21, out_24, shortcut_5, sigmoid_2, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_33.run(buf242, buf235, arg55_1, buf240, arg194_1, arg56_1, 4194304, grid=grid(4194304), stream=stream0)
        del arg194_1
        del arg55_1
        del arg56_1
        del buf235
        del buf240
        # Source Nodes: [out_25], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(buf242, buf243, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf244, (8, 768, 32, 32), (786432, 1024, 32, 1))
        del buf243
        buf245 = empty((8, 768, 33, 33), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_16, mul__19, out_25, out_26, x_7], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_34.run(buf244, arg62_1, buf245, 6690816, grid=grid(6690816), stream=stream0)
        del arg62_1
        del buf244
        # Source Nodes: [gelu_16, mul__19, out_25, out_26, x_7], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        buf247 = extern_kernels.convolution(buf245, buf246, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf247, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf245
        del buf246
        buf248 = buf247; del buf247  # reuse
        # Source Nodes: [gelu_16, gelu_17, mul__19, mul__20, out_25, out_26, out_27, x_7], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf248, arg65_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg65_1
        # Source Nodes: [gelu_16, gelu_17, mul__19, mul__20, out_25, out_26, out_27, x_7], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        buf250 = extern_kernels.convolution(buf248, buf249, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf250, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf248
        del buf249
        buf251 = buf250; del buf250  # reuse
        # Source Nodes: [gelu_16, gelu_17, gelu_18, mul__19, mul__20, mul__21, out_25, out_26, out_27, out_28, x_7], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf251, arg68_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg68_1
        # Source Nodes: [gelu_16, gelu_17, gelu_18, mul__19, mul__20, mul__21, out_25, out_26, out_27, out_28, x_7], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        buf253 = extern_kernels.convolution(buf251, buf252, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (8, 1536, 16, 16), (393216, 256, 16, 1))
        del buf251
        del buf252
        buf254 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cuda', dtype=torch.float32)
        buf255 = reinterpret_tensor(buf254, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf254  # reuse
        # Source Nodes: [gelu_16, gelu_17, gelu_18, mul__19, mul__20, mul__21, out_25, out_26, out_27, out_28, x_7, x_se_12, x_se_13], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mean, aten.mul]
        triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_36.run(buf255, buf253, arg71_1, 12288, 256, grid=grid(12288), stream=stream0)
        # Source Nodes: [gelu_16, gelu_17, gelu_18, mul__19, mul__20, mul__21, out_25, out_26, out_27, out_28, x_7, x_se_12, x_se_13], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mean, aten.mul]
        buf256 = extern_kernels.convolution(buf255, arg195_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (8, 768, 1, 1), (768, 1, 1, 1))
        del arg195_1
        del buf255
        buf257 = buf256; del buf256  # reuse
        # Source Nodes: [gelu_16, gelu_17, gelu_18, mul__19, mul__20, mul__21, out_25, out_26, out_27, out_28, x_7, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_37.run(buf257, arg196_1, 6144, grid=grid(6144), stream=stream0)
        del arg196_1
        # Source Nodes: [gelu_16, gelu_17, gelu_18, mul__19, mul__20, mul__21, out_25, out_26, out_27, out_28, x_7, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
        buf258 = extern_kernels.convolution(buf257, arg197_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf258, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg197_1
        del buf257
        buf259 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___downsample_pool, shortcut_6], Original ATen: [aten.avg_pool2d, aten.convolution]
        triton_poi_fused_avg_pool2d_convolution_38.run(buf242, buf259, 1048576, grid=grid(1048576), stream=stream0)
        del buf242
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___downsample_pool, shortcut_6], Original ATen: [aten.avg_pool2d, aten.convolution]
        buf261 = extern_kernels.convolution(buf259, buf260, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf261, (8, 1536, 16, 16), (393216, 256, 16, 1))
        del buf259
        buf262 = buf253; del buf253  # reuse
        buf263 = empty((8, 1536, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_16, gelu_17, gelu_18, gelu_19, getattr_getattr_l__mod___stages___2_____0___downsample_pool, mul_36, mul_38, mul__19, mul__20, mul__21, mul__22, mul__23, out_25, out_26, out_27, out_28, out_29, out_32, out_33, shortcut_6, shortcut_7, sigmoid_3, x_7, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.add, aten.avg_pool2d, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_add_avg_pool2d_constant_pad_nd_convolution_gelu_mean_mul_relu_sigmoid_39.run(buf262, arg71_1, buf258, arg198_1, arg72_1, buf261, arg59_1, buf263, 3145728, grid=grid(3145728), stream=stream0)
        del arg198_1
        del arg59_1
        del arg71_1
        del arg72_1
        del buf261
        # Source Nodes: [gelu_19, mul__23, out_32, out_33], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf265 = extern_kernels.convolution(buf263, buf264, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf265, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf264
        buf266 = buf265; del buf265  # reuse
        # Source Nodes: [gelu_19, gelu_20, mul__23, mul__24, out_32, out_33, out_34], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf266, arg75_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg75_1
        # Source Nodes: [gelu_19, gelu_20, mul__23, mul__24, out_32, out_33, out_34], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf268 = extern_kernels.convolution(buf266, buf267, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf268, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf266
        del buf267
        buf269 = buf268; del buf268  # reuse
        # Source Nodes: [gelu_19, gelu_20, gelu_21, mul__23, mul__24, mul__25, out_32, out_33, out_34, out_35], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf269, arg78_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg78_1
        # Source Nodes: [gelu_19, gelu_20, gelu_21, mul__23, mul__24, mul__25, out_32, out_33, out_34, out_35], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf271 = extern_kernels.convolution(buf269, buf270, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf271, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf269
        del buf270
        buf272 = buf271; del buf271  # reuse
        # Source Nodes: [gelu_19, gelu_20, gelu_21, gelu_22, mul__23, mul__24, mul__25, mul__26, out_32, out_33, out_34, out_35, out_36], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf272, arg81_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg81_1
        # Source Nodes: [gelu_19, gelu_20, gelu_21, gelu_22, mul__23, mul__24, mul__25, mul__26, out_32, out_33, out_34, out_35, out_36], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf274 = extern_kernels.convolution(buf272, buf273, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf274, (8, 1536, 16, 16), (393216, 256, 16, 1))
        del buf272
        del buf273
        buf275 = reinterpret_tensor(buf258, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf258  # reuse
        buf276 = reinterpret_tensor(buf275, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf275  # reuse
        # Source Nodes: [gelu_19, gelu_20, gelu_21, gelu_22, mul__23, mul__24, mul__25, mul__26, out_32, out_33, out_34, out_35, out_36, x_se_16, x_se_17], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul]
        triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_36.run(buf276, buf274, arg84_1, 12288, 256, grid=grid(12288), stream=stream0)
        # Source Nodes: [gelu_19, gelu_20, gelu_21, gelu_22, mul__23, mul__24, mul__25, mul__26, out_32, out_33, out_34, out_35, out_36, x_se_16, x_se_17], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul]
        buf277 = extern_kernels.convolution(buf276, arg199_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf277, (8, 768, 1, 1), (768, 1, 1, 1))
        del arg199_1
        del buf276
        buf278 = buf277; del buf277  # reuse
        # Source Nodes: [gelu_19, gelu_20, gelu_21, gelu_22, mul__23, mul__24, mul__25, mul__26, out_32, out_33, out_34, out_35, out_36, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_37.run(buf278, arg200_1, 6144, grid=grid(6144), stream=stream0)
        del arg200_1
        # Source Nodes: [gelu_19, gelu_20, gelu_21, gelu_22, mul__23, mul__24, mul__25, mul__26, out_32, out_33, out_34, out_35, out_36, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
        buf279 = extern_kernels.convolution(buf278, arg201_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf279, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg201_1
        del buf278
        buf280 = buf262; del buf262  # reuse
        buf281 = buf263; del buf263  # reuse
        # Source Nodes: [gelu_19, gelu_20, gelu_21, gelu_22, gelu_23, mul_44, mul_46, mul__23, mul__24, mul__25, mul__26, mul__27, mul__28, out_32, out_33, out_34, out_35, out_36, out_37, out_40, out_41, shortcut_8, sigmoid_4, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_40.run(buf280, buf274, arg84_1, buf279, arg202_1, arg85_1, buf281, 3145728, grid=grid(3145728), stream=stream0)
        del arg202_1
        del arg84_1
        del arg85_1
        del buf274
        # Source Nodes: [gelu_23, mul__28, out_40, out_41], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf283 = extern_kernels.convolution(buf281, buf282, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf282
        buf284 = buf283; del buf283  # reuse
        # Source Nodes: [gelu_23, gelu_24, mul__28, mul__29, out_40, out_41, out_42], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf284, arg88_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg88_1
        # Source Nodes: [gelu_23, gelu_24, mul__28, mul__29, out_40, out_41, out_42], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf286 = extern_kernels.convolution(buf284, buf285, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf286, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf284
        del buf285
        buf287 = buf286; del buf286  # reuse
        # Source Nodes: [gelu_23, gelu_24, gelu_25, mul__28, mul__29, mul__30, out_40, out_41, out_42, out_43], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf287, arg91_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg91_1
        # Source Nodes: [gelu_23, gelu_24, gelu_25, mul__28, mul__29, mul__30, out_40, out_41, out_42, out_43], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf289 = extern_kernels.convolution(buf287, buf288, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf289, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf287
        del buf288
        buf290 = buf289; del buf289  # reuse
        # Source Nodes: [gelu_23, gelu_24, gelu_25, gelu_26, mul__28, mul__29, mul__30, mul__31, out_40, out_41, out_42, out_43, out_44], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf290, arg94_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg94_1
        # Source Nodes: [gelu_23, gelu_24, gelu_25, gelu_26, mul__28, mul__29, mul__30, mul__31, out_40, out_41, out_42, out_43, out_44], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf292 = extern_kernels.convolution(buf290, buf291, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf292, (8, 1536, 16, 16), (393216, 256, 16, 1))
        del buf290
        del buf291
        buf293 = reinterpret_tensor(buf279, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf279  # reuse
        buf294 = reinterpret_tensor(buf293, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf293  # reuse
        # Source Nodes: [gelu_23, gelu_24, gelu_25, gelu_26, mul__28, mul__29, mul__30, mul__31, out_40, out_41, out_42, out_43, out_44, x_se_20, x_se_21], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul]
        triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_36.run(buf294, buf292, arg97_1, 12288, 256, grid=grid(12288), stream=stream0)
        # Source Nodes: [gelu_23, gelu_24, gelu_25, gelu_26, mul__28, mul__29, mul__30, mul__31, out_40, out_41, out_42, out_43, out_44, x_se_20, x_se_21], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul]
        buf295 = extern_kernels.convolution(buf294, arg203_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf295, (8, 768, 1, 1), (768, 1, 1, 1))
        del arg203_1
        del buf294
        buf296 = buf295; del buf295  # reuse
        # Source Nodes: [gelu_23, gelu_24, gelu_25, gelu_26, mul__28, mul__29, mul__30, mul__31, out_40, out_41, out_42, out_43, out_44, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_37.run(buf296, arg204_1, 6144, grid=grid(6144), stream=stream0)
        del arg204_1
        # Source Nodes: [gelu_23, gelu_24, gelu_25, gelu_26, mul__28, mul__29, mul__30, mul__31, out_40, out_41, out_42, out_43, out_44, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
        buf297 = extern_kernels.convolution(buf296, arg205_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf297, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg205_1
        del buf296
        buf298 = buf280; del buf280  # reuse
        buf299 = buf281; del buf281  # reuse
        # Source Nodes: [gelu_23, gelu_24, gelu_25, gelu_26, gelu_27, mul_52, mul_54, mul__28, mul__29, mul__30, mul__31, mul__32, mul__33, out_40, out_41, out_42, out_43, out_44, out_45, out_48, out_49, shortcut_9, sigmoid_5, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_41.run(buf298, buf292, arg97_1, buf297, arg206_1, arg98_1, buf299, 3145728, grid=grid(3145728), stream=stream0)
        del arg206_1
        del arg97_1
        del arg98_1
        del buf292
        # Source Nodes: [gelu_27, mul__33, out_48, out_49], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf301 = extern_kernels.convolution(buf299, buf300, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf301, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf300
        buf302 = buf301; del buf301  # reuse
        # Source Nodes: [gelu_27, gelu_28, mul__33, mul__34, out_48, out_49, out_50], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf302, arg101_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg101_1
        # Source Nodes: [gelu_27, gelu_28, mul__33, mul__34, out_48, out_49, out_50], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf304 = extern_kernels.convolution(buf302, buf303, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf304, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf302
        del buf303
        buf305 = buf304; del buf304  # reuse
        # Source Nodes: [gelu_27, gelu_28, gelu_29, mul__33, mul__34, mul__35, out_48, out_49, out_50, out_51], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf305, arg104_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg104_1
        # Source Nodes: [gelu_27, gelu_28, gelu_29, mul__33, mul__34, mul__35, out_48, out_49, out_50, out_51], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf307 = extern_kernels.convolution(buf305, buf306, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf307, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf305
        del buf306
        buf308 = buf307; del buf307  # reuse
        # Source Nodes: [gelu_27, gelu_28, gelu_29, gelu_30, mul__33, mul__34, mul__35, mul__36, out_48, out_49, out_50, out_51, out_52], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf308, arg107_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg107_1
        # Source Nodes: [gelu_27, gelu_28, gelu_29, gelu_30, mul__33, mul__34, mul__35, mul__36, out_48, out_49, out_50, out_51, out_52], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf310 = extern_kernels.convolution(buf308, buf309, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf310, (8, 1536, 16, 16), (393216, 256, 16, 1))
        del buf308
        del buf309
        buf311 = reinterpret_tensor(buf297, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf297  # reuse
        buf312 = reinterpret_tensor(buf311, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf311  # reuse
        # Source Nodes: [gelu_27, gelu_28, gelu_29, gelu_30, mul__33, mul__34, mul__35, mul__36, out_48, out_49, out_50, out_51, out_52, x_se_24, x_se_25], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul]
        triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_36.run(buf312, buf310, arg110_1, 12288, 256, grid=grid(12288), stream=stream0)
        # Source Nodes: [gelu_27, gelu_28, gelu_29, gelu_30, mul__33, mul__34, mul__35, mul__36, out_48, out_49, out_50, out_51, out_52, x_se_24, x_se_25], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul]
        buf313 = extern_kernels.convolution(buf312, arg207_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf313, (8, 768, 1, 1), (768, 1, 1, 1))
        del arg207_1
        del buf312
        buf314 = buf313; del buf313  # reuse
        # Source Nodes: [gelu_27, gelu_28, gelu_29, gelu_30, mul__33, mul__34, mul__35, mul__36, out_48, out_49, out_50, out_51, out_52, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_37.run(buf314, arg208_1, 6144, grid=grid(6144), stream=stream0)
        del arg208_1
        # Source Nodes: [gelu_27, gelu_28, gelu_29, gelu_30, mul__33, mul__34, mul__35, mul__36, out_48, out_49, out_50, out_51, out_52, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
        buf315 = extern_kernels.convolution(buf314, arg209_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf315, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg209_1
        del buf314
        buf316 = buf298; del buf298  # reuse
        buf317 = buf299; del buf299  # reuse
        # Source Nodes: [gelu_27, gelu_28, gelu_29, gelu_30, gelu_31, mul_60, mul_62, mul__33, mul__34, mul__35, mul__36, mul__37, mul__38, out_48, out_49, out_50, out_51, out_52, out_53, out_56, out_57, shortcut_10, sigmoid_6, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_42.run(buf316, buf310, arg110_1, buf315, arg210_1, arg111_1, buf317, 3145728, grid=grid(3145728), stream=stream0)
        del arg110_1
        del arg111_1
        del arg210_1
        del buf310
        # Source Nodes: [gelu_31, mul__38, out_56, out_57], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf319 = extern_kernels.convolution(buf317, buf318, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf319, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf318
        buf320 = buf319; del buf319  # reuse
        # Source Nodes: [gelu_31, gelu_32, mul__38, mul__39, out_56, out_57, out_58], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf320, arg114_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg114_1
        # Source Nodes: [gelu_31, gelu_32, mul__38, mul__39, out_56, out_57, out_58], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf322 = extern_kernels.convolution(buf320, buf321, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf322, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf320
        del buf321
        buf323 = buf322; del buf322  # reuse
        # Source Nodes: [gelu_31, gelu_32, gelu_33, mul__38, mul__39, mul__40, out_56, out_57, out_58, out_59], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf323, arg117_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg117_1
        # Source Nodes: [gelu_31, gelu_32, gelu_33, mul__38, mul__39, mul__40, out_56, out_57, out_58, out_59], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf325 = extern_kernels.convolution(buf323, buf324, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf325, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf323
        del buf324
        buf326 = buf325; del buf325  # reuse
        # Source Nodes: [gelu_31, gelu_32, gelu_33, gelu_34, mul__38, mul__39, mul__40, mul__41, out_56, out_57, out_58, out_59, out_60], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf326, arg120_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg120_1
        # Source Nodes: [gelu_31, gelu_32, gelu_33, gelu_34, mul__38, mul__39, mul__40, mul__41, out_56, out_57, out_58, out_59, out_60], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf328 = extern_kernels.convolution(buf326, buf327, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf328, (8, 1536, 16, 16), (393216, 256, 16, 1))
        del buf326
        del buf327
        buf329 = reinterpret_tensor(buf315, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf315  # reuse
        buf330 = reinterpret_tensor(buf329, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf329  # reuse
        # Source Nodes: [gelu_31, gelu_32, gelu_33, gelu_34, mul__38, mul__39, mul__40, mul__41, out_56, out_57, out_58, out_59, out_60, x_se_28, x_se_29], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul]
        triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_36.run(buf330, buf328, arg123_1, 12288, 256, grid=grid(12288), stream=stream0)
        # Source Nodes: [gelu_31, gelu_32, gelu_33, gelu_34, mul__38, mul__39, mul__40, mul__41, out_56, out_57, out_58, out_59, out_60, x_se_28, x_se_29], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul]
        buf331 = extern_kernels.convolution(buf330, arg211_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf331, (8, 768, 1, 1), (768, 1, 1, 1))
        del arg211_1
        del buf330
        buf332 = buf331; del buf331  # reuse
        # Source Nodes: [gelu_31, gelu_32, gelu_33, gelu_34, mul__38, mul__39, mul__40, mul__41, out_56, out_57, out_58, out_59, out_60, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_37.run(buf332, arg212_1, 6144, grid=grid(6144), stream=stream0)
        del arg212_1
        # Source Nodes: [gelu_31, gelu_32, gelu_33, gelu_34, mul__38, mul__39, mul__40, mul__41, out_56, out_57, out_58, out_59, out_60, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
        buf333 = extern_kernels.convolution(buf332, arg213_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf333, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg213_1
        del buf332
        buf334 = buf316; del buf316  # reuse
        buf335 = buf317; del buf317  # reuse
        # Source Nodes: [gelu_31, gelu_32, gelu_33, gelu_34, gelu_35, mul_68, mul_70, mul__38, mul__39, mul__40, mul__41, mul__42, mul__43, out_56, out_57, out_58, out_59, out_60, out_61, out_64, out_65, shortcut_11, sigmoid_7, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_43.run(buf334, buf328, arg123_1, buf333, arg214_1, arg124_1, buf335, 3145728, grid=grid(3145728), stream=stream0)
        del arg123_1
        del arg124_1
        del arg214_1
        del buf328
        # Source Nodes: [gelu_35, mul__43, out_64, out_65], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf337 = extern_kernels.convolution(buf335, buf336, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf337, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf335
        del buf336
        buf338 = buf337; del buf337  # reuse
        # Source Nodes: [gelu_35, gelu_36, mul__43, mul__44, out_64, out_65, out_66], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf338, arg127_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg127_1
        # Source Nodes: [gelu_35, gelu_36, mul__43, mul__44, out_64, out_65, out_66], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf340 = extern_kernels.convolution(buf338, buf339, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf340, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf338
        del buf339
        buf341 = buf340; del buf340  # reuse
        # Source Nodes: [gelu_35, gelu_36, gelu_37, mul__43, mul__44, mul__45, out_64, out_65, out_66, out_67], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf341, arg130_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg130_1
        # Source Nodes: [gelu_35, gelu_36, gelu_37, mul__43, mul__44, mul__45, out_64, out_65, out_66, out_67], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf343 = extern_kernels.convolution(buf341, buf342, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf343, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf341
        del buf342
        buf344 = buf343; del buf343  # reuse
        # Source Nodes: [gelu_35, gelu_36, gelu_37, gelu_38, mul__43, mul__44, mul__45, mul__46, out_64, out_65, out_66, out_67, out_68], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_35.run(buf344, arg133_1, 1572864, grid=grid(1572864), stream=stream0)
        del arg133_1
        # Source Nodes: [gelu_35, gelu_36, gelu_37, gelu_38, mul__43, mul__44, mul__45, mul__46, out_64, out_65, out_66, out_67, out_68], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf346 = extern_kernels.convolution(buf344, buf345, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf346, (8, 1536, 16, 16), (393216, 256, 16, 1))
        del buf344
        del buf345
        buf347 = reinterpret_tensor(buf333, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf333  # reuse
        buf348 = reinterpret_tensor(buf347, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf347  # reuse
        # Source Nodes: [gelu_35, gelu_36, gelu_37, gelu_38, mul__43, mul__44, mul__45, mul__46, out_64, out_65, out_66, out_67, out_68, x_se_32, x_se_33], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul]
        triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_36.run(buf348, buf346, arg136_1, 12288, 256, grid=grid(12288), stream=stream0)
        # Source Nodes: [gelu_35, gelu_36, gelu_37, gelu_38, mul__43, mul__44, mul__45, mul__46, out_64, out_65, out_66, out_67, out_68, x_se_32, x_se_33], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul]
        buf349 = extern_kernels.convolution(buf348, arg215_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf349, (8, 768, 1, 1), (768, 1, 1, 1))
        del arg215_1
        del buf348
        buf350 = buf349; del buf349  # reuse
        # Source Nodes: [gelu_35, gelu_36, gelu_37, gelu_38, mul__43, mul__44, mul__45, mul__46, out_64, out_65, out_66, out_67, out_68, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_37.run(buf350, arg216_1, 6144, grid=grid(6144), stream=stream0)
        del arg216_1
        # Source Nodes: [gelu_35, gelu_36, gelu_37, gelu_38, mul__43, mul__44, mul__45, mul__46, out_64, out_65, out_66, out_67, out_68, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
        buf351 = extern_kernels.convolution(buf350, arg217_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf351, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg217_1
        del buf350
        buf352 = buf334; del buf334  # reuse
        buf353 = buf352; del buf352  # reuse
        # Source Nodes: [gelu_35, gelu_36, gelu_37, gelu_38, gelu_39, mul_76, mul_78, mul__43, mul__44, mul__45, mul__46, mul__47, mul__48, out_64, out_65, out_66, out_67, out_68, out_69, out_72, shortcut_12, sigmoid_8, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_44.run(buf353, buf346, arg136_1, buf351, arg218_1, arg137_1, 3145728, grid=grid(3145728), stream=stream0)
        del arg136_1
        del arg137_1
        del arg218_1
        del buf346
        # Source Nodes: [out_73], Original ATen: [aten.convolution]
        buf355 = extern_kernels.convolution(buf353, buf354, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf355, (8, 768, 16, 16), (196608, 256, 16, 1))
        del buf354
        buf356 = empty((8, 768, 17, 17), device='cuda', dtype=torch.float32)
        # Source Nodes: [gelu_40, mul__49, out_73, out_74, x_9], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_45.run(buf355, arg143_1, buf356, 1775616, grid=grid(1775616), stream=stream0)
        del arg143_1
        del buf355
        # Source Nodes: [gelu_40, mul__49, out_73, out_74, x_9], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        buf358 = extern_kernels.convolution(buf356, buf357, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf358, (8, 768, 8, 8), (49152, 64, 8, 1))
        del buf356
        del buf357
        buf359 = buf358; del buf358  # reuse
        # Source Nodes: [gelu_40, gelu_41, mul__49, mul__50, out_73, out_74, out_75, x_9], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_46.run(buf359, arg146_1, 393216, grid=grid(393216), stream=stream0)
        del arg146_1
        # Source Nodes: [gelu_40, gelu_41, mul__49, mul__50, out_73, out_74, out_75, x_9], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        buf361 = extern_kernels.convolution(buf359, buf360, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf361, (8, 768, 8, 8), (49152, 64, 8, 1))
        del buf359
        del buf360
        buf362 = buf361; del buf361  # reuse
        # Source Nodes: [gelu_40, gelu_41, gelu_42, mul__49, mul__50, mul__51, out_73, out_74, out_75, out_76, x_9], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_46.run(buf362, arg149_1, 393216, grid=grid(393216), stream=stream0)
        del arg149_1
        # Source Nodes: [gelu_40, gelu_41, gelu_42, mul__49, mul__50, mul__51, out_73, out_74, out_75, out_76, x_9], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mul]
        buf364 = extern_kernels.convolution(buf362, buf363, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf364, (8, 1536, 8, 8), (98304, 64, 8, 1))
        del buf362
        del buf363
        buf365 = reinterpret_tensor(buf351, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf351  # reuse
        buf366 = reinterpret_tensor(buf365, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf365  # reuse
        # Source Nodes: [gelu_40, gelu_41, gelu_42, mul__49, mul__50, mul__51, out_73, out_74, out_75, out_76, x_9, x_se_36, x_se_37], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mean, aten.mul]
        triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_47.run(buf366, buf364, arg152_1, 12288, 64, grid=grid(12288), stream=stream0)
        # Source Nodes: [gelu_40, gelu_41, gelu_42, mul__49, mul__50, mul__51, out_73, out_74, out_75, out_76, x_9, x_se_36, x_se_37], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mean, aten.mul]
        buf367 = extern_kernels.convolution(buf366, arg219_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf367, (8, 768, 1, 1), (768, 1, 1, 1))
        del arg219_1
        del buf366
        buf368 = buf367; del buf367  # reuse
        # Source Nodes: [gelu_40, gelu_41, gelu_42, mul__49, mul__50, mul__51, out_73, out_74, out_75, out_76, x_9, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_37.run(buf368, arg220_1, 6144, grid=grid(6144), stream=stream0)
        del arg220_1
        # Source Nodes: [gelu_40, gelu_41, gelu_42, mul__49, mul__50, mul__51, out_73, out_74, out_75, out_76, x_9, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
        buf369 = extern_kernels.convolution(buf368, arg221_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf369, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg221_1
        del buf368
        buf370 = reinterpret_tensor(buf260, (8, 1536, 8, 8), (98304, 64, 8, 1), 0); del buf260  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____0___downsample_pool, shortcut_13], Original ATen: [aten.avg_pool2d, aten.convolution]
        triton_poi_fused_avg_pool2d_convolution_48.run(buf353, buf370, 786432, grid=grid(786432), stream=stream0)
        del buf353
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____0___downsample_pool, shortcut_13], Original ATen: [aten.avg_pool2d, aten.convolution]
        buf372 = extern_kernels.convolution(buf370, buf371, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf372, (8, 1536, 8, 8), (98304, 64, 8, 1))
        del buf371
        buf373 = buf364; del buf364  # reuse
        buf374 = buf370; del buf370  # reuse
        # Source Nodes: [gelu_40, gelu_41, gelu_42, gelu_43, getattr_getattr_l__mod___stages___3_____0___downsample_pool, mul_85, mul_87, mul__49, mul__50, mul__51, mul__52, mul__53, out_73, out_74, out_75, out_76, out_77, out_80, out_81, shortcut_13, shortcut_14, sigmoid_9, x_9, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.add, aten.avg_pool2d, aten.constant_pad_nd, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_add_avg_pool2d_constant_pad_nd_convolution_gelu_mean_mul_relu_sigmoid_49.run(buf373, arg152_1, buf369, arg222_1, arg153_1, buf372, arg140_1, buf374, 786432, grid=grid(786432), stream=stream0)
        del arg140_1
        del arg152_1
        del arg153_1
        del arg222_1
        del buf372
        # Source Nodes: [gelu_43, mul__53, out_80, out_81], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf376 = extern_kernels.convolution(buf374, buf375, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf376, (8, 768, 8, 8), (49152, 64, 8, 1))
        del buf375
        buf377 = buf376; del buf376  # reuse
        # Source Nodes: [gelu_43, gelu_44, mul__53, mul__54, out_80, out_81, out_82], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_46.run(buf377, arg156_1, 393216, grid=grid(393216), stream=stream0)
        del arg156_1
        # Source Nodes: [gelu_43, gelu_44, mul__53, mul__54, out_80, out_81, out_82], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf379 = extern_kernels.convolution(buf377, buf378, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf379, (8, 768, 8, 8), (49152, 64, 8, 1))
        del buf377
        del buf378
        buf380 = buf379; del buf379  # reuse
        # Source Nodes: [gelu_43, gelu_44, gelu_45, mul__53, mul__54, mul__55, out_80, out_81, out_82, out_83], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_46.run(buf380, arg159_1, 393216, grid=grid(393216), stream=stream0)
        del arg159_1
        # Source Nodes: [gelu_43, gelu_44, gelu_45, mul__53, mul__54, mul__55, out_80, out_81, out_82, out_83], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf382 = extern_kernels.convolution(buf380, buf381, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf382, (8, 768, 8, 8), (49152, 64, 8, 1))
        del buf380
        del buf381
        buf383 = buf382; del buf382  # reuse
        # Source Nodes: [gelu_43, gelu_44, gelu_45, gelu_46, mul__53, mul__54, mul__55, mul__56, out_80, out_81, out_82, out_83, out_84], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_46.run(buf383, arg162_1, 393216, grid=grid(393216), stream=stream0)
        del arg162_1
        # Source Nodes: [gelu_43, gelu_44, gelu_45, gelu_46, mul__53, mul__54, mul__55, mul__56, out_80, out_81, out_82, out_83, out_84], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf385 = extern_kernels.convolution(buf383, buf384, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf385, (8, 1536, 8, 8), (98304, 64, 8, 1))
        del buf383
        del buf384
        buf386 = reinterpret_tensor(buf369, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf369  # reuse
        buf387 = reinterpret_tensor(buf386, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf386  # reuse
        # Source Nodes: [gelu_43, gelu_44, gelu_45, gelu_46, mul__53, mul__54, mul__55, mul__56, out_80, out_81, out_82, out_83, out_84, x_se_40, x_se_41], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul]
        triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_47.run(buf387, buf385, arg165_1, 12288, 64, grid=grid(12288), stream=stream0)
        # Source Nodes: [gelu_43, gelu_44, gelu_45, gelu_46, mul__53, mul__54, mul__55, mul__56, out_80, out_81, out_82, out_83, out_84, x_se_40, x_se_41], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul]
        buf388 = extern_kernels.convolution(buf387, arg223_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf388, (8, 768, 1, 1), (768, 1, 1, 1))
        del arg223_1
        del buf387
        buf389 = buf388; del buf388  # reuse
        # Source Nodes: [gelu_43, gelu_44, gelu_45, gelu_46, mul__53, mul__54, mul__55, mul__56, out_80, out_81, out_82, out_83, out_84, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_37.run(buf389, arg224_1, 6144, grid=grid(6144), stream=stream0)
        del arg224_1
        # Source Nodes: [gelu_43, gelu_44, gelu_45, gelu_46, mul__53, mul__54, mul__55, mul__56, out_80, out_81, out_82, out_83, out_84, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
        buf390 = extern_kernels.convolution(buf389, arg225_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf390, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg225_1
        del buf389
        buf391 = buf373; del buf373  # reuse
        buf392 = buf374; del buf374  # reuse
        # Source Nodes: [gelu_43, gelu_44, gelu_45, gelu_46, gelu_47, mul_93, mul_95, mul__53, mul__54, mul__55, mul__56, mul__57, mul__58, out_80, out_81, out_82, out_83, out_84, out_85, out_88, out_89, shortcut_15, sigmoid_10, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_50.run(buf391, buf385, arg165_1, buf390, arg226_1, arg166_1, buf392, 786432, grid=grid(786432), stream=stream0)
        del arg165_1
        del arg166_1
        del arg226_1
        del buf385
        # Source Nodes: [gelu_47, mul__58, out_88, out_89], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf394 = extern_kernels.convolution(buf392, buf393, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf394, (8, 768, 8, 8), (49152, 64, 8, 1))
        del buf392
        del buf393
        buf395 = buf394; del buf394  # reuse
        # Source Nodes: [gelu_47, gelu_48, mul__58, mul__59, out_88, out_89, out_90], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_46.run(buf395, arg169_1, 393216, grid=grid(393216), stream=stream0)
        del arg169_1
        # Source Nodes: [gelu_47, gelu_48, mul__58, mul__59, out_88, out_89, out_90], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf397 = extern_kernels.convolution(buf395, buf396, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf397, (8, 768, 8, 8), (49152, 64, 8, 1))
        del buf395
        del buf396
        buf398 = buf397; del buf397  # reuse
        # Source Nodes: [gelu_47, gelu_48, gelu_49, mul__58, mul__59, mul__60, out_88, out_89, out_90, out_91], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_46.run(buf398, arg172_1, 393216, grid=grid(393216), stream=stream0)
        del arg172_1
        # Source Nodes: [gelu_47, gelu_48, gelu_49, mul__58, mul__59, mul__60, out_88, out_89, out_90, out_91], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf400 = extern_kernels.convolution(buf398, buf399, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf400, (8, 768, 8, 8), (49152, 64, 8, 1))
        del buf398
        del buf399
        buf401 = buf400; del buf400  # reuse
        # Source Nodes: [gelu_47, gelu_48, gelu_49, gelu_50, mul__58, mul__59, mul__60, mul__61, out_88, out_89, out_90, out_91, out_92], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mul_46.run(buf401, arg175_1, 393216, grid=grid(393216), stream=stream0)
        del arg175_1
        # Source Nodes: [gelu_47, gelu_48, gelu_49, gelu_50, mul__58, mul__59, mul__60, mul__61, out_88, out_89, out_90, out_91, out_92], Original ATen: [aten.convolution, aten.gelu, aten.mul]
        buf403 = extern_kernels.convolution(buf401, buf402, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf403, (8, 1536, 8, 8), (98304, 64, 8, 1))
        del buf401
        del buf402
        buf404 = reinterpret_tensor(buf390, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf390  # reuse
        buf405 = reinterpret_tensor(buf404, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf404  # reuse
        # Source Nodes: [gelu_47, gelu_48, gelu_49, gelu_50, mul__58, mul__59, mul__60, mul__61, out_88, out_89, out_90, out_91, out_92, x_se_44, x_se_45], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul]
        triton_per_fused_constant_pad_nd_convolution_gelu_mean_mul_47.run(buf405, buf403, arg178_1, 12288, 64, grid=grid(12288), stream=stream0)
        # Source Nodes: [gelu_47, gelu_48, gelu_49, gelu_50, mul__58, mul__59, mul__60, mul__61, out_88, out_89, out_90, out_91, out_92, x_se_44, x_se_45], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul]
        buf406 = extern_kernels.convolution(buf405, arg227_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf406, (8, 768, 1, 1), (768, 1, 1, 1))
        del arg227_1
        del buf405
        buf407 = buf406; del buf406  # reuse
        # Source Nodes: [gelu_47, gelu_48, gelu_49, gelu_50, mul__58, mul__59, mul__60, mul__61, out_88, out_89, out_90, out_91, out_92, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_constant_pad_nd_convolution_gelu_mean_mul_relu_37.run(buf407, arg228_1, 6144, grid=grid(6144), stream=stream0)
        del arg228_1
        # Source Nodes: [gelu_47, gelu_48, gelu_49, gelu_50, mul__58, mul__59, mul__60, mul__61, out_88, out_89, out_90, out_91, out_92, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu]
        buf408 = extern_kernels.convolution(buf407, arg229_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf408, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg229_1
        del buf407
        buf409 = buf391; del buf391  # reuse
        # Source Nodes: [gelu_47, gelu_48, gelu_49, gelu_50, mul_101, mul_103, mul__58, mul__59, mul__60, mul__61, mul__62, out_88, out_89, out_90, out_91, out_92, out_93, sigmoid_11, x_10, x_11, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_poi_fused_add_convolution_gelu_mean_mul_relu_sigmoid_51.run(buf409, buf403, arg178_1, buf408, arg230_1, arg179_1, 786432, grid=grid(786432), stream=stream0)
        del arg178_1
        del arg179_1
        del arg230_1
        del buf403
        del buf408
        # Source Nodes: [gelu_47, gelu_48, gelu_49, gelu_50, mul_101, mul_103, mul__58, mul__59, mul__60, mul__61, mul__62, out_88, out_89, out_90, out_91, out_92, out_93, sigmoid_11, x_10, x_11, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        buf411 = extern_kernels.convolution(buf409, buf410, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf411, (8, 3072, 8, 8), (196608, 64, 8, 1))
        del buf409
        del buf410
        buf412 = empty_strided((8, 3072, 1, 1), (3072, 1, 24576, 24576), device='cuda', dtype=torch.float32)
        buf413 = reinterpret_tensor(buf412, (8, 3072, 1, 1), (3072, 1, 1, 1), 0); del buf412  # reuse
        # Source Nodes: [gelu_47, gelu_48, gelu_49, gelu_50, gelu_51, mul_101, mul_103, mul__58, mul__59, mul__60, mul__61, mul__62, out_88, out_89, out_90, out_91, out_92, out_93, sigmoid_11, x_10, x_11, x_13, x_14, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.add, aten.convolution, aten.gelu, aten.mean, aten.mul, aten.relu, aten.sigmoid]
        triton_per_fused_add_convolution_gelu_mean_mul_relu_sigmoid_52.run(buf413, buf411, arg182_1, 24576, 64, grid=grid(24576), stream=stream0)
        del arg182_1
        del buf411
        buf414 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_18], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg232_1, reinterpret_tensor(buf413, (8, 3072), (3072, 1), 0), reinterpret_tensor(arg231_1, (3072, 1000), (1, 3072), 0), alpha=1, beta=1, out=buf414)
        del arg231_1
        del arg232_1
        return (buf414, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((32, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((768, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1536, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((768, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((768, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((), (), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((3072, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((3072, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((3072, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((768, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((1536, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((1000, 3072), (3072, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('dm_nfnet_f0', benchmark_compiled_module)
