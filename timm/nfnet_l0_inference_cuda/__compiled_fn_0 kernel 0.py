
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


# kernel path: /tmp/torchinductor_youkaichao/qp/cqpmp3o5ria6hmqfjwbbwkrqlv6seb4q27kplg4syx6eudtztsph.py
# Source Nodes: [batch_norm, conv2d], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
# batch_norm => var_mean
# conv2d => convolution
triton_per_fused__native_batch_norm_legit_convolution_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_0', 'mutated_arg_names': []}
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
    tmp25 = 0.34412564994580647
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (27*x0)), tmp27, rmask & xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/bx/cbxphtvt44eemelxa6xn4eihv22l3636k36db42o667v5ibbp2jz.py
# Source Nodes: [batch_norm_1, conv2d, conv2d_1, l__mod___stem_act2], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
# batch_norm_1 => var_mean_1
# conv2d => convolution
# conv2d_1 => convolution_1
# l__mod___stem_act2 => mul_3, sigmoid
triton_per_fused__native_batch_norm_legit_convolution_silu_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_silu_1', 'mutated_arg_names': []}
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
    tmp25 = 0.1490107774734497
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (144*x0)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/c4/cc4oghzlw6utkzbagoj47hjgcdbzzrn7r6udjalj74z6pzd2arfu.py
# Source Nodes: [batch_norm_2, conv2d, conv2d_1, conv2d_2, l__mod___stem_act2, l__mod___stem_act3], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
# batch_norm_2 => var_mean_2
# conv2d => convolution
# conv2d_1 => convolution_1
# conv2d_2 => convolution_2
# l__mod___stem_act2 => mul_3, sigmoid
# l__mod___stem_act3 => mul_7, sigmoid_1
triton_per_fused__native_batch_norm_legit_convolution_silu_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_silu_2', 'mutated_arg_names': []}
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
    tmp25 = 0.10536653122135592
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (288*x0)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cj/ccjar7ktr2sisku4n7ccmughdw5bmwtcodigaondwdg2xspykza2.py
# Source Nodes: [batch_norm_3, conv2d, conv2d_1, conv2d_2, l__mod___stem_act2, l__mod___stem_act3, l__mod___stem_act4, shortcut], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
# batch_norm_3 => var_mean_3
# conv2d => convolution
# conv2d_1 => convolution_1
# conv2d_2 => convolution_2
# l__mod___stem_act2 => mul_3, sigmoid
# l__mod___stem_act3 => mul_7, sigmoid_1
# l__mod___stem_act4 => mul_11, sigmoid_2
# shortcut => convolution_3
triton_per_fused__native_batch_norm_legit_convolution_silu_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_silu_3', 'mutated_arg_names': []}
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
    tmp25 = 0.07450538873672485
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (576*x0)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3n/c3n3o43rywoiw3k4a27afwmydcir32jd6ozo33fwaqnq2aljv7tf.py
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
    tmp25 = 0.1580497968320339
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fv/cfvnydbfllctcwqa7oscim5zhjoa6xoopywchnhftghfzrukrp3v.py
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
    size_hints=[64, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_5', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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
    tmp25 = 0.1580497968320339
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dy/cdyvyacjokclmvq5smja6wwnoh4omhy33lli344m3j3mgjnqbmyx.py
# Source Nodes: [batch_norm_6, getattr_getattr_l__mod___stages___0_____0___act2, out_1, out_2], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
# batch_norm_6 => var_mean_6
# getattr_getattr_l__mod___stages___0_____0___act2 => mul_23, sigmoid_4
# out_1 => convolution_5
# out_2 => convolution_6
triton_per_fused__native_batch_norm_legit_convolution_silu_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_silu_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 64
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
    tmp25 = 0.07450538873672485
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (576*x0)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fr/cfraaoldhs4m2ijmz6fare2bdxzakulbmwpt7kari373flsroinm.py
# Source Nodes: [batch_norm_8, getattr_getattr_l__mod___stages___0_____0___act2, getattr_getattr_l__mod___stages___0_____0___act2b, getattr_getattr_l__mod___stages___0_____0___act3, out_1, out_2, out_3, out_4], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
# batch_norm_8 => var_mean_8
# getattr_getattr_l__mod___stages___0_____0___act2 => mul_23, sigmoid_4
# getattr_getattr_l__mod___stages___0_____0___act2b => mul_27, sigmoid_5
# getattr_getattr_l__mod___stages___0_____0___act3 => mul_31, sigmoid_6
# out_1 => convolution_5
# out_2 => convolution_6
# out_3 => convolution_7
# out_4 => convolution_8
triton_per_fused__native_batch_norm_legit_convolution_silu_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_silu_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 64.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = 0.22351616621017456
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (64*x0)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zo/czoas57xjgnzkxiobn4pjikrsbrngj36wlfq57i5yslyosrvktdl.py
# Source Nodes: [batch_norm_9, getattr_getattr_l__mod___stages___1_____0___downsample_pool, shortcut_3], Original ATen: [aten._native_batch_norm_legit, aten.avg_pool2d, aten.convolution]
# batch_norm_9 => var_mean_9
# getattr_getattr_l__mod___stages___1_____0___downsample_pool => avg_pool2d
# shortcut_3 => convolution_11
triton_per_fused__native_batch_norm_legit_avg_pool2d_convolution_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_avg_pool2d_convolution_8', 'mutated_arg_names': []}
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
    tmp25 = 0.11175808310508728
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ao/caopl7tgxrrahzfrsiegjax7bfqtqqtvtojw26wk63c6qkegata7.py
# Source Nodes: [batch_norm_10, out_9], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
# batch_norm_10 => var_mean_10
# out_9 => convolution_12
triton_per_fused__native_batch_norm_legit_convolution_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 128
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
    tmp25 = 0.11175808310508728
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3e/c3eqiunwxw3spi3rvsuyvhxhpx7xqcod3cf6cr2dise6efwbi7e6.py
# Source Nodes: [batch_norm_13, getattr_getattr_l__mod___stages___1_____0___act2, getattr_getattr_l__mod___stages___1_____0___act2b, getattr_getattr_l__mod___stages___1_____0___act3, out_10, out_11, out_12, out_9], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
# batch_norm_13 => var_mean_13
# getattr_getattr_l__mod___stages___1_____0___act2 => mul_46, sigmoid_9
# getattr_getattr_l__mod___stages___1_____0___act2b => mul_50, sigmoid_10
# getattr_getattr_l__mod___stages___1_____0___act3 => mul_54, sigmoid_11
# out_10 => convolution_13
# out_11 => convolution_14
# out_12 => convolution_15
# out_9 => convolution_12
triton_per_fused__native_batch_norm_legit_convolution_silu_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_silu_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
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
    tmp25 = 0.1580497968320339
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ob/cob6my6my7rh5vebj3h56dkmxgjw46q2sybrwnesbfpfcip7gz4t.py
# Source Nodes: [batch_norm_14, getattr_getattr_l__mod___stages___1_____1___act1, out_16, out_17], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
# batch_norm_14 => var_mean_14
# getattr_getattr_l__mod___stages___1_____1___act1 => mul_61, sigmoid_13
# out_16 => mul_62
# out_17 => convolution_18
triton_per_fused__native_batch_norm_legit_convolution_mul_silu_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_mul_silu_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 128
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
    tmp25 = 0.07902489841601695
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eg/ceg72aixlb4mqqnjugjgdb4hhwta2jkubtckin3bhc7cq7qepfrp.py
# Source Nodes: [batch_norm_18, getattr_getattr_l__mod___stages___2_____0___downsample_pool, shortcut_6], Original ATen: [aten._native_batch_norm_legit, aten.avg_pool2d, aten.convolution]
# batch_norm_18 => var_mean_18
# getattr_getattr_l__mod___stages___2_____0___downsample_pool => avg_pool2d_1
# shortcut_6 => convolution_24
triton_per_fused__native_batch_norm_legit_avg_pool2d_convolution_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_avg_pool2d_convolution_12', 'mutated_arg_names': []}
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
    tmp25 = 0.07902489841601695
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4n/c4nwpeqghvtczk4wvtki75gclcomteokr3jg5p5vgbfrqvyx74ur.py
# Source Nodes: [batch_norm_19, out_25], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
# batch_norm_19 => var_mean_19
# out_25 => convolution_25
triton_per_fused__native_batch_norm_legit_convolution_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_13', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 384
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
    tmp25 = 0.07902489841601695
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yg/cygirgjf3d4znsp2ihi26y6naqjl5x4g3dmrnuogkuw643pf65bz.py
# Source Nodes: [batch_norm_20, getattr_getattr_l__mod___stages___2_____0___act2, out_25, out_26], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
# batch_norm_20 => var_mean_20
# getattr_getattr_l__mod___stages___2_____0___act2 => mul_89, sigmoid_19
# out_25 => convolution_25
# out_26 => convolution_26
triton_per_fused__native_batch_norm_legit_convolution_silu_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_silu_14', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 384
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
    tmp25 = 0.07450538873672485
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (576*x0)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/63/c63awopp7mxjtlthdg6la6tvmuzaev7hmd53ixlanxsxlmscw6iy.py
# Source Nodes: [batch_norm_22, getattr_getattr_l__mod___stages___2_____0___act2, getattr_getattr_l__mod___stages___2_____0___act2b, getattr_getattr_l__mod___stages___2_____0___act3, out_25, out_26, out_27, out_28], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
# batch_norm_22 => var_mean_22
# getattr_getattr_l__mod___stages___2_____0___act2 => mul_89, sigmoid_19
# getattr_getattr_l__mod___stages___2_____0___act2b => mul_93, sigmoid_20
# getattr_getattr_l__mod___stages___2_____0___act3 => mul_97, sigmoid_21
# out_25 => convolution_25
# out_26 => convolution_26
# out_27 => convolution_27
# out_28 => convolution_28
triton_per_fused__native_batch_norm_legit_convolution_silu_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_convolution_silu_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 1536
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
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 384, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = tmp0 - tmp10
    tmp18 = 384.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = 0.09125009274634042
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (384*x0)), tmp27, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hj/chjeihjf6hz2jjryotcb3phnph2c46m7nxavppcoizedfie5unnd.py
# Source Nodes: [batch_norm_23, getattr_getattr_l__mod___stages___2_____1___act1, out_32, out_33], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
# batch_norm_23 => var_mean_23
# getattr_getattr_l__mod___stages___2_____1___act1 => mul_104, sigmoid_23
# out_32 => mul_105
# out_33 => convolution_31
triton_red_fused__native_batch_norm_legit_convolution_mul_silu_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_convolution_mul_silu_16', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
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
        tmp14 = 0.04562504637317021
        tmp15 = tmp13 * tmp14
        tmp16 = tmp12 * tmp15
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp16, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2j/c2j62oiab3guuk63o2hbfplsox3yfrhdgsxzdlottlvi46a6bkbq.py
# Source Nodes: [batch_norm_43, getattr_getattr_l__mod___stages___3_____0___downsample_pool, shortcut_13], Original ATen: [aten._native_batch_norm_legit, aten.avg_pool2d, aten.convolution]
# batch_norm_43 => var_mean_43
# getattr_getattr_l__mod___stages___3_____0___downsample_pool => avg_pool2d_2
# shortcut_13 => convolution_61
triton_red_fused__native_batch_norm_legit_avg_pool2d_convolution_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_avg_pool2d_convolution_17', 'mutated_arg_names': []}
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
        tmp14 = 0.04562504637317021
        tmp15 = tmp13 * tmp14
        tmp16 = tmp12 * tmp15
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp16, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gr/cgr6iburtlwyszgig7b4ggewkf7g5k3qpyiyisrgoqr6djkmcfl2.py
# Source Nodes: [batch_norm_56, getattr_getattr_l__mod___stages___3_____2___act1, getattr_getattr_l__mod___stages___3_____2___act2, getattr_getattr_l__mod___stages___3_____2___act2b, getattr_getattr_l__mod___stages___3_____2___act3, mul_101, mul_103, out_88, out_89, out_90, out_91, out_92, out_93, sigmoid_11, x_1, x_2, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten._native_batch_norm_legit, aten.add, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
# batch_norm_56 => var_mean_56
# getattr_getattr_l__mod___stages___3_____2___act1 => mul_247, sigmoid_58
# getattr_getattr_l__mod___stages___3_____2___act2 => mul_252, sigmoid_59
# getattr_getattr_l__mod___stages___3_____2___act2b => mul_256, sigmoid_60
# getattr_getattr_l__mod___stages___3_____2___act3 => mul_260, sigmoid_61
# mul_101 => mul_264
# mul_103 => mul_266
# out_88 => mul_248
# out_89 => convolution_74
# out_90 => convolution_75
# out_91 => convolution_76
# out_92 => convolution_77
# out_93 => mul_265
# sigmoid_11 => sigmoid_62
# x_1 => add_67
# x_2 => convolution_80
# x_se_44 => mean_11
# x_se_45 => convolution_78
# x_se_46 => relu_11
# x_se_47 => convolution_79
triton_red_fused__native_batch_norm_legit_add_convolution_mean_mul_relu_sigmoid_silu_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_add_convolution_mean_mul_relu_sigmoid_silu_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2304
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
        tmp14 = 0.04562504637317021
        tmp15 = tmp13 * tmp14
        tmp16 = tmp12 * tmp15
        tl.store(out_ptr2 + (r1 + (1536*x0)), tmp16, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s2/cs2ie7lmjvxmgyo4rklx264ecjzvaqfpvowffdlqpqbrznrgiivw.py
# Source Nodes: [conv2d, conv2d_1, l__mod___stem_act2], Original ATen: [aten.convolution, aten.silu]
# conv2d => convolution
# conv2d_1 => convolution_1
# l__mod___stem_act2 => mul_3, sigmoid
triton_poi_fused_convolution_silu_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2654208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 20736) % 16
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jf/cjfjpkwjoy2is5iagbrurpf7kyzyxdbinsq3vskc3wmydxufd26y.py
# Source Nodes: [conv2d, conv2d_1, conv2d_2, l__mod___stem_act2, l__mod___stem_act3], Original ATen: [aten.convolution, aten.silu]
# conv2d => convolution
# conv2d_1 => convolution_1
# conv2d_2 => convolution_2
# l__mod___stem_act2 => mul_3, sigmoid
# l__mod___stem_act3 => mul_7, sigmoid_1
triton_poi_fused_convolution_silu_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5308416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 20736) % 32
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/v6/cv6rm3sakh2iaaof3vtoswutnodsozhztv2cmnendbll63n7cyis.py
# Source Nodes: [conv2d, conv2d_1, conv2d_2, l__mod___stem_act2, l__mod___stem_act3, l__mod___stem_act4, shortcut], Original ATen: [aten.convolution, aten.silu]
# conv2d => convolution
# conv2d_1 => convolution_1
# conv2d_2 => convolution_2
# l__mod___stem_act2 => mul_3, sigmoid
# l__mod___stem_act3 => mul_7, sigmoid_1
# l__mod___stem_act4 => mul_11, sigmoid_2
# shortcut => convolution_3
triton_poi_fused_convolution_silu_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10616832
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 20736) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lz/clzgw2rukwsmnjv4a76tmnjip4sc7pxtjumobhtzqqc6fdmovhwi.py
# Source Nodes: [conv2d, conv2d_1, conv2d_2, getattr_getattr_l__mod___stages___0_____0___act1, l__mod___stem_act2, l__mod___stem_act3, l__mod___stem_act4, out, shortcut], Original ATen: [aten.convolution, aten.mul, aten.silu]
# conv2d => convolution
# conv2d_1 => convolution_1
# conv2d_2 => convolution_2
# getattr_getattr_l__mod___stages___0_____0___act1 => mul_15, sigmoid_3
# l__mod___stem_act2 => mul_3, sigmoid
# l__mod___stem_act3 => mul_7, sigmoid_1
# l__mod___stem_act4 => mul_11, sigmoid_2
# out => mul_16
# shortcut => convolution_3
triton_poi_fused_convolution_mul_silu_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_silu_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5308416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 5184) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tl.store(in_out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ml/cmll76ddchcfpmviuwkkoqcmoud2rw4ghjra6oikcysfxg62ldu7.py
# Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act2, out_1, out_2], Original ATen: [aten.convolution, aten.silu]
# getattr_getattr_l__mod___stages___0_____0___act2 => mul_23, sigmoid_4
# out_1 => convolution_5
# out_2 => convolution_6
triton_poi_fused_convolution_silu_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2654208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 5184) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/of/cofupx7fzug7kxrx2aex7rhpc7alrv4netvxf6ojk7gscwix5p7o.py
# Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act2, getattr_getattr_l__mod___stages___0_____0___act2b, getattr_getattr_l__mod___stages___0_____0___act3, out_1, out_2, out_3, out_4, x_se, x_se_1], Original ATen: [aten.convolution, aten.mean, aten.silu]
# getattr_getattr_l__mod___stages___0_____0___act2 => mul_23, sigmoid_4
# getattr_getattr_l__mod___stages___0_____0___act2b => mul_27, sigmoid_5
# getattr_getattr_l__mod___stages___0_____0___act3 => mul_31, sigmoid_6
# out_1 => convolution_5
# out_2 => convolution_6
# out_3 => convolution_7
# out_4 => convolution_8
# x_se => mean
# x_se_1 => convolution_9
triton_red_fused_convolution_mean_silu_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[2048, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_mean_silu_24', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 5184
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
        tmp0 = tl.load(in_ptr0 + (r2 + (5184*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp6 = 5184.0
    tmp7 = tmp4 / tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/d5/cd5jw3jr4q4aabwqgzbh6eemra56x7cibza2php6dpjfzox2z2il.py
# Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act2, getattr_getattr_l__mod___stages___0_____0___act2b, getattr_getattr_l__mod___stages___0_____0___act3, out_1, out_2, out_3, out_4, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.silu]
# getattr_getattr_l__mod___stages___0_____0___act2 => mul_23, sigmoid_4
# getattr_getattr_l__mod___stages___0_____0___act2b => mul_27, sigmoid_5
# getattr_getattr_l__mod___stages___0_____0___act3 => mul_31, sigmoid_6
# out_1 => convolution_5
# out_2 => convolution_6
# out_3 => convolution_7
# out_4 => convolution_8
# x_se => mean
# x_se_1 => convolution_9
# x_se_2 => relu
# x_se_3 => convolution_10
triton_poi_fused_convolution_mean_relu_silu_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_silu_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xb/cxbr67euqiuzh3ey5y7fhj2bzfxr27kuywaoq6wh5spenosdlkfc.py
# Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act2, getattr_getattr_l__mod___stages___0_____0___act2b, getattr_getattr_l__mod___stages___0_____0___act3, getattr_getattr_l__mod___stages___1_____0___act1, mul_10, mul_12, out_1, out_2, out_3, out_4, out_5, out_8, shortcut_1, shortcut_2, sigmoid, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.add, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___stages___0_____0___act2 => mul_23, sigmoid_4
# getattr_getattr_l__mod___stages___0_____0___act2b => mul_27, sigmoid_5
# getattr_getattr_l__mod___stages___0_____0___act3 => mul_31, sigmoid_6
# getattr_getattr_l__mod___stages___1_____0___act1 => mul_38, sigmoid_8
# mul_10 => mul_35
# mul_12 => mul_37
# out_1 => convolution_5
# out_2 => convolution_6
# out_3 => convolution_7
# out_4 => convolution_8
# out_5 => mul_36
# out_8 => mul_39
# shortcut_1 => convolution_4
# shortcut_2 => add_9
# sigmoid => sigmoid_7
# x_se => mean
# x_se_1 => convolution_9
# x_se_2 => relu
# x_se_3 => convolution_10
triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_26', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10616832
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 5184) % 256
    x4 = (xindex // 5184)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x3), None)
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.2
    tmp11 = tmp9 * tmp10
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = 0.9805806756909201
    tmp19 = tmp17 * tmp18
    tl.store(in_out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/va/cva5x7ekjlywi6xfck7jk2jwy5dl7wyydvt6b4g7spuanjr37re5.py
# Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act2, out_10, out_9], Original ATen: [aten.convolution, aten.silu]
# getattr_getattr_l__mod___stages___1_____0___act2 => mul_46, sigmoid_9
# out_10 => convolution_13
# out_9 => convolution_12
triton_poi_fused_convolution_silu_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_27', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5308416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 5184) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qf/cqf2puzz5dzvqbiogeztdct4oungv5fyzb2r5xwudqwmqss6twlb.py
# Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act2, getattr_getattr_l__mod___stages___1_____0___act2b, out_10, out_11, out_9], Original ATen: [aten.convolution, aten.silu]
# getattr_getattr_l__mod___stages___1_____0___act2 => mul_46, sigmoid_9
# getattr_getattr_l__mod___stages___1_____0___act2b => mul_50, sigmoid_10
# out_10 => convolution_13
# out_11 => convolution_14
# out_9 => convolution_12
triton_poi_fused_convolution_silu_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_28', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1327104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1296) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vy/cvypqvz5txtnwpkke6oq3k57skrbgoznezkxyxmexqbci32s37uq.py
# Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act2, getattr_getattr_l__mod___stages___1_____0___act2b, getattr_getattr_l__mod___stages___1_____0___act3, out_10, out_11, out_12, out_9, x_se_4, x_se_5], Original ATen: [aten.convolution, aten.mean, aten.silu]
# getattr_getattr_l__mod___stages___1_____0___act2 => mul_46, sigmoid_9
# getattr_getattr_l__mod___stages___1_____0___act2b => mul_50, sigmoid_10
# getattr_getattr_l__mod___stages___1_____0___act3 => mul_54, sigmoid_11
# out_10 => convolution_13
# out_11 => convolution_14
# out_12 => convolution_15
# out_9 => convolution_12
# x_se_4 => mean_1
# x_se_5 => convolution_16
triton_red_fused_convolution_mean_silu_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_mean_silu_29', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 1296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x3 = xindex
    x0 = xindex % 512
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (1296*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp6 = 1296.0
    tmp7 = tmp4 / tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xd/cxdn46uu27x2krpb4kpy4tugnbof3kh4pnzz7zdqpkps5eholze2.py
# Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act2, getattr_getattr_l__mod___stages___1_____0___act2b, getattr_getattr_l__mod___stages___1_____0___act3, out_10, out_11, out_12, out_9, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.silu]
# getattr_getattr_l__mod___stages___1_____0___act2 => mul_46, sigmoid_9
# getattr_getattr_l__mod___stages___1_____0___act2b => mul_50, sigmoid_10
# getattr_getattr_l__mod___stages___1_____0___act3 => mul_54, sigmoid_11
# out_10 => convolution_13
# out_11 => convolution_14
# out_12 => convolution_15
# out_9 => convolution_12
# x_se_4 => mean_1
# x_se_5 => convolution_16
# x_se_6 => relu_1
# x_se_7 => convolution_17
triton_poi_fused_convolution_mean_relu_silu_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_silu_30', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/qp/cqpbrghz4qgdxaq6uh5gkioky5kofh6wye3pxwrakxmobf7wd2gd.py
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
    size_hints=[4194304], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_convolution_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2654208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 36
    x1 = (xindex // 36)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (144*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (144*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (72 + (2*x0) + (144*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (73 + (2*x0) + (144*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mo/cmoldwksgl22n7ja2cbnel7rapif5tkbakk6vcrpydml5yw6yq32.py
# Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act2, getattr_getattr_l__mod___stages___1_____0___act2b, getattr_getattr_l__mod___stages___1_____0___act3, getattr_getattr_l__mod___stages___1_____0___downsample_pool, getattr_getattr_l__mod___stages___1_____1___act1, mul_19, mul_21, out_10, out_11, out_12, out_13, out_16, out_17, out_9, shortcut_3, shortcut_4, sigmoid_1, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___stages___1_____0___act2 => mul_46, sigmoid_9
# getattr_getattr_l__mod___stages___1_____0___act2b => mul_50, sigmoid_10
# getattr_getattr_l__mod___stages___1_____0___act3 => mul_54, sigmoid_11
# getattr_getattr_l__mod___stages___1_____0___downsample_pool => avg_pool2d
# getattr_getattr_l__mod___stages___1_____1___act1 => mul_61, sigmoid_13
# mul_19 => mul_58
# mul_21 => mul_60
# out_10 => convolution_13
# out_11 => convolution_14
# out_12 => convolution_15
# out_13 => mul_59
# out_16 => mul_62
# out_17 => convolution_18
# out_9 => convolution_12
# shortcut_3 => convolution_11
# shortcut_4 => add_15
# sigmoid_1 => sigmoid_12
# x_se_4 => mean_1
# x_se_5 => convolution_16
# x_se_6 => relu_1
# x_se_7 => convolution_17
triton_poi_fused_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5308416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1296) % 512
    x4 = (xindex // 1296)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x3), None)
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.2
    tmp11 = tmp9 * tmp10
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = 0.9805806756909201
    tmp19 = tmp17 * tmp18
    tl.store(in_out_ptr0 + (x3), tmp15, None)
    tl.store(out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gm/cgmsxqn6ozjtriebymzctnijec233gu6fkbcalm5sy537lykjs7w.py
# Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act1, getattr_getattr_l__mod___stages___1_____1___act2, getattr_getattr_l__mod___stages___1_____1___act2b, getattr_getattr_l__mod___stages___1_____1___act3, getattr_getattr_l__mod___stages___2_____0___act1, mul_27, mul_29, out_16, out_17, out_18, out_19, out_20, out_21, out_24, shortcut_5, sigmoid_2, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.add, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___stages___1_____1___act1 => mul_61, sigmoid_13
# getattr_getattr_l__mod___stages___1_____1___act2 => mul_66, sigmoid_14
# getattr_getattr_l__mod___stages___1_____1___act2b => mul_70, sigmoid_15
# getattr_getattr_l__mod___stages___1_____1___act3 => mul_74, sigmoid_16
# getattr_getattr_l__mod___stages___2_____0___act1 => mul_81, sigmoid_18
# mul_27 => mul_78
# mul_29 => mul_80
# out_16 => mul_62
# out_17 => convolution_18
# out_18 => convolution_19
# out_19 => convolution_20
# out_20 => convolution_21
# out_21 => mul_79
# out_24 => mul_82
# shortcut_5 => add_20
# sigmoid_2 => sigmoid_17
# x_se_10 => relu_2
# x_se_11 => convolution_23
# x_se_8 => mean_2
# x_se_9 => convolution_22
triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_33', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5308416
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1296) % 512
    x4 = (xindex // 1296)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.2
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 0.9622504486493761
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4n/c4nbb673sp4mybvlumqbclr25lkig3p4igg6sixchtn57og5eup5.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act2, out_25, out_26], Original ATen: [aten.convolution, aten.silu]
# getattr_getattr_l__mod___stages___2_____0___act2 => mul_89, sigmoid_19
# out_25 => convolution_25
# out_26 => convolution_26
triton_poi_fused_convolution_silu_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_34', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3981312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1296) % 384
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/in/cinbganmpzawlba64sdyw546yd42vzew4ebxxsommufo73vx6xkh.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act2, getattr_getattr_l__mod___stages___2_____0___act2b, out_25, out_26, out_27], Original ATen: [aten.convolution, aten.silu]
# getattr_getattr_l__mod___stages___2_____0___act2 => mul_89, sigmoid_19
# getattr_getattr_l__mod___stages___2_____0___act2b => mul_93, sigmoid_20
# out_25 => convolution_25
# out_26 => convolution_26
# out_27 => convolution_27
triton_poi_fused_convolution_silu_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_35', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 995328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 324) % 384
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/z2/cz2xqnyjzjuw4g6o4dfuilxppcs3vmou33igaojkspzixdffbl52.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act2, getattr_getattr_l__mod___stages___2_____0___act2b, getattr_getattr_l__mod___stages___2_____0___act3, out_25, out_26, out_27, out_28, x_se_12, x_se_13], Original ATen: [aten.convolution, aten.mean, aten.silu]
# getattr_getattr_l__mod___stages___2_____0___act2 => mul_89, sigmoid_19
# getattr_getattr_l__mod___stages___2_____0___act2b => mul_93, sigmoid_20
# getattr_getattr_l__mod___stages___2_____0___act3 => mul_97, sigmoid_21
# out_25 => convolution_25
# out_26 => convolution_26
# out_27 => convolution_27
# out_28 => convolution_28
# x_se_12 => mean_3
# x_se_13 => convolution_29
triton_per_fused_convolution_mean_silu_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_mean_silu_36', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel):
    xnumel = 12288
    XBLOCK: tl.constexpr = 1
    rnumel = 324
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1536
    tmp0 = tl.load(in_ptr0 + (r2 + (324*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp7 = 324.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/v5/cv5i4molv5hprafcftpf3pszywf6a6mqorx3uuvm7rigeskzbiuf.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act2, getattr_getattr_l__mod___stages___2_____0___act2b, getattr_getattr_l__mod___stages___2_____0___act3, out_25, out_26, out_27, out_28, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.silu]
# getattr_getattr_l__mod___stages___2_____0___act2 => mul_89, sigmoid_19
# getattr_getattr_l__mod___stages___2_____0___act2b => mul_93, sigmoid_20
# getattr_getattr_l__mod___stages___2_____0___act3 => mul_97, sigmoid_21
# out_25 => convolution_25
# out_26 => convolution_26
# out_27 => convolution_27
# out_28 => convolution_28
# x_se_12 => mean_3
# x_se_13 => convolution_29
# x_se_14 => relu_3
# x_se_15 => convolution_30
triton_poi_fused_convolution_mean_relu_silu_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_silu_37', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vm/cvm3igc6cxjru22ka5oyew4zjy4aelnstbm2d4ifj7tp6uedbr7y.py
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
    size_hints=[2097152], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_convolution_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1327104
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 18
    x1 = (xindex // 18)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (72*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (72*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (36 + (2*x0) + (72*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (37 + (2*x0) + (72*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/r5/cr5gxqs3vvr4tykvqij5kvhohuuus56t7hw5fmjgmhnfn5txn7jp.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act2, getattr_getattr_l__mod___stages___2_____0___act2b, getattr_getattr_l__mod___stages___2_____0___act3, getattr_getattr_l__mod___stages___2_____0___downsample_pool, getattr_getattr_l__mod___stages___2_____1___act1, mul_36, mul_38, out_25, out_26, out_27, out_28, out_29, out_32, out_33, shortcut_6, shortcut_7, sigmoid_3, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___stages___2_____0___act2 => mul_89, sigmoid_19
# getattr_getattr_l__mod___stages___2_____0___act2b => mul_93, sigmoid_20
# getattr_getattr_l__mod___stages___2_____0___act3 => mul_97, sigmoid_21
# getattr_getattr_l__mod___stages___2_____0___downsample_pool => avg_pool2d_1
# getattr_getattr_l__mod___stages___2_____1___act1 => mul_104, sigmoid_23
# mul_36 => mul_101
# mul_38 => mul_103
# out_25 => convolution_25
# out_26 => convolution_26
# out_27 => convolution_27
# out_28 => convolution_28
# out_29 => mul_102
# out_32 => mul_105
# out_33 => convolution_31
# shortcut_6 => convolution_24
# shortcut_7 => add_26
# sigmoid_3 => sigmoid_22
# x_se_12 => mean_3
# x_se_13 => convolution_29
# x_se_14 => relu_3
# x_se_15 => convolution_30
triton_poi_fused_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_39', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3981312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 324) % 1536
    x4 = (xindex // 324)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x3), None)
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.2
    tmp11 = tmp9 * tmp10
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = 0.9805806756909201
    tmp19 = tmp17 * tmp18
    tl.store(in_out_ptr0 + (x3), tmp15, None)
    tl.store(out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ps/cpsu3waauj34ynkg2y23imgevkijgo2sjqze7sgtbfg2sbgvquor.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____1___act1, getattr_getattr_l__mod___stages___2_____1___act2, getattr_getattr_l__mod___stages___2_____1___act2b, getattr_getattr_l__mod___stages___2_____1___act3, getattr_getattr_l__mod___stages___2_____2___act1, mul_44, mul_46, out_32, out_33, out_34, out_35, out_36, out_37, out_40, out_41, shortcut_8, sigmoid_4, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.add, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___stages___2_____1___act1 => mul_104, sigmoid_23
# getattr_getattr_l__mod___stages___2_____1___act2 => mul_109, sigmoid_24
# getattr_getattr_l__mod___stages___2_____1___act2b => mul_113, sigmoid_25
# getattr_getattr_l__mod___stages___2_____1___act3 => mul_117, sigmoid_26
# getattr_getattr_l__mod___stages___2_____2___act1 => mul_124, sigmoid_28
# mul_44 => mul_121
# mul_46 => mul_123
# out_32 => mul_105
# out_33 => convolution_31
# out_34 => convolution_32
# out_35 => convolution_33
# out_36 => convolution_34
# out_37 => mul_122
# out_40 => mul_125
# out_41 => convolution_37
# shortcut_8 => add_31
# sigmoid_4 => sigmoid_27
# x_se_16 => mean_4
# x_se_17 => convolution_35
# x_se_18 => relu_4
# x_se_19 => convolution_36
triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_40', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3981312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 324) % 1536
    x4 = (xindex // 324)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.2
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 0.9622504486493761
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp13, None)
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/55/c55l4g56wl5yqzos2zlfcc3km3wru5ewaber2c4hwatitdz6scf6.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____2___act1, getattr_getattr_l__mod___stages___2_____2___act2, getattr_getattr_l__mod___stages___2_____2___act2b, getattr_getattr_l__mod___stages___2_____2___act3, getattr_getattr_l__mod___stages___2_____3___act1, mul_52, mul_54, out_40, out_41, out_42, out_43, out_44, out_45, out_48, out_49, shortcut_9, sigmoid_5, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.add, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___stages___2_____2___act1 => mul_124, sigmoid_28
# getattr_getattr_l__mod___stages___2_____2___act2 => mul_129, sigmoid_29
# getattr_getattr_l__mod___stages___2_____2___act2b => mul_133, sigmoid_30
# getattr_getattr_l__mod___stages___2_____2___act3 => mul_137, sigmoid_31
# getattr_getattr_l__mod___stages___2_____3___act1 => mul_144, sigmoid_33
# mul_52 => mul_141
# mul_54 => mul_143
# out_40 => mul_125
# out_41 => convolution_37
# out_42 => convolution_38
# out_43 => convolution_39
# out_44 => convolution_40
# out_45 => mul_142
# out_48 => mul_145
# out_49 => convolution_43
# shortcut_9 => add_36
# sigmoid_5 => sigmoid_32
# x_se_20 => mean_5
# x_se_21 => convolution_41
# x_se_22 => relu_5
# x_se_23 => convolution_42
triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_41', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3981312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 324) % 1536
    x4 = (xindex // 324)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.2
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 0.9449111825230679
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp13, None)
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/l6/cl6gy5hw2skodjjj2j6h7apicauzpzv3gxd32quptaiwya3tv6vb.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____3___act1, getattr_getattr_l__mod___stages___2_____3___act2, getattr_getattr_l__mod___stages___2_____3___act2b, getattr_getattr_l__mod___stages___2_____3___act3, getattr_getattr_l__mod___stages___2_____4___act1, mul_60, mul_62, out_48, out_49, out_50, out_51, out_52, out_53, out_56, out_57, shortcut_10, sigmoid_6, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.add, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___stages___2_____3___act1 => mul_144, sigmoid_33
# getattr_getattr_l__mod___stages___2_____3___act2 => mul_149, sigmoid_34
# getattr_getattr_l__mod___stages___2_____3___act2b => mul_153, sigmoid_35
# getattr_getattr_l__mod___stages___2_____3___act3 => mul_157, sigmoid_36
# getattr_getattr_l__mod___stages___2_____4___act1 => mul_164, sigmoid_38
# mul_60 => mul_161
# mul_62 => mul_163
# out_48 => mul_145
# out_49 => convolution_43
# out_50 => convolution_44
# out_51 => convolution_45
# out_52 => convolution_46
# out_53 => mul_162
# out_56 => mul_165
# out_57 => convolution_49
# shortcut_10 => add_41
# sigmoid_6 => sigmoid_37
# x_se_24 => mean_6
# x_se_25 => convolution_47
# x_se_26 => relu_6
# x_se_27 => convolution_48
triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_42', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3981312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 324) % 1536
    x4 = (xindex // 324)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.2
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 0.9284766908852592
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp13, None)
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qm/cqmuk2anwxsufi6ag7lbphmbv727434jrvatldrykc2nli32ok6g.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____4___act1, getattr_getattr_l__mod___stages___2_____4___act2, getattr_getattr_l__mod___stages___2_____4___act2b, getattr_getattr_l__mod___stages___2_____4___act3, getattr_getattr_l__mod___stages___2_____5___act1, mul_68, mul_70, out_56, out_57, out_58, out_59, out_60, out_61, out_64, out_65, shortcut_11, sigmoid_7, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.add, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___stages___2_____4___act1 => mul_164, sigmoid_38
# getattr_getattr_l__mod___stages___2_____4___act2 => mul_169, sigmoid_39
# getattr_getattr_l__mod___stages___2_____4___act2b => mul_173, sigmoid_40
# getattr_getattr_l__mod___stages___2_____4___act3 => mul_177, sigmoid_41
# getattr_getattr_l__mod___stages___2_____5___act1 => mul_184, sigmoid_43
# mul_68 => mul_181
# mul_70 => mul_183
# out_56 => mul_165
# out_57 => convolution_49
# out_58 => convolution_50
# out_59 => convolution_51
# out_60 => convolution_52
# out_61 => mul_182
# out_64 => mul_185
# out_65 => convolution_55
# shortcut_11 => add_46
# sigmoid_7 => sigmoid_42
# x_se_28 => mean_7
# x_se_29 => convolution_53
# x_se_30 => relu_7
# x_se_31 => convolution_54
triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_43', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3981312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 324) % 1536
    x4 = (xindex // 324)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.2
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 0.9128709291752768
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp13, None)
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wm/cwm7vvojdpwro2zgj7mrqb45sz4cbgd65bqr3f6zqxbpcta23lrr.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____5___act1, getattr_getattr_l__mod___stages___2_____5___act2, getattr_getattr_l__mod___stages___2_____5___act2b, getattr_getattr_l__mod___stages___2_____5___act3, getattr_getattr_l__mod___stages___3_____0___act1, mul_76, mul_78, out_64, out_65, out_66, out_67, out_68, out_69, out_72, shortcut_12, sigmoid_8, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.add, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___stages___2_____5___act1 => mul_184, sigmoid_43
# getattr_getattr_l__mod___stages___2_____5___act2 => mul_189, sigmoid_44
# getattr_getattr_l__mod___stages___2_____5___act2b => mul_193, sigmoid_45
# getattr_getattr_l__mod___stages___2_____5___act3 => mul_197, sigmoid_46
# getattr_getattr_l__mod___stages___3_____0___act1 => mul_204, sigmoid_48
# mul_76 => mul_201
# mul_78 => mul_203
# out_64 => mul_185
# out_65 => convolution_55
# out_66 => convolution_56
# out_67 => convolution_57
# out_68 => convolution_58
# out_69 => mul_202
# out_72 => mul_205
# shortcut_12 => add_51
# sigmoid_8 => sigmoid_47
# x_se_32 => mean_8
# x_se_33 => convolution_59
# x_se_34 => relu_8
# x_se_35 => convolution_60
triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_44', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3981312
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 324) % 1536
    x4 = (xindex // 324)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.2
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 0.8980265101338745
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zh/czht2p4vih2aiszlhu4clddnxt34uu24beiz4vfxyxtbfvq32eix.py
# Source Nodes: [getattr_getattr_l__mod___stages___3_____0___act2, getattr_getattr_l__mod___stages___3_____0___act2b, out_73, out_74, out_75], Original ATen: [aten.convolution, aten.silu]
# getattr_getattr_l__mod___stages___3_____0___act2 => mul_212, sigmoid_49
# getattr_getattr_l__mod___stages___3_____0___act2b => mul_216, sigmoid_50
# out_73 => convolution_62
# out_74 => convolution_63
# out_75 => convolution_64
triton_poi_fused_convolution_silu_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_45', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 248832
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 81) % 384
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wd/cwdgfatlspbpjg6ngdtdajtocrpy3fyqkj2mfquyyr4qtopciug4.py
# Source Nodes: [getattr_getattr_l__mod___stages___3_____0___act2, getattr_getattr_l__mod___stages___3_____0___act2b, getattr_getattr_l__mod___stages___3_____0___act3, out_73, out_74, out_75, out_76, x_se_36, x_se_37], Original ATen: [aten.convolution, aten.mean, aten.silu]
# getattr_getattr_l__mod___stages___3_____0___act2 => mul_212, sigmoid_49
# getattr_getattr_l__mod___stages___3_____0___act2b => mul_216, sigmoid_50
# getattr_getattr_l__mod___stages___3_____0___act3 => mul_220, sigmoid_51
# out_73 => convolution_62
# out_74 => convolution_63
# out_75 => convolution_64
# out_76 => convolution_65
# x_se_36 => mean_9
# x_se_37 => convolution_66
triton_per_fused_convolution_mean_silu_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_mean_silu_46', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 81
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1536
    tmp0 = tl.load(in_ptr0 + (r2 + (81*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 81.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4v/c4v7tebhisxnccfqh5cju6fjvng3ufoai2vjfzr6wzncdkz5qwtc.py
# Source Nodes: [getattr_getattr_l__mod___stages___3_____0___downsample_pool, shortcut_13], Original ATen: [aten.avg_pool2d, aten.convolution]
# getattr_getattr_l__mod___stages___3_____0___downsample_pool => avg_pool2d_2
# shortcut_13 => convolution_61
triton_poi_fused_avg_pool2d_convolution_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_convolution_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 995328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 9
    x1 = (xindex // 9)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (36*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (36*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (18 + (2*x0) + (36*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (19 + (2*x0) + (36*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fa/cfa4w2hjwmcz55ihs6chekvq72pkh3qv2j5mgm4m2jrmpjfp3mzh.py
# Source Nodes: [getattr_getattr_l__mod___stages___3_____0___act2, getattr_getattr_l__mod___stages___3_____0___act2b, getattr_getattr_l__mod___stages___3_____0___act3, getattr_getattr_l__mod___stages___3_____0___downsample_pool, getattr_getattr_l__mod___stages___3_____1___act1, mul_85, mul_87, out_73, out_74, out_75, out_76, out_77, out_80, out_81, shortcut_13, shortcut_14, sigmoid_9, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___stages___3_____0___act2 => mul_212, sigmoid_49
# getattr_getattr_l__mod___stages___3_____0___act2b => mul_216, sigmoid_50
# getattr_getattr_l__mod___stages___3_____0___act3 => mul_220, sigmoid_51
# getattr_getattr_l__mod___stages___3_____0___downsample_pool => avg_pool2d_2
# getattr_getattr_l__mod___stages___3_____1___act1 => mul_227, sigmoid_53
# mul_85 => mul_224
# mul_87 => mul_226
# out_73 => convolution_62
# out_74 => convolution_63
# out_75 => convolution_64
# out_76 => convolution_65
# out_77 => mul_225
# out_80 => mul_228
# out_81 => convolution_68
# shortcut_13 => convolution_61
# shortcut_14 => add_57
# sigmoid_9 => sigmoid_52
# x_se_36 => mean_9
# x_se_37 => convolution_66
# x_se_38 => relu_9
# x_se_39 => convolution_67
triton_poi_fused_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_48', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 995328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 81) % 1536
    x4 = (xindex // 81)
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr3 + (x3), None)
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.2
    tmp11 = tmp9 * tmp10
    tmp14 = tmp12 + tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = tl.sigmoid(tmp15)
    tmp17 = tmp15 * tmp16
    tmp18 = 0.9805806756909201
    tmp19 = tmp17 * tmp18
    tl.store(in_out_ptr0 + (x3), tmp15, None)
    tl.store(out_ptr0 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/az/cazyrtrzr2s2acl6mmtvhvgadctqk3j667y5ydixd5cqvceypn2v.py
# Source Nodes: [getattr_getattr_l__mod___stages___3_____1___act1, getattr_getattr_l__mod___stages___3_____1___act2, getattr_getattr_l__mod___stages___3_____1___act2b, getattr_getattr_l__mod___stages___3_____1___act3, getattr_getattr_l__mod___stages___3_____2___act1, mul_93, mul_95, out_80, out_81, out_82, out_83, out_84, out_85, out_88, out_89, shortcut_15, sigmoid_10, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.add, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___stages___3_____1___act1 => mul_227, sigmoid_53
# getattr_getattr_l__mod___stages___3_____1___act2 => mul_232, sigmoid_54
# getattr_getattr_l__mod___stages___3_____1___act2b => mul_236, sigmoid_55
# getattr_getattr_l__mod___stages___3_____1___act3 => mul_240, sigmoid_56
# getattr_getattr_l__mod___stages___3_____2___act1 => mul_247, sigmoid_58
# mul_93 => mul_244
# mul_95 => mul_246
# out_80 => mul_228
# out_81 => convolution_68
# out_82 => convolution_69
# out_83 => convolution_70
# out_84 => convolution_71
# out_85 => mul_245
# out_88 => mul_248
# out_89 => convolution_74
# shortcut_15 => add_62
# sigmoid_10 => sigmoid_57
# x_se_40 => mean_10
# x_se_41 => convolution_72
# x_se_42 => relu_10
# x_se_43 => convolution_73
triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_49', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 995328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 81) % 1536
    x4 = (xindex // 81)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.2
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 0.9622504486493761
    tmp17 = tmp15 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp13, None)
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hq/chq72m27xqf3hb5rgtqogafy3nxuqjirrbwbpokj4raskqttybm5.py
# Source Nodes: [getattr_getattr_l__mod___stages___3_____2___act1, getattr_getattr_l__mod___stages___3_____2___act2, getattr_getattr_l__mod___stages___3_____2___act2b, getattr_getattr_l__mod___stages___3_____2___act3, mul_101, mul_103, out_88, out_89, out_90, out_91, out_92, out_93, sigmoid_11, x_1, x_2, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.add, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___stages___3_____2___act1 => mul_247, sigmoid_58
# getattr_getattr_l__mod___stages___3_____2___act2 => mul_252, sigmoid_59
# getattr_getattr_l__mod___stages___3_____2___act2b => mul_256, sigmoid_60
# getattr_getattr_l__mod___stages___3_____2___act3 => mul_260, sigmoid_61
# mul_101 => mul_264
# mul_103 => mul_266
# out_88 => mul_248
# out_89 => convolution_74
# out_90 => convolution_75
# out_91 => convolution_76
# out_92 => convolution_77
# out_93 => mul_265
# sigmoid_11 => sigmoid_62
# x_1 => add_67
# x_2 => convolution_80
# x_se_44 => mean_11
# x_se_45 => convolution_78
# x_se_46 => relu_11
# x_se_47 => convolution_79
triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_50 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_50', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 995328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 81) % 1536
    x4 = (xindex // 81)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_out_ptr0 + (x3), None)
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sigmoid(tmp5)
    tmp7 = tmp2 * tmp6
    tmp8 = 2.0
    tmp9 = tmp7 * tmp8
    tmp10 = 0.2
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(in_out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2t/c2tvn75bmhxgma6cipx2vbmkrhd4bjopxewhxmixr7wy57emu6h6.py
# Source Nodes: [getattr_getattr_l__mod___stages___3_____2___act1, getattr_getattr_l__mod___stages___3_____2___act2, getattr_getattr_l__mod___stages___3_____2___act2b, getattr_getattr_l__mod___stages___3_____2___act3, mul_101, mul_103, out_88, out_89, out_90, out_91, out_92, out_93, sigmoid_11, x_1, x_2, x_4, x_5, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.add, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___stages___3_____2___act1 => mul_247, sigmoid_58
# getattr_getattr_l__mod___stages___3_____2___act2 => mul_252, sigmoid_59
# getattr_getattr_l__mod___stages___3_____2___act2b => mul_256, sigmoid_60
# getattr_getattr_l__mod___stages___3_____2___act3 => mul_260, sigmoid_61
# mul_101 => mul_264
# mul_103 => mul_266
# out_88 => mul_248
# out_89 => convolution_74
# out_90 => convolution_75
# out_91 => convolution_76
# out_92 => convolution_77
# out_93 => mul_265
# sigmoid_11 => sigmoid_62
# x_1 => add_67
# x_2 => convolution_80
# x_4 => mul_270, sigmoid_63
# x_5 => mean_12
# x_se_44 => mean_11
# x_se_45 => convolution_78
# x_se_46 => relu_11
# x_se_47 => convolution_79
triton_per_fused_add_convolution_mean_mul_relu_sigmoid_silu_51 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32768, 128],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_convolution_mean_mul_relu_sigmoid_silu_51', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 18432
    rnumel = 81
    RBLOCK: tl.constexpr = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 2304
    tmp0 = tl.load(in_ptr0 + (r2 + (81*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp9 = 81.0
    tmp10 = tmp8 / tmp9
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp10, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1 = args
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
    assert_size_stride(arg15_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg16_1, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg17_1, (64, ), (1, ))
    assert_size_stride(arg18_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg19_1, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg20_1, (64, ), (1, ))
    assert_size_stride(arg21_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg22_1, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg23_1, (64, ), (1, ))
    assert_size_stride(arg24_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg25_1, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg26_1, (256, ), (1, ))
    assert_size_stride(arg27_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg28_1, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg29_1, (512, ), (1, ))
    assert_size_stride(arg30_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg31_1, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg32_1, (128, ), (1, ))
    assert_size_stride(arg33_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg34_1, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg35_1, (128, ), (1, ))
    assert_size_stride(arg36_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg37_1, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg38_1, (128, ), (1, ))
    assert_size_stride(arg39_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg40_1, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg41_1, (512, ), (1, ))
    assert_size_stride(arg42_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg43_1, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg44_1, (128, ), (1, ))
    assert_size_stride(arg45_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg46_1, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg47_1, (128, ), (1, ))
    assert_size_stride(arg48_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg49_1, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg50_1, (128, ), (1, ))
    assert_size_stride(arg51_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg52_1, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg53_1, (512, ), (1, ))
    assert_size_stride(arg54_1, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg55_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg56_1, (1536, ), (1, ))
    assert_size_stride(arg57_1, (384, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg58_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg59_1, (384, ), (1, ))
    assert_size_stride(arg60_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg61_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg62_1, (384, ), (1, ))
    assert_size_stride(arg63_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg64_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg65_1, (384, ), (1, ))
    assert_size_stride(arg66_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg67_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg68_1, (1536, ), (1, ))
    assert_size_stride(arg69_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg70_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg71_1, (384, ), (1, ))
    assert_size_stride(arg72_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg73_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg74_1, (384, ), (1, ))
    assert_size_stride(arg75_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg76_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg77_1, (384, ), (1, ))
    assert_size_stride(arg78_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg79_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg80_1, (1536, ), (1, ))
    assert_size_stride(arg81_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg82_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg83_1, (384, ), (1, ))
    assert_size_stride(arg84_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg85_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg86_1, (384, ), (1, ))
    assert_size_stride(arg87_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg88_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg89_1, (384, ), (1, ))
    assert_size_stride(arg90_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg91_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg92_1, (1536, ), (1, ))
    assert_size_stride(arg93_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg94_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg95_1, (384, ), (1, ))
    assert_size_stride(arg96_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg97_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg98_1, (384, ), (1, ))
    assert_size_stride(arg99_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg100_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg101_1, (384, ), (1, ))
    assert_size_stride(arg102_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg103_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg104_1, (1536, ), (1, ))
    assert_size_stride(arg105_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg106_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg107_1, (384, ), (1, ))
    assert_size_stride(arg108_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg109_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg110_1, (384, ), (1, ))
    assert_size_stride(arg111_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg112_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg113_1, (384, ), (1, ))
    assert_size_stride(arg114_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg115_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg116_1, (1536, ), (1, ))
    assert_size_stride(arg117_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg118_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg119_1, (384, ), (1, ))
    assert_size_stride(arg120_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg121_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg122_1, (384, ), (1, ))
    assert_size_stride(arg123_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg124_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg125_1, (384, ), (1, ))
    assert_size_stride(arg126_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg127_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg128_1, (1536, ), (1, ))
    assert_size_stride(arg129_1, (1536, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg130_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg131_1, (1536, ), (1, ))
    assert_size_stride(arg132_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg133_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg134_1, (384, ), (1, ))
    assert_size_stride(arg135_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg136_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg137_1, (384, ), (1, ))
    assert_size_stride(arg138_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg139_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg140_1, (384, ), (1, ))
    assert_size_stride(arg141_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg142_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg143_1, (1536, ), (1, ))
    assert_size_stride(arg144_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg145_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg146_1, (384, ), (1, ))
    assert_size_stride(arg147_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg148_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg149_1, (384, ), (1, ))
    assert_size_stride(arg150_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg151_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg152_1, (384, ), (1, ))
    assert_size_stride(arg153_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg154_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg155_1, (1536, ), (1, ))
    assert_size_stride(arg156_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg157_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg158_1, (384, ), (1, ))
    assert_size_stride(arg159_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg160_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg161_1, (384, ), (1, ))
    assert_size_stride(arg162_1, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg163_1, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg164_1, (384, ), (1, ))
    assert_size_stride(arg165_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg166_1, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg167_1, (1536, ), (1, ))
    assert_size_stride(arg168_1, (2304, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg169_1, (2304, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(arg170_1, (2304, ), (1, ))
    assert_size_stride(arg171_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg172_1, (64, ), (1, ))
    assert_size_stride(arg173_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg174_1, (256, ), (1, ))
    assert_size_stride(arg175_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg176_1, (128, ), (1, ))
    assert_size_stride(arg177_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg178_1, (512, ), (1, ))
    assert_size_stride(arg179_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg180_1, (128, ), (1, ))
    assert_size_stride(arg181_1, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg182_1, (512, ), (1, ))
    assert_size_stride(arg183_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg184_1, (384, ), (1, ))
    assert_size_stride(arg185_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg186_1, (1536, ), (1, ))
    assert_size_stride(arg187_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg188_1, (384, ), (1, ))
    assert_size_stride(arg189_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg190_1, (1536, ), (1, ))
    assert_size_stride(arg191_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg192_1, (384, ), (1, ))
    assert_size_stride(arg193_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg194_1, (1536, ), (1, ))
    assert_size_stride(arg195_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg196_1, (384, ), (1, ))
    assert_size_stride(arg197_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg198_1, (1536, ), (1, ))
    assert_size_stride(arg199_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg200_1, (384, ), (1, ))
    assert_size_stride(arg201_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg202_1, (1536, ), (1, ))
    assert_size_stride(arg203_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg204_1, (384, ), (1, ))
    assert_size_stride(arg205_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg206_1, (1536, ), (1, ))
    assert_size_stride(arg207_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg208_1, (384, ), (1, ))
    assert_size_stride(arg209_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg210_1, (1536, ), (1, ))
    assert_size_stride(arg211_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg212_1, (384, ), (1, ))
    assert_size_stride(arg213_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg214_1, (1536, ), (1, ))
    assert_size_stride(arg215_1, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg216_1, (384, ), (1, ))
    assert_size_stride(arg217_1, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg218_1, (1536, ), (1, ))
    assert_size_stride(arg219_1, (1000, 2304), (2304, 1))
    assert_size_stride(arg220_1, (1000, ), (1, ))
    assert_size_stride(arg221_1, (8, 3, 288, 288), (248832, 82944, 288, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf171 = empty((16, 3, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm, conv2d], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_per_fused__native_batch_norm_legit_convolution_0.run(arg0_1, arg1_1, buf171, 16, 27, grid=grid(16), stream=stream0)
        del arg0_1
        del arg1_1
        buf174 = empty((32, 16, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_1, conv2d, conv2d_1, l__mod___stem_act2], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_1.run(arg3_1, arg4_1, buf174, 32, 144, grid=grid(32), stream=stream0)
        del arg3_1
        del arg4_1
        buf177 = empty((64, 32, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_2, conv2d, conv2d_1, conv2d_2, l__mod___stem_act2, l__mod___stem_act3], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_2.run(arg6_1, arg7_1, buf177, 64, 288, grid=grid(64), stream=stream0)
        del arg6_1
        del arg7_1
        buf180 = empty((128, 64, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_3, conv2d, conv2d_1, conv2d_2, l__mod___stem_act2, l__mod___stem_act3, l__mod___stem_act4, shortcut], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_3.run(arg9_1, arg10_1, buf180, 128, 576, grid=grid(128), stream=stream0)
        del arg10_1
        del arg9_1
        buf199 = empty((256, 128, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_4, shortcut_1], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
        triton_per_fused__native_batch_norm_legit_convolution_4.run(arg12_1, arg13_1, buf199, 256, 128, grid=grid(256), stream=stream0)
        del arg12_1
        del arg13_1
        buf183 = empty((64, 128, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_5, out_1], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
        triton_per_fused__native_batch_norm_legit_convolution_5.run(arg15_1, arg16_1, buf183, 64, 128, grid=grid(64), stream=stream0)
        del arg15_1
        del arg16_1
        buf186 = empty((64, 64, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_6, getattr_getattr_l__mod___stages___0_____0___act2, out_1, out_2], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_6.run(arg18_1, arg19_1, buf186, 64, 576, grid=grid(64), stream=stream0)
        del arg18_1
        del arg19_1
        buf189 = empty((64, 64, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_7, getattr_getattr_l__mod___stages___0_____0___act2, getattr_getattr_l__mod___stages___0_____0___act2b, out_1, out_2, out_3], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_6.run(arg21_1, arg22_1, buf189, 64, 576, grid=grid(64), stream=stream0)
        del arg21_1
        del arg22_1
        buf192 = empty((256, 64, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_8, getattr_getattr_l__mod___stages___0_____0___act2, getattr_getattr_l__mod___stages___0_____0___act2b, getattr_getattr_l__mod___stages___0_____0___act3, out_1, out_2, out_3, out_4], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_7.run(arg24_1, arg25_1, buf192, 256, 64, grid=grid(256), stream=stream0)
        del arg24_1
        del arg25_1
        buf220 = empty((512, 256, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_9, getattr_getattr_l__mod___stages___1_____0___downsample_pool, shortcut_3], Original ATen: [aten._native_batch_norm_legit, aten.avg_pool2d, aten.convolution]
        triton_per_fused__native_batch_norm_legit_avg_pool2d_convolution_8.run(arg27_1, arg28_1, buf220, 512, 256, grid=grid(512), stream=stream0)
        del arg27_1
        del arg28_1
        buf203 = empty((128, 256, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_10, out_9], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
        triton_per_fused__native_batch_norm_legit_convolution_9.run(arg30_1, arg31_1, buf203, 128, 256, grid=grid(128), stream=stream0)
        del arg30_1
        del arg31_1
        buf206 = empty((128, 64, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_11, getattr_getattr_l__mod___stages___1_____0___act2, out_10, out_9], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_3.run(arg33_1, arg34_1, buf206, 128, 576, grid=grid(128), stream=stream0)
        del arg33_1
        del arg34_1
        buf209 = empty((128, 64, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_12, getattr_getattr_l__mod___stages___1_____0___act2, getattr_getattr_l__mod___stages___1_____0___act2b, out_10, out_11, out_9], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_3.run(arg36_1, arg37_1, buf209, 128, 576, grid=grid(128), stream=stream0)
        del arg36_1
        del arg37_1
        buf212 = empty((512, 128, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_13, getattr_getattr_l__mod___stages___1_____0___act2, getattr_getattr_l__mod___stages___1_____0___act2b, getattr_getattr_l__mod___stages___1_____0___act3, out_10, out_11, out_12, out_9], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_10.run(arg39_1, arg40_1, buf212, 512, 128, grid=grid(512), stream=stream0)
        del arg39_1
        del arg40_1
        buf224 = empty((128, 512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_14, getattr_getattr_l__mod___stages___1_____1___act1, out_16, out_17], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_mul_silu_11.run(arg42_1, arg43_1, buf224, 128, 512, grid=grid(128), stream=stream0)
        del arg42_1
        del arg43_1
        buf227 = empty((128, 64, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_15, getattr_getattr_l__mod___stages___1_____1___act1, getattr_getattr_l__mod___stages___1_____1___act2, out_16, out_17, out_18], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_3.run(arg45_1, arg46_1, buf227, 128, 576, grid=grid(128), stream=stream0)
        del arg45_1
        del arg46_1
        buf230 = empty((128, 64, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_16, getattr_getattr_l__mod___stages___1_____1___act1, getattr_getattr_l__mod___stages___1_____1___act2, getattr_getattr_l__mod___stages___1_____1___act2b, out_16, out_17, out_18, out_19], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_3.run(arg48_1, arg49_1, buf230, 128, 576, grid=grid(128), stream=stream0)
        del arg48_1
        del arg49_1
        buf233 = empty((512, 128, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_17, getattr_getattr_l__mod___stages___1_____1___act1, getattr_getattr_l__mod___stages___1_____1___act2, getattr_getattr_l__mod___stages___1_____1___act2b, getattr_getattr_l__mod___stages___1_____1___act3, out_16, out_17, out_18, out_19, out_20], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_10.run(arg51_1, arg52_1, buf233, 512, 128, grid=grid(512), stream=stream0)
        del arg51_1
        del arg52_1
        buf259 = empty((1536, 512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_18, getattr_getattr_l__mod___stages___2_____0___downsample_pool, shortcut_6], Original ATen: [aten._native_batch_norm_legit, aten.avg_pool2d, aten.convolution]
        triton_per_fused__native_batch_norm_legit_avg_pool2d_convolution_12.run(arg54_1, arg55_1, buf259, 1536, 512, grid=grid(1536), stream=stream0)
        del arg54_1
        del arg55_1
        buf242 = empty((384, 512, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_19, out_25], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
        triton_per_fused__native_batch_norm_legit_convolution_13.run(arg57_1, arg58_1, buf242, 384, 512, grid=grid(384), stream=stream0)
        del arg57_1
        del arg58_1
        buf245 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_20, getattr_getattr_l__mod___stages___2_____0___act2, out_25, out_26], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_14.run(arg60_1, arg61_1, buf245, 384, 576, grid=grid(384), stream=stream0)
        del arg60_1
        del arg61_1
        buf248 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_21, getattr_getattr_l__mod___stages___2_____0___act2, getattr_getattr_l__mod___stages___2_____0___act2b, out_25, out_26, out_27], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_14.run(arg63_1, arg64_1, buf248, 384, 576, grid=grid(384), stream=stream0)
        del arg63_1
        del arg64_1
        buf251 = empty((1536, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_22, getattr_getattr_l__mod___stages___2_____0___act2, getattr_getattr_l__mod___stages___2_____0___act2b, getattr_getattr_l__mod___stages___2_____0___act3, out_25, out_26, out_27, out_28], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_15.run(arg66_1, arg67_1, buf251, 1536, 384, grid=grid(1536), stream=stream0)
        del arg66_1
        del arg67_1
        buf263 = empty((384, 1536, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_23, getattr_getattr_l__mod___stages___2_____1___act1, out_32, out_33], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_red_fused__native_batch_norm_legit_convolution_mul_silu_16.run(arg69_1, arg70_1, buf263, 384, 1536, grid=grid(384), stream=stream0)
        del arg69_1
        del arg70_1
        buf266 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_24, getattr_getattr_l__mod___stages___2_____1___act1, getattr_getattr_l__mod___stages___2_____1___act2, out_32, out_33, out_34], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_14.run(arg72_1, arg73_1, buf266, 384, 576, grid=grid(384), stream=stream0)
        del arg72_1
        del arg73_1
        buf269 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_25, getattr_getattr_l__mod___stages___2_____1___act1, getattr_getattr_l__mod___stages___2_____1___act2, getattr_getattr_l__mod___stages___2_____1___act2b, out_32, out_33, out_34, out_35], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_14.run(arg75_1, arg76_1, buf269, 384, 576, grid=grid(384), stream=stream0)
        del arg75_1
        del arg76_1
        buf272 = empty((1536, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_26, getattr_getattr_l__mod___stages___2_____1___act1, getattr_getattr_l__mod___stages___2_____1___act2, getattr_getattr_l__mod___stages___2_____1___act2b, getattr_getattr_l__mod___stages___2_____1___act3, out_32, out_33, out_34, out_35, out_36], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_15.run(arg78_1, arg79_1, buf272, 1536, 384, grid=grid(1536), stream=stream0)
        del arg78_1
        del arg79_1
        buf281 = empty((384, 1536, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_27, getattr_getattr_l__mod___stages___2_____2___act1, out_40, out_41], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_red_fused__native_batch_norm_legit_convolution_mul_silu_16.run(arg81_1, arg82_1, buf281, 384, 1536, grid=grid(384), stream=stream0)
        del arg81_1
        del arg82_1
        buf284 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_28, getattr_getattr_l__mod___stages___2_____2___act1, getattr_getattr_l__mod___stages___2_____2___act2, out_40, out_41, out_42], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_14.run(arg84_1, arg85_1, buf284, 384, 576, grid=grid(384), stream=stream0)
        del arg84_1
        del arg85_1
        buf287 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_29, getattr_getattr_l__mod___stages___2_____2___act1, getattr_getattr_l__mod___stages___2_____2___act2, getattr_getattr_l__mod___stages___2_____2___act2b, out_40, out_41, out_42, out_43], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_14.run(arg87_1, arg88_1, buf287, 384, 576, grid=grid(384), stream=stream0)
        del arg87_1
        del arg88_1
        buf290 = empty((1536, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_30, getattr_getattr_l__mod___stages___2_____2___act1, getattr_getattr_l__mod___stages___2_____2___act2, getattr_getattr_l__mod___stages___2_____2___act2b, getattr_getattr_l__mod___stages___2_____2___act3, out_40, out_41, out_42, out_43, out_44], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_15.run(arg90_1, arg91_1, buf290, 1536, 384, grid=grid(1536), stream=stream0)
        del arg90_1
        del arg91_1
        buf299 = empty((384, 1536, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_31, getattr_getattr_l__mod___stages___2_____3___act1, out_48, out_49], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_red_fused__native_batch_norm_legit_convolution_mul_silu_16.run(arg93_1, arg94_1, buf299, 384, 1536, grid=grid(384), stream=stream0)
        del arg93_1
        del arg94_1
        buf302 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_32, getattr_getattr_l__mod___stages___2_____3___act1, getattr_getattr_l__mod___stages___2_____3___act2, out_48, out_49, out_50], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_14.run(arg96_1, arg97_1, buf302, 384, 576, grid=grid(384), stream=stream0)
        del arg96_1
        del arg97_1
        buf305 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_33, getattr_getattr_l__mod___stages___2_____3___act1, getattr_getattr_l__mod___stages___2_____3___act2, getattr_getattr_l__mod___stages___2_____3___act2b, out_48, out_49, out_50, out_51], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_14.run(arg99_1, arg100_1, buf305, 384, 576, grid=grid(384), stream=stream0)
        del arg100_1
        del arg99_1
        buf308 = empty((1536, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_34, getattr_getattr_l__mod___stages___2_____3___act1, getattr_getattr_l__mod___stages___2_____3___act2, getattr_getattr_l__mod___stages___2_____3___act2b, getattr_getattr_l__mod___stages___2_____3___act3, out_48, out_49, out_50, out_51, out_52], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_15.run(arg102_1, arg103_1, buf308, 1536, 384, grid=grid(1536), stream=stream0)
        del arg102_1
        del arg103_1
        buf317 = empty((384, 1536, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_35, getattr_getattr_l__mod___stages___2_____4___act1, out_56, out_57], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_red_fused__native_batch_norm_legit_convolution_mul_silu_16.run(arg105_1, arg106_1, buf317, 384, 1536, grid=grid(384), stream=stream0)
        del arg105_1
        del arg106_1
        buf320 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_36, getattr_getattr_l__mod___stages___2_____4___act1, getattr_getattr_l__mod___stages___2_____4___act2, out_56, out_57, out_58], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_14.run(arg108_1, arg109_1, buf320, 384, 576, grid=grid(384), stream=stream0)
        del arg108_1
        del arg109_1
        buf323 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_37, getattr_getattr_l__mod___stages___2_____4___act1, getattr_getattr_l__mod___stages___2_____4___act2, getattr_getattr_l__mod___stages___2_____4___act2b, out_56, out_57, out_58, out_59], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_14.run(arg111_1, arg112_1, buf323, 384, 576, grid=grid(384), stream=stream0)
        del arg111_1
        del arg112_1
        buf326 = empty((1536, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_38, getattr_getattr_l__mod___stages___2_____4___act1, getattr_getattr_l__mod___stages___2_____4___act2, getattr_getattr_l__mod___stages___2_____4___act2b, getattr_getattr_l__mod___stages___2_____4___act3, out_56, out_57, out_58, out_59, out_60], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_15.run(arg114_1, arg115_1, buf326, 1536, 384, grid=grid(1536), stream=stream0)
        del arg114_1
        del arg115_1
        buf335 = empty((384, 1536, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_39, getattr_getattr_l__mod___stages___2_____5___act1, out_64, out_65], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_red_fused__native_batch_norm_legit_convolution_mul_silu_16.run(arg117_1, arg118_1, buf335, 384, 1536, grid=grid(384), stream=stream0)
        del arg117_1
        del arg118_1
        buf338 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_40, getattr_getattr_l__mod___stages___2_____5___act1, getattr_getattr_l__mod___stages___2_____5___act2, out_64, out_65, out_66], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_14.run(arg120_1, arg121_1, buf338, 384, 576, grid=grid(384), stream=stream0)
        del arg120_1
        del arg121_1
        buf341 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_41, getattr_getattr_l__mod___stages___2_____5___act1, getattr_getattr_l__mod___stages___2_____5___act2, getattr_getattr_l__mod___stages___2_____5___act2b, out_64, out_65, out_66, out_67], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_14.run(arg123_1, arg124_1, buf341, 384, 576, grid=grid(384), stream=stream0)
        del arg123_1
        del arg124_1
        buf344 = empty((1536, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_42, getattr_getattr_l__mod___stages___2_____5___act1, getattr_getattr_l__mod___stages___2_____5___act2, getattr_getattr_l__mod___stages___2_____5___act2b, getattr_getattr_l__mod___stages___2_____5___act3, out_64, out_65, out_66, out_67, out_68], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_15.run(arg126_1, arg127_1, buf344, 1536, 384, grid=grid(1536), stream=stream0)
        del arg126_1
        del arg127_1
        buf370 = empty((1536, 1536, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_43, getattr_getattr_l__mod___stages___3_____0___downsample_pool, shortcut_13], Original ATen: [aten._native_batch_norm_legit, aten.avg_pool2d, aten.convolution]
        triton_red_fused__native_batch_norm_legit_avg_pool2d_convolution_17.run(arg129_1, arg130_1, buf370, 1536, 1536, grid=grid(1536), stream=stream0)
        del arg129_1
        del arg130_1
        buf353 = empty((384, 1536, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_44, out_73], Original ATen: [aten._native_batch_norm_legit, aten.convolution]
        triton_red_fused__native_batch_norm_legit_convolution_mul_silu_16.run(arg132_1, arg133_1, buf353, 384, 1536, grid=grid(384), stream=stream0)
        del arg132_1
        del arg133_1
        buf356 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_45, getattr_getattr_l__mod___stages___3_____0___act2, out_73, out_74], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_14.run(arg135_1, arg136_1, buf356, 384, 576, grid=grid(384), stream=stream0)
        del arg135_1
        del arg136_1
        buf359 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_46, getattr_getattr_l__mod___stages___3_____0___act2, getattr_getattr_l__mod___stages___3_____0___act2b, out_73, out_74, out_75], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_14.run(arg138_1, arg139_1, buf359, 384, 576, grid=grid(384), stream=stream0)
        del arg138_1
        del arg139_1
        buf362 = empty((1536, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_47, getattr_getattr_l__mod___stages___3_____0___act2, getattr_getattr_l__mod___stages___3_____0___act2b, getattr_getattr_l__mod___stages___3_____0___act3, out_73, out_74, out_75, out_76], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_15.run(arg141_1, arg142_1, buf362, 1536, 384, grid=grid(1536), stream=stream0)
        del arg141_1
        del arg142_1
        buf374 = empty((384, 1536, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_48, getattr_getattr_l__mod___stages___3_____1___act1, out_80, out_81], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_red_fused__native_batch_norm_legit_convolution_mul_silu_16.run(arg144_1, arg145_1, buf374, 384, 1536, grid=grid(384), stream=stream0)
        del arg144_1
        del arg145_1
        buf377 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_49, getattr_getattr_l__mod___stages___3_____1___act1, getattr_getattr_l__mod___stages___3_____1___act2, out_80, out_81, out_82], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_14.run(arg147_1, arg148_1, buf377, 384, 576, grid=grid(384), stream=stream0)
        del arg147_1
        del arg148_1
        buf380 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_50, getattr_getattr_l__mod___stages___3_____1___act1, getattr_getattr_l__mod___stages___3_____1___act2, getattr_getattr_l__mod___stages___3_____1___act2b, out_80, out_81, out_82, out_83], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_14.run(arg150_1, arg151_1, buf380, 384, 576, grid=grid(384), stream=stream0)
        del arg150_1
        del arg151_1
        buf383 = empty((1536, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_51, getattr_getattr_l__mod___stages___3_____1___act1, getattr_getattr_l__mod___stages___3_____1___act2, getattr_getattr_l__mod___stages___3_____1___act2b, getattr_getattr_l__mod___stages___3_____1___act3, out_80, out_81, out_82, out_83, out_84], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_15.run(arg153_1, arg154_1, buf383, 1536, 384, grid=grid(1536), stream=stream0)
        del arg153_1
        del arg154_1
        buf392 = empty((384, 1536, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_52, getattr_getattr_l__mod___stages___3_____2___act1, out_88, out_89], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_red_fused__native_batch_norm_legit_convolution_mul_silu_16.run(arg156_1, arg157_1, buf392, 384, 1536, grid=grid(384), stream=stream0)
        del arg156_1
        del arg157_1
        buf395 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_53, getattr_getattr_l__mod___stages___3_____2___act1, getattr_getattr_l__mod___stages___3_____2___act2, out_88, out_89, out_90], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_14.run(arg159_1, arg160_1, buf395, 384, 576, grid=grid(384), stream=stream0)
        del arg159_1
        del arg160_1
        buf398 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_54, getattr_getattr_l__mod___stages___3_____2___act1, getattr_getattr_l__mod___stages___3_____2___act2, getattr_getattr_l__mod___stages___3_____2___act2b, out_88, out_89, out_90, out_91], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_14.run(arg162_1, arg163_1, buf398, 384, 576, grid=grid(384), stream=stream0)
        del arg162_1
        del arg163_1
        buf401 = empty((1536, 384, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_55, getattr_getattr_l__mod___stages___3_____2___act1, getattr_getattr_l__mod___stages___3_____2___act2, getattr_getattr_l__mod___stages___3_____2___act2b, getattr_getattr_l__mod___stages___3_____2___act3, out_88, out_89, out_90, out_91, out_92], Original ATen: [aten._native_batch_norm_legit, aten.convolution, aten.mul, aten.silu]
        triton_per_fused__native_batch_norm_legit_convolution_silu_15.run(arg165_1, arg166_1, buf401, 1536, 384, grid=grid(1536), stream=stream0)
        del arg165_1
        del arg166_1
        buf409 = empty((2304, 1536, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_56, getattr_getattr_l__mod___stages___3_____2___act1, getattr_getattr_l__mod___stages___3_____2___act2, getattr_getattr_l__mod___stages___3_____2___act2b, getattr_getattr_l__mod___stages___3_____2___act3, mul_101, mul_103, out_88, out_89, out_90, out_91, out_92, out_93, sigmoid_11, x_1, x_2, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten._native_batch_norm_legit, aten.add, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        triton_red_fused__native_batch_norm_legit_add_convolution_mean_mul_relu_sigmoid_silu_18.run(arg168_1, arg169_1, buf409, 2304, 1536, grid=grid(2304), stream=stream0)
        del arg168_1
        del arg169_1
        # Source Nodes: [conv2d], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(arg221_1, buf171, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (8, 16, 144, 144), (331776, 20736, 144, 1))
        del arg221_1
        del buf171
        buf173 = buf172; del buf172  # reuse
        # Source Nodes: [conv2d, conv2d_1, l__mod___stem_act2], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_19.run(buf173, arg2_1, 2654208, grid=grid(2654208), stream=stream0)
        del arg2_1
        # Source Nodes: [conv2d, conv2d_1, l__mod___stem_act2], Original ATen: [aten.convolution, aten.silu]
        buf175 = extern_kernels.convolution(buf173, buf174, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (8, 32, 144, 144), (663552, 20736, 144, 1))
        del buf173
        del buf174
        buf176 = buf175; del buf175  # reuse
        # Source Nodes: [conv2d, conv2d_1, conv2d_2, l__mod___stem_act2, l__mod___stem_act3], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_20.run(buf176, arg5_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg5_1
        # Source Nodes: [conv2d, conv2d_1, conv2d_2, l__mod___stem_act2, l__mod___stem_act3], Original ATen: [aten.convolution, aten.silu]
        buf178 = extern_kernels.convolution(buf176, buf177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (8, 64, 144, 144), (1327104, 20736, 144, 1))
        del buf176
        buf179 = buf178; del buf178  # reuse
        # Source Nodes: [conv2d, conv2d_1, conv2d_2, l__mod___stem_act2, l__mod___stem_act3, l__mod___stem_act4, shortcut], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_21.run(buf179, arg8_1, 10616832, grid=grid(10616832), stream=stream0)
        del arg8_1
        # Source Nodes: [conv2d, conv2d_1, conv2d_2, l__mod___stem_act2, l__mod___stem_act3, l__mod___stem_act4, shortcut], Original ATen: [aten.convolution, aten.silu]
        buf181 = extern_kernels.convolution(buf179, buf180, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (8, 128, 72, 72), (663552, 5184, 72, 1))
        del buf179
        del buf180
        buf182 = buf181; del buf181  # reuse
        # Source Nodes: [conv2d, conv2d_1, conv2d_2, getattr_getattr_l__mod___stages___0_____0___act1, l__mod___stem_act2, l__mod___stem_act3, l__mod___stem_act4, out, shortcut], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_mul_silu_22.run(buf182, arg11_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg11_1
        # Source Nodes: [out_1], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf182, buf183, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (8, 64, 72, 72), (331776, 5184, 72, 1))
        del buf183
        buf185 = buf184; del buf184  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act2, out_1, out_2], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_23.run(buf185, arg17_1, 2654208, grid=grid(2654208), stream=stream0)
        del arg17_1
        # Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act2, out_1, out_2], Original ATen: [aten.convolution, aten.silu]
        buf187 = extern_kernels.convolution(buf185, buf186, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (8, 64, 72, 72), (331776, 5184, 72, 1))
        del buf185
        del buf186
        buf188 = buf187; del buf187  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act2, getattr_getattr_l__mod___stages___0_____0___act2b, out_1, out_2, out_3], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_23.run(buf188, arg20_1, 2654208, grid=grid(2654208), stream=stream0)
        del arg20_1
        # Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act2, getattr_getattr_l__mod___stages___0_____0___act2b, out_1, out_2, out_3], Original ATen: [aten.convolution, aten.silu]
        buf190 = extern_kernels.convolution(buf188, buf189, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf190, (8, 64, 72, 72), (331776, 5184, 72, 1))
        del buf188
        del buf189
        buf191 = buf190; del buf190  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act2, getattr_getattr_l__mod___stages___0_____0___act2b, getattr_getattr_l__mod___stages___0_____0___act3, out_1, out_2, out_3, out_4], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_23.run(buf191, arg23_1, 2654208, grid=grid(2654208), stream=stream0)
        del arg23_1
        # Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act2, getattr_getattr_l__mod___stages___0_____0___act2b, getattr_getattr_l__mod___stages___0_____0___act3, out_1, out_2, out_3, out_4], Original ATen: [aten.convolution, aten.silu]
        buf193 = extern_kernels.convolution(buf191, buf192, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (8, 256, 72, 72), (1327104, 5184, 72, 1))
        del buf192
        buf194 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf195 = reinterpret_tensor(buf194, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf194  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act2, getattr_getattr_l__mod___stages___0_____0___act2b, getattr_getattr_l__mod___stages___0_____0___act3, out_1, out_2, out_3, out_4, x_se, x_se_1], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_red_fused_convolution_mean_silu_24.run(buf195, buf193, arg26_1, 2048, 5184, grid=grid(2048), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act2, getattr_getattr_l__mod___stages___0_____0___act2b, getattr_getattr_l__mod___stages___0_____0___act3, out_1, out_2, out_3, out_4, x_se, x_se_1], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf196 = extern_kernels.convolution(buf195, arg171_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (8, 64, 1, 1), (64, 1, 1, 1))
        del arg171_1
        del buf195
        buf197 = buf196; del buf196  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act2, getattr_getattr_l__mod___stages___0_____0___act2b, getattr_getattr_l__mod___stages___0_____0___act3, out_1, out_2, out_3, out_4, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.silu]
        triton_poi_fused_convolution_mean_relu_silu_25.run(buf197, arg172_1, 512, grid=grid(512), stream=stream0)
        del arg172_1
        # Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act2, getattr_getattr_l__mod___stages___0_____0___act2b, getattr_getattr_l__mod___stages___0_____0___act3, out_1, out_2, out_3, out_4, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.silu]
        buf198 = extern_kernels.convolution(buf197, arg173_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg173_1
        del buf197
        # Source Nodes: [shortcut_1], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf182, buf199, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (8, 256, 72, 72), (1327104, 5184, 72, 1))
        del buf182
        del buf199
        buf201 = buf193; del buf193  # reuse
        buf202 = buf201; del buf201  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act2, getattr_getattr_l__mod___stages___0_____0___act2b, getattr_getattr_l__mod___stages___0_____0___act3, getattr_getattr_l__mod___stages___1_____0___act1, mul_10, mul_12, out_1, out_2, out_3, out_4, out_5, out_8, shortcut_1, shortcut_2, sigmoid, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.add, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_26.run(buf202, arg26_1, buf198, arg174_1, buf200, arg14_1, 10616832, grid=grid(10616832), stream=stream0)
        del arg14_1
        del arg174_1
        del arg26_1
        del buf198
        del buf200
        # Source Nodes: [out_9], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf202, buf203, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (8, 128, 72, 72), (663552, 5184, 72, 1))
        del buf203
        buf205 = buf204; del buf204  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act2, out_10, out_9], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_27.run(buf205, arg32_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg32_1
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act2, out_10, out_9], Original ATen: [aten.convolution, aten.silu]
        buf207 = extern_kernels.convolution(buf205, buf206, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf207, (8, 128, 36, 36), (165888, 1296, 36, 1))
        del buf206
        buf208 = buf207; del buf207  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act2, getattr_getattr_l__mod___stages___1_____0___act2b, out_10, out_11, out_9], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_28.run(buf208, arg35_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg35_1
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act2, getattr_getattr_l__mod___stages___1_____0___act2b, out_10, out_11, out_9], Original ATen: [aten.convolution, aten.silu]
        buf210 = extern_kernels.convolution(buf208, buf209, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf210, (8, 128, 36, 36), (165888, 1296, 36, 1))
        del buf208
        del buf209
        buf211 = buf210; del buf210  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act2, getattr_getattr_l__mod___stages___1_____0___act2b, getattr_getattr_l__mod___stages___1_____0___act3, out_10, out_11, out_12, out_9], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_28.run(buf211, arg38_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg38_1
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act2, getattr_getattr_l__mod___stages___1_____0___act2b, getattr_getattr_l__mod___stages___1_____0___act3, out_10, out_11, out_12, out_9], Original ATen: [aten.convolution, aten.silu]
        buf213 = extern_kernels.convolution(buf211, buf212, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (8, 512, 36, 36), (663552, 1296, 36, 1))
        del buf211
        del buf212
        buf214 = empty_strided((8, 512, 1, 1), (512, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf215 = reinterpret_tensor(buf214, (8, 512, 1, 1), (512, 1, 1, 1), 0); del buf214  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act2, getattr_getattr_l__mod___stages___1_____0___act2b, getattr_getattr_l__mod___stages___1_____0___act3, out_10, out_11, out_12, out_9, x_se_4, x_se_5], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_red_fused_convolution_mean_silu_29.run(buf215, buf213, arg41_1, 4096, 1296, grid=grid(4096), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act2, getattr_getattr_l__mod___stages___1_____0___act2b, getattr_getattr_l__mod___stages___1_____0___act3, out_10, out_11, out_12, out_9, x_se_4, x_se_5], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf216 = extern_kernels.convolution(buf215, arg175_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg175_1
        del buf215
        buf217 = buf216; del buf216  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act2, getattr_getattr_l__mod___stages___1_____0___act2b, getattr_getattr_l__mod___stages___1_____0___act3, out_10, out_11, out_12, out_9, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.silu]
        triton_poi_fused_convolution_mean_relu_silu_30.run(buf217, arg176_1, 1024, grid=grid(1024), stream=stream0)
        del arg176_1
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act2, getattr_getattr_l__mod___stages___1_____0___act2b, getattr_getattr_l__mod___stages___1_____0___act3, out_10, out_11, out_12, out_9, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.silu]
        buf218 = extern_kernels.convolution(buf217, arg177_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg177_1
        del buf217
        buf219 = reinterpret_tensor(buf191, (8, 256, 36, 36), (331776, 1296, 36, 1), 0); del buf191  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___downsample_pool, shortcut_3], Original ATen: [aten.avg_pool2d, aten.convolution]
        triton_poi_fused_avg_pool2d_convolution_31.run(buf202, buf219, 2654208, grid=grid(2654208), stream=stream0)
        del buf202
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___downsample_pool, shortcut_3], Original ATen: [aten.avg_pool2d, aten.convolution]
        buf221 = extern_kernels.convolution(buf219, buf220, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (8, 512, 36, 36), (663552, 1296, 36, 1))
        del buf219
        del buf220
        buf222 = buf213; del buf213  # reuse
        buf223 = reinterpret_tensor(buf205, (8, 512, 36, 36), (663552, 1296, 36, 1), 0); del buf205  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act2, getattr_getattr_l__mod___stages___1_____0___act2b, getattr_getattr_l__mod___stages___1_____0___act3, getattr_getattr_l__mod___stages___1_____0___downsample_pool, getattr_getattr_l__mod___stages___1_____1___act1, mul_19, mul_21, out_10, out_11, out_12, out_13, out_16, out_17, out_9, shortcut_3, shortcut_4, sigmoid_1, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        triton_poi_fused_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_32.run(buf222, arg41_1, buf218, arg178_1, buf221, arg29_1, buf223, 5308416, grid=grid(5308416), stream=stream0)
        del arg178_1
        del arg29_1
        del arg41_1
        del buf221
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act1, out_16, out_17], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf225 = extern_kernels.convolution(buf223, buf224, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (8, 128, 36, 36), (165888, 1296, 36, 1))
        del buf223
        del buf224
        buf226 = buf225; del buf225  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act1, getattr_getattr_l__mod___stages___1_____1___act2, out_16, out_17, out_18], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_silu_28.run(buf226, arg44_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg44_1
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act1, getattr_getattr_l__mod___stages___1_____1___act2, out_16, out_17, out_18], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf228 = extern_kernels.convolution(buf226, buf227, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf228, (8, 128, 36, 36), (165888, 1296, 36, 1))
        del buf226
        del buf227
        buf229 = buf228; del buf228  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act1, getattr_getattr_l__mod___stages___1_____1___act2, getattr_getattr_l__mod___stages___1_____1___act2b, out_16, out_17, out_18, out_19], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_silu_28.run(buf229, arg47_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg47_1
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act1, getattr_getattr_l__mod___stages___1_____1___act2, getattr_getattr_l__mod___stages___1_____1___act2b, out_16, out_17, out_18, out_19], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf231 = extern_kernels.convolution(buf229, buf230, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf231, (8, 128, 36, 36), (165888, 1296, 36, 1))
        del buf229
        del buf230
        buf232 = buf231; del buf231  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act1, getattr_getattr_l__mod___stages___1_____1___act2, getattr_getattr_l__mod___stages___1_____1___act2b, getattr_getattr_l__mod___stages___1_____1___act3, out_16, out_17, out_18, out_19, out_20], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_silu_28.run(buf232, arg50_1, 1327104, grid=grid(1327104), stream=stream0)
        del arg50_1
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act1, getattr_getattr_l__mod___stages___1_____1___act2, getattr_getattr_l__mod___stages___1_____1___act2b, getattr_getattr_l__mod___stages___1_____1___act3, out_16, out_17, out_18, out_19, out_20], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf234 = extern_kernels.convolution(buf232, buf233, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf234, (8, 512, 36, 36), (663552, 1296, 36, 1))
        del buf233
        buf235 = reinterpret_tensor(buf218, (8, 512, 1, 1), (512, 1, 4096, 4096), 0); del buf218  # reuse
        buf236 = reinterpret_tensor(buf235, (8, 512, 1, 1), (512, 1, 1, 1), 0); del buf235  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act1, getattr_getattr_l__mod___stages___1_____1___act2, getattr_getattr_l__mod___stages___1_____1___act2b, getattr_getattr_l__mod___stages___1_____1___act3, out_16, out_17, out_18, out_19, out_20, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.silu]
        triton_red_fused_convolution_mean_silu_29.run(buf236, buf234, arg53_1, 4096, 1296, grid=grid(4096), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act1, getattr_getattr_l__mod___stages___1_____1___act2, getattr_getattr_l__mod___stages___1_____1___act2b, getattr_getattr_l__mod___stages___1_____1___act3, out_16, out_17, out_18, out_19, out_20, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.silu]
        buf237 = extern_kernels.convolution(buf236, arg179_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg179_1
        del buf236
        buf238 = buf237; del buf237  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act1, getattr_getattr_l__mod___stages___1_____1___act2, getattr_getattr_l__mod___stages___1_____1___act2b, getattr_getattr_l__mod___stages___1_____1___act3, out_16, out_17, out_18, out_19, out_20, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.silu]
        triton_poi_fused_convolution_mean_relu_silu_30.run(buf238, arg180_1, 1024, grid=grid(1024), stream=stream0)
        del arg180_1
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act1, getattr_getattr_l__mod___stages___1_____1___act2, getattr_getattr_l__mod___stages___1_____1___act2b, getattr_getattr_l__mod___stages___1_____1___act3, out_16, out_17, out_18, out_19, out_20, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.silu]
        buf239 = extern_kernels.convolution(buf238, arg181_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (8, 512, 1, 1), (512, 1, 1, 1))
        del arg181_1
        del buf238
        buf240 = buf222; del buf222  # reuse
        buf241 = buf240; del buf240  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act1, getattr_getattr_l__mod___stages___1_____1___act2, getattr_getattr_l__mod___stages___1_____1___act2b, getattr_getattr_l__mod___stages___1_____1___act3, getattr_getattr_l__mod___stages___2_____0___act1, mul_27, mul_29, out_16, out_17, out_18, out_19, out_20, out_21, out_24, shortcut_5, sigmoid_2, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.add, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_33.run(buf241, buf234, arg53_1, buf239, arg182_1, 5308416, grid=grid(5308416), stream=stream0)
        del arg182_1
        del arg53_1
        del buf234
        del buf239
        # Source Nodes: [out_25], Original ATen: [aten.convolution]
        buf243 = extern_kernels.convolution(buf241, buf242, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (8, 384, 36, 36), (497664, 1296, 36, 1))
        del buf242
        buf244 = buf243; del buf243  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act2, out_25, out_26], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_34.run(buf244, arg59_1, 3981312, grid=grid(3981312), stream=stream0)
        del arg59_1
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act2, out_25, out_26], Original ATen: [aten.convolution, aten.silu]
        buf246 = extern_kernels.convolution(buf244, buf245, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf246, (8, 384, 18, 18), (124416, 324, 18, 1))
        del buf245
        buf247 = buf246; del buf246  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act2, getattr_getattr_l__mod___stages___2_____0___act2b, out_25, out_26, out_27], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_35.run(buf247, arg62_1, 995328, grid=grid(995328), stream=stream0)
        del arg62_1
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act2, getattr_getattr_l__mod___stages___2_____0___act2b, out_25, out_26, out_27], Original ATen: [aten.convolution, aten.silu]
        buf249 = extern_kernels.convolution(buf247, buf248, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf249, (8, 384, 18, 18), (124416, 324, 18, 1))
        del buf247
        del buf248
        buf250 = buf249; del buf249  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act2, getattr_getattr_l__mod___stages___2_____0___act2b, getattr_getattr_l__mod___stages___2_____0___act3, out_25, out_26, out_27, out_28], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_35.run(buf250, arg65_1, 995328, grid=grid(995328), stream=stream0)
        del arg65_1
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act2, getattr_getattr_l__mod___stages___2_____0___act2b, getattr_getattr_l__mod___stages___2_____0___act3, out_25, out_26, out_27, out_28], Original ATen: [aten.convolution, aten.silu]
        buf252 = extern_kernels.convolution(buf250, buf251, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf252, (8, 1536, 18, 18), (497664, 324, 18, 1))
        del buf250
        del buf251
        buf253 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cuda', dtype=torch.float32)
        buf254 = reinterpret_tensor(buf253, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf253  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act2, getattr_getattr_l__mod___stages___2_____0___act2b, getattr_getattr_l__mod___stages___2_____0___act3, out_25, out_26, out_27, out_28, x_se_12, x_se_13], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_per_fused_convolution_mean_silu_36.run(buf254, buf252, arg68_1, 12288, 324, grid=grid(12288), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act2, getattr_getattr_l__mod___stages___2_____0___act2b, getattr_getattr_l__mod___stages___2_____0___act3, out_25, out_26, out_27, out_28, x_se_12, x_se_13], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf255 = extern_kernels.convolution(buf254, arg183_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf255, (8, 384, 1, 1), (384, 1, 1, 1))
        del arg183_1
        del buf254
        buf256 = buf255; del buf255  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act2, getattr_getattr_l__mod___stages___2_____0___act2b, getattr_getattr_l__mod___stages___2_____0___act3, out_25, out_26, out_27, out_28, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.silu]
        triton_poi_fused_convolution_mean_relu_silu_37.run(buf256, arg184_1, 3072, grid=grid(3072), stream=stream0)
        del arg184_1
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act2, getattr_getattr_l__mod___stages___2_____0___act2b, getattr_getattr_l__mod___stages___2_____0___act3, out_25, out_26, out_27, out_28, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.silu]
        buf257 = extern_kernels.convolution(buf256, arg185_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf257, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg185_1
        del buf256
        buf258 = reinterpret_tensor(buf232, (8, 512, 18, 18), (165888, 324, 18, 1), 0); del buf232  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___downsample_pool, shortcut_6], Original ATen: [aten.avg_pool2d, aten.convolution]
        triton_poi_fused_avg_pool2d_convolution_38.run(buf241, buf258, 1327104, grid=grid(1327104), stream=stream0)
        del buf241
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___downsample_pool, shortcut_6], Original ATen: [aten.avg_pool2d, aten.convolution]
        buf260 = extern_kernels.convolution(buf258, buf259, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf260, (8, 1536, 18, 18), (497664, 324, 18, 1))
        del buf258
        del buf259
        buf261 = buf252; del buf252  # reuse
        buf262 = reinterpret_tensor(buf244, (8, 1536, 18, 18), (497664, 324, 18, 1), 0); del buf244  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act2, getattr_getattr_l__mod___stages___2_____0___act2b, getattr_getattr_l__mod___stages___2_____0___act3, getattr_getattr_l__mod___stages___2_____0___downsample_pool, getattr_getattr_l__mod___stages___2_____1___act1, mul_36, mul_38, out_25, out_26, out_27, out_28, out_29, out_32, out_33, shortcut_6, shortcut_7, sigmoid_3, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        triton_poi_fused_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_39.run(buf261, arg68_1, buf257, arg186_1, buf260, arg56_1, buf262, 3981312, grid=grid(3981312), stream=stream0)
        del arg186_1
        del arg56_1
        del arg68_1
        del buf260
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____1___act1, out_32, out_33], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf264 = extern_kernels.convolution(buf262, buf263, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (8, 384, 18, 18), (124416, 324, 18, 1))
        del buf263
        buf265 = buf264; del buf264  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____1___act1, getattr_getattr_l__mod___stages___2_____1___act2, out_32, out_33, out_34], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_silu_35.run(buf265, arg71_1, 995328, grid=grid(995328), stream=stream0)
        del arg71_1
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____1___act1, getattr_getattr_l__mod___stages___2_____1___act2, out_32, out_33, out_34], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf267 = extern_kernels.convolution(buf265, buf266, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf267, (8, 384, 18, 18), (124416, 324, 18, 1))
        del buf265
        del buf266
        buf268 = buf267; del buf267  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____1___act1, getattr_getattr_l__mod___stages___2_____1___act2, getattr_getattr_l__mod___stages___2_____1___act2b, out_32, out_33, out_34, out_35], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_silu_35.run(buf268, arg74_1, 995328, grid=grid(995328), stream=stream0)
        del arg74_1
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____1___act1, getattr_getattr_l__mod___stages___2_____1___act2, getattr_getattr_l__mod___stages___2_____1___act2b, out_32, out_33, out_34, out_35], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf270 = extern_kernels.convolution(buf268, buf269, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf270, (8, 384, 18, 18), (124416, 324, 18, 1))
        del buf268
        del buf269
        buf271 = buf270; del buf270  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____1___act1, getattr_getattr_l__mod___stages___2_____1___act2, getattr_getattr_l__mod___stages___2_____1___act2b, getattr_getattr_l__mod___stages___2_____1___act3, out_32, out_33, out_34, out_35, out_36], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_silu_35.run(buf271, arg77_1, 995328, grid=grid(995328), stream=stream0)
        del arg77_1
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____1___act1, getattr_getattr_l__mod___stages___2_____1___act2, getattr_getattr_l__mod___stages___2_____1___act2b, getattr_getattr_l__mod___stages___2_____1___act3, out_32, out_33, out_34, out_35, out_36], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf273 = extern_kernels.convolution(buf271, buf272, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf273, (8, 1536, 18, 18), (497664, 324, 18, 1))
        del buf271
        del buf272
        buf274 = reinterpret_tensor(buf257, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf257  # reuse
        buf275 = reinterpret_tensor(buf274, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf274  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____1___act1, getattr_getattr_l__mod___stages___2_____1___act2, getattr_getattr_l__mod___stages___2_____1___act2b, getattr_getattr_l__mod___stages___2_____1___act3, out_32, out_33, out_34, out_35, out_36, x_se_16, x_se_17], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.silu]
        triton_per_fused_convolution_mean_silu_36.run(buf275, buf273, arg80_1, 12288, 324, grid=grid(12288), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____1___act1, getattr_getattr_l__mod___stages___2_____1___act2, getattr_getattr_l__mod___stages___2_____1___act2b, getattr_getattr_l__mod___stages___2_____1___act3, out_32, out_33, out_34, out_35, out_36, x_se_16, x_se_17], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.silu]
        buf276 = extern_kernels.convolution(buf275, arg187_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf276, (8, 384, 1, 1), (384, 1, 1, 1))
        del arg187_1
        del buf275
        buf277 = buf276; del buf276  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____1___act1, getattr_getattr_l__mod___stages___2_____1___act2, getattr_getattr_l__mod___stages___2_____1___act2b, getattr_getattr_l__mod___stages___2_____1___act3, out_32, out_33, out_34, out_35, out_36, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.silu]
        triton_poi_fused_convolution_mean_relu_silu_37.run(buf277, arg188_1, 3072, grid=grid(3072), stream=stream0)
        del arg188_1
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____1___act1, getattr_getattr_l__mod___stages___2_____1___act2, getattr_getattr_l__mod___stages___2_____1___act2b, getattr_getattr_l__mod___stages___2_____1___act3, out_32, out_33, out_34, out_35, out_36, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.silu]
        buf278 = extern_kernels.convolution(buf277, arg189_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg189_1
        del buf277
        buf279 = buf261; del buf261  # reuse
        buf280 = buf262; del buf262  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____1___act1, getattr_getattr_l__mod___stages___2_____1___act2, getattr_getattr_l__mod___stages___2_____1___act2b, getattr_getattr_l__mod___stages___2_____1___act3, getattr_getattr_l__mod___stages___2_____2___act1, mul_44, mul_46, out_32, out_33, out_34, out_35, out_36, out_37, out_40, out_41, shortcut_8, sigmoid_4, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.add, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_40.run(buf279, buf273, arg80_1, buf278, arg190_1, buf280, 3981312, grid=grid(3981312), stream=stream0)
        del arg190_1
        del arg80_1
        del buf273
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____2___act1, out_40, out_41], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf282 = extern_kernels.convolution(buf280, buf281, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf282, (8, 384, 18, 18), (124416, 324, 18, 1))
        del buf281
        buf283 = buf282; del buf282  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____2___act1, getattr_getattr_l__mod___stages___2_____2___act2, out_40, out_41, out_42], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_silu_35.run(buf283, arg83_1, 995328, grid=grid(995328), stream=stream0)
        del arg83_1
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____2___act1, getattr_getattr_l__mod___stages___2_____2___act2, out_40, out_41, out_42], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf285 = extern_kernels.convolution(buf283, buf284, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf285, (8, 384, 18, 18), (124416, 324, 18, 1))
        del buf283
        del buf284
        buf286 = buf285; del buf285  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____2___act1, getattr_getattr_l__mod___stages___2_____2___act2, getattr_getattr_l__mod___stages___2_____2___act2b, out_40, out_41, out_42, out_43], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_silu_35.run(buf286, arg86_1, 995328, grid=grid(995328), stream=stream0)
        del arg86_1
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____2___act1, getattr_getattr_l__mod___stages___2_____2___act2, getattr_getattr_l__mod___stages___2_____2___act2b, out_40, out_41, out_42, out_43], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf288 = extern_kernels.convolution(buf286, buf287, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf288, (8, 384, 18, 18), (124416, 324, 18, 1))
        del buf286
        del buf287
        buf289 = buf288; del buf288  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____2___act1, getattr_getattr_l__mod___stages___2_____2___act2, getattr_getattr_l__mod___stages___2_____2___act2b, getattr_getattr_l__mod___stages___2_____2___act3, out_40, out_41, out_42, out_43, out_44], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_silu_35.run(buf289, arg89_1, 995328, grid=grid(995328), stream=stream0)
        del arg89_1
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____2___act1, getattr_getattr_l__mod___stages___2_____2___act2, getattr_getattr_l__mod___stages___2_____2___act2b, getattr_getattr_l__mod___stages___2_____2___act3, out_40, out_41, out_42, out_43, out_44], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf291 = extern_kernels.convolution(buf289, buf290, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (8, 1536, 18, 18), (497664, 324, 18, 1))
        del buf289
        del buf290
        buf292 = reinterpret_tensor(buf278, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf278  # reuse
        buf293 = reinterpret_tensor(buf292, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf292  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____2___act1, getattr_getattr_l__mod___stages___2_____2___act2, getattr_getattr_l__mod___stages___2_____2___act2b, getattr_getattr_l__mod___stages___2_____2___act3, out_40, out_41, out_42, out_43, out_44, x_se_20, x_se_21], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.silu]
        triton_per_fused_convolution_mean_silu_36.run(buf293, buf291, arg92_1, 12288, 324, grid=grid(12288), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____2___act1, getattr_getattr_l__mod___stages___2_____2___act2, getattr_getattr_l__mod___stages___2_____2___act2b, getattr_getattr_l__mod___stages___2_____2___act3, out_40, out_41, out_42, out_43, out_44, x_se_20, x_se_21], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.silu]
        buf294 = extern_kernels.convolution(buf293, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (8, 384, 1, 1), (384, 1, 1, 1))
        del arg191_1
        del buf293
        buf295 = buf294; del buf294  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____2___act1, getattr_getattr_l__mod___stages___2_____2___act2, getattr_getattr_l__mod___stages___2_____2___act2b, getattr_getattr_l__mod___stages___2_____2___act3, out_40, out_41, out_42, out_43, out_44, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.silu]
        triton_poi_fused_convolution_mean_relu_silu_37.run(buf295, arg192_1, 3072, grid=grid(3072), stream=stream0)
        del arg192_1
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____2___act1, getattr_getattr_l__mod___stages___2_____2___act2, getattr_getattr_l__mod___stages___2_____2___act2b, getattr_getattr_l__mod___stages___2_____2___act3, out_40, out_41, out_42, out_43, out_44, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.silu]
        buf296 = extern_kernels.convolution(buf295, arg193_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf296, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg193_1
        del buf295
        buf297 = buf279; del buf279  # reuse
        buf298 = buf280; del buf280  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____2___act1, getattr_getattr_l__mod___stages___2_____2___act2, getattr_getattr_l__mod___stages___2_____2___act2b, getattr_getattr_l__mod___stages___2_____2___act3, getattr_getattr_l__mod___stages___2_____3___act1, mul_52, mul_54, out_40, out_41, out_42, out_43, out_44, out_45, out_48, out_49, shortcut_9, sigmoid_5, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.add, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_41.run(buf297, buf291, arg92_1, buf296, arg194_1, buf298, 3981312, grid=grid(3981312), stream=stream0)
        del arg194_1
        del arg92_1
        del buf291
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____3___act1, out_48, out_49], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf300 = extern_kernels.convolution(buf298, buf299, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf300, (8, 384, 18, 18), (124416, 324, 18, 1))
        del buf299
        buf301 = buf300; del buf300  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____3___act1, getattr_getattr_l__mod___stages___2_____3___act2, out_48, out_49, out_50], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_silu_35.run(buf301, arg95_1, 995328, grid=grid(995328), stream=stream0)
        del arg95_1
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____3___act1, getattr_getattr_l__mod___stages___2_____3___act2, out_48, out_49, out_50], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf303 = extern_kernels.convolution(buf301, buf302, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf303, (8, 384, 18, 18), (124416, 324, 18, 1))
        del buf301
        del buf302
        buf304 = buf303; del buf303  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____3___act1, getattr_getattr_l__mod___stages___2_____3___act2, getattr_getattr_l__mod___stages___2_____3___act2b, out_48, out_49, out_50, out_51], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_silu_35.run(buf304, arg98_1, 995328, grid=grid(995328), stream=stream0)
        del arg98_1
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____3___act1, getattr_getattr_l__mod___stages___2_____3___act2, getattr_getattr_l__mod___stages___2_____3___act2b, out_48, out_49, out_50, out_51], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf306 = extern_kernels.convolution(buf304, buf305, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf306, (8, 384, 18, 18), (124416, 324, 18, 1))
        del buf304
        del buf305
        buf307 = buf306; del buf306  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____3___act1, getattr_getattr_l__mod___stages___2_____3___act2, getattr_getattr_l__mod___stages___2_____3___act2b, getattr_getattr_l__mod___stages___2_____3___act3, out_48, out_49, out_50, out_51, out_52], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_silu_35.run(buf307, arg101_1, 995328, grid=grid(995328), stream=stream0)
        del arg101_1
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____3___act1, getattr_getattr_l__mod___stages___2_____3___act2, getattr_getattr_l__mod___stages___2_____3___act2b, getattr_getattr_l__mod___stages___2_____3___act3, out_48, out_49, out_50, out_51, out_52], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf309 = extern_kernels.convolution(buf307, buf308, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf309, (8, 1536, 18, 18), (497664, 324, 18, 1))
        del buf307
        del buf308
        buf310 = reinterpret_tensor(buf296, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf296  # reuse
        buf311 = reinterpret_tensor(buf310, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf310  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____3___act1, getattr_getattr_l__mod___stages___2_____3___act2, getattr_getattr_l__mod___stages___2_____3___act2b, getattr_getattr_l__mod___stages___2_____3___act3, out_48, out_49, out_50, out_51, out_52, x_se_24, x_se_25], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.silu]
        triton_per_fused_convolution_mean_silu_36.run(buf311, buf309, arg104_1, 12288, 324, grid=grid(12288), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____3___act1, getattr_getattr_l__mod___stages___2_____3___act2, getattr_getattr_l__mod___stages___2_____3___act2b, getattr_getattr_l__mod___stages___2_____3___act3, out_48, out_49, out_50, out_51, out_52, x_se_24, x_se_25], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.silu]
        buf312 = extern_kernels.convolution(buf311, arg195_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf312, (8, 384, 1, 1), (384, 1, 1, 1))
        del arg195_1
        del buf311
        buf313 = buf312; del buf312  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____3___act1, getattr_getattr_l__mod___stages___2_____3___act2, getattr_getattr_l__mod___stages___2_____3___act2b, getattr_getattr_l__mod___stages___2_____3___act3, out_48, out_49, out_50, out_51, out_52, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.silu]
        triton_poi_fused_convolution_mean_relu_silu_37.run(buf313, arg196_1, 3072, grid=grid(3072), stream=stream0)
        del arg196_1
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____3___act1, getattr_getattr_l__mod___stages___2_____3___act2, getattr_getattr_l__mod___stages___2_____3___act2b, getattr_getattr_l__mod___stages___2_____3___act3, out_48, out_49, out_50, out_51, out_52, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.silu]
        buf314 = extern_kernels.convolution(buf313, arg197_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf314, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg197_1
        del buf313
        buf315 = buf297; del buf297  # reuse
        buf316 = buf298; del buf298  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____3___act1, getattr_getattr_l__mod___stages___2_____3___act2, getattr_getattr_l__mod___stages___2_____3___act2b, getattr_getattr_l__mod___stages___2_____3___act3, getattr_getattr_l__mod___stages___2_____4___act1, mul_60, mul_62, out_48, out_49, out_50, out_51, out_52, out_53, out_56, out_57, shortcut_10, sigmoid_6, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.add, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_42.run(buf315, buf309, arg104_1, buf314, arg198_1, buf316, 3981312, grid=grid(3981312), stream=stream0)
        del arg104_1
        del arg198_1
        del buf309
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____4___act1, out_56, out_57], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf318 = extern_kernels.convolution(buf316, buf317, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf318, (8, 384, 18, 18), (124416, 324, 18, 1))
        del buf317
        buf319 = buf318; del buf318  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____4___act1, getattr_getattr_l__mod___stages___2_____4___act2, out_56, out_57, out_58], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_silu_35.run(buf319, arg107_1, 995328, grid=grid(995328), stream=stream0)
        del arg107_1
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____4___act1, getattr_getattr_l__mod___stages___2_____4___act2, out_56, out_57, out_58], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf321 = extern_kernels.convolution(buf319, buf320, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf321, (8, 384, 18, 18), (124416, 324, 18, 1))
        del buf319
        del buf320
        buf322 = buf321; del buf321  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____4___act1, getattr_getattr_l__mod___stages___2_____4___act2, getattr_getattr_l__mod___stages___2_____4___act2b, out_56, out_57, out_58, out_59], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_silu_35.run(buf322, arg110_1, 995328, grid=grid(995328), stream=stream0)
        del arg110_1
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____4___act1, getattr_getattr_l__mod___stages___2_____4___act2, getattr_getattr_l__mod___stages___2_____4___act2b, out_56, out_57, out_58, out_59], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf324 = extern_kernels.convolution(buf322, buf323, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf324, (8, 384, 18, 18), (124416, 324, 18, 1))
        del buf322
        del buf323
        buf325 = buf324; del buf324  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____4___act1, getattr_getattr_l__mod___stages___2_____4___act2, getattr_getattr_l__mod___stages___2_____4___act2b, getattr_getattr_l__mod___stages___2_____4___act3, out_56, out_57, out_58, out_59, out_60], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_silu_35.run(buf325, arg113_1, 995328, grid=grid(995328), stream=stream0)
        del arg113_1
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____4___act1, getattr_getattr_l__mod___stages___2_____4___act2, getattr_getattr_l__mod___stages___2_____4___act2b, getattr_getattr_l__mod___stages___2_____4___act3, out_56, out_57, out_58, out_59, out_60], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf327 = extern_kernels.convolution(buf325, buf326, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf327, (8, 1536, 18, 18), (497664, 324, 18, 1))
        del buf325
        del buf326
        buf328 = reinterpret_tensor(buf314, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf314  # reuse
        buf329 = reinterpret_tensor(buf328, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf328  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____4___act1, getattr_getattr_l__mod___stages___2_____4___act2, getattr_getattr_l__mod___stages___2_____4___act2b, getattr_getattr_l__mod___stages___2_____4___act3, out_56, out_57, out_58, out_59, out_60, x_se_28, x_se_29], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.silu]
        triton_per_fused_convolution_mean_silu_36.run(buf329, buf327, arg116_1, 12288, 324, grid=grid(12288), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____4___act1, getattr_getattr_l__mod___stages___2_____4___act2, getattr_getattr_l__mod___stages___2_____4___act2b, getattr_getattr_l__mod___stages___2_____4___act3, out_56, out_57, out_58, out_59, out_60, x_se_28, x_se_29], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.silu]
        buf330 = extern_kernels.convolution(buf329, arg199_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf330, (8, 384, 1, 1), (384, 1, 1, 1))
        del arg199_1
        del buf329
        buf331 = buf330; del buf330  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____4___act1, getattr_getattr_l__mod___stages___2_____4___act2, getattr_getattr_l__mod___stages___2_____4___act2b, getattr_getattr_l__mod___stages___2_____4___act3, out_56, out_57, out_58, out_59, out_60, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.silu]
        triton_poi_fused_convolution_mean_relu_silu_37.run(buf331, arg200_1, 3072, grid=grid(3072), stream=stream0)
        del arg200_1
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____4___act1, getattr_getattr_l__mod___stages___2_____4___act2, getattr_getattr_l__mod___stages___2_____4___act2b, getattr_getattr_l__mod___stages___2_____4___act3, out_56, out_57, out_58, out_59, out_60, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.silu]
        buf332 = extern_kernels.convolution(buf331, arg201_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf332, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg201_1
        del buf331
        buf333 = buf315; del buf315  # reuse
        buf334 = buf316; del buf316  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____4___act1, getattr_getattr_l__mod___stages___2_____4___act2, getattr_getattr_l__mod___stages___2_____4___act2b, getattr_getattr_l__mod___stages___2_____4___act3, getattr_getattr_l__mod___stages___2_____5___act1, mul_68, mul_70, out_56, out_57, out_58, out_59, out_60, out_61, out_64, out_65, shortcut_11, sigmoid_7, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.add, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_43.run(buf333, buf327, arg116_1, buf332, arg202_1, buf334, 3981312, grid=grid(3981312), stream=stream0)
        del arg116_1
        del arg202_1
        del buf327
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____5___act1, out_64, out_65], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf336 = extern_kernels.convolution(buf334, buf335, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (8, 384, 18, 18), (124416, 324, 18, 1))
        del buf334
        del buf335
        buf337 = buf336; del buf336  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____5___act1, getattr_getattr_l__mod___stages___2_____5___act2, out_64, out_65, out_66], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_silu_35.run(buf337, arg119_1, 995328, grid=grid(995328), stream=stream0)
        del arg119_1
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____5___act1, getattr_getattr_l__mod___stages___2_____5___act2, out_64, out_65, out_66], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf339 = extern_kernels.convolution(buf337, buf338, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf339, (8, 384, 18, 18), (124416, 324, 18, 1))
        del buf337
        del buf338
        buf340 = buf339; del buf339  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____5___act1, getattr_getattr_l__mod___stages___2_____5___act2, getattr_getattr_l__mod___stages___2_____5___act2b, out_64, out_65, out_66, out_67], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_silu_35.run(buf340, arg122_1, 995328, grid=grid(995328), stream=stream0)
        del arg122_1
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____5___act1, getattr_getattr_l__mod___stages___2_____5___act2, getattr_getattr_l__mod___stages___2_____5___act2b, out_64, out_65, out_66, out_67], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf342 = extern_kernels.convolution(buf340, buf341, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf342, (8, 384, 18, 18), (124416, 324, 18, 1))
        del buf340
        del buf341
        buf343 = buf342; del buf342  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____5___act1, getattr_getattr_l__mod___stages___2_____5___act2, getattr_getattr_l__mod___stages___2_____5___act2b, getattr_getattr_l__mod___stages___2_____5___act3, out_64, out_65, out_66, out_67, out_68], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_silu_35.run(buf343, arg125_1, 995328, grid=grid(995328), stream=stream0)
        del arg125_1
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____5___act1, getattr_getattr_l__mod___stages___2_____5___act2, getattr_getattr_l__mod___stages___2_____5___act2b, getattr_getattr_l__mod___stages___2_____5___act3, out_64, out_65, out_66, out_67, out_68], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf345 = extern_kernels.convolution(buf343, buf344, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf345, (8, 1536, 18, 18), (497664, 324, 18, 1))
        del buf343
        del buf344
        buf346 = reinterpret_tensor(buf332, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf332  # reuse
        buf347 = reinterpret_tensor(buf346, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf346  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____5___act1, getattr_getattr_l__mod___stages___2_____5___act2, getattr_getattr_l__mod___stages___2_____5___act2b, getattr_getattr_l__mod___stages___2_____5___act3, out_64, out_65, out_66, out_67, out_68, x_se_32, x_se_33], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.silu]
        triton_per_fused_convolution_mean_silu_36.run(buf347, buf345, arg128_1, 12288, 324, grid=grid(12288), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____5___act1, getattr_getattr_l__mod___stages___2_____5___act2, getattr_getattr_l__mod___stages___2_____5___act2b, getattr_getattr_l__mod___stages___2_____5___act3, out_64, out_65, out_66, out_67, out_68, x_se_32, x_se_33], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.silu]
        buf348 = extern_kernels.convolution(buf347, arg203_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf348, (8, 384, 1, 1), (384, 1, 1, 1))
        del arg203_1
        del buf347
        buf349 = buf348; del buf348  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____5___act1, getattr_getattr_l__mod___stages___2_____5___act2, getattr_getattr_l__mod___stages___2_____5___act2b, getattr_getattr_l__mod___stages___2_____5___act3, out_64, out_65, out_66, out_67, out_68, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.silu]
        triton_poi_fused_convolution_mean_relu_silu_37.run(buf349, arg204_1, 3072, grid=grid(3072), stream=stream0)
        del arg204_1
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____5___act1, getattr_getattr_l__mod___stages___2_____5___act2, getattr_getattr_l__mod___stages___2_____5___act2b, getattr_getattr_l__mod___stages___2_____5___act3, out_64, out_65, out_66, out_67, out_68, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.silu]
        buf350 = extern_kernels.convolution(buf349, arg205_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf350, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg205_1
        del buf349
        buf351 = buf333; del buf333  # reuse
        buf352 = buf351; del buf351  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____5___act1, getattr_getattr_l__mod___stages___2_____5___act2, getattr_getattr_l__mod___stages___2_____5___act2b, getattr_getattr_l__mod___stages___2_____5___act3, getattr_getattr_l__mod___stages___3_____0___act1, mul_76, mul_78, out_64, out_65, out_66, out_67, out_68, out_69, out_72, shortcut_12, sigmoid_8, x_se_32, x_se_33, x_se_34, x_se_35], Original ATen: [aten.add, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_44.run(buf352, buf345, arg128_1, buf350, arg206_1, 3981312, grid=grid(3981312), stream=stream0)
        del arg128_1
        del arg206_1
        del buf345
        # Source Nodes: [out_73], Original ATen: [aten.convolution]
        buf354 = extern_kernels.convolution(buf352, buf353, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf354, (8, 384, 18, 18), (124416, 324, 18, 1))
        del buf353
        buf355 = buf354; del buf354  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____0___act2, out_73, out_74], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_35.run(buf355, arg134_1, 995328, grid=grid(995328), stream=stream0)
        del arg134_1
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____0___act2, out_73, out_74], Original ATen: [aten.convolution, aten.silu]
        buf357 = extern_kernels.convolution(buf355, buf356, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf357, (8, 384, 9, 9), (31104, 81, 9, 1))
        del buf356
        buf358 = buf357; del buf357  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____0___act2, getattr_getattr_l__mod___stages___3_____0___act2b, out_73, out_74, out_75], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_45.run(buf358, arg137_1, 248832, grid=grid(248832), stream=stream0)
        del arg137_1
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____0___act2, getattr_getattr_l__mod___stages___3_____0___act2b, out_73, out_74, out_75], Original ATen: [aten.convolution, aten.silu]
        buf360 = extern_kernels.convolution(buf358, buf359, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf360, (8, 384, 9, 9), (31104, 81, 9, 1))
        del buf358
        del buf359
        buf361 = buf360; del buf360  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____0___act2, getattr_getattr_l__mod___stages___3_____0___act2b, getattr_getattr_l__mod___stages___3_____0___act3, out_73, out_74, out_75, out_76], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_45.run(buf361, arg140_1, 248832, grid=grid(248832), stream=stream0)
        del arg140_1
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____0___act2, getattr_getattr_l__mod___stages___3_____0___act2b, getattr_getattr_l__mod___stages___3_____0___act3, out_73, out_74, out_75, out_76], Original ATen: [aten.convolution, aten.silu]
        buf363 = extern_kernels.convolution(buf361, buf362, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf363, (8, 1536, 9, 9), (124416, 81, 9, 1))
        del buf361
        del buf362
        buf364 = reinterpret_tensor(buf350, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf350  # reuse
        buf365 = reinterpret_tensor(buf364, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf364  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____0___act2, getattr_getattr_l__mod___stages___3_____0___act2b, getattr_getattr_l__mod___stages___3_____0___act3, out_73, out_74, out_75, out_76, x_se_36, x_se_37], Original ATen: [aten.convolution, aten.mean, aten.silu]
        triton_per_fused_convolution_mean_silu_46.run(buf365, buf363, arg143_1, 12288, 81, grid=grid(12288), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____0___act2, getattr_getattr_l__mod___stages___3_____0___act2b, getattr_getattr_l__mod___stages___3_____0___act3, out_73, out_74, out_75, out_76, x_se_36, x_se_37], Original ATen: [aten.convolution, aten.mean, aten.silu]
        buf366 = extern_kernels.convolution(buf365, arg207_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf366, (8, 384, 1, 1), (384, 1, 1, 1))
        del arg207_1
        del buf365
        buf367 = buf366; del buf366  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____0___act2, getattr_getattr_l__mod___stages___3_____0___act2b, getattr_getattr_l__mod___stages___3_____0___act3, out_73, out_74, out_75, out_76, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.silu]
        triton_poi_fused_convolution_mean_relu_silu_37.run(buf367, arg208_1, 3072, grid=grid(3072), stream=stream0)
        del arg208_1
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____0___act2, getattr_getattr_l__mod___stages___3_____0___act2b, getattr_getattr_l__mod___stages___3_____0___act3, out_73, out_74, out_75, out_76, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.silu]
        buf368 = extern_kernels.convolution(buf367, arg209_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf368, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg209_1
        del buf367
        buf369 = reinterpret_tensor(buf355, (8, 1536, 9, 9), (124416, 81, 9, 1), 0); del buf355  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____0___downsample_pool, shortcut_13], Original ATen: [aten.avg_pool2d, aten.convolution]
        triton_poi_fused_avg_pool2d_convolution_47.run(buf352, buf369, 995328, grid=grid(995328), stream=stream0)
        del buf352
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____0___downsample_pool, shortcut_13], Original ATen: [aten.avg_pool2d, aten.convolution]
        buf371 = extern_kernels.convolution(buf369, buf370, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf371, (8, 1536, 9, 9), (124416, 81, 9, 1))
        del buf370
        buf372 = buf363; del buf363  # reuse
        buf373 = buf369; del buf369  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____0___act2, getattr_getattr_l__mod___stages___3_____0___act2b, getattr_getattr_l__mod___stages___3_____0___act3, getattr_getattr_l__mod___stages___3_____0___downsample_pool, getattr_getattr_l__mod___stages___3_____1___act1, mul_85, mul_87, out_73, out_74, out_75, out_76, out_77, out_80, out_81, shortcut_13, shortcut_14, sigmoid_9, x_se_36, x_se_37, x_se_38, x_se_39], Original ATen: [aten.add, aten.avg_pool2d, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        triton_poi_fused_add_avg_pool2d_convolution_mean_mul_relu_sigmoid_silu_48.run(buf372, arg143_1, buf368, arg210_1, buf371, arg131_1, buf373, 995328, grid=grid(995328), stream=stream0)
        del arg131_1
        del arg143_1
        del arg210_1
        del buf371
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____1___act1, out_80, out_81], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf375 = extern_kernels.convolution(buf373, buf374, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf375, (8, 384, 9, 9), (31104, 81, 9, 1))
        del buf374
        buf376 = buf375; del buf375  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____1___act1, getattr_getattr_l__mod___stages___3_____1___act2, out_80, out_81, out_82], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_silu_45.run(buf376, arg146_1, 248832, grid=grid(248832), stream=stream0)
        del arg146_1
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____1___act1, getattr_getattr_l__mod___stages___3_____1___act2, out_80, out_81, out_82], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf378 = extern_kernels.convolution(buf376, buf377, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf378, (8, 384, 9, 9), (31104, 81, 9, 1))
        del buf376
        del buf377
        buf379 = buf378; del buf378  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____1___act1, getattr_getattr_l__mod___stages___3_____1___act2, getattr_getattr_l__mod___stages___3_____1___act2b, out_80, out_81, out_82, out_83], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_silu_45.run(buf379, arg149_1, 248832, grid=grid(248832), stream=stream0)
        del arg149_1
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____1___act1, getattr_getattr_l__mod___stages___3_____1___act2, getattr_getattr_l__mod___stages___3_____1___act2b, out_80, out_81, out_82, out_83], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf381 = extern_kernels.convolution(buf379, buf380, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf381, (8, 384, 9, 9), (31104, 81, 9, 1))
        del buf379
        del buf380
        buf382 = buf381; del buf381  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____1___act1, getattr_getattr_l__mod___stages___3_____1___act2, getattr_getattr_l__mod___stages___3_____1___act2b, getattr_getattr_l__mod___stages___3_____1___act3, out_80, out_81, out_82, out_83, out_84], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_silu_45.run(buf382, arg152_1, 248832, grid=grid(248832), stream=stream0)
        del arg152_1
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____1___act1, getattr_getattr_l__mod___stages___3_____1___act2, getattr_getattr_l__mod___stages___3_____1___act2b, getattr_getattr_l__mod___stages___3_____1___act3, out_80, out_81, out_82, out_83, out_84], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf384 = extern_kernels.convolution(buf382, buf383, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf384, (8, 1536, 9, 9), (124416, 81, 9, 1))
        del buf382
        del buf383
        buf385 = reinterpret_tensor(buf368, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf368  # reuse
        buf386 = reinterpret_tensor(buf385, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf385  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____1___act1, getattr_getattr_l__mod___stages___3_____1___act2, getattr_getattr_l__mod___stages___3_____1___act2b, getattr_getattr_l__mod___stages___3_____1___act3, out_80, out_81, out_82, out_83, out_84, x_se_40, x_se_41], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.silu]
        triton_per_fused_convolution_mean_silu_46.run(buf386, buf384, arg155_1, 12288, 81, grid=grid(12288), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____1___act1, getattr_getattr_l__mod___stages___3_____1___act2, getattr_getattr_l__mod___stages___3_____1___act2b, getattr_getattr_l__mod___stages___3_____1___act3, out_80, out_81, out_82, out_83, out_84, x_se_40, x_se_41], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.silu]
        buf387 = extern_kernels.convolution(buf386, arg211_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf387, (8, 384, 1, 1), (384, 1, 1, 1))
        del arg211_1
        del buf386
        buf388 = buf387; del buf387  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____1___act1, getattr_getattr_l__mod___stages___3_____1___act2, getattr_getattr_l__mod___stages___3_____1___act2b, getattr_getattr_l__mod___stages___3_____1___act3, out_80, out_81, out_82, out_83, out_84, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.silu]
        triton_poi_fused_convolution_mean_relu_silu_37.run(buf388, arg212_1, 3072, grid=grid(3072), stream=stream0)
        del arg212_1
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____1___act1, getattr_getattr_l__mod___stages___3_____1___act2, getattr_getattr_l__mod___stages___3_____1___act2b, getattr_getattr_l__mod___stages___3_____1___act3, out_80, out_81, out_82, out_83, out_84, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.silu]
        buf389 = extern_kernels.convolution(buf388, arg213_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf389, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg213_1
        del buf388
        buf390 = buf372; del buf372  # reuse
        buf391 = buf373; del buf373  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____1___act1, getattr_getattr_l__mod___stages___3_____1___act2, getattr_getattr_l__mod___stages___3_____1___act2b, getattr_getattr_l__mod___stages___3_____1___act3, getattr_getattr_l__mod___stages___3_____2___act1, mul_93, mul_95, out_80, out_81, out_82, out_83, out_84, out_85, out_88, out_89, shortcut_15, sigmoid_10, x_se_40, x_se_41, x_se_42, x_se_43], Original ATen: [aten.add, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_49.run(buf390, buf384, arg155_1, buf389, arg214_1, buf391, 995328, grid=grid(995328), stream=stream0)
        del arg155_1
        del arg214_1
        del buf384
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____2___act1, out_88, out_89], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf393 = extern_kernels.convolution(buf391, buf392, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf393, (8, 384, 9, 9), (31104, 81, 9, 1))
        del buf391
        del buf392
        buf394 = buf393; del buf393  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____2___act1, getattr_getattr_l__mod___stages___3_____2___act2, out_88, out_89, out_90], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_silu_45.run(buf394, arg158_1, 248832, grid=grid(248832), stream=stream0)
        del arg158_1
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____2___act1, getattr_getattr_l__mod___stages___3_____2___act2, out_88, out_89, out_90], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf396 = extern_kernels.convolution(buf394, buf395, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf396, (8, 384, 9, 9), (31104, 81, 9, 1))
        del buf394
        del buf395
        buf397 = buf396; del buf396  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____2___act1, getattr_getattr_l__mod___stages___3_____2___act2, getattr_getattr_l__mod___stages___3_____2___act2b, out_88, out_89, out_90, out_91], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_silu_45.run(buf397, arg161_1, 248832, grid=grid(248832), stream=stream0)
        del arg161_1
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____2___act1, getattr_getattr_l__mod___stages___3_____2___act2, getattr_getattr_l__mod___stages___3_____2___act2b, out_88, out_89, out_90, out_91], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf399 = extern_kernels.convolution(buf397, buf398, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf399, (8, 384, 9, 9), (31104, 81, 9, 1))
        del buf397
        del buf398
        buf400 = buf399; del buf399  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____2___act1, getattr_getattr_l__mod___stages___3_____2___act2, getattr_getattr_l__mod___stages___3_____2___act2b, getattr_getattr_l__mod___stages___3_____2___act3, out_88, out_89, out_90, out_91, out_92], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_silu_45.run(buf400, arg164_1, 248832, grid=grid(248832), stream=stream0)
        del arg164_1
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____2___act1, getattr_getattr_l__mod___stages___3_____2___act2, getattr_getattr_l__mod___stages___3_____2___act2b, getattr_getattr_l__mod___stages___3_____2___act3, out_88, out_89, out_90, out_91, out_92], Original ATen: [aten.convolution, aten.mul, aten.silu]
        buf402 = extern_kernels.convolution(buf400, buf401, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf402, (8, 1536, 9, 9), (124416, 81, 9, 1))
        del buf400
        del buf401
        buf403 = reinterpret_tensor(buf389, (8, 1536, 1, 1), (1536, 1, 12288, 12288), 0); del buf389  # reuse
        buf404 = reinterpret_tensor(buf403, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf403  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____2___act1, getattr_getattr_l__mod___stages___3_____2___act2, getattr_getattr_l__mod___stages___3_____2___act2b, getattr_getattr_l__mod___stages___3_____2___act3, out_88, out_89, out_90, out_91, out_92, x_se_44, x_se_45], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.silu]
        triton_per_fused_convolution_mean_silu_46.run(buf404, buf402, arg167_1, 12288, 81, grid=grid(12288), stream=stream0)
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____2___act1, getattr_getattr_l__mod___stages___3_____2___act2, getattr_getattr_l__mod___stages___3_____2___act2b, getattr_getattr_l__mod___stages___3_____2___act3, out_88, out_89, out_90, out_91, out_92, x_se_44, x_se_45], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.silu]
        buf405 = extern_kernels.convolution(buf404, arg215_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf405, (8, 384, 1, 1), (384, 1, 1, 1))
        del arg215_1
        del buf404
        buf406 = buf405; del buf405  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____2___act1, getattr_getattr_l__mod___stages___3_____2___act2, getattr_getattr_l__mod___stages___3_____2___act2b, getattr_getattr_l__mod___stages___3_____2___act3, out_88, out_89, out_90, out_91, out_92, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.silu]
        triton_poi_fused_convolution_mean_relu_silu_37.run(buf406, arg216_1, 3072, grid=grid(3072), stream=stream0)
        del arg216_1
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____2___act1, getattr_getattr_l__mod___stages___3_____2___act2, getattr_getattr_l__mod___stages___3_____2___act2b, getattr_getattr_l__mod___stages___3_____2___act3, out_88, out_89, out_90, out_91, out_92, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.convolution, aten.mean, aten.mul, aten.relu, aten.silu]
        buf407 = extern_kernels.convolution(buf406, arg217_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf407, (8, 1536, 1, 1), (1536, 1, 1, 1))
        del arg217_1
        del buf406
        buf408 = buf390; del buf390  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____2___act1, getattr_getattr_l__mod___stages___3_____2___act2, getattr_getattr_l__mod___stages___3_____2___act2b, getattr_getattr_l__mod___stages___3_____2___act3, mul_101, mul_103, out_88, out_89, out_90, out_91, out_92, out_93, sigmoid_11, x_1, x_2, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.add, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        triton_poi_fused_add_convolution_mean_mul_relu_sigmoid_silu_50.run(buf408, buf402, arg167_1, buf407, arg218_1, 995328, grid=grid(995328), stream=stream0)
        del arg167_1
        del arg218_1
        del buf402
        del buf407
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____2___act1, getattr_getattr_l__mod___stages___3_____2___act2, getattr_getattr_l__mod___stages___3_____2___act2b, getattr_getattr_l__mod___stages___3_____2___act3, mul_101, mul_103, out_88, out_89, out_90, out_91, out_92, out_93, sigmoid_11, x_1, x_2, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.add, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        buf410 = extern_kernels.convolution(buf408, buf409, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf410, (8, 2304, 9, 9), (186624, 81, 9, 1))
        del buf408
        del buf409
        buf411 = reinterpret_tensor(buf177, (8, 2304, 1, 1), (2304, 1, 18432, 18432), 0); del buf177  # reuse
        buf412 = reinterpret_tensor(buf411, (8, 2304, 1, 1), (2304, 1, 1, 1), 0); del buf411  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____2___act1, getattr_getattr_l__mod___stages___3_____2___act2, getattr_getattr_l__mod___stages___3_____2___act2b, getattr_getattr_l__mod___stages___3_____2___act3, mul_101, mul_103, out_88, out_89, out_90, out_91, out_92, out_93, sigmoid_11, x_1, x_2, x_4, x_5, x_se_44, x_se_45, x_se_46, x_se_47], Original ATen: [aten.add, aten.convolution, aten.mean, aten.mul, aten.relu, aten.sigmoid, aten.silu]
        triton_per_fused_add_convolution_mean_mul_relu_sigmoid_silu_51.run(buf412, buf410, arg170_1, 18432, 81, grid=grid(18432), stream=stream0)
        del arg170_1
        del buf410
        buf413 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg220_1, reinterpret_tensor(buf412, (8, 2304), (2304, 1), 0), reinterpret_tensor(arg219_1, (2304, 1000), (1, 2304), 0), alpha=1, beta=1, out=buf413)
        del arg219_1
        del arg220_1
        return (buf413, )


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
    arg15_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((384, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((1536, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((2304, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((2304, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((1000, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((8, 3, 288, 288), (248832, 82944, 288, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('nfnet_l0', benchmark_compiled_module)
