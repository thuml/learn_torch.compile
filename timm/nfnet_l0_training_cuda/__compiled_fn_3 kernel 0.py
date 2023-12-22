
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


# kernel path: /tmp/torchinductor_youkaichao/7c/c7ctiubojrueqxjksbc6uobjqb36nvev55zveszq5cgjche5kfft.py
# Source Nodes: [batch_norm, weight], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm => add, mul_1, mul_2, rsqrt, squeeze_1, sub, var_mean
# weight => view_2
triton_per_fused__native_batch_norm_legit_view_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_0', 'mutated_arg_names': []}
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
    tmp25 = 0.34412564994580647
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (27*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/rh/crh55g6obf4gqcnzfwjza3hczhpysulwcdc3ci45de5w2djj3jsp.py
# Source Nodes: [conv2d, l__mod___stem_act2], Original ATen: [aten.convolution, aten.silu]
# conv2d => convolution
# l__mod___stem_act2 => mul_3, sigmoid
triton_poi_fused_convolution_silu_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 16
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qv/cqvy7a5mtvzelrwkpgq6iz3dhdl7f5nshvbi4zevg3qvhwpei7iy.py
# Source Nodes: [batch_norm_1, weight_1], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_1 => add_1, mul_5, mul_6, rsqrt_1, squeeze_3, sub_1, var_mean_1
# weight_1 => view_5
triton_per_fused__native_batch_norm_legit_view_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_2', 'mutated_arg_names': []}
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
    tmp25 = 0.1490107774734497
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (144*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5g/c5gv7xz5khjrwgvgl34pt3z36mbu4ou7q6me5onks76ob3vl6n5e.py
# Source Nodes: [conv2d_1, l__mod___stem_act3], Original ATen: [aten.convolution, aten.silu]
# conv2d_1 => convolution_1
# l__mod___stem_act3 => mul_7, sigmoid_1
triton_poi_fused_convolution_silu_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 32
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zv/czvpb4c5xkf7vuawuexh4rc65ixxqx2dbdq66slclenxa5ibuvqd.py
# Source Nodes: [batch_norm_2, weight_2], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_2 => add_2, mul_10, mul_9, rsqrt_2, squeeze_5, sub_2, var_mean_2
# weight_2 => view_8
triton_per_fused__native_batch_norm_legit_view_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_4', 'mutated_arg_names': []}
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
    tmp25 = 0.10536653122135592
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (288*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wp/cwpu5zf6jxlcnqx277xkgbwpqxxzsc4kuh6ixmdcnmri6tdcgk4e.py
# Source Nodes: [conv2d_2, l__mod___stem_act4], Original ATen: [aten.convolution, aten.silu]
# conv2d_2 => convolution_2
# l__mod___stem_act4 => mul_11, sigmoid_2
triton_poi_fused_convolution_silu_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ja/cjaaeftfnqjf4azb352i4uzx2zqjolfk47k7jholiyasyjxlk5tv.py
# Source Nodes: [batch_norm_3, weight_3], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_3 => add_3, mul_13, mul_14, rsqrt_3, squeeze_7, sub_3, var_mean_3
# weight_3 => view_11
triton_per_fused__native_batch_norm_legit_view_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_6', 'mutated_arg_names': []}
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
    tmp25 = 0.07450538873672485
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (576*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2c/c2cmqxqv32wzuh7aq7xfoy47ber7ypcdjflbpj3qomzqyqzl3lfy.py
# Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act1, out, shortcut], Original ATen: [aten.convolution, aten.mul, aten.silu]
# getattr_getattr_l__mod___stages___0_____0___act1 => mul_15, sigmoid_3
# out => mul_16
# shortcut => convolution_3
triton_poi_fused_convolution_mul_silu_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mul_silu_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tmp5 = 1.0
    tmp6 = tmp4 * tmp5
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ca/ccasx52fszgko4mbo7joba5wwmcj66oozow5egr7whym2sqptggj.py
# Source Nodes: [batch_norm_4, weight_4], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_4 => add_4, mul_18, mul_19, rsqrt_4, squeeze_9, sub_4, var_mean_4
# weight_4 => view_14
triton_per_fused__native_batch_norm_legit_view_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_8', 'mutated_arg_names': []}
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
    tmp25 = 0.1580497968320339
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (128*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2q/c2qahsiyrla55jffqh5smyjkq5ddiirdrzguvaetxwz7evzzyg7g.py
# Source Nodes: [batch_norm_5, weight_5], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_5 => add_5, mul_21, mul_22, rsqrt_5, squeeze_11, sub_5, var_mean_5
# weight_5 => view_17
triton_per_fused__native_batch_norm_legit_view_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_9', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ib/cib5c5tvliouk6phwzkcozfs3oarqgksrwjnqhdeknmdw5lgo36x.py
# Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act2, out_1], Original ATen: [aten.convolution, aten.silu]
# getattr_getattr_l__mod___stages___0_____0___act2 => mul_23, sigmoid_4
# out_1 => convolution_5
triton_poi_fused_convolution_silu_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wq/cwq6rmdvq7xwojnjbmprax657u4fcvvorvwtty2bpgsqkk5utxbf.py
# Source Nodes: [batch_norm_6, weight_6], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_6 => add_6, mul_25, mul_26, rsqrt_6, squeeze_13, sub_6, var_mean_6
# weight_6 => view_20
triton_per_fused__native_batch_norm_legit_view_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iz/cizc6cjw5lfx3vpfzr3sxun4ykk2u2yzqo7kewwxntjnymq2nrfr.py
# Source Nodes: [batch_norm_8, weight_8], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_8 => add_8, mul_33, mul_34, rsqrt_8, squeeze_17, sub_8, var_mean_8
# weight_8 => view_26
triton_per_fused__native_batch_norm_legit_view_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_12', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ju/cjuu6dvkpmmhl3k7me3vdfbsgninqnl5bxbgryhe2diwkfxtverr.py
# Source Nodes: [out_4, x_se], Original ATen: [aten.convolution, aten.mean]
# out_4 => convolution_8
# x_se => mean
triton_red_fused_convolution_mean_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_convolution_mean_13', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 3136
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
        tmp0 = tl.load(in_out_ptr0 + (r2 + (3136*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
        tl.store(in_out_ptr0 + (r2 + (3136*x3)), tmp2, rmask)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp6 = 3136.0
    tmp7 = tmp4 / tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp7, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2f/c2fknzjolminx2o236sdcpq7qtnhjoolfsqiy5srfymd7gmsldzl.py
# Source Nodes: [x_se_1, x_se_2], Original ATen: [aten.convolution, aten.relu]
# x_se_1 => convolution_9
# x_se_2 => relu
triton_poi_fused_convolution_relu_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_14', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/7i/c7ir4mptpnrfzjqd623urlogwcnfrwa2ekmoov6qccjuhnirqxkz.py
# Source Nodes: [x_se_3], Original ATen: [aten.convolution]
# x_se_3 => convolution_10
triton_poi_fused_convolution_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_15', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/jp/cjpi7ptnoage37unizawctpunsceh7wq2dow3jijzhqlye4fogww.py
# Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act1, mul_10, mul_12, out_5, out_8, shortcut_1, shortcut_2, sigmoid], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___stages___1_____0___act1 => mul_38, sigmoid_8
# mul_10 => mul_35
# mul_12 => mul_37
# out_5 => mul_36
# out_8 => mul_39
# shortcut_1 => convolution_4
# shortcut_2 => add_9
# sigmoid => sigmoid_7
triton_poi_fused_add_convolution_mul_sigmoid_silu_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_sigmoid_silu_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 3136)
    x1 = (xindex // 3136) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x3), None)
    tmp9 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.2
    tmp7 = tmp5 * tmp6
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = 0.9805806756909201
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ag/caglxsovtl4bchrya6kwyhzz5ixc2rrpx2lopbacv2wsqhlhmkjp.py
# Source Nodes: [getattr_getattr_l__mod___stages___1_____0___downsample_pool], Original ATen: [aten.avg_pool2d]
# getattr_getattr_l__mod___stages___1_____0___downsample_pool => avg_pool2d
triton_poi_fused_avg_pool2d_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 28
    x1 = (xindex // 28)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (112*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (112*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (56 + (2*x0) + (112*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (57 + (2*x0) + (112*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ju/cjunhrtroz4vcwx2vsazbc45pav2u3vz6jwraplr5fbjfauxeu2h.py
# Source Nodes: [batch_norm_9, weight_9], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_9 => add_10, mul_41, mul_42, rsqrt_9, squeeze_19, sub_9, var_mean_9
# weight_9 => view_29
triton_per_fused__native_batch_norm_legit_view_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_18', 'mutated_arg_names': []}
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
    tmp25 = 0.11175808310508728
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (256*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ct/cctuwkluzglczhyb7ogjg2ulpuecok5xs3lwfmie3ildfgwditx7.py
# Source Nodes: [batch_norm_10, weight_10], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_10 => add_11, mul_44, mul_45, rsqrt_10, squeeze_21, sub_10, var_mean_10
# weight_10 => view_32
triton_per_fused__native_batch_norm_legit_view_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_19', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/47/c47yktmosppgytqj3ve5sibhkuenarpifw6sj3rmufhjodujxxh5.py
# Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act2, out_9], Original ATen: [aten.convolution, aten.silu]
# getattr_getattr_l__mod___stages___1_____0___act2 => mul_46, sigmoid_9
# out_9 => convolution_12
triton_poi_fused_convolution_silu_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4w/c4wpy5ezp2sl2q34g6wbbbgxi64dxzfydjhwnriqktgksmkfvazt.py
# Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act2b, out_10], Original ATen: [aten.convolution, aten.silu]
# getattr_getattr_l__mod___stages___1_____0___act2b => mul_50, sigmoid_10
# out_10 => convolution_13
triton_poi_fused_convolution_silu_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/a5/ca5ix5lfufspw6pibmwgx7wspmavnry3dqddriqjdzn5y4mn6y73.py
# Source Nodes: [batch_norm_13, weight_13], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_13 => add_14, mul_56, mul_57, rsqrt_13, squeeze_27, sub_13, var_mean_13
# weight_13 => view_41
triton_per_fused__native_batch_norm_legit_view_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_22', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yy/cyybj267ty2xpigcycnpbbgeuulmex43qvmcmiwaa5nir4cxwwey.py
# Source Nodes: [out_12, x_se_4], Original ATen: [aten.convolution, aten.mean]
# out_12 => convolution_15
# x_se_4 => mean_1
triton_per_fused_convolution_mean_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_mean_23', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_out_ptr0 + (r2 + (784*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp7 = 784.0
    tmp8 = tmp6 / tmp7
    tl.store(in_out_ptr0 + (r2 + (784*x3)), tmp2, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/57/c57naukr3j3pys2lfsctbiouxoofvorfx2tuhfb4h3yvntxlh5ac.py
# Source Nodes: [x_se_5, x_se_6], Original ATen: [aten.convolution, aten.relu]
# x_se_5 => convolution_16
# x_se_6 => relu_1
triton_poi_fused_convolution_relu_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_24', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/hd/chdompjeseiypdvh3lwhkynxbbfuujxw67f6t4ztkzhokr2fvpqa.py
# Source Nodes: [x_se_7], Original ATen: [aten.convolution]
# x_se_7 => convolution_17
triton_poi_fused_convolution_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_25', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/hp/chp626indurvoq6zeuxwiv2wjb7ux4zboy3wfmlucrzf6bqkj4s6.py
# Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act1, mul_19, mul_21, out_13, out_16, shortcut_3, shortcut_4, sigmoid_1], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___stages___1_____1___act1 => mul_61, sigmoid_13
# mul_19 => mul_58
# mul_21 => mul_60
# out_13 => mul_59
# out_16 => mul_62
# shortcut_3 => convolution_11
# shortcut_4 => add_15
# sigmoid_1 => sigmoid_12
triton_poi_fused_add_convolution_mul_sigmoid_silu_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_sigmoid_silu_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 784)
    x1 = (xindex // 784) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x3), None)
    tmp9 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.2
    tmp7 = tmp5 * tmp6
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = 0.9805806756909201
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qn/cqn7sqmux5pmjlz4wuzktuouyjiacj4rp3556kmtzcplrpql4fzz.py
# Source Nodes: [batch_norm_14, weight_14], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_14 => add_16, mul_64, mul_65, rsqrt_14, squeeze_29, sub_14, var_mean_14
# weight_14 => view_44
triton_per_fused__native_batch_norm_legit_view_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_27', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sg/csgnhs7jthyizj7pkfizrq5h7yw26kjmvo7cax7rs7qym5shkvxd.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act1, mul_19, mul_21, mul_27, mul_29, out_13, out_21, out_24, shortcut_3, shortcut_4, shortcut_5, sigmoid_1, sigmoid_2], Original ATen: [aten.add, aten.convolution, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# getattr_getattr_l__mod___stages___2_____0___act1 => mul_81, sigmoid_18
# mul_19 => mul_58
# mul_21 => mul_60
# mul_27 => mul_78
# mul_29 => mul_80
# out_13 => mul_59
# out_21 => mul_79
# out_24 => mul_82
# shortcut_3 => convolution_11
# shortcut_4 => add_15
# shortcut_5 => add_20
# sigmoid_1 => sigmoid_12
# sigmoid_2 => sigmoid_17
triton_poi_fused_add_convolution_fill_mul_sigmoid_silu_sub_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_fill_mul_sigmoid_silu_sub_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 784)
    x1 = (xindex // 784) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x3), None)
    tmp9 = tl.load(in_ptr3 + (x4), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x3), None)
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.2
    tmp7 = tmp5 * tmp6
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp8 * tmp10
    tmp12 = tmp11 * tmp4
    tmp13 = tmp12 * tmp6
    tmp16 = tmp14 + tmp15
    tmp17 = tmp13 + tmp16
    tmp18 = tmp7 + tmp17
    tmp19 = tl.sigmoid(tmp18)
    tmp20 = tmp18 * tmp19
    tmp21 = 0.9622504486493761
    tmp22 = tmp20 * tmp21
    tmp23 = 1.0
    tmp24 = tmp23 - tmp19
    tmp25 = tmp18 * tmp24
    tmp26 = tmp25 + tmp23
    tmp27 = tmp19 * tmp26
    tl.store(out_ptr1 + (x3), tmp22, None)
    tl.store(out_ptr2 + (x3), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/w2/cw2ywvrd3ql2veavql6yuua265l3ftckwwshkl2r2r3xvafqatry.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____0___downsample_pool], Original ATen: [aten.avg_pool2d]
# getattr_getattr_l__mod___stages___2_____0___downsample_pool => avg_pool2d_1
triton_poi_fused_avg_pool2d_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 14
    x1 = (xindex // 14)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (56*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (56*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (28 + (2*x0) + (56*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (29 + (2*x0) + (56*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4y/c4yjc76tweoyryt4ctormglevaghxbrwj6otrekkkwzxcoddg6wg.py
# Source Nodes: [batch_norm_18, weight_18], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_18 => add_21, mul_84, mul_85, rsqrt_18, squeeze_37, sub_18, var_mean_18
# weight_18 => view_56
triton_per_fused__native_batch_norm_legit_view_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_30', 'mutated_arg_names': []}
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
    tmp25 = 0.07902489841601695
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 * tmp26
    tl.store(out_ptr2 + (r1 + (512*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bb/cbb3mv3uwo6qlaleh5bs6fp4r6di5wg5dqqwnt3z7umzltjth5j3.py
# Source Nodes: [batch_norm_19, weight_19], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_19 => add_22, mul_87, mul_88, rsqrt_19, squeeze_39, sub_19, var_mean_19
# weight_19 => view_59
triton_per_fused__native_batch_norm_legit_view_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_31', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tu/ctu2y7ynrlkpykioybmwvlll7u5srobznfxa5km7d7a3mh73enha.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act2, out_25], Original ATen: [aten.convolution, aten.silu]
# getattr_getattr_l__mod___stages___2_____0___act2 => mul_89, sigmoid_19
# out_25 => convolution_25
triton_poi_fused_convolution_silu_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 384
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xn/cxntqdshs3gfjs6bytssj6rsvqiwhw5yhkwlycnzyeoy5ezkqifi.py
# Source Nodes: [batch_norm_20, weight_20], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_20 => add_23, mul_91, mul_92, rsqrt_20, squeeze_41, sub_20, var_mean_20
# weight_20 => view_62
triton_per_fused__native_batch_norm_legit_view_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xb/cxbvq7ftjaazdxxio2k6rxl6g2yh4rnu7m4af2z2jridcugvxzaq.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act2b, out_26], Original ATen: [aten.convolution, aten.silu]
# getattr_getattr_l__mod___stages___2_____0___act2b => mul_93, sigmoid_20
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_34', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 384
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ju/cjuivvxwdephkairy4alreac2sfn6tvra7bu562e7dj7f6ku5pbf.py
# Source Nodes: [batch_norm_22, weight_22], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_22 => add_25, mul_100, mul_99, rsqrt_22, squeeze_45, sub_22, var_mean_22
# weight_22 => view_68
triton_per_fused__native_batch_norm_legit_view_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_view_35', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel):
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
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hu/chujtc7ks5aggajygomj5uw6bazd7mdwl5iyz6yv66bdglnt5zyp.py
# Source Nodes: [out_28, x_se_12], Original ATen: [aten.convolution, aten.mean]
# out_28 => convolution_28
# x_se_12 => mean_3
triton_per_fused_convolution_mean_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_mean_36', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1536
    tmp0 = tl.load(in_out_ptr0 + (r2 + (196*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 196.0
    tmp8 = tmp6 / tmp7
    tl.store(in_out_ptr0 + (r2 + (196*x3)), tmp2, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/x7/cx77ddeuyyu5uxzrxfzfj2tgogqcrxlomg66yuak2unfm6ehbh6t.py
# Source Nodes: [x_se_13, x_se_14], Original ATen: [aten.convolution, aten.relu]
# x_se_13 => convolution_29
# x_se_14 => relu_3
triton_poi_fused_convolution_relu_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_37', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/ek/cekympdjalcrb5g3wwrvhnmp662i7cszkop2umzxm3sal24hyais.py
# Source Nodes: [x_se_15], Original ATen: [aten.convolution]
# x_se_15 => convolution_30
triton_poi_fused_convolution_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_38', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/ll/cllqxemkcrftbog5mbretzyrxhgyeyjcfargdnvmb4sizuuwwunw.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____1___act1, mul_36, mul_38, out_29, out_32, shortcut_6, shortcut_7, sigmoid_3], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___stages___2_____1___act1 => mul_104, sigmoid_23
# mul_36 => mul_101
# mul_38 => mul_103
# out_29 => mul_102
# out_32 => mul_105
# shortcut_6 => convolution_24
# shortcut_7 => add_26
# sigmoid_3 => sigmoid_22
triton_poi_fused_add_convolution_mul_sigmoid_silu_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_sigmoid_silu_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 1536
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x3), None)
    tmp9 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.2
    tmp7 = tmp5 * tmp6
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = 0.9805806756909201
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/iw/ciwlvgbzu4jbeltuducrmpx7mxrus2ozo4jaiva7rqcef6hz65zk.py
# Source Nodes: [batch_norm_23, weight_23], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_23 => add_27, mul_107, mul_108, rsqrt_23, squeeze_47, sub_23, var_mean_23
# weight_23 => view_71
triton_red_fused__native_batch_norm_legit_view_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_view_40', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        tmp14 = 0.04562504637317021
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


# kernel path: /tmp/torchinductor_youkaichao/js/cjs4eey2xxgvtydryhqgbzsgwbr2tjjf5ncprglvkssqtc2kstcl.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____2___act1, mul_36, mul_38, mul_44, mul_46, out_29, out_37, out_40, shortcut_6, shortcut_7, shortcut_8, sigmoid_3, sigmoid_4], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___stages___2_____2___act1 => mul_124, sigmoid_28
# mul_36 => mul_101
# mul_38 => mul_103
# mul_44 => mul_121
# mul_46 => mul_123
# out_29 => mul_102
# out_37 => mul_122
# out_40 => mul_125
# shortcut_6 => convolution_24
# shortcut_7 => add_26
# shortcut_8 => add_31
# sigmoid_3 => sigmoid_22
# sigmoid_4 => sigmoid_27
triton_poi_fused_add_convolution_mul_sigmoid_silu_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_sigmoid_silu_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 1536
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x3), None)
    tmp9 = tl.load(in_ptr3 + (x4), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x3), None)
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.2
    tmp7 = tmp5 * tmp6
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp8 * tmp10
    tmp12 = tmp11 * tmp4
    tmp13 = tmp12 * tmp6
    tmp16 = tmp14 + tmp15
    tmp17 = tmp13 + tmp16
    tmp18 = tmp7 + tmp17
    tmp19 = tl.sigmoid(tmp18)
    tmp20 = tmp18 * tmp19
    tmp21 = 0.9622504486493761
    tmp22 = tmp20 * tmp21
    tl.store(out_ptr0 + (x3), tmp18, None)
    tl.store(out_ptr1 + (x3), tmp22, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/al/cal3fcc2poucnixnacy5iufr3aqkr7xxtr3rbxfkedz6hcjmuhln.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____3___act1, mul_52, mul_54, out_45, out_48, shortcut_9, sigmoid_5], Original ATen: [aten.add, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___stages___2_____3___act1 => mul_144, sigmoid_33
# mul_52 => mul_141
# mul_54 => mul_143
# out_45 => mul_142
# out_48 => mul_145
# shortcut_9 => add_36
# sigmoid_5 => sigmoid_32
triton_poi_fused_add_mul_sigmoid_silu_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_silu_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x2), None)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.2
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp9 * tmp10
    tmp12 = 0.9449111825230679
    tmp13 = tmp11 * tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rb/crbtbdp2omy7zk2dzgkjpc44fx3qynlfxcuzohpibveph7rd4skb.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____4___act1, mul_52, mul_54, mul_60, mul_62, out_45, out_53, out_56, shortcut_10, shortcut_9, sigmoid_5, sigmoid_6], Original ATen: [aten.add, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___stages___2_____4___act1 => mul_164, sigmoid_38
# mul_52 => mul_141
# mul_54 => mul_143
# mul_60 => mul_161
# mul_62 => mul_163
# out_45 => mul_142
# out_53 => mul_162
# out_56 => mul_165
# shortcut_10 => add_41
# shortcut_9 => add_36
# sigmoid_5 => sigmoid_32
# sigmoid_6 => sigmoid_37
triton_poi_fused_add_mul_sigmoid_silu_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_silu_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x2), None)
    tmp9 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), None)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.2
    tmp7 = tmp5 * tmp6
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp8 * tmp10
    tmp12 = tmp11 * tmp4
    tmp13 = tmp12 * tmp6
    tmp15 = tmp13 + tmp14
    tmp16 = tmp7 + tmp15
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tmp19 = 0.9284766908852592
    tmp20 = tmp18 * tmp19
    tl.store(out_ptr0 + (x2), tmp16, None)
    tl.store(out_ptr1 + (x2), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/md/cmd2aawulcwynl77qeqwbywpyb3smzsomln2hmppjcuyhzgl64de.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____5___act1, mul_68, mul_70, out_61, out_64, shortcut_11, sigmoid_7], Original ATen: [aten.add, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___stages___2_____5___act1 => mul_184, sigmoid_43
# mul_68 => mul_181
# mul_70 => mul_183
# out_61 => mul_182
# out_64 => mul_185
# shortcut_11 => add_46
# sigmoid_7 => sigmoid_42
triton_poi_fused_add_mul_sigmoid_silu_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_mul_sigmoid_silu_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x2), None)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.2
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp9 * tmp10
    tmp12 = 0.9128709291752768
    tmp13 = tmp11 * tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sp/csp5x6wqer65v6likzce76xbvqum5oz4gxtxodvlmjsnooauv44b.py
# Source Nodes: [getattr_getattr_l__mod___stages___3_____0___act1, mul_68, mul_70, mul_76, mul_78, out_61, out_69, out_72, shortcut_11, shortcut_12, sigmoid_7, sigmoid_8], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# getattr_getattr_l__mod___stages___3_____0___act1 => mul_204, sigmoid_48
# mul_68 => mul_181
# mul_70 => mul_183
# mul_76 => mul_201
# mul_78 => mul_203
# out_61 => mul_182
# out_69 => mul_202
# out_72 => mul_205
# shortcut_11 => add_46
# shortcut_12 => add_51
# sigmoid_7 => sigmoid_42
# sigmoid_8 => sigmoid_47
triton_poi_fused_add_fill_mul_sigmoid_silu_sub_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_silu_sub_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x2), None)
    tmp9 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x2), None)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.2
    tmp7 = tmp5 * tmp6
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp8 * tmp10
    tmp12 = tmp11 * tmp4
    tmp13 = tmp12 * tmp6
    tmp15 = tmp13 + tmp14
    tmp16 = tmp7 + tmp15
    tmp17 = tl.sigmoid(tmp16)
    tmp18 = tmp16 * tmp17
    tmp19 = 0.8980265101338745
    tmp20 = tmp18 * tmp19
    tmp21 = 1.0
    tmp22 = tmp21 - tmp17
    tmp23 = tmp16 * tmp22
    tmp24 = tmp23 + tmp21
    tmp25 = tmp17 * tmp24
    tl.store(out_ptr1 + (x2), tmp20, None)
    tl.store(out_ptr2 + (x2), tmp25, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jl/cjl6gpmzc5h4zvfiyaxivxfermiz32wdpctw4hnarf4hj55nwzsj.py
# Source Nodes: [getattr_getattr_l__mod___stages___3_____0___downsample_pool], Original ATen: [aten.avg_pool2d]
# getattr_getattr_l__mod___stages___3_____0___downsample_pool => avg_pool2d_2
triton_poi_fused_avg_pool2d_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 7
    x1 = (xindex // 7)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (28*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (28*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (14 + (2*x0) + (28*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (15 + (2*x0) + (28*x1)), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tl.store(out_ptr0 + (x2), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fx/cfxdwgfokrnkqdnxfwt6ijoxdlr62tqttmjjddsjaf2hikapkhrn.py
# Source Nodes: [batch_norm_43, weight_43], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_43 => add_52, mul_207, mul_208, rsqrt_43, squeeze_87, sub_43, var_mean_43
# weight_43 => view_131
triton_red_fused__native_batch_norm_legit_view_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_view_47', 'mutated_arg_names': []}
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
        tmp14 = 0.04562504637317021
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


# kernel path: /tmp/torchinductor_youkaichao/5j/c5jkno7kev3ysuhws2czwdciqaivipvxpnxb7isuxeivbyw3kbmz.py
# Source Nodes: [getattr_getattr_l__mod___stages___3_____0___act2b, out_74], Original ATen: [aten.convolution, aten.silu]
# getattr_getattr_l__mod___stages___3_____0___act2b => mul_216, sigmoid_50
# out_74 => convolution_63
triton_poi_fused_convolution_silu_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_silu_48', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 384
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aj/cajwf3hnpsavule25rxgcsigg6wsmpq3evyxmwoogwqqixwljrsz.py
# Source Nodes: [out_76, x_se_36], Original ATen: [aten.convolution, aten.mean]
# out_76 => convolution_65
# x_se_36 => mean_9
triton_per_fused_convolution_mean_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_mean_49', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12288
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1536
    tmp0 = tl.load(in_out_ptr0 + (r2 + (49*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 49.0
    tmp8 = tmp6 / tmp7
    tl.store(in_out_ptr0 + (r2 + (49*x3)), tmp2, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5d/c5d5pw3kny54ymovyptyvxdwinuuhk3ioytzf5xx2vypyhqvumpm.py
# Source Nodes: [getattr_getattr_l__mod___stages___3_____1___act1, mul_85, mul_87, out_77, out_80, shortcut_13, shortcut_14, sigmoid_9], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sigmoid, aten.silu]
# getattr_getattr_l__mod___stages___3_____1___act1 => mul_227, sigmoid_53
# mul_85 => mul_224
# mul_87 => mul_226
# out_77 => mul_225
# out_80 => mul_228
# shortcut_13 => convolution_61
# shortcut_14 => add_57
# sigmoid_9 => sigmoid_52
triton_poi_fused_add_convolution_mul_sigmoid_silu_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_mul_sigmoid_silu_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 49)
    x1 = (xindex // 49) % 1536
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x3), None)
    tmp9 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.2
    tmp7 = tmp5 * tmp6
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = 0.9805806756909201
    tmp15 = tmp13 * tmp14
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/er/cer25umlrmxdffxfigd3mnca534nt3sfshp5pyigafvzncntt6lp.py
# Source Nodes: [getattr_getattr_l__mod___stages___3_____1___act1, getattr_getattr_l__mod___stages___3_____2___act1, mul_85, mul_87, mul_93, mul_95, out_77, out_85, out_88, shortcut_13, shortcut_14, shortcut_15, sigmoid_10, sigmoid_9], Original ATen: [aten.add, aten.convolution, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# getattr_getattr_l__mod___stages___3_____1___act1 => sigmoid_53
# getattr_getattr_l__mod___stages___3_____2___act1 => mul_247, sigmoid_58
# mul_85 => mul_224
# mul_87 => mul_226
# mul_93 => mul_244
# mul_95 => mul_246
# out_77 => mul_225
# out_85 => mul_245
# out_88 => mul_248
# shortcut_13 => convolution_61
# shortcut_14 => add_57
# shortcut_15 => add_62
# sigmoid_10 => sigmoid_57
# sigmoid_9 => sigmoid_52
triton_poi_fused_add_convolution_fill_mul_sigmoid_silu_sub_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_fill_mul_sigmoid_silu_sub_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 49)
    x1 = (xindex // 49) % 1536
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x3), None)
    tmp9 = tl.load(in_ptr3 + (x4), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr4 + (x3), None)
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.2
    tmp7 = tmp5 * tmp6
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = tmp8 * tmp10
    tmp12 = tmp11 * tmp4
    tmp13 = tmp12 * tmp6
    tmp16 = tmp14 + tmp15
    tmp17 = tmp13 + tmp16
    tmp18 = tmp7 + tmp17
    tmp19 = tl.sigmoid(tmp17)
    tmp20 = 1.0
    tmp21 = tmp20 - tmp19
    tmp22 = tmp17 * tmp21
    tmp23 = tmp22 + tmp20
    tmp24 = tmp19 * tmp23
    tmp25 = tl.sigmoid(tmp18)
    tmp26 = tmp18 * tmp25
    tmp27 = 0.9622504486493761
    tmp28 = tmp26 * tmp27
    tl.store(out_ptr0 + (x3), tmp18, None)
    tl.store(out_ptr1 + (x3), tmp24, None)
    tl.store(out_ptr2 + (x3), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ae/caeeg3ns5mnmy2eq6e3gqcglchurqm34zetsypp6kf7pmcui6ko7.py
# Source Nodes: [getattr_getattr_l__mod___stages___3_____2___act1, mul_101, mul_103, out_93, sigmoid_11, x_1], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# getattr_getattr_l__mod___stages___3_____2___act1 => sigmoid_58
# mul_101 => mul_264
# mul_103 => mul_266
# out_93 => mul_265
# sigmoid_11 => sigmoid_62
# x_1 => add_67
triton_poi_fused_add_fill_mul_sigmoid_silu_sub_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_silu_sub_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 49)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x2), None)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.2
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tl.sigmoid(tmp8)
    tmp11 = 1.0
    tmp12 = tmp11 - tmp10
    tmp13 = tmp8 * tmp12
    tmp14 = tmp13 + tmp11
    tmp15 = tmp10 * tmp14
    tl.store(out_ptr0 + (x2), tmp9, None)
    tl.store(out_ptr1 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4c/c4cmphelf2zve2zrpyg2kqi7emhdd7ws4ggk67sqzecozxmmjkyz.py
# Source Nodes: [batch_norm_56, weight_56], Original ATen: [aten._native_batch_norm_legit, aten.view]
# batch_norm_56 => add_68, mul_268, mul_269, rsqrt_56, squeeze_113, sub_56, var_mean_56
# weight_56 => view_170
triton_red_fused__native_batch_norm_legit_view_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_view_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        tmp14 = 0.04562504637317021
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


# kernel path: /tmp/torchinductor_youkaichao/7x/c7xyceqrqlwrjuwaosxze6djmswacghczai7tztp6dmx53rayrod.py
# Source Nodes: [x_2, x_4, x_5, x_7], Original ATen: [aten.convolution, aten.mean, aten.silu, aten.view]
# x_2 => convolution_80
# x_4 => mul_270, sigmoid_63
# x_5 => mean_12
# x_7 => view_171
triton_per_fused_convolution_mean_silu_view_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_mean_silu_view_54', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 18432
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 2304
    tmp0 = tl.load(in_out_ptr0 + (r2 + (49*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.sum(tmp7, 1)[:, None]
    tmp9 = 49.0
    tmp10 = tmp8 / tmp9
    tl.store(in_out_ptr0 + (r2 + (49*x3)), tmp2, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp10, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3r/c3rucecrb6eqqzw5dkl2oluqjamerb63qq4e44qut4aajs3twyhs.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____4___act1, getattr_getattr_l__mod___stages___2_____5___act1, mul_68, mul_70, out_61, shortcut_11, sigmoid_7], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# getattr_getattr_l__mod___stages___2_____4___act1 => sigmoid_38
# getattr_getattr_l__mod___stages___2_____5___act1 => sigmoid_43
# mul_68 => mul_181
# mul_70 => mul_183
# out_61 => mul_182
# shortcut_11 => add_46
# sigmoid_7 => sigmoid_42
triton_poi_fused_add_fill_mul_sigmoid_silu_sub_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_silu_sub_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 196)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_ptr2 + (x2), None)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.2
    tmp7 = tmp5 * tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tl.sigmoid(tmp9)
    tmp11 = 1.0
    tmp12 = tmp11 - tmp10
    tmp13 = tmp9 * tmp12
    tmp14 = tmp13 + tmp11
    tmp15 = tmp10 * tmp14
    tmp16 = tl.sigmoid(tmp8)
    tmp17 = tmp11 - tmp16
    tmp18 = tmp8 * tmp17
    tmp19 = tmp18 + tmp11
    tmp20 = tmp16 * tmp19
    tl.store(out_ptr0 + (x2), tmp15, None)
    tl.store(out_ptr1 + (x2), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bq/cbqxmwzxcjkrkwjb6lpvfn646wkyhtanzqehd6gejqqhpfg4gpmv.py
# Source Nodes: [getattr_getattr_l__mod___stages___2_____1___act1, mul_36, mul_38, out_29, shortcut_6, shortcut_7, sigmoid_3], Original ATen: [aten.add, aten.convolution, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# getattr_getattr_l__mod___stages___2_____1___act1 => sigmoid_23
# mul_36 => mul_101
# mul_38 => mul_103
# out_29 => mul_102
# shortcut_6 => convolution_24
# shortcut_7 => add_26
# sigmoid_3 => sigmoid_22
triton_poi_fused_add_convolution_fill_mul_sigmoid_silu_sub_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_fill_mul_sigmoid_silu_sub_56', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 196)
    x1 = (xindex // 196) % 1536
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr0 + (x3), None)
    tmp9 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.2
    tmp7 = tmp5 * tmp6
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = 1.0
    tmp14 = tmp13 - tmp12
    tmp15 = tmp11 * tmp14
    tmp16 = tmp15 + tmp13
    tmp17 = tmp12 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6h/c6hcj3tdtyykkqbgaxedstfme6fcs6advatb5gkzct5ejyptmws2.py
# Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act1, mul_19, mul_21, out_13, shortcut_3, shortcut_4, sigmoid_1], Original ATen: [aten.add, aten.convolution, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# getattr_getattr_l__mod___stages___1_____1___act1 => sigmoid_13
# mul_19 => mul_58
# mul_21 => mul_60
# out_13 => mul_59
# shortcut_3 => convolution_11
# shortcut_4 => add_15
# sigmoid_1 => sigmoid_12
triton_poi_fused_add_convolution_fill_mul_sigmoid_silu_sub_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_fill_mul_sigmoid_silu_sub_57', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 784)
    x1 = (xindex // 784) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr0 + (x3), None)
    tmp9 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.2
    tmp7 = tmp5 * tmp6
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = 1.0
    tmp14 = tmp13 - tmp12
    tmp15 = tmp11 * tmp14
    tmp16 = tmp15 + tmp13
    tmp17 = tmp12 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nn/cnndbtqwidq7wtoutgu6m2beo6uzdpthx3u4hprjm7hanfk6wymb.py
# Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act1, mul_10, mul_12, out_5, shortcut_1, shortcut_2, sigmoid], Original ATen: [aten.add, aten.convolution, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# getattr_getattr_l__mod___stages___1_____0___act1 => sigmoid_8
# mul_10 => mul_35
# mul_12 => mul_37
# out_5 => mul_36
# shortcut_1 => convolution_4
# shortcut_2 => add_9
# sigmoid => sigmoid_7
triton_poi_fused_add_convolution_fill_mul_sigmoid_silu_sub_58 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_convolution_fill_mul_sigmoid_silu_sub_58', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = (xindex // 3136)
    x1 = (xindex // 3136) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp8 = tl.load(in_out_ptr0 + (x3), None)
    tmp9 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp0 * tmp2
    tmp4 = 2.0
    tmp5 = tmp3 * tmp4
    tmp6 = 0.2
    tmp7 = tmp5 * tmp6
    tmp10 = tmp8 + tmp9
    tmp11 = tmp7 + tmp10
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = 1.0
    tmp14 = tmp13 - tmp12
    tmp15 = tmp11 * tmp14
    tmp16 = tmp15 + tmp13
    tmp17 = tmp12 * tmp16
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222 = args
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
    assert_size_stride(primals_16, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_17, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_18, (64, ), (1, ))
    assert_size_stride(primals_19, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_20, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_21, (64, ), (1, ))
    assert_size_stride(primals_22, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_23, (64, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_24, (64, ), (1, ))
    assert_size_stride(primals_25, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_26, (256, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_27, (256, ), (1, ))
    assert_size_stride(primals_28, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_29, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_30, (512, ), (1, ))
    assert_size_stride(primals_31, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_32, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_33, (128, ), (1, ))
    assert_size_stride(primals_34, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_35, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_36, (128, ), (1, ))
    assert_size_stride(primals_37, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_38, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_40, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_41, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_42, (512, ), (1, ))
    assert_size_stride(primals_43, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_44, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_46, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_47, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_48, (128, ), (1, ))
    assert_size_stride(primals_49, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_50, (128, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_51, (128, ), (1, ))
    assert_size_stride(primals_52, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_53, (512, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_54, (512, ), (1, ))
    assert_size_stride(primals_55, (1536, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_56, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_57, (1536, ), (1, ))
    assert_size_stride(primals_58, (384, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_59, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_60, (384, ), (1, ))
    assert_size_stride(primals_61, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_62, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_63, (384, ), (1, ))
    assert_size_stride(primals_64, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_65, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_66, (384, ), (1, ))
    assert_size_stride(primals_67, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_68, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_69, (1536, ), (1, ))
    assert_size_stride(primals_70, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_71, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_72, (384, ), (1, ))
    assert_size_stride(primals_73, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_74, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_75, (384, ), (1, ))
    assert_size_stride(primals_76, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_77, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_78, (384, ), (1, ))
    assert_size_stride(primals_79, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_80, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_81, (1536, ), (1, ))
    assert_size_stride(primals_82, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_83, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_84, (384, ), (1, ))
    assert_size_stride(primals_85, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_86, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_87, (384, ), (1, ))
    assert_size_stride(primals_88, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_89, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_90, (384, ), (1, ))
    assert_size_stride(primals_91, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_92, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_93, (1536, ), (1, ))
    assert_size_stride(primals_94, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_95, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_96, (384, ), (1, ))
    assert_size_stride(primals_97, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_98, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_99, (384, ), (1, ))
    assert_size_stride(primals_100, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_101, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_102, (384, ), (1, ))
    assert_size_stride(primals_103, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_104, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_105, (1536, ), (1, ))
    assert_size_stride(primals_106, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_107, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_108, (384, ), (1, ))
    assert_size_stride(primals_109, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_110, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_111, (384, ), (1, ))
    assert_size_stride(primals_112, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_113, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_114, (384, ), (1, ))
    assert_size_stride(primals_115, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_116, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_117, (1536, ), (1, ))
    assert_size_stride(primals_118, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_119, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_120, (384, ), (1, ))
    assert_size_stride(primals_121, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_122, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_123, (384, ), (1, ))
    assert_size_stride(primals_124, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_125, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_126, (384, ), (1, ))
    assert_size_stride(primals_127, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_128, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_129, (1536, ), (1, ))
    assert_size_stride(primals_130, (1536, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_131, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_132, (1536, ), (1, ))
    assert_size_stride(primals_133, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_134, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_135, (384, ), (1, ))
    assert_size_stride(primals_136, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_137, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_138, (384, ), (1, ))
    assert_size_stride(primals_139, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_140, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_141, (384, ), (1, ))
    assert_size_stride(primals_142, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_143, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_144, (1536, ), (1, ))
    assert_size_stride(primals_145, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_146, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_147, (384, ), (1, ))
    assert_size_stride(primals_148, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_149, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_150, (384, ), (1, ))
    assert_size_stride(primals_151, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_152, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_153, (384, ), (1, ))
    assert_size_stride(primals_154, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_155, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_156, (1536, ), (1, ))
    assert_size_stride(primals_157, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_158, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_159, (384, ), (1, ))
    assert_size_stride(primals_160, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_161, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_162, (384, ), (1, ))
    assert_size_stride(primals_163, (384, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_164, (384, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_165, (384, ), (1, ))
    assert_size_stride(primals_166, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_167, (1536, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_168, (1536, ), (1, ))
    assert_size_stride(primals_169, (2304, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_170, (2304, 1, 1, 1), (1, 1, 1, 1))
    assert_size_stride(primals_171, (2304, ), (1, ))
    assert_size_stride(primals_172, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_173, (64, ), (1, ))
    assert_size_stride(primals_174, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_175, (256, ), (1, ))
    assert_size_stride(primals_176, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_177, (128, ), (1, ))
    assert_size_stride(primals_178, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_179, (512, ), (1, ))
    assert_size_stride(primals_180, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_181, (128, ), (1, ))
    assert_size_stride(primals_182, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_183, (512, ), (1, ))
    assert_size_stride(primals_184, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_185, (384, ), (1, ))
    assert_size_stride(primals_186, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_187, (1536, ), (1, ))
    assert_size_stride(primals_188, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_189, (384, ), (1, ))
    assert_size_stride(primals_190, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_191, (1536, ), (1, ))
    assert_size_stride(primals_192, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_193, (384, ), (1, ))
    assert_size_stride(primals_194, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_195, (1536, ), (1, ))
    assert_size_stride(primals_196, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_197, (384, ), (1, ))
    assert_size_stride(primals_198, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_199, (1536, ), (1, ))
    assert_size_stride(primals_200, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_201, (384, ), (1, ))
    assert_size_stride(primals_202, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_203, (1536, ), (1, ))
    assert_size_stride(primals_204, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_205, (384, ), (1, ))
    assert_size_stride(primals_206, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_207, (1536, ), (1, ))
    assert_size_stride(primals_208, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_209, (384, ), (1, ))
    assert_size_stride(primals_210, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_211, (1536, ), (1, ))
    assert_size_stride(primals_212, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_213, (384, ), (1, ))
    assert_size_stride(primals_214, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_215, (1536, ), (1, ))
    assert_size_stride(primals_216, (384, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_217, (384, ), (1, ))
    assert_size_stride(primals_218, (1536, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_219, (1536, ), (1, ))
    assert_size_stride(primals_220, (1000, 2304), (2304, 1))
    assert_size_stride(primals_221, (1000, ), (1, ))
    assert_size_stride(primals_222, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((1, 16, 1), (16, 1, 16), device='cuda', dtype=torch.float32)
        buf4 = empty((16, 3, 3, 3), device='cuda', dtype=torch.float32)
        buf3 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm, weight], Original ATen: [aten._native_batch_norm_legit, aten.view]
        stream0 = get_cuda_stream(0)
        triton_per_fused__native_batch_norm_legit_view_0.run(primals_1, primals_2, buf0, buf4, buf3, 16, 27, grid=grid(16), stream=stream0)
        # Source Nodes: [conv2d], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(primals_222, buf4, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (8, 16, 112, 112), (200704, 12544, 112, 1))
        buf6 = buf5; del buf5  # reuse
        buf7 = empty((8, 16, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv2d, l__mod___stem_act2], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_1.run(buf6, primals_3, buf7, 1605632, grid=grid(1605632), stream=stream0)
        del primals_3
        buf8 = empty_strided((1, 32, 1), (32, 1, 32), device='cuda', dtype=torch.float32)
        buf12 = empty((32, 16, 3, 3), device='cuda', dtype=torch.float32)
        buf11 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_1, weight_1], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_2.run(primals_4, primals_5, buf8, buf12, buf11, 32, 144, grid=grid(32), stream=stream0)
        # Source Nodes: [conv2d_1], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf7, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 32, 112, 112), (401408, 12544, 112, 1))
        buf14 = buf13; del buf13  # reuse
        buf15 = empty((8, 32, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv2d_1, l__mod___stem_act3], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_3.run(buf14, primals_6, buf15, 3211264, grid=grid(3211264), stream=stream0)
        del primals_6
        buf16 = empty_strided((1, 64, 1), (64, 1, 64), device='cuda', dtype=torch.float32)
        buf20 = empty((64, 32, 3, 3), device='cuda', dtype=torch.float32)
        buf19 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_2, weight_2], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_4.run(primals_7, primals_8, buf16, buf20, buf19, 64, 288, grid=grid(64), stream=stream0)
        # Source Nodes: [conv2d_2], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf15, buf20, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (8, 64, 112, 112), (802816, 12544, 112, 1))
        buf22 = buf21; del buf21  # reuse
        buf23 = empty((8, 64, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [conv2d_2, l__mod___stem_act4], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_5.run(buf22, primals_9, buf23, 6422528, grid=grid(6422528), stream=stream0)
        del primals_9
        buf24 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf28 = empty((128, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf27 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_3, weight_3], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_6.run(primals_10, primals_11, buf24, buf28, buf27, 128, 576, grid=grid(128), stream=stream0)
        # Source Nodes: [shortcut], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf23, buf28, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf30 = buf29; del buf29  # reuse
        buf31 = empty((8, 128, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act1, out, shortcut], Original ATen: [aten.convolution, aten.mul, aten.silu]
        triton_poi_fused_convolution_mul_silu_7.run(buf30, primals_12, buf31, 3211264, grid=grid(3211264), stream=stream0)
        del primals_12
        buf32 = empty_strided((1, 256, 1), (256, 1, 256), device='cuda', dtype=torch.float32)
        buf36 = empty((256, 128, 1, 1), device='cuda', dtype=torch.float32)
        buf35 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_4, weight_4], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_8.run(primals_13, primals_14, buf32, buf36, buf35, 256, 128, grid=grid(256), stream=stream0)
        # Source Nodes: [shortcut_1], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf31, buf36, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (8, 256, 56, 56), (802816, 3136, 56, 1))
        buf38 = empty_strided((1, 64, 1), (64, 1, 64), device='cuda', dtype=torch.float32)
        buf42 = empty((64, 128, 1, 1), device='cuda', dtype=torch.float32)
        buf41 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_5, weight_5], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_9.run(primals_16, primals_17, buf38, buf42, buf41, 64, 128, grid=grid(64), stream=stream0)
        # Source Nodes: [out_1], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf31, buf42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf44 = buf43; del buf43  # reuse
        buf45 = empty((8, 64, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act2, out_1], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_10.run(buf44, primals_18, buf45, 1605632, grid=grid(1605632), stream=stream0)
        del primals_18
        buf46 = empty_strided((1, 64, 1), (64, 1, 64), device='cuda', dtype=torch.float32)
        buf50 = empty((64, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf49 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_6, weight_6], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_11.run(primals_19, primals_20, buf46, buf50, buf49, 64, 576, grid=grid(64), stream=stream0)
        # Source Nodes: [out_2], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf45, buf50, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf52 = buf51; del buf51  # reuse
        buf53 = empty((8, 64, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act2b, out_2], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_10.run(buf52, primals_21, buf53, 1605632, grid=grid(1605632), stream=stream0)
        del primals_21
        buf54 = empty_strided((1, 64, 1), (64, 1, 64), device='cuda', dtype=torch.float32)
        buf58 = empty((64, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf57 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_7, weight_7], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_11.run(primals_22, primals_23, buf54, buf58, buf57, 64, 576, grid=grid(64), stream=stream0)
        # Source Nodes: [out_3], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf53, buf58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (8, 64, 56, 56), (200704, 3136, 56, 1))
        buf60 = buf59; del buf59  # reuse
        buf61 = empty((8, 64, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0_____0___act3, out_3], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_10.run(buf60, primals_24, buf61, 1605632, grid=grid(1605632), stream=stream0)
        del primals_24
        buf62 = empty_strided((1, 256, 1), (256, 1, 256), device='cuda', dtype=torch.float32)
        buf66 = empty((256, 64, 1, 1), device='cuda', dtype=torch.float32)
        buf65 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_8, weight_8], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_12.run(primals_25, primals_26, buf62, buf66, buf65, 256, 64, grid=grid(256), stream=stream0)
        # Source Nodes: [out_4], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf61, buf66, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (8, 256, 56, 56), (802816, 3136, 56, 1))
        buf68 = buf67; del buf67  # reuse
        buf69 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf70 = reinterpret_tensor(buf69, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf69  # reuse
        # Source Nodes: [out_4, x_se], Original ATen: [aten.convolution, aten.mean]
        triton_red_fused_convolution_mean_13.run(buf68, buf70, primals_27, 2048, 3136, grid=grid(2048), stream=stream0)
        del primals_27
        # Source Nodes: [x_se_1], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (8, 64, 1, 1), (64, 1, 1, 1))
        buf72 = buf71; del buf71  # reuse
        # Source Nodes: [x_se_1, x_se_2], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_14.run(buf72, primals_173, 512, grid=grid(512), stream=stream0)
        del primals_173
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, primals_174, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (8, 256, 1, 1), (256, 1, 1, 1))
        buf74 = buf73; del buf73  # reuse
        # Source Nodes: [x_se_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(buf74, primals_175, 2048, grid=grid(2048), stream=stream0)
        del primals_175
        buf75 = empty((8, 256, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act1, mul_10, mul_12, out_5, out_8, shortcut_1, shortcut_2, sigmoid], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_add_convolution_mul_sigmoid_silu_16.run(buf68, buf74, buf37, primals_15, buf75, 6422528, grid=grid(6422528), stream=stream0)
        buf76 = empty((8, 256, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___downsample_pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_17.run(buf75, buf76, 1605632, grid=grid(1605632), stream=stream0)
        buf77 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf81 = empty((512, 256, 1, 1), device='cuda', dtype=torch.float32)
        buf80 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_9, weight_9], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_18.run(primals_28, primals_29, buf77, buf81, buf80, 512, 256, grid=grid(512), stream=stream0)
        # Source Nodes: [shortcut_3], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf76, buf81, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (8, 512, 28, 28), (401408, 784, 28, 1))
        buf83 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf87 = empty((128, 256, 1, 1), device='cuda', dtype=torch.float32)
        buf86 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_10, weight_10], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_19.run(primals_31, primals_32, buf83, buf87, buf86, 128, 256, grid=grid(128), stream=stream0)
        # Source Nodes: [out_9], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf75, buf87, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf89 = buf88; del buf88  # reuse
        buf90 = empty((8, 128, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act2, out_9], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_20.run(buf89, primals_33, buf90, 3211264, grid=grid(3211264), stream=stream0)
        del primals_33
        buf91 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf95 = empty((128, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf94 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_11, weight_11], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_6.run(primals_34, primals_35, buf91, buf95, buf94, 128, 576, grid=grid(128), stream=stream0)
        # Source Nodes: [out_10], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf90, buf95, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf96, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf97 = buf96; del buf96  # reuse
        buf98 = empty((8, 128, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act2b, out_10], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_21.run(buf97, primals_36, buf98, 802816, grid=grid(802816), stream=stream0)
        del primals_36
        buf99 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf103 = empty((128, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf102 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_12, weight_12], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_6.run(primals_37, primals_38, buf99, buf103, buf102, 128, 576, grid=grid(128), stream=stream0)
        # Source Nodes: [out_11], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf98, buf103, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf104, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf105 = buf104; del buf104  # reuse
        buf106 = empty((8, 128, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act3, out_11], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_21.run(buf105, primals_39, buf106, 802816, grid=grid(802816), stream=stream0)
        del primals_39
        buf107 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf111 = empty((512, 128, 1, 1), device='cuda', dtype=torch.float32)
        buf110 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_13, weight_13], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_22.run(primals_40, primals_41, buf107, buf111, buf110, 512, 128, grid=grid(512), stream=stream0)
        # Source Nodes: [out_12], Original ATen: [aten.convolution]
        buf112 = extern_kernels.convolution(buf106, buf111, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 512, 28, 28), (401408, 784, 28, 1))
        buf113 = buf112; del buf112  # reuse
        buf114 = empty_strided((8, 512, 1, 1), (512, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf115 = reinterpret_tensor(buf114, (8, 512, 1, 1), (512, 1, 1, 1), 0); del buf114  # reuse
        # Source Nodes: [out_12, x_se_4], Original ATen: [aten.convolution, aten.mean]
        triton_per_fused_convolution_mean_23.run(buf113, buf115, primals_42, 4096, 784, grid=grid(4096), stream=stream0)
        del primals_42
        # Source Nodes: [x_se_5], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, primals_176, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (8, 128, 1, 1), (128, 1, 1, 1))
        buf117 = buf116; del buf116  # reuse
        # Source Nodes: [x_se_5, x_se_6], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_24.run(buf117, primals_177, 1024, grid=grid(1024), stream=stream0)
        del primals_177
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, primals_178, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (8, 512, 1, 1), (512, 1, 1, 1))
        buf119 = buf118; del buf118  # reuse
        # Source Nodes: [x_se_7], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf119, primals_179, 4096, grid=grid(4096), stream=stream0)
        del primals_179
        buf120 = empty((8, 512, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act1, mul_19, mul_21, out_13, out_16, shortcut_3, shortcut_4, sigmoid_1], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_add_convolution_mul_sigmoid_silu_26.run(buf113, buf119, buf82, primals_30, buf120, 3211264, grid=grid(3211264), stream=stream0)
        buf121 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf125 = empty((128, 512, 1, 1), device='cuda', dtype=torch.float32)
        buf124 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_14, weight_14], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_27.run(primals_43, primals_44, buf121, buf125, buf124, 128, 512, grid=grid(128), stream=stream0)
        # Source Nodes: [out_17], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf120, buf125, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf127 = buf126; del buf126  # reuse
        buf128 = empty((8, 128, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act2, out_17], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_21.run(buf127, primals_45, buf128, 802816, grid=grid(802816), stream=stream0)
        del primals_45
        buf129 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf133 = empty((128, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf132 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_15, weight_15], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_6.run(primals_46, primals_47, buf129, buf133, buf132, 128, 576, grid=grid(128), stream=stream0)
        # Source Nodes: [out_18], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf128, buf133, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf134, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf135 = buf134; del buf134  # reuse
        buf136 = empty((8, 128, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act2b, out_18], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_21.run(buf135, primals_48, buf136, 802816, grid=grid(802816), stream=stream0)
        del primals_48
        buf137 = empty_strided((1, 128, 1), (128, 1, 128), device='cuda', dtype=torch.float32)
        buf141 = empty((128, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf140 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_16, weight_16], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_6.run(primals_49, primals_50, buf137, buf141, buf140, 128, 576, grid=grid(128), stream=stream0)
        # Source Nodes: [out_19], Original ATen: [aten.convolution]
        buf142 = extern_kernels.convolution(buf136, buf141, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf142, (8, 128, 28, 28), (100352, 784, 28, 1))
        buf143 = buf142; del buf142  # reuse
        buf144 = empty((8, 128, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act3, out_19], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_21.run(buf143, primals_51, buf144, 802816, grid=grid(802816), stream=stream0)
        del primals_51
        buf145 = empty_strided((1, 512, 1), (512, 1, 512), device='cuda', dtype=torch.float32)
        buf149 = empty((512, 128, 1, 1), device='cuda', dtype=torch.float32)
        buf148 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_17, weight_17], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_22.run(primals_52, primals_53, buf145, buf149, buf148, 512, 128, grid=grid(512), stream=stream0)
        # Source Nodes: [out_20], Original ATen: [aten.convolution]
        buf150 = extern_kernels.convolution(buf144, buf149, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (8, 512, 28, 28), (401408, 784, 28, 1))
        buf151 = buf150; del buf150  # reuse
        buf152 = empty_strided((8, 512, 1, 1), (512, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf153 = reinterpret_tensor(buf152, (8, 512, 1, 1), (512, 1, 1, 1), 0); del buf152  # reuse
        # Source Nodes: [out_20, x_se_8], Original ATen: [aten.convolution, aten.mean]
        triton_per_fused_convolution_mean_23.run(buf151, buf153, primals_54, 4096, 784, grid=grid(4096), stream=stream0)
        del primals_54
        # Source Nodes: [x_se_9], Original ATen: [aten.convolution]
        buf154 = extern_kernels.convolution(buf153, primals_180, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (8, 128, 1, 1), (128, 1, 1, 1))
        buf155 = buf154; del buf154  # reuse
        # Source Nodes: [x_se_10, x_se_9], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_24.run(buf155, primals_181, 1024, grid=grid(1024), stream=stream0)
        del primals_181
        # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, primals_182, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (8, 512, 1, 1), (512, 1, 1, 1))
        buf157 = buf156; del buf156  # reuse
        # Source Nodes: [x_se_11], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf157, primals_183, 4096, grid=grid(4096), stream=stream0)
        del primals_183
        buf159 = empty((8, 512, 28, 28), device='cuda', dtype=torch.float32)
        buf538 = empty((8, 512, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act1, mul_19, mul_21, mul_27, mul_29, out_13, out_21, out_24, shortcut_3, shortcut_4, shortcut_5, sigmoid_1, sigmoid_2], Original ATen: [aten.add, aten.convolution, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_convolution_fill_mul_sigmoid_silu_sub_28.run(buf151, buf157, buf113, buf119, buf82, primals_30, buf159, buf538, 3211264, grid=grid(3211264), stream=stream0)
        buf160 = empty((8, 512, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___downsample_pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_29.run(buf159, buf160, 802816, grid=grid(802816), stream=stream0)
        buf161 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cuda', dtype=torch.float32)
        buf165 = empty((1536, 512, 1, 1), device='cuda', dtype=torch.float32)
        buf164 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_18, weight_18], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_30.run(primals_55, primals_56, buf161, buf165, buf164, 1536, 512, grid=grid(1536), stream=stream0)
        # Source Nodes: [shortcut_6], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf160, buf165, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf167 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf171 = empty((384, 512, 1, 1), device='cuda', dtype=torch.float32)
        buf170 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_19, weight_19], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_31.run(primals_58, primals_59, buf167, buf171, buf170, 384, 512, grid=grid(384), stream=stream0)
        # Source Nodes: [out_25], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf159, buf171, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (8, 384, 28, 28), (301056, 784, 28, 1))
        buf173 = buf172; del buf172  # reuse
        buf174 = empty((8, 384, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act2, out_25], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_32.run(buf173, primals_60, buf174, 2408448, grid=grid(2408448), stream=stream0)
        del primals_60
        buf175 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf179 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf178 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_20, weight_20], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_33.run(primals_61, primals_62, buf175, buf179, buf178, 384, 576, grid=grid(384), stream=stream0)
        # Source Nodes: [out_26], Original ATen: [aten.convolution]
        buf180 = extern_kernels.convolution(buf174, buf179, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf180, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf181 = buf180; del buf180  # reuse
        buf182 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act2b, out_26], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_34.run(buf181, primals_63, buf182, 602112, grid=grid(602112), stream=stream0)
        del primals_63
        buf183 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf187 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf186 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_21, weight_21], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_33.run(primals_64, primals_65, buf183, buf187, buf186, 384, 576, grid=grid(384), stream=stream0)
        # Source Nodes: [out_27], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf182, buf187, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf188, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf189 = buf188; del buf188  # reuse
        buf190 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____0___act3, out_27], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_34.run(buf189, primals_66, buf190, 602112, grid=grid(602112), stream=stream0)
        del primals_66
        buf191 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cuda', dtype=torch.float32)
        buf195 = empty((1536, 384, 1, 1), device='cuda', dtype=torch.float32)
        buf194 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_22, weight_22], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_35.run(primals_67, primals_68, buf191, buf195, buf194, 1536, 384, grid=grid(1536), stream=stream0)
        # Source Nodes: [out_28], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf190, buf195, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf197 = buf196; del buf196  # reuse
        buf198 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cuda', dtype=torch.float32)
        buf199 = reinterpret_tensor(buf198, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf198  # reuse
        # Source Nodes: [out_28, x_se_12], Original ATen: [aten.convolution, aten.mean]
        triton_per_fused_convolution_mean_36.run(buf197, buf199, primals_69, 12288, 196, grid=grid(12288), stream=stream0)
        del primals_69
        # Source Nodes: [x_se_13], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf199, primals_184, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (8, 384, 1, 1), (384, 1, 1, 1))
        buf201 = buf200; del buf200  # reuse
        # Source Nodes: [x_se_13, x_se_14], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_37.run(buf201, primals_185, 3072, grid=grid(3072), stream=stream0)
        del primals_185
        # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
        buf202 = extern_kernels.convolution(buf201, primals_186, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (8, 1536, 1, 1), (1536, 1, 1, 1))
        buf203 = buf202; del buf202  # reuse
        # Source Nodes: [x_se_15], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf203, primals_187, 12288, grid=grid(12288), stream=stream0)
        del primals_187
        buf204 = empty((8, 1536, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____1___act1, mul_36, mul_38, out_29, out_32, shortcut_6, shortcut_7, sigmoid_3], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_add_convolution_mul_sigmoid_silu_39.run(buf197, buf203, buf166, primals_57, buf204, 2408448, grid=grid(2408448), stream=stream0)
        buf205 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf209 = empty((384, 1536, 1, 1), device='cuda', dtype=torch.float32)
        buf208 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_23, weight_23], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_40.run(primals_70, primals_71, buf205, buf209, buf208, 384, 1536, grid=grid(384), stream=stream0)
        # Source Nodes: [out_33], Original ATen: [aten.convolution]
        buf210 = extern_kernels.convolution(buf204, buf209, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf211 = buf210; del buf210  # reuse
        buf212 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____1___act2, out_33], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_34.run(buf211, primals_72, buf212, 602112, grid=grid(602112), stream=stream0)
        del primals_72
        buf213 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf217 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf216 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_24, weight_24], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_33.run(primals_73, primals_74, buf213, buf217, buf216, 384, 576, grid=grid(384), stream=stream0)
        # Source Nodes: [out_34], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(buf212, buf217, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf218, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf219 = buf218; del buf218  # reuse
        buf220 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____1___act2b, out_34], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_34.run(buf219, primals_75, buf220, 602112, grid=grid(602112), stream=stream0)
        del primals_75
        buf221 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf225 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf224 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_25, weight_25], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_33.run(primals_76, primals_77, buf221, buf225, buf224, 384, 576, grid=grid(384), stream=stream0)
        # Source Nodes: [out_35], Original ATen: [aten.convolution]
        buf226 = extern_kernels.convolution(buf220, buf225, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf226, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf227 = buf226; del buf226  # reuse
        buf228 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____1___act3, out_35], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_34.run(buf227, primals_78, buf228, 602112, grid=grid(602112), stream=stream0)
        del primals_78
        buf229 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cuda', dtype=torch.float32)
        buf233 = empty((1536, 384, 1, 1), device='cuda', dtype=torch.float32)
        buf232 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_26, weight_26], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_35.run(primals_79, primals_80, buf229, buf233, buf232, 1536, 384, grid=grid(1536), stream=stream0)
        # Source Nodes: [out_36], Original ATen: [aten.convolution]
        buf234 = extern_kernels.convolution(buf228, buf233, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf234, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf235 = buf234; del buf234  # reuse
        buf236 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cuda', dtype=torch.float32)
        buf237 = reinterpret_tensor(buf236, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf236  # reuse
        # Source Nodes: [out_36, x_se_16], Original ATen: [aten.convolution, aten.mean]
        triton_per_fused_convolution_mean_36.run(buf235, buf237, primals_81, 12288, 196, grid=grid(12288), stream=stream0)
        del primals_81
        # Source Nodes: [x_se_17], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf237, primals_188, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (8, 384, 1, 1), (384, 1, 1, 1))
        buf239 = buf238; del buf238  # reuse
        # Source Nodes: [x_se_17, x_se_18], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_37.run(buf239, primals_189, 3072, grid=grid(3072), stream=stream0)
        del primals_189
        # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
        buf240 = extern_kernels.convolution(buf239, primals_190, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf240, (8, 1536, 1, 1), (1536, 1, 1, 1))
        buf241 = buf240; del buf240  # reuse
        # Source Nodes: [x_se_19], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf241, primals_191, 12288, grid=grid(12288), stream=stream0)
        del primals_191
        buf242 = empty((8, 1536, 14, 14), device='cuda', dtype=torch.float32)
        buf243 = empty((8, 1536, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____2___act1, mul_36, mul_38, mul_44, mul_46, out_29, out_37, out_40, shortcut_6, shortcut_7, shortcut_8, sigmoid_3, sigmoid_4], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_add_convolution_mul_sigmoid_silu_41.run(buf235, buf241, buf197, buf203, buf166, primals_57, buf242, buf243, 2408448, grid=grid(2408448), stream=stream0)
        buf244 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf248 = empty((384, 1536, 1, 1), device='cuda', dtype=torch.float32)
        buf247 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_27, weight_27], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_40.run(primals_82, primals_83, buf244, buf248, buf247, 384, 1536, grid=grid(384), stream=stream0)
        # Source Nodes: [out_41], Original ATen: [aten.convolution]
        buf249 = extern_kernels.convolution(buf243, buf248, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf249, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf250 = buf249; del buf249  # reuse
        buf251 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____2___act2, out_41], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_34.run(buf250, primals_84, buf251, 602112, grid=grid(602112), stream=stream0)
        del primals_84
        buf252 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf256 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf255 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_28, weight_28], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_33.run(primals_85, primals_86, buf252, buf256, buf255, 384, 576, grid=grid(384), stream=stream0)
        # Source Nodes: [out_42], Original ATen: [aten.convolution]
        buf257 = extern_kernels.convolution(buf251, buf256, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf257, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf258 = buf257; del buf257  # reuse
        buf259 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____2___act2b, out_42], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_34.run(buf258, primals_87, buf259, 602112, grid=grid(602112), stream=stream0)
        del primals_87
        buf260 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf264 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf263 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_29, weight_29], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_33.run(primals_88, primals_89, buf260, buf264, buf263, 384, 576, grid=grid(384), stream=stream0)
        # Source Nodes: [out_43], Original ATen: [aten.convolution]
        buf265 = extern_kernels.convolution(buf259, buf264, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf265, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf266 = buf265; del buf265  # reuse
        buf267 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____2___act3, out_43], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_34.run(buf266, primals_90, buf267, 602112, grid=grid(602112), stream=stream0)
        del primals_90
        buf268 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cuda', dtype=torch.float32)
        buf272 = empty((1536, 384, 1, 1), device='cuda', dtype=torch.float32)
        buf271 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_30, weight_30], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_35.run(primals_91, primals_92, buf268, buf272, buf271, 1536, 384, grid=grid(1536), stream=stream0)
        # Source Nodes: [out_44], Original ATen: [aten.convolution]
        buf273 = extern_kernels.convolution(buf267, buf272, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf273, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf274 = buf273; del buf273  # reuse
        buf275 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cuda', dtype=torch.float32)
        buf276 = reinterpret_tensor(buf275, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf275  # reuse
        # Source Nodes: [out_44, x_se_20], Original ATen: [aten.convolution, aten.mean]
        triton_per_fused_convolution_mean_36.run(buf274, buf276, primals_93, 12288, 196, grid=grid(12288), stream=stream0)
        del primals_93
        # Source Nodes: [x_se_21], Original ATen: [aten.convolution]
        buf277 = extern_kernels.convolution(buf276, primals_192, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf277, (8, 384, 1, 1), (384, 1, 1, 1))
        buf278 = buf277; del buf277  # reuse
        # Source Nodes: [x_se_21, x_se_22], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_37.run(buf278, primals_193, 3072, grid=grid(3072), stream=stream0)
        del primals_193
        # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
        buf279 = extern_kernels.convolution(buf278, primals_194, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf279, (8, 1536, 1, 1), (1536, 1, 1, 1))
        buf280 = buf279; del buf279  # reuse
        # Source Nodes: [x_se_23], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf280, primals_195, 12288, grid=grid(12288), stream=stream0)
        del primals_195
        buf281 = empty((8, 1536, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____3___act1, mul_52, mul_54, out_45, out_48, shortcut_9, sigmoid_5], Original ATen: [aten.add, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_add_mul_sigmoid_silu_42.run(buf274, buf280, buf242, buf281, 2408448, grid=grid(2408448), stream=stream0)
        buf282 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf286 = empty((384, 1536, 1, 1), device='cuda', dtype=torch.float32)
        buf285 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_31, weight_31], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_40.run(primals_94, primals_95, buf282, buf286, buf285, 384, 1536, grid=grid(384), stream=stream0)
        # Source Nodes: [out_49], Original ATen: [aten.convolution]
        buf287 = extern_kernels.convolution(buf281, buf286, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf287, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf288 = buf287; del buf287  # reuse
        buf289 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____3___act2, out_49], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_34.run(buf288, primals_96, buf289, 602112, grid=grid(602112), stream=stream0)
        del primals_96
        buf290 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf294 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf293 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_32, weight_32], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_33.run(primals_97, primals_98, buf290, buf294, buf293, 384, 576, grid=grid(384), stream=stream0)
        # Source Nodes: [out_50], Original ATen: [aten.convolution]
        buf295 = extern_kernels.convolution(buf289, buf294, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf295, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf296 = buf295; del buf295  # reuse
        buf297 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____3___act2b, out_50], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_34.run(buf296, primals_99, buf297, 602112, grid=grid(602112), stream=stream0)
        del primals_99
        buf298 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf302 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf301 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_33, weight_33], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_33.run(primals_100, primals_101, buf298, buf302, buf301, 384, 576, grid=grid(384), stream=stream0)
        # Source Nodes: [out_51], Original ATen: [aten.convolution]
        buf303 = extern_kernels.convolution(buf297, buf302, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf303, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf304 = buf303; del buf303  # reuse
        buf305 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____3___act3, out_51], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_34.run(buf304, primals_102, buf305, 602112, grid=grid(602112), stream=stream0)
        del primals_102
        buf306 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cuda', dtype=torch.float32)
        buf310 = empty((1536, 384, 1, 1), device='cuda', dtype=torch.float32)
        buf309 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_34, weight_34], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_35.run(primals_103, primals_104, buf306, buf310, buf309, 1536, 384, grid=grid(1536), stream=stream0)
        # Source Nodes: [out_52], Original ATen: [aten.convolution]
        buf311 = extern_kernels.convolution(buf305, buf310, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf311, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf312 = buf311; del buf311  # reuse
        buf313 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cuda', dtype=torch.float32)
        buf314 = reinterpret_tensor(buf313, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf313  # reuse
        # Source Nodes: [out_52, x_se_24], Original ATen: [aten.convolution, aten.mean]
        triton_per_fused_convolution_mean_36.run(buf312, buf314, primals_105, 12288, 196, grid=grid(12288), stream=stream0)
        del primals_105
        # Source Nodes: [x_se_25], Original ATen: [aten.convolution]
        buf315 = extern_kernels.convolution(buf314, primals_196, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf315, (8, 384, 1, 1), (384, 1, 1, 1))
        buf316 = buf315; del buf315  # reuse
        # Source Nodes: [x_se_25, x_se_26], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_37.run(buf316, primals_197, 3072, grid=grid(3072), stream=stream0)
        del primals_197
        # Source Nodes: [x_se_27], Original ATen: [aten.convolution]
        buf317 = extern_kernels.convolution(buf316, primals_198, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf317, (8, 1536, 1, 1), (1536, 1, 1, 1))
        buf318 = buf317; del buf317  # reuse
        # Source Nodes: [x_se_27], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf318, primals_199, 12288, grid=grid(12288), stream=stream0)
        del primals_199
        buf319 = empty((8, 1536, 14, 14), device='cuda', dtype=torch.float32)
        buf320 = empty((8, 1536, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____4___act1, mul_52, mul_54, mul_60, mul_62, out_45, out_53, out_56, shortcut_10, shortcut_9, sigmoid_5, sigmoid_6], Original ATen: [aten.add, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_add_mul_sigmoid_silu_43.run(buf312, buf318, buf274, buf280, buf242, buf319, buf320, 2408448, grid=grid(2408448), stream=stream0)
        buf321 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf325 = empty((384, 1536, 1, 1), device='cuda', dtype=torch.float32)
        buf324 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_35, weight_35], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_40.run(primals_106, primals_107, buf321, buf325, buf324, 384, 1536, grid=grid(384), stream=stream0)
        # Source Nodes: [out_57], Original ATen: [aten.convolution]
        buf326 = extern_kernels.convolution(buf320, buf325, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf326, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf327 = buf326; del buf326  # reuse
        buf328 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____4___act2, out_57], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_34.run(buf327, primals_108, buf328, 602112, grid=grid(602112), stream=stream0)
        del primals_108
        buf329 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf333 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf332 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_36, weight_36], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_33.run(primals_109, primals_110, buf329, buf333, buf332, 384, 576, grid=grid(384), stream=stream0)
        # Source Nodes: [out_58], Original ATen: [aten.convolution]
        buf334 = extern_kernels.convolution(buf328, buf333, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf334, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf335 = buf334; del buf334  # reuse
        buf336 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____4___act2b, out_58], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_34.run(buf335, primals_111, buf336, 602112, grid=grid(602112), stream=stream0)
        del primals_111
        buf337 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf341 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf340 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_37, weight_37], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_33.run(primals_112, primals_113, buf337, buf341, buf340, 384, 576, grid=grid(384), stream=stream0)
        # Source Nodes: [out_59], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf336, buf341, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf342, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf343 = buf342; del buf342  # reuse
        buf344 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____4___act3, out_59], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_34.run(buf343, primals_114, buf344, 602112, grid=grid(602112), stream=stream0)
        del primals_114
        buf345 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cuda', dtype=torch.float32)
        buf349 = empty((1536, 384, 1, 1), device='cuda', dtype=torch.float32)
        buf348 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_38, weight_38], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_35.run(primals_115, primals_116, buf345, buf349, buf348, 1536, 384, grid=grid(1536), stream=stream0)
        # Source Nodes: [out_60], Original ATen: [aten.convolution]
        buf350 = extern_kernels.convolution(buf344, buf349, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf350, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf351 = buf350; del buf350  # reuse
        buf352 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cuda', dtype=torch.float32)
        buf353 = reinterpret_tensor(buf352, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf352  # reuse
        # Source Nodes: [out_60, x_se_28], Original ATen: [aten.convolution, aten.mean]
        triton_per_fused_convolution_mean_36.run(buf351, buf353, primals_117, 12288, 196, grid=grid(12288), stream=stream0)
        del primals_117
        # Source Nodes: [x_se_29], Original ATen: [aten.convolution]
        buf354 = extern_kernels.convolution(buf353, primals_200, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf354, (8, 384, 1, 1), (384, 1, 1, 1))
        buf355 = buf354; del buf354  # reuse
        # Source Nodes: [x_se_29, x_se_30], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_37.run(buf355, primals_201, 3072, grid=grid(3072), stream=stream0)
        del primals_201
        # Source Nodes: [x_se_31], Original ATen: [aten.convolution]
        buf356 = extern_kernels.convolution(buf355, primals_202, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf356, (8, 1536, 1, 1), (1536, 1, 1, 1))
        buf357 = buf356; del buf356  # reuse
        # Source Nodes: [x_se_31], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf357, primals_203, 12288, grid=grid(12288), stream=stream0)
        del primals_203
        buf358 = empty((8, 1536, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____5___act1, mul_68, mul_70, out_61, out_64, shortcut_11, sigmoid_7], Original ATen: [aten.add, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_add_mul_sigmoid_silu_44.run(buf351, buf357, buf319, buf358, 2408448, grid=grid(2408448), stream=stream0)
        buf359 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf363 = empty((384, 1536, 1, 1), device='cuda', dtype=torch.float32)
        buf362 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_39, weight_39], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_40.run(primals_118, primals_119, buf359, buf363, buf362, 384, 1536, grid=grid(384), stream=stream0)
        # Source Nodes: [out_65], Original ATen: [aten.convolution]
        buf364 = extern_kernels.convolution(buf358, buf363, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf364, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf365 = buf364; del buf364  # reuse
        buf366 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____5___act2, out_65], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_34.run(buf365, primals_120, buf366, 602112, grid=grid(602112), stream=stream0)
        del primals_120
        buf367 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf371 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf370 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_40, weight_40], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_33.run(primals_121, primals_122, buf367, buf371, buf370, 384, 576, grid=grid(384), stream=stream0)
        # Source Nodes: [out_66], Original ATen: [aten.convolution]
        buf372 = extern_kernels.convolution(buf366, buf371, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf372, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf373 = buf372; del buf372  # reuse
        buf374 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____5___act2b, out_66], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_34.run(buf373, primals_123, buf374, 602112, grid=grid(602112), stream=stream0)
        del primals_123
        buf375 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf379 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf378 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_41, weight_41], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_33.run(primals_124, primals_125, buf375, buf379, buf378, 384, 576, grid=grid(384), stream=stream0)
        # Source Nodes: [out_67], Original ATen: [aten.convolution]
        buf380 = extern_kernels.convolution(buf374, buf379, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf380, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf381 = buf380; del buf380  # reuse
        buf382 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____5___act3, out_67], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_34.run(buf381, primals_126, buf382, 602112, grid=grid(602112), stream=stream0)
        del primals_126
        buf383 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cuda', dtype=torch.float32)
        buf387 = empty((1536, 384, 1, 1), device='cuda', dtype=torch.float32)
        buf386 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_42, weight_42], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_35.run(primals_127, primals_128, buf383, buf387, buf386, 1536, 384, grid=grid(1536), stream=stream0)
        # Source Nodes: [out_68], Original ATen: [aten.convolution]
        buf388 = extern_kernels.convolution(buf382, buf387, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf388, (8, 1536, 14, 14), (301056, 196, 14, 1))
        buf389 = buf388; del buf388  # reuse
        buf390 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cuda', dtype=torch.float32)
        buf391 = reinterpret_tensor(buf390, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf390  # reuse
        # Source Nodes: [out_68, x_se_32], Original ATen: [aten.convolution, aten.mean]
        triton_per_fused_convolution_mean_36.run(buf389, buf391, primals_129, 12288, 196, grid=grid(12288), stream=stream0)
        del primals_129
        # Source Nodes: [x_se_33], Original ATen: [aten.convolution]
        buf392 = extern_kernels.convolution(buf391, primals_204, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf392, (8, 384, 1, 1), (384, 1, 1, 1))
        buf393 = buf392; del buf392  # reuse
        # Source Nodes: [x_se_33, x_se_34], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_37.run(buf393, primals_205, 3072, grid=grid(3072), stream=stream0)
        del primals_205
        # Source Nodes: [x_se_35], Original ATen: [aten.convolution]
        buf394 = extern_kernels.convolution(buf393, primals_206, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf394, (8, 1536, 1, 1), (1536, 1, 1, 1))
        buf395 = buf394; del buf394  # reuse
        # Source Nodes: [x_se_35], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf395, primals_207, 12288, grid=grid(12288), stream=stream0)
        del primals_207
        buf397 = empty((8, 1536, 14, 14), device='cuda', dtype=torch.float32)
        buf532 = empty((8, 1536, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____0___act1, mul_68, mul_70, mul_76, mul_78, out_61, out_69, out_72, shortcut_11, shortcut_12, sigmoid_7, sigmoid_8], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_45.run(buf389, buf395, buf351, buf357, buf319, buf397, buf532, 2408448, grid=grid(2408448), stream=stream0)
        buf398 = empty((8, 1536, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____0___downsample_pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_46.run(buf397, buf398, 602112, grid=grid(602112), stream=stream0)
        buf399 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cuda', dtype=torch.float32)
        buf403 = empty((1536, 1536, 1, 1), device='cuda', dtype=torch.float32)
        buf402 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_43, weight_43], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_47.run(primals_130, primals_131, buf399, buf403, buf402, 1536, 1536, grid=grid(1536), stream=stream0)
        # Source Nodes: [shortcut_13], Original ATen: [aten.convolution]
        buf404 = extern_kernels.convolution(buf398, buf403, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf404, (8, 1536, 7, 7), (75264, 49, 7, 1))
        buf405 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf409 = empty((384, 1536, 1, 1), device='cuda', dtype=torch.float32)
        buf408 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_44, weight_44], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_40.run(primals_133, primals_134, buf405, buf409, buf408, 384, 1536, grid=grid(384), stream=stream0)
        # Source Nodes: [out_73], Original ATen: [aten.convolution]
        buf410 = extern_kernels.convolution(buf397, buf409, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf410, (8, 384, 14, 14), (75264, 196, 14, 1))
        buf411 = buf410; del buf410  # reuse
        buf412 = empty((8, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____0___act2, out_73], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_34.run(buf411, primals_135, buf412, 602112, grid=grid(602112), stream=stream0)
        del primals_135
        buf413 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf417 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf416 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_45, weight_45], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_33.run(primals_136, primals_137, buf413, buf417, buf416, 384, 576, grid=grid(384), stream=stream0)
        # Source Nodes: [out_74], Original ATen: [aten.convolution]
        buf418 = extern_kernels.convolution(buf412, buf417, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf418, (8, 384, 7, 7), (18816, 49, 7, 1))
        buf419 = buf418; del buf418  # reuse
        buf420 = empty((8, 384, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____0___act2b, out_74], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_48.run(buf419, primals_138, buf420, 150528, grid=grid(150528), stream=stream0)
        del primals_138
        buf421 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf425 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf424 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_46, weight_46], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_33.run(primals_139, primals_140, buf421, buf425, buf424, 384, 576, grid=grid(384), stream=stream0)
        # Source Nodes: [out_75], Original ATen: [aten.convolution]
        buf426 = extern_kernels.convolution(buf420, buf425, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf426, (8, 384, 7, 7), (18816, 49, 7, 1))
        buf427 = buf426; del buf426  # reuse
        buf428 = empty((8, 384, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____0___act3, out_75], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_48.run(buf427, primals_141, buf428, 150528, grid=grid(150528), stream=stream0)
        del primals_141
        buf429 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cuda', dtype=torch.float32)
        buf433 = empty((1536, 384, 1, 1), device='cuda', dtype=torch.float32)
        buf432 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_47, weight_47], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_35.run(primals_142, primals_143, buf429, buf433, buf432, 1536, 384, grid=grid(1536), stream=stream0)
        # Source Nodes: [out_76], Original ATen: [aten.convolution]
        buf434 = extern_kernels.convolution(buf428, buf433, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf434, (8, 1536, 7, 7), (75264, 49, 7, 1))
        buf435 = buf434; del buf434  # reuse
        buf436 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cuda', dtype=torch.float32)
        buf437 = reinterpret_tensor(buf436, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf436  # reuse
        # Source Nodes: [out_76, x_se_36], Original ATen: [aten.convolution, aten.mean]
        triton_per_fused_convolution_mean_49.run(buf435, buf437, primals_144, 12288, 49, grid=grid(12288), stream=stream0)
        del primals_144
        # Source Nodes: [x_se_37], Original ATen: [aten.convolution]
        buf438 = extern_kernels.convolution(buf437, primals_208, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf438, (8, 384, 1, 1), (384, 1, 1, 1))
        buf439 = buf438; del buf438  # reuse
        # Source Nodes: [x_se_37, x_se_38], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_37.run(buf439, primals_209, 3072, grid=grid(3072), stream=stream0)
        del primals_209
        # Source Nodes: [x_se_39], Original ATen: [aten.convolution]
        buf440 = extern_kernels.convolution(buf439, primals_210, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf440, (8, 1536, 1, 1), (1536, 1, 1, 1))
        buf441 = buf440; del buf440  # reuse
        # Source Nodes: [x_se_39], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf441, primals_211, 12288, grid=grid(12288), stream=stream0)
        del primals_211
        buf442 = empty((8, 1536, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____1___act1, mul_85, mul_87, out_77, out_80, shortcut_13, shortcut_14, sigmoid_9], Original ATen: [aten.add, aten.convolution, aten.mul, aten.sigmoid, aten.silu]
        triton_poi_fused_add_convolution_mul_sigmoid_silu_50.run(buf435, buf441, buf404, primals_132, buf442, 602112, grid=grid(602112), stream=stream0)
        buf443 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf447 = empty((384, 1536, 1, 1), device='cuda', dtype=torch.float32)
        buf446 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_48, weight_48], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_40.run(primals_145, primals_146, buf443, buf447, buf446, 384, 1536, grid=grid(384), stream=stream0)
        # Source Nodes: [out_81], Original ATen: [aten.convolution]
        buf448 = extern_kernels.convolution(buf442, buf447, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf448, (8, 384, 7, 7), (18816, 49, 7, 1))
        buf449 = buf448; del buf448  # reuse
        buf450 = empty((8, 384, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____1___act2, out_81], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_48.run(buf449, primals_147, buf450, 150528, grid=grid(150528), stream=stream0)
        del primals_147
        buf451 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf455 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf454 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_49, weight_49], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_33.run(primals_148, primals_149, buf451, buf455, buf454, 384, 576, grid=grid(384), stream=stream0)
        # Source Nodes: [out_82], Original ATen: [aten.convolution]
        buf456 = extern_kernels.convolution(buf450, buf455, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf456, (8, 384, 7, 7), (18816, 49, 7, 1))
        buf457 = buf456; del buf456  # reuse
        buf458 = empty((8, 384, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____1___act2b, out_82], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_48.run(buf457, primals_150, buf458, 150528, grid=grid(150528), stream=stream0)
        del primals_150
        buf459 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf463 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf462 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_50, weight_50], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_33.run(primals_151, primals_152, buf459, buf463, buf462, 384, 576, grid=grid(384), stream=stream0)
        # Source Nodes: [out_83], Original ATen: [aten.convolution]
        buf464 = extern_kernels.convolution(buf458, buf463, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf464, (8, 384, 7, 7), (18816, 49, 7, 1))
        buf465 = buf464; del buf464  # reuse
        buf466 = empty((8, 384, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____1___act3, out_83], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_48.run(buf465, primals_153, buf466, 150528, grid=grid(150528), stream=stream0)
        del primals_153
        buf467 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cuda', dtype=torch.float32)
        buf471 = empty((1536, 384, 1, 1), device='cuda', dtype=torch.float32)
        buf470 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_51, weight_51], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_35.run(primals_154, primals_155, buf467, buf471, buf470, 1536, 384, grid=grid(1536), stream=stream0)
        # Source Nodes: [out_84], Original ATen: [aten.convolution]
        buf472 = extern_kernels.convolution(buf466, buf471, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf472, (8, 1536, 7, 7), (75264, 49, 7, 1))
        buf473 = buf472; del buf472  # reuse
        buf474 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cuda', dtype=torch.float32)
        buf475 = reinterpret_tensor(buf474, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf474  # reuse
        # Source Nodes: [out_84, x_se_40], Original ATen: [aten.convolution, aten.mean]
        triton_per_fused_convolution_mean_49.run(buf473, buf475, primals_156, 12288, 49, grid=grid(12288), stream=stream0)
        del primals_156
        # Source Nodes: [x_se_41], Original ATen: [aten.convolution]
        buf476 = extern_kernels.convolution(buf475, primals_212, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf476, (8, 384, 1, 1), (384, 1, 1, 1))
        buf477 = buf476; del buf476  # reuse
        # Source Nodes: [x_se_41, x_se_42], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_37.run(buf477, primals_213, 3072, grid=grid(3072), stream=stream0)
        del primals_213
        # Source Nodes: [x_se_43], Original ATen: [aten.convolution]
        buf478 = extern_kernels.convolution(buf477, primals_214, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf478, (8, 1536, 1, 1), (1536, 1, 1, 1))
        buf479 = buf478; del buf478  # reuse
        # Source Nodes: [x_se_43], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf479, primals_215, 12288, grid=grid(12288), stream=stream0)
        del primals_215
        buf480 = empty((8, 1536, 7, 7), device='cuda', dtype=torch.float32)
        buf531 = empty((8, 1536, 7, 7), device='cuda', dtype=torch.float32)
        buf481 = empty((8, 1536, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____1___act1, getattr_getattr_l__mod___stages___3_____2___act1, mul_85, mul_87, mul_93, mul_95, out_77, out_85, out_88, shortcut_13, shortcut_14, shortcut_15, sigmoid_10, sigmoid_9], Original ATen: [aten.add, aten.convolution, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_convolution_fill_mul_sigmoid_silu_sub_51.run(buf473, buf479, buf435, buf441, buf404, primals_132, buf480, buf531, buf481, 602112, grid=grid(602112), stream=stream0)
        del primals_132
        buf482 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf486 = empty((384, 1536, 1, 1), device='cuda', dtype=torch.float32)
        buf485 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_52, weight_52], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_40.run(primals_157, primals_158, buf482, buf486, buf485, 384, 1536, grid=grid(384), stream=stream0)
        # Source Nodes: [out_89], Original ATen: [aten.convolution]
        buf487 = extern_kernels.convolution(buf481, buf486, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf487, (8, 384, 7, 7), (18816, 49, 7, 1))
        buf488 = buf487; del buf487  # reuse
        buf489 = empty((8, 384, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____2___act2, out_89], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_48.run(buf488, primals_159, buf489, 150528, grid=grid(150528), stream=stream0)
        del primals_159
        buf490 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf494 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf493 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_53, weight_53], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_33.run(primals_160, primals_161, buf490, buf494, buf493, 384, 576, grid=grid(384), stream=stream0)
        # Source Nodes: [out_90], Original ATen: [aten.convolution]
        buf495 = extern_kernels.convolution(buf489, buf494, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf495, (8, 384, 7, 7), (18816, 49, 7, 1))
        buf496 = buf495; del buf495  # reuse
        buf497 = empty((8, 384, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____2___act2b, out_90], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_48.run(buf496, primals_162, buf497, 150528, grid=grid(150528), stream=stream0)
        del primals_162
        buf498 = empty_strided((1, 384, 1), (384, 1, 384), device='cuda', dtype=torch.float32)
        buf502 = empty((384, 64, 3, 3), device='cuda', dtype=torch.float32)
        buf501 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_54, weight_54], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_33.run(primals_163, primals_164, buf498, buf502, buf501, 384, 576, grid=grid(384), stream=stream0)
        # Source Nodes: [out_91], Original ATen: [aten.convolution]
        buf503 = extern_kernels.convolution(buf497, buf502, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=6, bias=None)
        assert_size_stride(buf503, (8, 384, 7, 7), (18816, 49, 7, 1))
        buf504 = buf503; del buf503  # reuse
        buf505 = empty((8, 384, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____2___act3, out_91], Original ATen: [aten.convolution, aten.silu]
        triton_poi_fused_convolution_silu_48.run(buf504, primals_165, buf505, 150528, grid=grid(150528), stream=stream0)
        del primals_165
        buf506 = empty_strided((1, 1536, 1), (1536, 1, 1536), device='cuda', dtype=torch.float32)
        buf510 = empty((1536, 384, 1, 1), device='cuda', dtype=torch.float32)
        buf509 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_55, weight_55], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_per_fused__native_batch_norm_legit_view_35.run(primals_166, primals_167, buf506, buf510, buf509, 1536, 384, grid=grid(1536), stream=stream0)
        # Source Nodes: [out_92], Original ATen: [aten.convolution]
        buf511 = extern_kernels.convolution(buf505, buf510, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf511, (8, 1536, 7, 7), (75264, 49, 7, 1))
        buf512 = buf511; del buf511  # reuse
        buf513 = empty_strided((8, 1536, 1, 1), (1536, 1, 12288, 12288), device='cuda', dtype=torch.float32)
        buf514 = reinterpret_tensor(buf513, (8, 1536, 1, 1), (1536, 1, 1, 1), 0); del buf513  # reuse
        # Source Nodes: [out_92, x_se_44], Original ATen: [aten.convolution, aten.mean]
        triton_per_fused_convolution_mean_49.run(buf512, buf514, primals_168, 12288, 49, grid=grid(12288), stream=stream0)
        del primals_168
        # Source Nodes: [x_se_45], Original ATen: [aten.convolution]
        buf515 = extern_kernels.convolution(buf514, primals_216, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf515, (8, 384, 1, 1), (384, 1, 1, 1))
        buf516 = buf515; del buf515  # reuse
        # Source Nodes: [x_se_45, x_se_46], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_37.run(buf516, primals_217, 3072, grid=grid(3072), stream=stream0)
        del primals_217
        # Source Nodes: [x_se_47], Original ATen: [aten.convolution]
        buf517 = extern_kernels.convolution(buf516, primals_218, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf517, (8, 1536, 1, 1), (1536, 1, 1, 1))
        buf518 = buf517; del buf517  # reuse
        # Source Nodes: [x_se_47], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_38.run(buf518, primals_219, 12288, grid=grid(12288), stream=stream0)
        del primals_219
        buf519 = buf404; del buf404  # reuse
        buf530 = empty((8, 1536, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___3_____2___act1, mul_101, mul_103, out_93, sigmoid_11, x_1], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_52.run(buf512, buf518, buf480, buf519, buf530, 602112, grid=grid(602112), stream=stream0)
        del buf480
        buf520 = empty_strided((1, 2304, 1), (2304, 1, 2304), device='cuda', dtype=torch.float32)
        buf524 = empty((2304, 1536, 1, 1), device='cuda', dtype=torch.float32)
        buf523 = empty((2304, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [batch_norm_56, weight_56], Original ATen: [aten._native_batch_norm_legit, aten.view]
        triton_red_fused__native_batch_norm_legit_view_53.run(primals_169, primals_170, buf520, buf524, buf523, 2304, 1536, grid=grid(2304), stream=stream0)
        # Source Nodes: [x_2], Original ATen: [aten.convolution]
        buf525 = extern_kernels.convolution(buf519, buf524, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf525, (8, 2304, 7, 7), (112896, 49, 7, 1))
        buf526 = buf525; del buf525  # reuse
        buf527 = empty_strided((8, 2304, 1, 1), (2304, 1, 18432, 18432), device='cuda', dtype=torch.float32)
        buf528 = reinterpret_tensor(buf527, (8, 2304), (2304, 1), 0); del buf527  # reuse
        # Source Nodes: [x_2, x_4, x_5, x_7], Original ATen: [aten.convolution, aten.mean, aten.silu, aten.view]
        triton_per_fused_convolution_mean_silu_view_54.run(buf526, buf528, primals_171, 18432, 49, grid=grid(18432), stream=stream0)
        del primals_171
        buf529 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_9], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_221, buf528, reinterpret_tensor(primals_220, (2304, 1000), (1, 2304), 0), alpha=1, beta=1, out=buf529)
        del primals_221
        buf533 = empty((8, 1536, 14, 14), device='cuda', dtype=torch.float32)
        buf534 = empty((8, 1536, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____4___act1, getattr_getattr_l__mod___stages___2_____5___act1, mul_68, mul_70, out_61, shortcut_11, sigmoid_7], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_55.run(buf351, buf357, buf319, buf533, buf534, 2408448, grid=grid(2408448), stream=stream0)
        buf535 = buf319; del buf319  # reuse
        buf536 = empty((8, 1536, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____2___act1, getattr_getattr_l__mod___stages___2_____3___act1, mul_52, mul_54, out_45, shortcut_9, sigmoid_5], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_55.run(buf274, buf280, buf242, buf535, buf536, 2408448, grid=grid(2408448), stream=stream0)
        del buf242
        buf537 = buf166; del buf166  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___2_____1___act1, mul_36, mul_38, out_29, shortcut_6, shortcut_7, sigmoid_3], Original ATen: [aten.add, aten.convolution, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_convolution_fill_mul_sigmoid_silu_sub_56.run(buf537, buf197, buf203, primals_57, 2408448, grid=grid(2408448), stream=stream0)
        del primals_57
        buf539 = buf82; del buf82  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____1___act1, mul_19, mul_21, out_13, shortcut_3, shortcut_4, sigmoid_1], Original ATen: [aten.add, aten.convolution, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_convolution_fill_mul_sigmoid_silu_sub_57.run(buf539, buf113, buf119, primals_30, 3211264, grid=grid(3211264), stream=stream0)
        del primals_30
        buf540 = buf37; del buf37  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1_____0___act1, mul_10, mul_12, out_5, shortcut_1, shortcut_2, sigmoid], Original ATen: [aten.add, aten.convolution, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_convolution_fill_mul_sigmoid_silu_sub_58.run(buf540, buf68, buf74, primals_15, 6422528, grid=grid(6422528), stream=stream0)
        del primals_15
        return (buf529, primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_174, primals_176, primals_178, primals_180, primals_182, primals_184, primals_186, primals_188, primals_190, primals_192, primals_194, primals_196, primals_198, primals_200, primals_202, primals_204, primals_206, primals_208, primals_210, primals_212, primals_214, primals_216, primals_218, primals_222, buf3, buf4, buf6, buf7, buf11, buf12, buf14, buf15, buf19, buf20, buf22, buf23, buf27, buf28, buf30, buf31, buf35, buf36, buf41, buf42, buf44, buf45, buf49, buf50, buf52, buf53, buf57, buf58, buf60, buf61, buf65, buf66, buf68, buf70, buf72, buf74, buf75, buf76, buf80, buf81, buf86, buf87, buf89, buf90, buf94, buf95, buf97, buf98, buf102, buf103, buf105, buf106, buf110, buf111, buf113, buf115, buf117, buf119, buf120, buf124, buf125, buf127, buf128, buf132, buf133, buf135, buf136, buf140, buf141, buf143, buf144, buf148, buf149, buf151, buf153, buf155, buf157, buf159, buf160, buf164, buf165, buf170, buf171, buf173, buf174, buf178, buf179, buf181, buf182, buf186, buf187, buf189, buf190, buf194, buf195, buf197, buf199, buf201, buf203, buf204, buf208, buf209, buf211, buf212, buf216, buf217, buf219, buf220, buf224, buf225, buf227, buf228, buf232, buf233, buf235, buf237, buf239, buf241, buf243, buf247, buf248, buf250, buf251, buf255, buf256, buf258, buf259, buf263, buf264, buf266, buf267, buf271, buf272, buf274, buf276, buf278, buf280, buf281, buf285, buf286, buf288, buf289, buf293, buf294, buf296, buf297, buf301, buf302, buf304, buf305, buf309, buf310, buf312, buf314, buf316, buf318, buf320, buf324, buf325, buf327, buf328, buf332, buf333, buf335, buf336, buf340, buf341, buf343, buf344, buf348, buf349, buf351, buf353, buf355, buf357, buf358, buf362, buf363, buf365, buf366, buf370, buf371, buf373, buf374, buf378, buf379, buf381, buf382, buf386, buf387, buf389, buf391, buf393, buf395, buf397, buf398, buf402, buf403, buf408, buf409, buf411, buf412, buf416, buf417, buf419, buf420, buf424, buf425, buf427, buf428, buf432, buf433, buf435, buf437, buf439, buf441, buf442, buf446, buf447, buf449, buf450, buf454, buf455, buf457, buf458, buf462, buf463, buf465, buf466, buf470, buf471, buf473, buf475, buf477, buf479, buf481, buf485, buf486, buf488, buf489, buf493, buf494, buf496, buf497, buf501, buf502, buf504, buf505, buf509, buf510, buf512, buf514, buf516, buf518, buf519, buf523, buf524, buf526, buf528, reinterpret_tensor(primals_220, (1000, 2304), (2304, 1), 0), reinterpret_tensor(buf520, (1, 2304, 1), (2304, 1, 1), 0), reinterpret_tensor(buf506, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf498, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf490, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf482, (1, 384, 1), (384, 1, 1), 0), buf530, reinterpret_tensor(buf467, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf459, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf451, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf443, (1, 384, 1), (384, 1, 1), 0), buf531, reinterpret_tensor(buf429, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf421, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf413, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf405, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf399, (1, 1536, 1), (1536, 1, 1), 0), buf532, reinterpret_tensor(buf383, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf375, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf367, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf359, (1, 384, 1), (384, 1, 1), 0), buf533, reinterpret_tensor(buf345, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf337, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf329, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf321, (1, 384, 1), (384, 1, 1), 0), buf534, reinterpret_tensor(buf306, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf298, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf290, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf282, (1, 384, 1), (384, 1, 1), 0), buf535, reinterpret_tensor(buf268, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf260, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf252, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf244, (1, 384, 1), (384, 1, 1), 0), buf536, reinterpret_tensor(buf229, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf221, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf213, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf205, (1, 384, 1), (384, 1, 1), 0), buf537, reinterpret_tensor(buf191, (1, 1536, 1), (1536, 1, 1), 0), reinterpret_tensor(buf183, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf175, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf167, (1, 384, 1), (384, 1, 1), 0), reinterpret_tensor(buf161, (1, 1536, 1), (1536, 1, 1), 0), buf538, reinterpret_tensor(buf145, (1, 512, 1), (512, 1, 1), 0), reinterpret_tensor(buf137, (1, 128, 1), (128, 1, 1), 0), reinterpret_tensor(buf129, (1, 128, 1), (128, 1, 1), 0), reinterpret_tensor(buf121, (1, 128, 1), (128, 1, 1), 0), buf539, reinterpret_tensor(buf107, (1, 512, 1), (512, 1, 1), 0), reinterpret_tensor(buf99, (1, 128, 1), (128, 1, 1), 0), reinterpret_tensor(buf91, (1, 128, 1), (128, 1, 1), 0), reinterpret_tensor(buf83, (1, 128, 1), (128, 1, 1), 0), reinterpret_tensor(buf77, (1, 512, 1), (512, 1, 1), 0), buf540, reinterpret_tensor(buf62, (1, 256, 1), (256, 1, 1), 0), reinterpret_tensor(buf54, (1, 64, 1), (64, 1, 1), 0), reinterpret_tensor(buf46, (1, 64, 1), (64, 1, 1), 0), reinterpret_tensor(buf38, (1, 64, 1), (64, 1, 1), 0), reinterpret_tensor(buf32, (1, 256, 1), (256, 1, 1), 0), reinterpret_tensor(buf24, (1, 128, 1), (128, 1, 1), 0), reinterpret_tensor(buf16, (1, 64, 1), (64, 1, 1), 0), reinterpret_tensor(buf8, (1, 32, 1), (32, 1, 1), 0), reinterpret_tensor(buf0, (1, 16, 1), (16, 1, 1), 0), )


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
    primals_16 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((256, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((512, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((1536, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((384, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((1536, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((384, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((384, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((1536, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((2304, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((2304, 1, 1, 1), (1, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((384, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((1536, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((1000, 2304), (2304, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('nfnet_l0', benchmark_compiled_module)
