
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


# kernel path: /tmp/torchinductor_youkaichao/5h/c5hypntpo4wb7k72ppyjttf7xvgpskg6curghx4r3anlmvtuq2mi.py
# Source Nodes: [l__mod___stem_conv1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___stem_conv1_bn => var_mean
triton_red_fused__native_batch_norm_legit_functional_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 208
    rnumel = 7720
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 13
    x1 = (xindex // 13)
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (7720*x0)
        tmp1 = tl.full([1, 1], 100352, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + ((12544*x1) + (200704*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/rb/crbagzdbyuymi6e3c7fv4zh2dfakjexo2glilhexay5s7hdzdmby.py
# Source Nodes: [l__mod___stem_conv1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___stem_conv1_bn => add_1, add_2, add_3, mul_1, mul_2, mul_3, mul_4, mul_5, rsqrt, squeeze_1, var_mean
triton_per_fused__native_batch_norm_legit_functional_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_1', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (r1 + (13*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (13*x0)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 100352.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.00000996502277
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4x/c4xut74sbdd4bi737jzugmy3poij37wvutwkpbqyjiierblxlzgh.py
# Source Nodes: [l__mod___stem_act1, l__mod___stem_conv1_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
# l__mod___stem_act1 => add_5, clamp_max, clamp_min, div, mul_7
# l__mod___stem_conv1_bn => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
triton_poi_fused__native_batch_norm_legit_functional_hardswish_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 16
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 100352.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 3.0
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tmp20 / tmp18
    tl.store(out_ptr0 + (x3), tmp13, None)
    tl.store(out_ptr1 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cw/ccw3b66eu7gkpbruabvcv4lyly2grpqsi2cfnbz66z6vp5pyfk3c.py
# Source Nodes: [l__mod___stem_conv2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___stem_conv2_bn => var_mean_1
triton_red_fused__native_batch_norm_legit_functional_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (100352*(r2 // 3136)) + (200704*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mo/cmoxdooqit3hldrg5chq6yzctewznidgwsaew74h5tdaevvs4gju.py
# Source Nodes: [l__mod___stem_conv2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___stem_conv2_bn => add_7, add_8, add_9, mul_10, mul_11, mul_12, mul_13, mul_9, rsqrt_1, squeeze_4, var_mean_1
triton_per_fused__native_batch_norm_legit_functional_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_4', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 25088.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000398612827361
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/av/cavpl3n6mwfn7a3lwu63yqa7hji5qucwjel2vhx77crekbcwbs4f.py
# Source Nodes: [l__mod___stem_act2, l__mod___stem_conv2_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
# l__mod___stem_act2 => add_11, clamp_max_1, clamp_min_1, div_1, mul_15
# l__mod___stem_conv2_bn => add_10, add_7, mul_14, mul_8, rsqrt_1, sub_1, var_mean_1
triton_poi_fused__native_batch_norm_legit_functional_hardswish_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 32
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 25088.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 3.0
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tmp20 / tmp18
    tl.store(out_ptr0 + (x3), tmp13, None)
    tl.store(out_ptr1 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2u/c2uokvndo56q4hbaiim6poqwwl35aeucoaffzud3t6cwdikajvib.py
# Source Nodes: [l__mod___stem_conv3_bn], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___stem_conv3_bn => add_13, add_14, add_15, mul_17, mul_18, mul_19, mul_20, mul_21, rsqrt_2, squeeze_7, var_mean_2
triton_red_fused__native_batch_norm_legit_functional_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_6', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 6272
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
        r1 = rindex % 784
        r2 = (rindex // 784)
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (50176*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr1 + (x0), tmp3, xmask)
    tmp12 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = 6272.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = 0.1
    tmp11 = tmp2 * tmp10
    tmp13 = 0.9
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = 1.0001594642002871
    tmp17 = tmp6 * tmp16
    tmp18 = tmp17 * tmp10
    tmp20 = tmp19 * tmp13
    tmp21 = tmp18 + tmp20
    tl.store(out_ptr2 + (x0), tmp9, xmask)
    tl.store(out_ptr4 + (x0), tmp15, xmask)
    tl.store(out_ptr6 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oa/coadlutzxtmronkmrssjmxrcsjf7kpqs77rdddmqcl6huhxbdtgs.py
# Source Nodes: [l__mod___stem_act3, l__mod___stem_conv3_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
# l__mod___stem_act3 => add_17, clamp_max_2, clamp_min_2, div_2, mul_23
# l__mod___stem_conv3_bn => add_13, add_16, mul_16, mul_22, rsqrt_2, sub_2, var_mean_2
triton_poi_fused__native_batch_norm_legit_functional_hardswish_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 64
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 3.0
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tmp20 / tmp18
    tl.store(out_ptr0 + (x3), tmp13, None)
    tl.store(out_ptr1 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/la/clar64wsemmpm2b7pu7usdy5mehfda3f5qyqteo2rxcqttcsxwsz.py
# Source Nodes: [x], Original ATen: [aten._native_batch_norm_legit_functional]
# x => add_19, add_20, add_21, mul_25, mul_26, mul_27, mul_28, mul_29, rsqrt_3, squeeze_10, var_mean_3
triton_red_fused__native_batch_norm_legit_functional_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_8', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 1568
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
        r1 = rindex % 196
        r2 = (rindex // 196)
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (25088*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr1 + (x0), tmp3, xmask)
    tmp12 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = 1568.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = 0.1
    tmp11 = tmp2 * tmp10
    tmp13 = 0.9
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = 1.0006381620931717
    tmp17 = tmp6 * tmp16
    tmp18 = tmp17 * tmp10
    tmp20 = tmp19 * tmp13
    tmp21 = tmp18 + tmp20
    tl.store(out_ptr2 + (x0), tmp9, xmask)
    tl.store(out_ptr4 + (x0), tmp15, xmask)
    tl.store(out_ptr6 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sa/csazu2lxq2dtcz6eqbz6megkfn5v7ds7za3unuxpx3am5cb7zei6.py
# Source Nodes: [x], Original ATen: [aten._native_batch_norm_legit_functional]
# x => add_19, add_22, mul_24, mul_30, rsqrt_3, sub_3, var_mean_3
triton_poi_fused__native_batch_norm_legit_functional_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fo/cfojjsthuyxw4wsfudrr5ulpena5x5vxqpmbkprl5jgzrw2jl7fb.py
# Source Nodes: [x_3], Original ATen: [aten._unsafe_view, aten.clone]
# x_3 => clone, view_1
triton_poi_fused__unsafe_view_clone_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + ((196*x1) + (25088*(y0 // 196)) + (y0 % 196)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x1 + (128*y0)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sg/csgh3cqme6ydmat6hjz7dhk7hgbao37ggsadstshqgu5a5mcuz2a.py
# Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___attn_qkv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___stages___0___blocks___0___attn_qkv_bn => var_mean_4
triton_red_fused__native_batch_norm_legit_functional_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_11', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 3328
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 256)
    x0 = xindex % 256
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (256*r2) + (30976*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dj/cdjy6py3lr5fkewk5dpg2424gd7h6iacc4vp3hulr23rbwxxutbz.py
# Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___attn_qkv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___stages___0___blocks___0___attn_qkv_bn => add_24, add_25, add_26, mul_32, mul_33, mul_34, mul_35, mul_36, rsqrt_4, squeeze_13, var_mean_4
triton_per_fused__native_batch_norm_legit_functional_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_12', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 1568.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0006381620931717
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d3/cd3p26x5otcz5sphia4rmgodn3jvyslhuev2wbizgn35pjrdokkq.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_1
triton_poi_fused_clone_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 196
    x2 = (xindex // 3136) % 4
    x3 = (xindex // 12544)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (256*x1) + (50176*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x4), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/an/canffctwu72xtdi5tjjlxoqcstkutjvvw2olon2ryvck7jpwbtgf.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_2
triton_poi_fused_clone_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16) % 4
    y2 = (yindex // 64)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (16 + y0 + (64*y1) + (256*x3) + (50176*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (16 + y0 + (64*y1)), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (16 + y0 + (64*y1)), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (16 + y0 + (64*y1)), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (16 + y0 + (64*y1)), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3 + (196*y4)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ld/cldukkpulra7gdpw3zxrx5ap333eeynt2widfazdbqurgzla2mfw.py
# Source Nodes: [attn, attn_1, getitem_3, mul], Original ATen: [aten._softmax, aten.add, aten.index, aten.mul]
# attn => add_28
# attn_1 => amax, div_3, exp, sub_5, sum_1
# getitem_3 => index
# mul => mul_38
triton_per_fused__softmax_add_index_mul_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_index_mul_15', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 196
    x1 = (xindex // 196) % 4
    tmp0 = tl.load(in_ptr0 + (r3 + (196*x4)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r3 + (196*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 + 196
    tmp5 = tmp3 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp3)
    tl.device_assert(((0 <= tmp6) & (tmp6 < 196)) | ~(rmask & xmask), "index out of bounds: 0 <= tmp6 < 196")
    tmp7 = tl.load(in_ptr2 + (tmp6 + (196*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp2 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, float("-inf"))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp8 - tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp14 / tmp18
    tl.store(out_ptr2 + (r3 + (196*x4)), tmp19, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w6/cw6ibu6l7uvkowm3tcxjrdzp2ux3bnad3cp3ezbybtnffkagt3ve.py
# Source Nodes: [matmul_1], Original ATen: [aten.clone]
# matmul_1 => clone_3
triton_poi_fused_clone_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 196
    x2 = (xindex // 6272) % 4
    x3 = (xindex // 25088)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (32 + x0 + (64*x2) + (256*x1) + (50176*x3)), None)
    tmp1 = tl.load(in_ptr1 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x4), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dk/cdkfdlvl2ilnd26rl2iwkehhn4pffqv55zr6kquyjpf6cuxghn66.py
# Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___attn_proj_act, x_4, x_5], Original ATen: [aten._unsafe_view, aten.clone, aten.hardswish, aten.view]
# getattr_getattr_l__mod___stages___0___blocks___0___attn_proj_act => add_29, clamp_max_3, clamp_min_3, div_4, mul_39
# x_4 => clone_4, view_12
# x_5 => view_13
triton_poi_fused__unsafe_view_clone_hardswish_view_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_hardswish_view_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 196
    x2 = (xindex // 25088)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*x1) + (6272*(x0 // 32)) + (25088*x2) + (x0 % 32)), None)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tl.store(out_ptr0 + (x3), tmp0, None)
    tl.store(out_ptr1 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pr/cprswdbphbyuuk63zvot3ywgws3lhquu6ioxaomfx4fdrd7euoq6.py
# Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___attn_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___stages___0___blocks___0___attn_proj_ln_bn => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_18', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1664
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 128)
    x0 = xindex % 128
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
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
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gl/cgla3bz3ubf7fvo6v5dldpsfwmzhc6shywkl7glbt236f4ljmudo.py
# Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___attn_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_getattr_l__mod___stages___0___blocks___0___attn_proj_ln_bn => add_31, add_32, add_33, mul_41, mul_42, mul_43, mul_44, mul_45, rsqrt_5, squeeze_16, var_mean_5
triton_per_fused__native_batch_norm_legit_functional_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_19', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 1568.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0006381620931717
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6t/c6tsxer6ctip6lg4q2rmtlt2dg744yecink73umpjgoudur73jbk.py
# Source Nodes: [x_7, x_8], Original ATen: [aten._unsafe_view, aten.add, aten.clone]
# x_7 => add_35
# x_8 => clone_5, view_17
triton_poi_fused__unsafe_view_add_clone_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_clone_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
    xnumel = 128
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (25088*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (128*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1568.0
    tmp6 = tmp4 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp0 + tmp14
    tl.store(out_ptr0 + (x2 + (128*y3)), tmp15, xmask & ymask)
    tl.store(out_ptr1 + (x2 + (128*y3)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ld/cldp4bycupeltz53pp2bwz7ztk2pgrl6ocpkiuyob2qtbeleesax.py
# Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___mlp_ln1_bn, x_10, x_12, x_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish, aten.view]
# getattr_getattr_l__mod___stages___0___blocks___0___mlp_ln1_bn => add_37, add_40, mul_47, mul_53, rsqrt_6, sub_7, var_mean_6
# x_10 => add_41, clamp_max_4, clamp_min_4, div_5, mul_54
# x_12 => view_21
# x_9 => view_20
triton_poi_fused__native_batch_norm_legit_functional_hardswish_view_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_view_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 3.0
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tmp20 / tmp18
    tl.store(out_ptr0 + (x2), tmp13, None)
    tl.store(out_ptr1 + (x2), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bj/cbjmdcs3bf24awlma4ro4ci5iakmxe5kdm6wqskiyrvqa6yvdsng.py
# Source Nodes: [x_14, x_15], Original ATen: [aten._unsafe_view, aten.add, aten.clone]
# x_14 => add_47
# x_15 => clone_7, view_25
triton_poi_fused__unsafe_view_add_clone_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_add_clone_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp2 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 1568.0
    tmp6 = tmp4 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp0 + tmp14
    tl.store(in_out_ptr0 + (x2), tmp15, None)
    tl.store(out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fz/cfzd3eh4oluqftjglctgpatontiv4rmiffbrfqorvwte6ogmobwu.py
# Source Nodes: [getattr_l__mod___stages___1___downsample_attn_downsample_kv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_l__mod___stages___1___downsample_attn_downsample_kv_bn => var_mean_20
triton_red_fused__native_batch_norm_legit_functional_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_23', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8320
    rnumel = 121
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 640)
    x0 = xindex % 640
    tmp15_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp15_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = r2 + (121*x1)
        tmp1 = tl.full([1, 1], 1568, tl.int32)
        tmp2 = tmp0 < tmp1
        tmp3 = tl.load(in_ptr0 + (x0 + (640*r2) + (77440*x1)), rmask & tmp2 & xmask, eviction_policy='evict_first', other=0.0)
        tmp4 = tl.full(tmp3.shape, 0, tmp3.dtype)
        tmp5 = tl.where(tmp2, tmp3, tmp4)
        tmp6 = 0.0
        tmp7 = tl.full(tmp6.shape, 0, tmp6.dtype)
        tmp8 = tl.where(tmp2, tmp6, tmp7)
        tmp9 = 1.0
        tmp10 = tl.full(tmp9.shape, 0, tmp9.dtype)
        tmp11 = tl.where(tmp2, tmp9, tmp10)
        tmp12 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp13 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp14 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp15_mean_next, tmp15_m2_next, tmp15_weight_next = triton_helpers.welford_combine(
            tmp15_mean, tmp15_m2, tmp15_weight,
            tmp12, tmp13, tmp14
        )
        tmp15_mean = tl.where(rmask & xmask, tmp15_mean_next, tmp15_mean)
        tmp15_m2 = tl.where(rmask & xmask, tmp15_m2_next, tmp15_m2)
        tmp15_weight = tl.where(rmask & xmask, tmp15_weight_next, tmp15_weight)
    tmp15_tmp, tmp16_tmp, tmp17_tmp = triton_helpers.welford(
        tmp15_mean, tmp15_m2, tmp15_weight, 1
    )
    tmp15 = tmp15_tmp[:, None]
    tmp16 = tmp16_tmp[:, None]
    tmp17 = tmp17_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp16, xmask)
    tl.store(out_ptr2 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/he/cheida6ugjnknaureop77kchoh3t4uier4jxdwqlquwijzd67pmb.py
# Source Nodes: [getattr_l__mod___stages___1___downsample_attn_downsample_kv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_l__mod___stages___1___downsample_attn_downsample_kv_bn => add_124, add_125, add_126, mul_156, mul_157, mul_158, mul_159, mul_160, rsqrt_20, squeeze_61, var_mean_20
triton_per_fused__native_batch_norm_legit_functional_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_24', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (640*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (640*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 1568.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0006381620931717
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/33/c333mxudwszdjfn664r76xxvoqypiawftehxosr2k2tscegqp23v.py
# Source Nodes: [x_55], Original ATen: [aten.view]
# x_55 => view_104
triton_poi_fused_view_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*((x1 % 49) % 7)) + (3584*((x1 % 49) // 7)) + (25088*(x1 // 49))), xmask)
    tl.store(out_ptr0 + (x2), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wy/cwydzavmxyjpquzotxp4blal66k3qn3dilpoyrwnu4xhfofsgc7m.py
# Source Nodes: [getattr_l__mod___stages___1___downsample_attn_downsample_q_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_l__mod___stages___1___downsample_attn_downsample_q_ln_bn => var_mean_21
triton_red_fused__native_batch_norm_legit_functional_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (128*r2) + (12544*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fe/cfer25osnyt6nunpf27zpksyybvmjlh4ygesdebm4stc6c7in4rj.py
# Source Nodes: [getattr_l__mod___stages___1___downsample_attn_downsample_q_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_l__mod___stages___1___downsample_attn_downsample_q_ln_bn => add_129, add_130, add_131, mul_163, mul_164, mul_165, mul_166, mul_167, rsqrt_21, squeeze_64, var_mean_21
triton_per_fused__native_batch_norm_legit_functional_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_27', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 392.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0025575447570332
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sq/csq4jkux65apfwufgucleq7f52iccnfhsrvk3iyq3vqhopucs6fg.py
# Source Nodes: [matmul_8], Original ATen: [aten.clone]
# matmul_8 => clone_30
triton_poi_fused_clone_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 49
    x2 = (xindex // 784) % 8
    x3 = (xindex // 6272)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16*x2) + (128*x1) + (6272*x3)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (16*x2)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (16*x2)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0 + (16*x2)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0 + (16*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x4), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nu/cnuhdadjb5kocyuthfrb37vehsmamps4634qh5qppktbwi2fb2qr.py
# Source Nodes: [matmul_8], Original ATen: [aten.clone]
# matmul_8 => clone_31
triton_poi_fused_clone_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16) % 8
    y2 = (yindex // 128)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (80*y1) + (640*x3) + (125440*y2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (80*y1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (80*y1)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0 + (80*y1)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0 + (80*y1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3 + (196*y4)), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bt/cbtion7abwdiloosc7zhbdfy5sw4nmxsmkz7mooqfgzhti23gzpo.py
# Source Nodes: [attn_8, attn_9, getitem_19, mul_4], Original ATen: [aten._softmax, aten.add, aten.index, aten.mul]
# attn_8 => add_133
# attn_9 => amax_4, div_15, exp_4, sub_26, sum_5
# getitem_19 => index_4
# mul_4 => mul_169
triton_per_fused__softmax_add_index_mul_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_index_mul_30', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3136
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 49
    x1 = (xindex // 49) % 8
    tmp0 = tl.load(in_ptr0 + (r3 + (196*x4)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r3 + (196*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 + 196
    tmp5 = tmp3 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp3)
    tl.device_assert(((0 <= tmp6) & (tmp6 < 196)) | ~(rmask & xmask), "index out of bounds: 0 <= tmp6 < 196")
    tmp7 = tl.load(in_ptr2 + (tmp6 + (196*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp2 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, float("-inf"))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp8 - tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp14 / tmp18
    tl.store(out_ptr2 + (r3 + (196*x4)), tmp19, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kg/ckgcqgbt4u46oi7xxz6ahalbk3h4gqz7g3m2rwbkiqmfdtjghjob.py
# Source Nodes: [matmul_9], Original ATen: [aten.clone]
# matmul_9 => clone_32
triton_poi_fused_clone_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 196
    x2 = (xindex // 12544) % 8
    x3 = (xindex // 100352)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (16 + x0 + (80*x2) + (640*x1) + (125440*x3)), None)
    tmp1 = tl.load(in_ptr1 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x4), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gn/cgnrcppmmgiolbsah62zeyh5igsb5xfzobe7h6bfc35ocmyrrdzz.py
# Source Nodes: [getattr_l__mod___stages___1___downsample_attn_downsample_proj_act, x_56, x_57], Original ATen: [aten._unsafe_view, aten.clone, aten.hardswish, aten.view]
# getattr_l__mod___stages___1___downsample_attn_downsample_proj_act => add_134, clamp_max_11, clamp_min_11, div_16, mul_170
# x_56 => clone_33, view_115
# x_57 => view_116
triton_poi_fused__unsafe_view_clone_hardswish_view_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_hardswish_view_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 512
    x1 = (xindex // 512) % 49
    x2 = (xindex // 25088)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (3136*(x0 // 64)) + (25088*x2) + (x0 % 64)), None)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tl.store(out_ptr0 + (x3), tmp0, None)
    tl.store(out_ptr1 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/br/cbrzwl5r7ee76hehyq76bfpnfsywluygo6r2tk6mpnki6pd66uh2.py
# Source Nodes: [getattr_l__mod___stages___1___downsample_attn_downsample_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_l__mod___stages___1___downsample_attn_downsample_proj_ln_bn => var_mean_22
triton_red_fused__native_batch_norm_legit_functional_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r2) + (25088*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wd/cwde5r4e2m4orgitrpfo7rxxj6yppdydzur2mhu44kc7y6biqrbn.py
# Source Nodes: [getattr_l__mod___stages___1___downsample_attn_downsample_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_l__mod___stages___1___downsample_attn_downsample_proj_ln_bn => add_136, add_137, add_138, mul_172, mul_173, mul_174, mul_175, mul_176, rsqrt_22, squeeze_67, var_mean_22
triton_per_fused__native_batch_norm_legit_functional_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_34', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 392.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0025575447570332
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zf/czfbe2cgrlxzhv4tbeppr5srmrpk6tx4yu53by3iqlekxhtv5wco.py
# Source Nodes: [getattr_l__mod___stages___1___downsample_attn_downsample_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_l__mod___stages___1___downsample_attn_downsample_proj_ln_bn => add_136, add_139, mul_171, mul_177, rsqrt_22, sub_27, var_mean_22
triton_poi_fused__native_batch_norm_legit_functional_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/di/cdijcfsi5273siqtl5xq5nnvy3khsicbfnmxsrw63f2nijzbdgni.py
# Source Nodes: [getattr_l__mod___stages___1___downsample_mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_l__mod___stages___1___downsample_mlp_ln1_bn => var_mean_23
triton_red_fused__native_batch_norm_legit_functional_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_36', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 512
    x1 = (xindex // 512)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (512*r2) + (50176*x1)), rmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
        tmp2_mean_next, tmp2_m2_next, tmp2_weight_next = triton_helpers.welford_reduce(
            tmp1, tmp2_mean, tmp2_m2, tmp2_weight,
        )
        tmp2_mean = tl.where(rmask, tmp2_mean_next, tmp2_mean)
        tmp2_m2 = tl.where(rmask, tmp2_m2_next, tmp2_m2)
        tmp2_weight = tl.where(rmask, tmp2_weight_next, tmp2_weight)
    tmp2_tmp, tmp3_tmp, tmp4_tmp = triton_helpers.welford(
        tmp2_mean, tmp2_m2, tmp2_weight, 1
    )
    tmp2 = tmp2_tmp[:, None]
    tmp3 = tmp3_tmp[:, None]
    tmp4 = tmp4_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp2, None)
    tl.store(out_ptr1 + (x3), tmp3, None)
    tl.store(out_ptr2 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/e7/ce7ln2kgag3gqgjikrwtcrjc3exizuky6nbf5mlggno76bkwsgvd.py
# Source Nodes: [getattr_l__mod___stages___1___downsample_mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_l__mod___stages___1___downsample_mlp_ln1_bn => add_141, add_142, add_143, mul_179, mul_180, mul_181, mul_182, mul_183, rsqrt_23, squeeze_70, var_mean_23
triton_per_fused__native_batch_norm_legit_functional_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_37', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (512*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 392.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0025575447570332
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3k/c3kbk2b35cenck5yns6vxf4dvkvmvd4gatibfer7l5c3uyhtmiim.py
# Source Nodes: [getattr_l__mod___stages___1___downsample_mlp_ln1_bn, x_61, x_62, x_64], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish, aten.view]
# getattr_l__mod___stages___1___downsample_mlp_ln1_bn => add_141, add_144, mul_178, mul_184, rsqrt_23, sub_28, var_mean_23
# x_61 => view_123
# x_62 => add_145, clamp_max_12, clamp_min_12, div_17, mul_185
# x_64 => view_124
triton_poi_fused__native_batch_norm_legit_functional_hardswish_view_38 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_view_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 512
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 3.0
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tmp20 / tmp18
    tl.store(out_ptr0 + (x2), tmp13, None)
    tl.store(out_ptr1 + (x2), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tt/cttbte3zoz7bo3tbqvj7so5nqiq2w6fkyg2u4c7hy7l73vi7tfrr.py
# Source Nodes: [x_67], Original ATen: [aten.add]
# x_67 => add_151
triton_poi_fused_add_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 392.0
    tmp6 = tmp4 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp0 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rq/crqtft5vji3ohxgxbirwipnunmpvhhap4etjsrdi5awnaj4qsjwj.py
# Source Nodes: [matmul_10], Original ATen: [aten.clone]
# matmul_10 => clone_35
triton_poi_fused_clone_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 50176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 49
    x2 = (xindex // 784) % 8
    x3 = (xindex // 6272)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (512*x1) + (25088*x3)), xmask)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (64*x2)), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0 + (64*x2)), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0 + (64*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x4), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oj/cojamozduig6jubpz7dokanm4txq22pvm6m2bfhlyxd7gl3avghx.py
# Source Nodes: [matmul_10], Original ATen: [aten.clone]
# matmul_10 => clone_36
triton_poi_fused_clone_41 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16) % 8
    y2 = (yindex // 128)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (16 + y0 + (64*y1) + (512*x3) + (25088*y2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (16 + y0 + (64*y1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (16 + y0 + (64*y1)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (16 + y0 + (64*y1)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (16 + y0 + (64*y1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3 + (49*y4)), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/27/c27kmql4ftrvmgy4zhjte5zwwz5lwrradbzhpy3hxtxnentt6de2.py
# Source Nodes: [attn_10, attn_11, getitem_23, mul_5], Original ATen: [aten._softmax, aten.add, aten.index, aten.mul]
# attn_10 => add_157
# attn_11 => amax_5, div_18, exp_5, sub_31, sum_6
# getitem_23 => index_5
# mul_5 => mul_200
triton_per_fused__softmax_add_index_mul_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_index_mul_42', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3136
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 49
    x1 = (xindex // 49) % 8
    tmp0 = tl.load(in_ptr0 + (r3 + (49*x4)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r3 + (49*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 + 49
    tmp5 = tmp3 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp3)
    tl.device_assert(((0 <= tmp6) & (tmp6 < 49)) | ~(rmask & xmask), "index out of bounds: 0 <= tmp6 < 49")
    tmp7 = tl.load(in_ptr2 + (tmp6 + (49*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp2 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, float("-inf"))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp8 - tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp14 / tmp18
    tl.store(out_ptr2 + (r3 + (49*x4)), tmp19, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ro/cronfvapuckj5zfmsord4mogsb65qu26fzcwyoinnpoay5w55x26.py
# Source Nodes: [matmul_11], Original ATen: [aten.clone]
# matmul_11 => clone_37
triton_poi_fused_clone_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 49
    x2 = (xindex // 1568) % 8
    x3 = (xindex // 12544)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (32 + x0 + (64*x2) + (512*x1) + (25088*x3)), None)
    tmp1 = tl.load(in_ptr1 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x4), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/i6/ci65zvwntze3o6i6zsjcg2d7cvjxx3zn6hl3u62gpyziamqwmzya.py
# Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___0___attn_proj_act, x_69, x_70], Original ATen: [aten._unsafe_view, aten.clone, aten.hardswish, aten.view]
# getattr_getattr_l__mod___stages___1___blocks___0___attn_proj_act => add_158, clamp_max_13, clamp_min_13, div_19, mul_201
# x_69 => clone_38, view_139
# x_70 => view_140
triton_poi_fused__unsafe_view_clone_hardswish_view_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_hardswish_view_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256) % 49
    x2 = (xindex // 12544)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*x1) + (1568*(x0 // 32)) + (12544*x2) + (x0 % 32)), None)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tl.store(out_ptr0 + (x3), tmp0, None)
    tl.store(out_ptr1 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4l/c4li5wtn36w5ie7by2sxuxbqo2wfpdqfavragbmz45cjhoqnujm3.py
# Source Nodes: [getattr_l__mod___stages___2___downsample_attn_downsample_kv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_l__mod___stages___2___downsample_attn_downsample_kv_bn => var_mean_41
triton_red_fused__native_batch_norm_legit_functional_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 5120
    rnumel = 98
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1280
    x1 = (xindex // 1280)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1280*r2) + (125440*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x3), tmp2, xmask)
    tl.store(out_ptr1 + (x3), tmp3, xmask)
    tl.store(out_ptr2 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/eh/cehgqhlcl6ohxl77na36avselicu4ihgbbdew6dai7v2h7afg6gi.py
# Source Nodes: [getattr_l__mod___stages___2___downsample_attn_downsample_kv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_l__mod___stages___2___downsample_attn_downsample_kv_bn => add_253, add_254, add_255, mul_318, mul_319, mul_320, mul_321, mul_322, rsqrt_41, squeeze_124, var_mean_41
triton_per_fused__native_batch_norm_legit_functional_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_46', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp1 = tl.load(in_ptr1 + (x0 + (1280*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (1280*r1)), rmask & xmask, other=0.0)
    tmp23 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp30 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp3, 0)
    tmp8 = tl.where(rmask & xmask, tmp4, 0)
    tmp9 = tl.where(rmask & xmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 392.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0025575447570332
    tmp28 = tmp17 * tmp27
    tmp29 = tmp28 * tmp21
    tmp31 = tmp30 * tmp24
    tmp32 = tmp29 + tmp31
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp32, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sp/cspe7kjbfedll5couqwvi4s4lcsew56pxjg3ue6om4hro6dgralf.py
# Source Nodes: [x_120], Original ATen: [aten.view]
# x_120 => view_231
triton_poi_fused_view_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*((x1 % 16) % 4)) + (3584*((x1 % 16) // 4)) + (12544*(x1 // 16))), None)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/c3/cc33oud73ghsdjd2eh7vb7juf2ty42opbflwjj7fmm2iyqgq3zmn.py
# Source Nodes: [getattr_l__mod___stages___2___downsample_attn_downsample_q_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_l__mod___stages___2___downsample_attn_downsample_q_ln_bn => add_258, add_259, add_260, mul_325, mul_326, mul_327, mul_328, mul_329, rsqrt_42, squeeze_127, var_mean_42
triton_red_fused__native_batch_norm_legit_functional_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_48', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr1 + (x0), tmp3, xmask)
    tmp12 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = 128.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = 0.1
    tmp11 = tmp2 * tmp10
    tmp13 = 0.9
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = 1.0078740157480315
    tmp17 = tmp6 * tmp16
    tmp18 = tmp17 * tmp10
    tmp20 = tmp19 * tmp13
    tmp21 = tmp18 + tmp20
    tl.store(out_ptr2 + (x0), tmp9, xmask)
    tl.store(out_ptr4 + (x0), tmp15, xmask)
    tl.store(out_ptr6 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zy/czymg32sf2awqrg5p2fp5hfrl2q5blcd4nwcvqcvfpmehjbr6nbc.py
# Source Nodes: [matmul_18], Original ATen: [aten.clone]
# matmul_18 => clone_56
triton_poi_fused_clone_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32768
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 16
    x2 = (xindex // 256) % 16
    x3 = (xindex // 4096)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (16*x2) + (256*x1) + (4096*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (16*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (16*x2)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0 + (16*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0 + (16*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 128.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x4), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/p2/cp2ds2kdywzeketikuzkfheuujtgsvvfonoz5dqcw564pyucnx4l.py
# Source Nodes: [matmul_18], Original ATen: [aten.clone]
# matmul_18 => clone_57
triton_poi_fused_clone_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16) % 16
    y2 = (yindex // 256)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (80*y1) + (1280*x3) + (62720*y2)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0 + (80*y1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (80*y1)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (y0 + (80*y1)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (y0 + (80*y1)), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3 + (49*y4)), tmp13, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/we/cwefkby55n65jks4lkw6pze2biodvyk3mrykx47sswmqywmff4w2.py
# Source Nodes: [attn_18, attn_19, getitem_39, mul_9], Original ATen: [aten._softmax, aten.add, aten.index, aten.mul]
# attn_18 => add_262
# attn_19 => amax_9, div_30, exp_9, sub_52, sum_10
# getitem_39 => index_9
# mul_9 => mul_331
triton_per_fused__softmax_add_index_mul_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_index_mul_51', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 16
    x1 = (xindex // 16) % 16
    tmp0 = tl.load(in_ptr0 + (r3 + (49*x4)), rmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r3 + (49*x0)), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 + 49
    tmp5 = tmp3 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp3)
    tl.device_assert(((0 <= tmp6) & (tmp6 < 49)) | ~rmask, "index out of bounds: 0 <= tmp6 < 49")
    tmp7 = tl.load(in_ptr2 + (tmp6 + (49*x1)), rmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp2 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, float("-inf"))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp8 - tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp14 / tmp18
    tl.store(out_ptr2 + (r3 + (49*x4)), tmp19, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fe/cfempccxovdrdhaxxmjh3iklxj6kuekzjluhlg5cnk2th2srelrg.py
# Source Nodes: [matmul_19], Original ATen: [aten.clone]
# matmul_19 => clone_58
triton_poi_fused_clone_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 49
    x2 = (xindex // 3136) % 16
    x3 = (xindex // 50176)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (16 + x0 + (80*x2) + (1280*x1) + (62720*x3)), None)
    tmp1 = tl.load(in_ptr1 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (16 + x0 + (80*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x4), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ai/caitmm5zog2fs7ztamrlkghxwrgwsxwg5zjau6zyc7s2z4iiej6j.py
# Source Nodes: [getattr_l__mod___stages___2___downsample_attn_downsample_proj_act, x_121, x_122], Original ATen: [aten._unsafe_view, aten.clone, aten.hardswish, aten.view]
# getattr_l__mod___stages___2___downsample_attn_downsample_proj_act => add_263, clamp_max_21, clamp_min_21, div_31, mul_332
# x_121 => clone_59, view_242
# x_122 => view_243
triton_poi_fused__unsafe_view_clone_hardswish_view_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_hardswish_view_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = (xindex // 1024) % 16
    x2 = (xindex // 16384)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*x1) + (1024*(x0 // 64)) + (16384*x2) + (x0 % 64)), None)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tl.store(out_ptr0 + (x3), tmp0, None)
    tl.store(out_ptr1 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ej/cejg4ybcouyfksohtfcfvpzdm5amoqxhdpgkmnhxanki7djkskpo.py
# Source Nodes: [getattr_l__mod___stages___2___downsample_attn_downsample_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_l__mod___stages___2___downsample_attn_downsample_proj_ln_bn => add_265, add_266, add_267, mul_334, mul_335, mul_336, mul_337, mul_338, rsqrt_43, squeeze_130, var_mean_43
triton_red_fused__native_batch_norm_legit_functional_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_54', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (384*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr1 + (x0), tmp3, xmask)
    tmp12 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = 128.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = 0.1
    tmp11 = tmp2 * tmp10
    tmp13 = 0.9
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = 1.0078740157480315
    tmp17 = tmp6 * tmp16
    tmp18 = tmp17 * tmp10
    tmp20 = tmp19 * tmp13
    tmp21 = tmp18 + tmp20
    tl.store(out_ptr2 + (x0), tmp9, xmask)
    tl.store(out_ptr4 + (x0), tmp15, xmask)
    tl.store(out_ptr6 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u3/cu3lhr6whxn7ska6tfjybuicghsvejmji4nr2rghb6uadk5t6a2l.py
# Source Nodes: [getattr_l__mod___stages___2___downsample_attn_downsample_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_l__mod___stages___2___downsample_attn_downsample_proj_ln_bn => add_265, add_268, mul_333, mul_339, rsqrt_43, sub_53, var_mean_43
triton_poi_fused__native_batch_norm_legit_functional_55 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 128.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x2), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/42/c42zwkgx2sbcgzofbh7s4tnjxqlnzecmxux3j7r663uygwkalgzl.py
# Source Nodes: [getattr_l__mod___stages___2___downsample_mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
# getattr_l__mod___stages___2___downsample_mlp_ln1_bn => add_270, add_271, add_272, mul_341, mul_342, mul_343, mul_344, mul_345, rsqrt_44, squeeze_133, var_mean_44
triton_red_fused__native_batch_norm_legit_functional_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_56', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (x0 + (768*r1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr1 + (x0), tmp3, xmask)
    tmp12 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = 128.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = 0.1
    tmp11 = tmp2 * tmp10
    tmp13 = 0.9
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = 1.0078740157480315
    tmp17 = tmp6 * tmp16
    tmp18 = tmp17 * tmp10
    tmp20 = tmp19 * tmp13
    tmp21 = tmp18 + tmp20
    tl.store(out_ptr2 + (x0), tmp9, xmask)
    tl.store(out_ptr4 + (x0), tmp15, xmask)
    tl.store(out_ptr6 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fg/cfgf6gngymnlvnptk5ispa4sr4vts63hhltq476i2crvoxmv5zi3.py
# Source Nodes: [getattr_l__mod___stages___2___downsample_mlp_ln1_bn, x_126, x_127, x_129], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish, aten.view]
# getattr_l__mod___stages___2___downsample_mlp_ln1_bn => add_270, add_273, mul_340, mul_346, rsqrt_44, sub_54, var_mean_44
# x_126 => view_250
# x_127 => add_274, clamp_max_22, clamp_min_22, div_32, mul_347
# x_129 => view_251
triton_poi_fused__native_batch_norm_legit_functional_hardswish_view_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_hardswish_view_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 768
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 128.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = 3.0
    tmp15 = tmp13 + tmp14
    tmp16 = 0.0
    tmp17 = triton_helpers.maximum(tmp15, tmp16)
    tmp18 = 6.0
    tmp19 = triton_helpers.minimum(tmp17, tmp18)
    tmp20 = tmp13 * tmp19
    tmp21 = tmp20 / tmp18
    tl.store(out_ptr0 + (x2), tmp13, None)
    tl.store(out_ptr1 + (x2), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/l4/cl4okanxr7zacz3ys2tblf5iwp6lai4j3i5yhujsqixl6j4dhwkl.py
# Source Nodes: [x_132], Original ATen: [aten.add]
# x_132 => add_280
triton_poi_fused_add_58 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 128.0
    tmp6 = tmp4 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp0 + tmp14
    tl.store(out_ptr0 + (x2), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5v/c5vjsfq7mnivzulxm6mdhukk2h4kxdcxzt7yskdupjkctql36nbp.py
# Source Nodes: [matmul_20], Original ATen: [aten.clone]
# matmul_20 => clone_61
triton_poi_fused_clone_59 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 24576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 16
    x1 = (xindex // 16) % 16
    x2 = (xindex // 256) % 12
    x3 = (xindex // 3072)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (768*x1) + (12288*x3)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 128.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x4), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/eu/ceuhlt3qps2bm5n3hchrfj4dmu3fm3jr7mx6n2uvoibqwkgbhs5e.py
# Source Nodes: [matmul_20], Original ATen: [aten.clone]
# matmul_20 => clone_62
triton_poi_fused_clone_60 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = xindex
    y0 = yindex % 16
    y1 = (yindex // 16) % 12
    y2 = (yindex // 192)
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + (16 + y0 + (64*y1) + (768*x3) + (12288*y2)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (16 + y0 + (64*y1)), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (16 + y0 + (64*y1)), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (16 + y0 + (64*y1)), ymask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (16 + y0 + (64*y1)), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 128.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3 + (16*y4)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lr/clrosi7js6uwhzfzv76jklyr575sygvgrwvacknag22uasyeol5e.py
# Source Nodes: [attn_20, attn_21, getitem_43, mul_10], Original ATen: [aten._softmax, aten.add, aten.index, aten.mul]
# attn_20 => add_286
# attn_21 => amax_10, div_33, exp_10, sub_57, sum_11
# getitem_43 => index_10
# mul_10 => mul_362
triton_per_fused__softmax_add_index_mul_61 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*i64', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_add_index_mul_61', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1536
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r3 = rindex
    x4 = xindex
    x0 = xindex % 16
    x1 = (xindex // 16) % 12
    tmp0 = tl.load(in_ptr0 + (r3 + (16*x4)), rmask & xmask, other=0.0)
    tmp3 = tl.load(in_ptr1 + (r3 + (16*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp1 = 0.25
    tmp2 = tmp0 * tmp1
    tmp4 = tmp3 + 16
    tmp5 = tmp3 < 0
    tmp6 = tl.where(tmp5, tmp4, tmp3)
    tl.device_assert(((0 <= tmp6) & (tmp6 < 16)) | ~(rmask & xmask), "index out of bounds: 0 <= tmp6 < 16")
    tmp7 = tl.load(in_ptr2 + (tmp6 + (16*x1)), rmask & xmask, eviction_policy='evict_last', other=0.0)
    tmp8 = tmp2 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, float("-inf"))
    tmp12 = triton_helpers.max2(tmp11, 1)[:, None]
    tmp13 = tmp8 - tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp17 = tl.where(rmask & xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp14 / tmp18
    tl.store(out_ptr2 + (r3 + (16*x4)), tmp19, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g5/cg5pqbtidcuowmbtsn6vpllncmhyvrbug3xxzxfit6igusj7lycl.py
# Source Nodes: [matmul_21], Original ATen: [aten.clone]
# matmul_21 => clone_63
triton_poi_fused_clone_62 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_62', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x1 = (xindex // 32) % 16
    x2 = (xindex // 512) % 12
    x3 = (xindex // 6144)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (32 + x0 + (64*x2) + (768*x1) + (12288*x3)), None)
    tmp1 = tl.load(in_ptr1 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (32 + x0 + (64*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 128.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x4), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pq/cpqslpzykpheuyismtksqceomsirhwuj6xr4m2kcynxablngf5m7.py
# Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___0___attn_proj_act, x_134, x_135], Original ATen: [aten._unsafe_view, aten.clone, aten.hardswish, aten.view]
# getattr_getattr_l__mod___stages___2___blocks___0___attn_proj_act => add_287, clamp_max_23, clamp_min_23, div_34, mul_363
# x_134 => clone_64, view_266
# x_135 => view_267
triton_poi_fused__unsafe_view_clone_hardswish_view_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_hardswish_view_63', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 384
    x1 = (xindex // 384) % 16
    x2 = (xindex // 6144)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((32*x1) + (512*(x0 // 32)) + (6144*x2) + (x0 % 32)), None)
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tl.store(out_ptr0 + (x3), tmp0, None)
    tl.store(out_ptr1 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/f2/cf2pbq6je43ilmnmbdjbidneihbeogiblhegmizgztkedhmulsbu.py
# Source Nodes: [x_183, x_184], Original ATen: [aten.add, aten.mean]
# x_183 => add_380
# x_184 => mean
triton_per_fused_add_mean_64 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 16],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_64', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 384
    x1 = (xindex // 384)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*r2) + (6144*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (384*r2) + (6144*x1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 - tmp2
    tmp5 = 128.0
    tmp6 = tmp4 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = tmp3 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = tmp0 + tmp14
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = 16.0
    tmp21 = tmp19 / tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w4/cw4jgoofmddfwbyiqmlv6wgjceakjtvrtfuytbxahaiovb342aco.py
# Source Nodes: [l__mod___head_bn, l__mod___head_dist_bn], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___head_bn => add_383, add_384, mul_480, mul_481, mul_482, mul_483, mul_484, var_mean_62
# l__mod___head_dist_bn => add_388, add_389, mul_488, mul_491
triton_per_fused__native_batch_norm_legit_functional_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_65', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'in_ptr3', 'in_ptr4', 'out_ptr3', 'out_ptr5', 'out_ptr7', 'out_ptr9']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr3, out_ptr5, out_ptr7, out_ptr9, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp19 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp23 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 8, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = 0.1
    tmp18 = tmp10 * tmp17
    tmp20 = 0.9
    tmp21 = tmp19 * tmp20
    tmp22 = tmp18 + tmp21
    tmp24 = tmp23 * tmp20
    tmp25 = tmp18 + tmp24
    tmp26 = 8.0
    tmp27 = tmp16 / tmp26
    tmp28 = 1.1428571428571428
    tmp29 = tmp27 * tmp28
    tmp30 = tmp29 * tmp17
    tmp32 = tmp31 * tmp20
    tmp33 = tmp30 + tmp32
    tmp35 = tmp34 * tmp20
    tmp36 = tmp30 + tmp35
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr5 + (x0), tmp25, xmask)
    tl.store(out_ptr7 + (x0), tmp33, xmask)
    tl.store(out_ptr9 + (x0), tmp36, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o3/co3i4jmzqdpiccgz7cdjqeqjqk6upecuz7nfs6fhe5q5ln7vzasz.py
# Source Nodes: [l__mod___head_bn, l__mod___head_dist_bn], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___head_bn => add_382, add_385, mul_479, mul_485, rsqrt_62, sub_76, var_mean_62
# l__mod___head_dist_bn => add_390, mul_492
triton_poi_fused__native_batch_norm_legit_functional_66 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_66', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 384
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 8.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp9 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x2), tmp13, xmask)
    tl.store(out_ptr1 + (x2), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e5/ce5kqt47z7xhn667aia3u4jiu6kav3l5ftce5ucoe44k7nnxv6yg.py
# Source Nodes: [add_40, pred], Original ATen: [aten.add, aten.div]
# add_40 => add_391
# pred => div_45
triton_poi_fused_add_div_67 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_div_67', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1000
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x2), xmask)
    tmp4 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp6 = tmp2 + tmp5
    tmp7 = 2.0
    tmp8 = tmp6 / tmp7
    tl.store(in_out_ptr0 + (x2), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m4/cm4b4oi3zkuaw4i6phicrt6lr5uuq5jk5xfumyvrfcp3omr2e5vm.py
# Source Nodes: [l__mod___stem_conv1_bn], Original ATen: [aten.add]
# l__mod___stem_conv1_bn => add
triton_poi_fused_add_68 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1], 
    filename=__file__,
    triton_meta={'signature': {0: '*i64', 1: '*i64', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_68', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp2 = tl.full([1], 1, tl.int64)
    tmp3 = tmp1 + tmp2
    tl.store(out_ptr1 + (tl.full([XBLOCK], 0, tl.int32)), tmp3, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415 = args
    args.clear()
    assert_size_stride(primals_1, (4, 196), (196, 1))
    assert_size_stride(primals_2, (4, 196), (196, 1))
    assert_size_stride(primals_3, (4, 196), (196, 1))
    assert_size_stride(primals_4, (4, 196), (196, 1))
    assert_size_stride(primals_5, (8, 196), (196, 1))
    assert_size_stride(primals_6, (8, 49), (49, 1))
    assert_size_stride(primals_7, (8, 49), (49, 1))
    assert_size_stride(primals_8, (8, 49), (49, 1))
    assert_size_stride(primals_9, (8, 49), (49, 1))
    assert_size_stride(primals_10, (16, 49), (49, 1))
    assert_size_stride(primals_11, (12, 16), (16, 1))
    assert_size_stride(primals_12, (12, 16), (16, 1))
    assert_size_stride(primals_13, (12, 16), (16, 1))
    assert_size_stride(primals_14, (12, 16), (16, 1))
    assert_size_stride(primals_15, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_16, (16, ), (1, ))
    assert_size_stride(primals_17, (16, ), (1, ))
    assert_size_stride(primals_18, (32, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_19, (32, ), (1, ))
    assert_size_stride(primals_20, (32, ), (1, ))
    assert_size_stride(primals_21, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_22, (64, ), (1, ))
    assert_size_stride(primals_23, (64, ), (1, ))
    assert_size_stride(primals_24, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_25, (128, ), (1, ))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_27, (256, 128), (128, 1))
    assert_size_stride(primals_28, (256, ), (1, ))
    assert_size_stride(primals_29, (256, ), (1, ))
    assert_size_stride(primals_30, (128, 128), (128, 1))
    assert_size_stride(primals_31, (128, ), (1, ))
    assert_size_stride(primals_32, (128, ), (1, ))
    assert_size_stride(primals_33, (256, 128), (128, 1))
    assert_size_stride(primals_34, (256, ), (1, ))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_36, (128, 256), (256, 1))
    assert_size_stride(primals_37, (128, ), (1, ))
    assert_size_stride(primals_38, (128, ), (1, ))
    assert_size_stride(primals_39, (256, 128), (128, 1))
    assert_size_stride(primals_40, (256, ), (1, ))
    assert_size_stride(primals_41, (256, ), (1, ))
    assert_size_stride(primals_42, (128, 128), (128, 1))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_44, (128, ), (1, ))
    assert_size_stride(primals_45, (256, 128), (128, 1))
    assert_size_stride(primals_46, (256, ), (1, ))
    assert_size_stride(primals_47, (256, ), (1, ))
    assert_size_stride(primals_48, (128, 256), (256, 1))
    assert_size_stride(primals_49, (128, ), (1, ))
    assert_size_stride(primals_50, (128, ), (1, ))
    assert_size_stride(primals_51, (256, 128), (128, 1))
    assert_size_stride(primals_52, (256, ), (1, ))
    assert_size_stride(primals_53, (256, ), (1, ))
    assert_size_stride(primals_54, (128, 128), (128, 1))
    assert_size_stride(primals_55, (128, ), (1, ))
    assert_size_stride(primals_56, (128, ), (1, ))
    assert_size_stride(primals_57, (256, 128), (128, 1))
    assert_size_stride(primals_58, (256, ), (1, ))
    assert_size_stride(primals_59, (256, ), (1, ))
    assert_size_stride(primals_60, (128, 256), (256, 1))
    assert_size_stride(primals_61, (128, ), (1, ))
    assert_size_stride(primals_62, (128, ), (1, ))
    assert_size_stride(primals_63, (256, 128), (128, 1))
    assert_size_stride(primals_64, (256, ), (1, ))
    assert_size_stride(primals_65, (256, ), (1, ))
    assert_size_stride(primals_66, (128, 128), (128, 1))
    assert_size_stride(primals_67, (128, ), (1, ))
    assert_size_stride(primals_68, (128, ), (1, ))
    assert_size_stride(primals_69, (256, 128), (128, 1))
    assert_size_stride(primals_70, (256, ), (1, ))
    assert_size_stride(primals_71, (256, ), (1, ))
    assert_size_stride(primals_72, (128, 256), (256, 1))
    assert_size_stride(primals_73, (128, ), (1, ))
    assert_size_stride(primals_74, (128, ), (1, ))
    assert_size_stride(primals_75, (640, 128), (128, 1))
    assert_size_stride(primals_76, (640, ), (1, ))
    assert_size_stride(primals_77, (640, ), (1, ))
    assert_size_stride(primals_78, (128, 128), (128, 1))
    assert_size_stride(primals_79, (128, ), (1, ))
    assert_size_stride(primals_80, (128, ), (1, ))
    assert_size_stride(primals_81, (256, 512), (512, 1))
    assert_size_stride(primals_82, (256, ), (1, ))
    assert_size_stride(primals_83, (256, ), (1, ))
    assert_size_stride(primals_84, (512, 256), (256, 1))
    assert_size_stride(primals_85, (512, ), (1, ))
    assert_size_stride(primals_86, (512, ), (1, ))
    assert_size_stride(primals_87, (256, 512), (512, 1))
    assert_size_stride(primals_88, (256, ), (1, ))
    assert_size_stride(primals_89, (256, ), (1, ))
    assert_size_stride(primals_90, (512, 256), (256, 1))
    assert_size_stride(primals_91, (512, ), (1, ))
    assert_size_stride(primals_92, (512, ), (1, ))
    assert_size_stride(primals_93, (256, 256), (256, 1))
    assert_size_stride(primals_94, (256, ), (1, ))
    assert_size_stride(primals_95, (256, ), (1, ))
    assert_size_stride(primals_96, (512, 256), (256, 1))
    assert_size_stride(primals_97, (512, ), (1, ))
    assert_size_stride(primals_98, (512, ), (1, ))
    assert_size_stride(primals_99, (256, 512), (512, 1))
    assert_size_stride(primals_100, (256, ), (1, ))
    assert_size_stride(primals_101, (256, ), (1, ))
    assert_size_stride(primals_102, (512, 256), (256, 1))
    assert_size_stride(primals_103, (512, ), (1, ))
    assert_size_stride(primals_104, (512, ), (1, ))
    assert_size_stride(primals_105, (256, 256), (256, 1))
    assert_size_stride(primals_106, (256, ), (1, ))
    assert_size_stride(primals_107, (256, ), (1, ))
    assert_size_stride(primals_108, (512, 256), (256, 1))
    assert_size_stride(primals_109, (512, ), (1, ))
    assert_size_stride(primals_110, (512, ), (1, ))
    assert_size_stride(primals_111, (256, 512), (512, 1))
    assert_size_stride(primals_112, (256, ), (1, ))
    assert_size_stride(primals_113, (256, ), (1, ))
    assert_size_stride(primals_114, (512, 256), (256, 1))
    assert_size_stride(primals_115, (512, ), (1, ))
    assert_size_stride(primals_116, (512, ), (1, ))
    assert_size_stride(primals_117, (256, 256), (256, 1))
    assert_size_stride(primals_118, (256, ), (1, ))
    assert_size_stride(primals_119, (256, ), (1, ))
    assert_size_stride(primals_120, (512, 256), (256, 1))
    assert_size_stride(primals_121, (512, ), (1, ))
    assert_size_stride(primals_122, (512, ), (1, ))
    assert_size_stride(primals_123, (256, 512), (512, 1))
    assert_size_stride(primals_124, (256, ), (1, ))
    assert_size_stride(primals_125, (256, ), (1, ))
    assert_size_stride(primals_126, (512, 256), (256, 1))
    assert_size_stride(primals_127, (512, ), (1, ))
    assert_size_stride(primals_128, (512, ), (1, ))
    assert_size_stride(primals_129, (256, 256), (256, 1))
    assert_size_stride(primals_130, (256, ), (1, ))
    assert_size_stride(primals_131, (256, ), (1, ))
    assert_size_stride(primals_132, (512, 256), (256, 1))
    assert_size_stride(primals_133, (512, ), (1, ))
    assert_size_stride(primals_134, (512, ), (1, ))
    assert_size_stride(primals_135, (256, 512), (512, 1))
    assert_size_stride(primals_136, (256, ), (1, ))
    assert_size_stride(primals_137, (256, ), (1, ))
    assert_size_stride(primals_138, (1280, 256), (256, 1))
    assert_size_stride(primals_139, (1280, ), (1, ))
    assert_size_stride(primals_140, (1280, ), (1, ))
    assert_size_stride(primals_141, (256, 256), (256, 1))
    assert_size_stride(primals_142, (256, ), (1, ))
    assert_size_stride(primals_143, (256, ), (1, ))
    assert_size_stride(primals_144, (384, 1024), (1024, 1))
    assert_size_stride(primals_145, (384, ), (1, ))
    assert_size_stride(primals_146, (384, ), (1, ))
    assert_size_stride(primals_147, (768, 384), (384, 1))
    assert_size_stride(primals_148, (768, ), (1, ))
    assert_size_stride(primals_149, (768, ), (1, ))
    assert_size_stride(primals_150, (384, 768), (768, 1))
    assert_size_stride(primals_151, (384, ), (1, ))
    assert_size_stride(primals_152, (384, ), (1, ))
    assert_size_stride(primals_153, (768, 384), (384, 1))
    assert_size_stride(primals_154, (768, ), (1, ))
    assert_size_stride(primals_155, (768, ), (1, ))
    assert_size_stride(primals_156, (384, 384), (384, 1))
    assert_size_stride(primals_157, (384, ), (1, ))
    assert_size_stride(primals_158, (384, ), (1, ))
    assert_size_stride(primals_159, (768, 384), (384, 1))
    assert_size_stride(primals_160, (768, ), (1, ))
    assert_size_stride(primals_161, (768, ), (1, ))
    assert_size_stride(primals_162, (384, 768), (768, 1))
    assert_size_stride(primals_163, (384, ), (1, ))
    assert_size_stride(primals_164, (384, ), (1, ))
    assert_size_stride(primals_165, (768, 384), (384, 1))
    assert_size_stride(primals_166, (768, ), (1, ))
    assert_size_stride(primals_167, (768, ), (1, ))
    assert_size_stride(primals_168, (384, 384), (384, 1))
    assert_size_stride(primals_169, (384, ), (1, ))
    assert_size_stride(primals_170, (384, ), (1, ))
    assert_size_stride(primals_171, (768, 384), (384, 1))
    assert_size_stride(primals_172, (768, ), (1, ))
    assert_size_stride(primals_173, (768, ), (1, ))
    assert_size_stride(primals_174, (384, 768), (768, 1))
    assert_size_stride(primals_175, (384, ), (1, ))
    assert_size_stride(primals_176, (384, ), (1, ))
    assert_size_stride(primals_177, (768, 384), (384, 1))
    assert_size_stride(primals_178, (768, ), (1, ))
    assert_size_stride(primals_179, (768, ), (1, ))
    assert_size_stride(primals_180, (384, 384), (384, 1))
    assert_size_stride(primals_181, (384, ), (1, ))
    assert_size_stride(primals_182, (384, ), (1, ))
    assert_size_stride(primals_183, (768, 384), (384, 1))
    assert_size_stride(primals_184, (768, ), (1, ))
    assert_size_stride(primals_185, (768, ), (1, ))
    assert_size_stride(primals_186, (384, 768), (768, 1))
    assert_size_stride(primals_187, (384, ), (1, ))
    assert_size_stride(primals_188, (384, ), (1, ))
    assert_size_stride(primals_189, (768, 384), (384, 1))
    assert_size_stride(primals_190, (768, ), (1, ))
    assert_size_stride(primals_191, (768, ), (1, ))
    assert_size_stride(primals_192, (384, 384), (384, 1))
    assert_size_stride(primals_193, (384, ), (1, ))
    assert_size_stride(primals_194, (384, ), (1, ))
    assert_size_stride(primals_195, (768, 384), (384, 1))
    assert_size_stride(primals_196, (768, ), (1, ))
    assert_size_stride(primals_197, (768, ), (1, ))
    assert_size_stride(primals_198, (384, 768), (768, 1))
    assert_size_stride(primals_199, (384, ), (1, ))
    assert_size_stride(primals_200, (384, ), (1, ))
    assert_size_stride(primals_201, (384, ), (1, ))
    assert_size_stride(primals_202, (384, ), (1, ))
    assert_size_stride(primals_203, (1000, 384), (384, 1))
    assert_size_stride(primals_204, (1000, ), (1, ))
    assert_size_stride(primals_205, (384, ), (1, ))
    assert_size_stride(primals_206, (384, ), (1, ))
    assert_size_stride(primals_207, (1000, 384), (384, 1))
    assert_size_stride(primals_208, (1000, ), (1, ))
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
    assert_size_stride(primals_223, (16, ), (1, ))
    assert_size_stride(primals_224, (16, ), (1, ))
    assert_size_stride(primals_225, (), ())
    assert_size_stride(primals_226, (32, ), (1, ))
    assert_size_stride(primals_227, (32, ), (1, ))
    assert_size_stride(primals_228, (), ())
    assert_size_stride(primals_229, (64, ), (1, ))
    assert_size_stride(primals_230, (64, ), (1, ))
    assert_size_stride(primals_231, (), ())
    assert_size_stride(primals_232, (128, ), (1, ))
    assert_size_stride(primals_233, (128, ), (1, ))
    assert_size_stride(primals_234, (), ())
    assert_size_stride(primals_235, (256, ), (1, ))
    assert_size_stride(primals_236, (256, ), (1, ))
    assert_size_stride(primals_237, (), ())
    assert_size_stride(primals_238, (128, ), (1, ))
    assert_size_stride(primals_239, (128, ), (1, ))
    assert_size_stride(primals_240, (), ())
    assert_size_stride(primals_241, (256, ), (1, ))
    assert_size_stride(primals_242, (256, ), (1, ))
    assert_size_stride(primals_243, (), ())
    assert_size_stride(primals_244, (128, ), (1, ))
    assert_size_stride(primals_245, (128, ), (1, ))
    assert_size_stride(primals_246, (), ())
    assert_size_stride(primals_247, (256, ), (1, ))
    assert_size_stride(primals_248, (256, ), (1, ))
    assert_size_stride(primals_249, (), ())
    assert_size_stride(primals_250, (128, ), (1, ))
    assert_size_stride(primals_251, (128, ), (1, ))
    assert_size_stride(primals_252, (), ())
    assert_size_stride(primals_253, (256, ), (1, ))
    assert_size_stride(primals_254, (256, ), (1, ))
    assert_size_stride(primals_255, (), ())
    assert_size_stride(primals_256, (128, ), (1, ))
    assert_size_stride(primals_257, (128, ), (1, ))
    assert_size_stride(primals_258, (), ())
    assert_size_stride(primals_259, (256, ), (1, ))
    assert_size_stride(primals_260, (256, ), (1, ))
    assert_size_stride(primals_261, (), ())
    assert_size_stride(primals_262, (128, ), (1, ))
    assert_size_stride(primals_263, (128, ), (1, ))
    assert_size_stride(primals_264, (), ())
    assert_size_stride(primals_265, (256, ), (1, ))
    assert_size_stride(primals_266, (256, ), (1, ))
    assert_size_stride(primals_267, (), ())
    assert_size_stride(primals_268, (128, ), (1, ))
    assert_size_stride(primals_269, (128, ), (1, ))
    assert_size_stride(primals_270, (), ())
    assert_size_stride(primals_271, (256, ), (1, ))
    assert_size_stride(primals_272, (256, ), (1, ))
    assert_size_stride(primals_273, (), ())
    assert_size_stride(primals_274, (128, ), (1, ))
    assert_size_stride(primals_275, (128, ), (1, ))
    assert_size_stride(primals_276, (), ())
    assert_size_stride(primals_277, (256, ), (1, ))
    assert_size_stride(primals_278, (256, ), (1, ))
    assert_size_stride(primals_279, (), ())
    assert_size_stride(primals_280, (128, ), (1, ))
    assert_size_stride(primals_281, (128, ), (1, ))
    assert_size_stride(primals_282, (), ())
    assert_size_stride(primals_283, (640, ), (1, ))
    assert_size_stride(primals_284, (640, ), (1, ))
    assert_size_stride(primals_285, (), ())
    assert_size_stride(primals_286, (128, ), (1, ))
    assert_size_stride(primals_287, (128, ), (1, ))
    assert_size_stride(primals_288, (), ())
    assert_size_stride(primals_289, (256, ), (1, ))
    assert_size_stride(primals_290, (256, ), (1, ))
    assert_size_stride(primals_291, (), ())
    assert_size_stride(primals_292, (512, ), (1, ))
    assert_size_stride(primals_293, (512, ), (1, ))
    assert_size_stride(primals_294, (), ())
    assert_size_stride(primals_295, (256, ), (1, ))
    assert_size_stride(primals_296, (256, ), (1, ))
    assert_size_stride(primals_297, (), ())
    assert_size_stride(primals_298, (512, ), (1, ))
    assert_size_stride(primals_299, (512, ), (1, ))
    assert_size_stride(primals_300, (), ())
    assert_size_stride(primals_301, (256, ), (1, ))
    assert_size_stride(primals_302, (256, ), (1, ))
    assert_size_stride(primals_303, (), ())
    assert_size_stride(primals_304, (512, ), (1, ))
    assert_size_stride(primals_305, (512, ), (1, ))
    assert_size_stride(primals_306, (), ())
    assert_size_stride(primals_307, (256, ), (1, ))
    assert_size_stride(primals_308, (256, ), (1, ))
    assert_size_stride(primals_309, (), ())
    assert_size_stride(primals_310, (512, ), (1, ))
    assert_size_stride(primals_311, (512, ), (1, ))
    assert_size_stride(primals_312, (), ())
    assert_size_stride(primals_313, (256, ), (1, ))
    assert_size_stride(primals_314, (256, ), (1, ))
    assert_size_stride(primals_315, (), ())
    assert_size_stride(primals_316, (512, ), (1, ))
    assert_size_stride(primals_317, (512, ), (1, ))
    assert_size_stride(primals_318, (), ())
    assert_size_stride(primals_319, (256, ), (1, ))
    assert_size_stride(primals_320, (256, ), (1, ))
    assert_size_stride(primals_321, (), ())
    assert_size_stride(primals_322, (512, ), (1, ))
    assert_size_stride(primals_323, (512, ), (1, ))
    assert_size_stride(primals_324, (), ())
    assert_size_stride(primals_325, (256, ), (1, ))
    assert_size_stride(primals_326, (256, ), (1, ))
    assert_size_stride(primals_327, (), ())
    assert_size_stride(primals_328, (512, ), (1, ))
    assert_size_stride(primals_329, (512, ), (1, ))
    assert_size_stride(primals_330, (), ())
    assert_size_stride(primals_331, (256, ), (1, ))
    assert_size_stride(primals_332, (256, ), (1, ))
    assert_size_stride(primals_333, (), ())
    assert_size_stride(primals_334, (512, ), (1, ))
    assert_size_stride(primals_335, (512, ), (1, ))
    assert_size_stride(primals_336, (), ())
    assert_size_stride(primals_337, (256, ), (1, ))
    assert_size_stride(primals_338, (256, ), (1, ))
    assert_size_stride(primals_339, (), ())
    assert_size_stride(primals_340, (512, ), (1, ))
    assert_size_stride(primals_341, (512, ), (1, ))
    assert_size_stride(primals_342, (), ())
    assert_size_stride(primals_343, (256, ), (1, ))
    assert_size_stride(primals_344, (256, ), (1, ))
    assert_size_stride(primals_345, (), ())
    assert_size_stride(primals_346, (1280, ), (1, ))
    assert_size_stride(primals_347, (1280, ), (1, ))
    assert_size_stride(primals_348, (), ())
    assert_size_stride(primals_349, (256, ), (1, ))
    assert_size_stride(primals_350, (256, ), (1, ))
    assert_size_stride(primals_351, (), ())
    assert_size_stride(primals_352, (384, ), (1, ))
    assert_size_stride(primals_353, (384, ), (1, ))
    assert_size_stride(primals_354, (), ())
    assert_size_stride(primals_355, (768, ), (1, ))
    assert_size_stride(primals_356, (768, ), (1, ))
    assert_size_stride(primals_357, (), ())
    assert_size_stride(primals_358, (384, ), (1, ))
    assert_size_stride(primals_359, (384, ), (1, ))
    assert_size_stride(primals_360, (), ())
    assert_size_stride(primals_361, (768, ), (1, ))
    assert_size_stride(primals_362, (768, ), (1, ))
    assert_size_stride(primals_363, (), ())
    assert_size_stride(primals_364, (384, ), (1, ))
    assert_size_stride(primals_365, (384, ), (1, ))
    assert_size_stride(primals_366, (), ())
    assert_size_stride(primals_367, (768, ), (1, ))
    assert_size_stride(primals_368, (768, ), (1, ))
    assert_size_stride(primals_369, (), ())
    assert_size_stride(primals_370, (384, ), (1, ))
    assert_size_stride(primals_371, (384, ), (1, ))
    assert_size_stride(primals_372, (), ())
    assert_size_stride(primals_373, (768, ), (1, ))
    assert_size_stride(primals_374, (768, ), (1, ))
    assert_size_stride(primals_375, (), ())
    assert_size_stride(primals_376, (384, ), (1, ))
    assert_size_stride(primals_377, (384, ), (1, ))
    assert_size_stride(primals_378, (), ())
    assert_size_stride(primals_379, (768, ), (1, ))
    assert_size_stride(primals_380, (768, ), (1, ))
    assert_size_stride(primals_381, (), ())
    assert_size_stride(primals_382, (384, ), (1, ))
    assert_size_stride(primals_383, (384, ), (1, ))
    assert_size_stride(primals_384, (), ())
    assert_size_stride(primals_385, (768, ), (1, ))
    assert_size_stride(primals_386, (768, ), (1, ))
    assert_size_stride(primals_387, (), ())
    assert_size_stride(primals_388, (384, ), (1, ))
    assert_size_stride(primals_389, (384, ), (1, ))
    assert_size_stride(primals_390, (), ())
    assert_size_stride(primals_391, (768, ), (1, ))
    assert_size_stride(primals_392, (768, ), (1, ))
    assert_size_stride(primals_393, (), ())
    assert_size_stride(primals_394, (384, ), (1, ))
    assert_size_stride(primals_395, (384, ), (1, ))
    assert_size_stride(primals_396, (), ())
    assert_size_stride(primals_397, (768, ), (1, ))
    assert_size_stride(primals_398, (768, ), (1, ))
    assert_size_stride(primals_399, (), ())
    assert_size_stride(primals_400, (384, ), (1, ))
    assert_size_stride(primals_401, (384, ), (1, ))
    assert_size_stride(primals_402, (), ())
    assert_size_stride(primals_403, (768, ), (1, ))
    assert_size_stride(primals_404, (768, ), (1, ))
    assert_size_stride(primals_405, (), ())
    assert_size_stride(primals_406, (384, ), (1, ))
    assert_size_stride(primals_407, (384, ), (1, ))
    assert_size_stride(primals_408, (), ())
    assert_size_stride(primals_409, (384, ), (1, ))
    assert_size_stride(primals_410, (384, ), (1, ))
    assert_size_stride(primals_411, (), ())
    assert_size_stride(primals_412, (384, ), (1, ))
    assert_size_stride(primals_413, (384, ), (1, ))
    assert_size_stride(primals_414, (), ())
    assert_size_stride(primals_415, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [l__mod___stem_conv1_linear], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_415, primals_15, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 16, 112, 112), (200704, 12544, 112, 1))
        buf1 = empty_strided((1, 16, 1, 1, 13), (208, 13, 208, 208, 1), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((1, 16, 1, 1, 13), (208, 13, 208, 208, 1), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((1, 16, 1, 1, 13), (208, 13, 208, 208, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___stem_conv1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_cuda_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_0.run(buf0, buf1, buf2, buf3, 208, 7720, grid=grid(208), stream=stream0)
        buf4 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf7 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___stem_conv1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_1.run(buf1, buf2, buf3, primals_223, primals_224, buf4, buf5, buf7, primals_223, primals_224, 16, 13, grid=grid(16), stream=stream0)
        del buf1
        del buf2
        del buf3
        del primals_223
        del primals_224
        buf8 = empty((8, 16, 112, 112), device='cuda', dtype=torch.float32)
        buf9 = empty((8, 16, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___stem_act1, l__mod___stem_conv1_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_2.run(buf0, buf4, buf5, primals_16, primals_17, buf8, buf9, 1605632, grid=grid(1605632), stream=stream0)
        del buf5
        del primals_17
        # Source Nodes: [l__mod___stem_conv2_linear], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_18, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (8, 32, 56, 56), (100352, 3136, 56, 1))
        buf11 = empty_strided((1, 32, 1, 1, 4), (128, 1, 128, 128, 32), device='cuda', dtype=torch.float32)
        buf12 = empty_strided((1, 32, 1, 1, 4), (128, 1, 128, 128, 32), device='cuda', dtype=torch.float32)
        buf13 = empty_strided((1, 32, 1, 1, 4), (128, 1, 128, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___stem_conv2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf10, buf11, buf12, buf13, 128, 6272, grid=grid(128), stream=stream0)
        buf14 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf15 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf17 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___stem_conv2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_4.run(buf11, buf12, buf13, primals_226, primals_227, buf14, buf15, buf17, primals_226, primals_227, 32, 4, grid=grid(32), stream=stream0)
        del primals_226
        del primals_227
        buf18 = empty((8, 32, 56, 56), device='cuda', dtype=torch.float32)
        buf19 = empty((8, 32, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___stem_act2, l__mod___stem_conv2_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_5.run(buf10, buf14, buf15, primals_19, primals_20, buf18, buf19, 802816, grid=grid(802816), stream=stream0)
        del buf15
        del primals_20
        # Source Nodes: [l__mod___stem_conv3_linear], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_21, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (8, 64, 28, 28), (50176, 784, 28, 1))
        buf21 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf22 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf24 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___stem_conv3_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_6.run(buf20, primals_229, primals_230, buf21, buf22, buf24, primals_229, primals_230, 64, 6272, grid=grid(64), stream=stream0)
        del primals_229
        del primals_230
        buf25 = empty((8, 64, 28, 28), device='cuda', dtype=torch.float32)
        buf26 = empty((8, 64, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___stem_act3, l__mod___stem_conv3_bn], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_7.run(buf20, buf21, buf22, primals_22, primals_23, buf25, buf26, 401408, grid=grid(401408), stream=stream0)
        del buf22
        del primals_23
        # Source Nodes: [l__mod___stem_conv4_linear], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, primals_24, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (8, 128, 14, 14), (25088, 196, 14, 1))
        buf28 = reinterpret_tensor(buf13, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf13  # reuse
        buf29 = reinterpret_tensor(buf12, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf12  # reuse
        buf31 = reinterpret_tensor(buf11, (128, ), (1, ), 0); del buf11  # reuse
        # Source Nodes: [x], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_8.run(buf27, primals_232, primals_233, buf28, buf29, buf31, primals_232, primals_233, 128, 1568, grid=grid(128), stream=stream0)
        del primals_232
        del primals_233
        buf32 = empty((8, 128, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_9.run(buf27, buf28, buf29, primals_25, primals_26, buf32, 200704, grid=grid(200704), stream=stream0)
        del primals_26
        buf33 = empty((1568, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_3], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_10.run(buf32, buf33, 1568, 128, grid=grid(1568, 128), stream=stream0)
        buf34 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf33, reinterpret_tensor(primals_27, (128, 256), (1, 128), 0), out=buf34)
        buf35 = empty_strided((1, 256, 13), (3328, 1, 256), device='cuda', dtype=torch.float32)
        buf36 = empty_strided((1, 256, 13), (3328, 1, 256), device='cuda', dtype=torch.float32)
        buf37 = empty_strided((1, 256, 13), (3328, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___attn_qkv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf34, buf35, buf36, buf37, 3328, 121, grid=grid(3328), stream=stream0)
        buf38 = empty((1, 256), device='cuda', dtype=torch.float32)
        buf39 = empty((1, 256), device='cuda', dtype=torch.float32)
        buf41 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___attn_qkv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_12.run(buf35, buf36, buf37, primals_235, primals_236, buf38, buf39, buf41, primals_235, primals_236, 256, 13, grid=grid(256), stream=stream0)
        del primals_235
        del primals_236
        buf42 = empty((8, 4, 196, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf34, buf38, buf39, primals_28, primals_29, buf42, 100352, grid=grid(100352), stream=stream0)
        buf43 = empty((8, 4, 16, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf34, buf38, buf39, primals_28, primals_29, buf43, 512, 196, grid=grid(512, 196), stream=stream0)
        buf44 = empty((32, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf42, (32, 196, 16), (3136, 16, 1), 0), reinterpret_tensor(buf43, (32, 16, 196), (3136, 196, 1), 0), out=buf44)
        buf47 = empty((8, 4, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn, attn_1, getitem_3, mul], Original ATen: [aten._softmax, aten.add, aten.index, aten.mul]
        triton_per_fused__softmax_add_index_mul_15.run(buf44, primals_209, primals_1, buf47, 6272, 196, grid=grid(6272), stream=stream0)
        del primals_1
        buf48 = empty((8, 4, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf34, buf38, buf39, primals_28, primals_29, buf48, 200704, grid=grid(200704), stream=stream0)
        del primals_29
        buf49 = empty((32, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf47, (32, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf48, (32, 196, 32), (6272, 32, 1), 0), out=buf49)
        buf50 = empty((8, 196, 128), device='cuda', dtype=torch.float32)
        buf51 = empty((1568, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___attn_proj_act, x_4, x_5], Original ATen: [aten._unsafe_view, aten.clone, aten.hardswish, aten.view]
        triton_poi_fused__unsafe_view_clone_hardswish_view_17.run(buf49, buf50, buf51, 200704, grid=grid(200704), stream=stream0)
        buf52 = reinterpret_tensor(buf49, (1568, 128), (128, 1), 0); del buf49  # reuse
        # Source Nodes: [x_5], Original ATen: [aten.mm]
        extern_kernels.mm(buf51, reinterpret_tensor(primals_30, (128, 128), (1, 128), 0), out=buf52)
        buf53 = empty_strided((1, 128, 13), (1664, 1, 128), device='cuda', dtype=torch.float32)
        buf54 = empty_strided((1, 128, 13), (1664, 1, 128), device='cuda', dtype=torch.float32)
        buf55 = empty_strided((1, 128, 13), (1664, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___attn_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf52, buf53, buf54, buf55, 1664, 121, grid=grid(1664), stream=stream0)
        buf56 = reinterpret_tensor(buf29, (1, 128), (128, 1), 0); del buf29  # reuse
        buf57 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf59 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___attn_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_19.run(buf53, buf54, buf55, primals_238, primals_239, buf56, buf57, buf59, primals_238, primals_239, 128, 13, grid=grid(128), stream=stream0)
        del primals_238
        del primals_239
        buf60 = empty((8, 196, 128), device='cuda', dtype=torch.float32)
        buf61 = empty((1568, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_7, x_8], Original ATen: [aten._unsafe_view, aten.add, aten.clone]
        triton_poi_fused__unsafe_view_add_clone_20.run(buf32, buf52, buf56, buf57, primals_31, primals_32, buf60, buf61, 1568, 128, grid=grid(1568, 128), stream=stream0)
        del primals_32
        buf62 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_8], Original ATen: [aten.mm]
        extern_kernels.mm(buf61, reinterpret_tensor(primals_33, (128, 256), (1, 128), 0), out=buf62)
        buf63 = buf37; del buf37  # reuse
        buf64 = buf36; del buf36  # reuse
        buf65 = buf35; del buf35  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf62, buf63, buf64, buf65, 3328, 121, grid=grid(3328), stream=stream0)
        buf66 = buf39; del buf39  # reuse
        buf67 = empty((1, 256), device='cuda', dtype=torch.float32)
        buf69 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_12.run(buf63, buf64, buf65, primals_241, primals_242, buf66, buf67, buf69, primals_241, primals_242, 256, 13, grid=grid(256), stream=stream0)
        del primals_241
        del primals_242
        buf70 = empty((8, 196, 256), device='cuda', dtype=torch.float32)
        buf71 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___mlp_ln1_bn, x_10, x_12, x_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish, aten.view]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_view_21.run(buf62, buf66, buf67, primals_34, primals_35, buf70, buf71, 401408, grid=grid(401408), stream=stream0)
        del primals_35
        buf72 = reinterpret_tensor(buf32, (1568, 128), (128, 1), 0); del buf32  # reuse
        # Source Nodes: [x_12], Original ATen: [aten.mm]
        extern_kernels.mm(buf71, reinterpret_tensor(primals_36, (256, 128), (1, 256), 0), out=buf72)
        buf73 = buf55; del buf55  # reuse
        buf74 = buf54; del buf54  # reuse
        buf75 = buf53; del buf53  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___mlp_ln2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf72, buf73, buf74, buf75, 1664, 121, grid=grid(1664), stream=stream0)
        buf76 = buf57; del buf57  # reuse
        buf77 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf79 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___mlp_ln2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_19.run(buf73, buf74, buf75, primals_244, primals_245, buf76, buf77, buf79, primals_244, primals_245, 128, 13, grid=grid(128), stream=stream0)
        del primals_244
        del primals_245
        buf80 = buf60; del buf60  # reuse
        buf81 = empty((1568, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_14, x_15], Original ATen: [aten._unsafe_view, aten.add, aten.clone]
        triton_poi_fused__unsafe_view_add_clone_22.run(buf80, buf72, buf76, buf77, primals_37, primals_38, buf81, 200704, grid=grid(200704), stream=stream0)
        del primals_38
        buf82 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_15], Original ATen: [aten.mm]
        extern_kernels.mm(buf81, reinterpret_tensor(primals_39, (128, 256), (1, 128), 0), out=buf82)
        buf83 = buf65; del buf65  # reuse
        buf84 = buf64; del buf64  # reuse
        buf85 = buf63; del buf63  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___1___attn_qkv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf82, buf83, buf84, buf85, 3328, 121, grid=grid(3328), stream=stream0)
        buf86 = buf67; del buf67  # reuse
        buf87 = empty((1, 256), device='cuda', dtype=torch.float32)
        buf89 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___1___attn_qkv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_12.run(buf83, buf84, buf85, primals_247, primals_248, buf86, buf87, buf89, primals_247, primals_248, 256, 13, grid=grid(256), stream=stream0)
        del primals_247
        del primals_248
        buf90 = empty((8, 4, 196, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf82, buf86, buf87, primals_40, primals_41, buf90, 100352, grid=grid(100352), stream=stream0)
        buf91 = empty((8, 4, 16, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_2], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf82, buf86, buf87, primals_40, primals_41, buf91, 512, 196, grid=grid(512, 196), stream=stream0)
        buf92 = buf44; del buf44  # reuse
        # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf90, (32, 196, 16), (3136, 16, 1), 0), reinterpret_tensor(buf91, (32, 16, 196), (3136, 196, 1), 0), out=buf92)
        buf95 = empty((8, 4, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_2, attn_3, getitem_7, mul_1], Original ATen: [aten._softmax, aten.add, aten.index, aten.mul]
        triton_per_fused__softmax_add_index_mul_15.run(buf92, primals_210, primals_2, buf95, 6272, 196, grid=grid(6272), stream=stream0)
        del primals_2
        buf96 = empty((8, 4, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf82, buf86, buf87, primals_40, primals_41, buf96, 200704, grid=grid(200704), stream=stream0)
        del primals_41
        buf97 = empty((32, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf95, (32, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf96, (32, 196, 32), (6272, 32, 1), 0), out=buf97)
        buf98 = empty((8, 196, 128), device='cuda', dtype=torch.float32)
        buf99 = empty((1568, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___1___attn_proj_act, x_16, x_17], Original ATen: [aten._unsafe_view, aten.clone, aten.hardswish, aten.view]
        triton_poi_fused__unsafe_view_clone_hardswish_view_17.run(buf97, buf98, buf99, 200704, grid=grid(200704), stream=stream0)
        buf100 = reinterpret_tensor(buf97, (1568, 128), (128, 1), 0); del buf97  # reuse
        # Source Nodes: [x_17], Original ATen: [aten.mm]
        extern_kernels.mm(buf99, reinterpret_tensor(primals_42, (128, 128), (1, 128), 0), out=buf100)
        buf101 = buf75; del buf75  # reuse
        buf102 = buf74; del buf74  # reuse
        buf103 = buf73; del buf73  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___1___attn_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf100, buf101, buf102, buf103, 1664, 121, grid=grid(1664), stream=stream0)
        buf104 = buf77; del buf77  # reuse
        buf105 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf107 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___1___attn_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_19.run(buf101, buf102, buf103, primals_250, primals_251, buf104, buf105, buf107, primals_250, primals_251, 128, 13, grid=grid(128), stream=stream0)
        del primals_250
        del primals_251
        buf108 = buf80; del buf80  # reuse
        buf109 = empty((1568, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_19, x_20], Original ATen: [aten._unsafe_view, aten.add, aten.clone]
        triton_poi_fused__unsafe_view_add_clone_22.run(buf108, buf100, buf104, buf105, primals_43, primals_44, buf109, 200704, grid=grid(200704), stream=stream0)
        del primals_44
        buf110 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_20], Original ATen: [aten.mm]
        extern_kernels.mm(buf109, reinterpret_tensor(primals_45, (128, 256), (1, 128), 0), out=buf110)
        buf111 = buf85; del buf85  # reuse
        buf112 = buf84; del buf84  # reuse
        buf113 = buf83; del buf83  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___1___mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf110, buf111, buf112, buf113, 3328, 121, grid=grid(3328), stream=stream0)
        buf114 = buf87; del buf87  # reuse
        buf115 = empty((1, 256), device='cuda', dtype=torch.float32)
        buf117 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___1___mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_12.run(buf111, buf112, buf113, primals_253, primals_254, buf114, buf115, buf117, primals_253, primals_254, 256, 13, grid=grid(256), stream=stream0)
        del primals_253
        del primals_254
        buf118 = empty((8, 196, 256), device='cuda', dtype=torch.float32)
        buf119 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___1___mlp_ln1_bn, x_21, x_22, x_24], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish, aten.view]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_view_21.run(buf110, buf114, buf115, primals_46, primals_47, buf118, buf119, 401408, grid=grid(401408), stream=stream0)
        del primals_47
        buf120 = empty((1568, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_24], Original ATen: [aten.mm]
        extern_kernels.mm(buf119, reinterpret_tensor(primals_48, (256, 128), (1, 256), 0), out=buf120)
        buf121 = buf103; del buf103  # reuse
        buf122 = buf102; del buf102  # reuse
        buf123 = buf101; del buf101  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___1___mlp_ln2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf120, buf121, buf122, buf123, 1664, 121, grid=grid(1664), stream=stream0)
        buf124 = buf105; del buf105  # reuse
        buf125 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf127 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___1___mlp_ln2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_19.run(buf121, buf122, buf123, primals_256, primals_257, buf124, buf125, buf127, primals_256, primals_257, 128, 13, grid=grid(128), stream=stream0)
        del primals_256
        del primals_257
        buf128 = buf108; del buf108  # reuse
        buf129 = empty((1568, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_26, x_27], Original ATen: [aten._unsafe_view, aten.add, aten.clone]
        triton_poi_fused__unsafe_view_add_clone_22.run(buf128, buf120, buf124, buf125, primals_49, primals_50, buf129, 200704, grid=grid(200704), stream=stream0)
        del primals_50
        buf130 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_27], Original ATen: [aten.mm]
        extern_kernels.mm(buf129, reinterpret_tensor(primals_51, (128, 256), (1, 128), 0), out=buf130)
        buf131 = buf113; del buf113  # reuse
        buf132 = buf112; del buf112  # reuse
        buf133 = buf111; del buf111  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___2___attn_qkv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf130, buf131, buf132, buf133, 3328, 121, grid=grid(3328), stream=stream0)
        buf134 = buf115; del buf115  # reuse
        buf135 = empty((1, 256), device='cuda', dtype=torch.float32)
        buf137 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___2___attn_qkv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_12.run(buf131, buf132, buf133, primals_259, primals_260, buf134, buf135, buf137, primals_259, primals_260, 256, 13, grid=grid(256), stream=stream0)
        del primals_259
        del primals_260
        buf138 = empty((8, 4, 196, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf130, buf134, buf135, primals_52, primals_53, buf138, 100352, grid=grid(100352), stream=stream0)
        buf139 = empty((8, 4, 16, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_4], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf130, buf134, buf135, primals_52, primals_53, buf139, 512, 196, grid=grid(512, 196), stream=stream0)
        buf140 = buf92; del buf92  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf138, (32, 196, 16), (3136, 16, 1), 0), reinterpret_tensor(buf139, (32, 16, 196), (3136, 196, 1), 0), out=buf140)
        buf143 = empty((8, 4, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_4, attn_5, getitem_11, mul_2], Original ATen: [aten._softmax, aten.add, aten.index, aten.mul]
        triton_per_fused__softmax_add_index_mul_15.run(buf140, primals_211, primals_3, buf143, 6272, 196, grid=grid(6272), stream=stream0)
        del primals_3
        buf144 = empty((8, 4, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_5], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf130, buf134, buf135, primals_52, primals_53, buf144, 200704, grid=grid(200704), stream=stream0)
        del primals_53
        buf145 = empty((32, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf143, (32, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf144, (32, 196, 32), (6272, 32, 1), 0), out=buf145)
        buf146 = empty((8, 196, 128), device='cuda', dtype=torch.float32)
        buf147 = empty((1568, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___2___attn_proj_act, x_28, x_29], Original ATen: [aten._unsafe_view, aten.clone, aten.hardswish, aten.view]
        triton_poi_fused__unsafe_view_clone_hardswish_view_17.run(buf145, buf146, buf147, 200704, grid=grid(200704), stream=stream0)
        buf148 = reinterpret_tensor(buf145, (1568, 128), (128, 1), 0); del buf145  # reuse
        # Source Nodes: [x_29], Original ATen: [aten.mm]
        extern_kernels.mm(buf147, reinterpret_tensor(primals_54, (128, 128), (1, 128), 0), out=buf148)
        buf149 = buf123; del buf123  # reuse
        buf150 = buf122; del buf122  # reuse
        buf151 = buf121; del buf121  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___2___attn_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf148, buf149, buf150, buf151, 1664, 121, grid=grid(1664), stream=stream0)
        buf152 = buf125; del buf125  # reuse
        buf153 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf155 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___2___attn_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_19.run(buf149, buf150, buf151, primals_262, primals_263, buf152, buf153, buf155, primals_262, primals_263, 128, 13, grid=grid(128), stream=stream0)
        del primals_262
        del primals_263
        buf156 = buf128; del buf128  # reuse
        buf157 = empty((1568, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_31, x_32], Original ATen: [aten._unsafe_view, aten.add, aten.clone]
        triton_poi_fused__unsafe_view_add_clone_22.run(buf156, buf148, buf152, buf153, primals_55, primals_56, buf157, 200704, grid=grid(200704), stream=stream0)
        del primals_56
        buf158 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten.mm]
        extern_kernels.mm(buf157, reinterpret_tensor(primals_57, (128, 256), (1, 128), 0), out=buf158)
        buf159 = buf133; del buf133  # reuse
        buf160 = buf132; del buf132  # reuse
        buf161 = buf131; del buf131  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___2___mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf158, buf159, buf160, buf161, 3328, 121, grid=grid(3328), stream=stream0)
        buf162 = buf135; del buf135  # reuse
        buf163 = empty((1, 256), device='cuda', dtype=torch.float32)
        buf165 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___2___mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_12.run(buf159, buf160, buf161, primals_265, primals_266, buf162, buf163, buf165, primals_265, primals_266, 256, 13, grid=grid(256), stream=stream0)
        del primals_265
        del primals_266
        buf166 = empty((8, 196, 256), device='cuda', dtype=torch.float32)
        buf167 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___2___mlp_ln1_bn, x_33, x_34, x_36], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish, aten.view]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_view_21.run(buf158, buf162, buf163, primals_58, primals_59, buf166, buf167, 401408, grid=grid(401408), stream=stream0)
        del primals_59
        buf168 = empty((1568, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_36], Original ATen: [aten.mm]
        extern_kernels.mm(buf167, reinterpret_tensor(primals_60, (256, 128), (1, 256), 0), out=buf168)
        buf169 = buf151; del buf151  # reuse
        buf170 = buf150; del buf150  # reuse
        buf171 = buf149; del buf149  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___2___mlp_ln2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf168, buf169, buf170, buf171, 1664, 121, grid=grid(1664), stream=stream0)
        buf172 = buf153; del buf153  # reuse
        buf173 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf175 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___2___mlp_ln2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_19.run(buf169, buf170, buf171, primals_268, primals_269, buf172, buf173, buf175, primals_268, primals_269, 128, 13, grid=grid(128), stream=stream0)
        del primals_268
        del primals_269
        buf176 = buf156; del buf156  # reuse
        buf177 = empty((1568, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_38, x_39], Original ATen: [aten._unsafe_view, aten.add, aten.clone]
        triton_poi_fused__unsafe_view_add_clone_22.run(buf176, buf168, buf172, buf173, primals_61, primals_62, buf177, 200704, grid=grid(200704), stream=stream0)
        del primals_62
        buf178 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_39], Original ATen: [aten.mm]
        extern_kernels.mm(buf177, reinterpret_tensor(primals_63, (128, 256), (1, 128), 0), out=buf178)
        buf179 = buf161; del buf161  # reuse
        buf180 = buf160; del buf160  # reuse
        buf181 = buf159; del buf159  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___3___attn_qkv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf178, buf179, buf180, buf181, 3328, 121, grid=grid(3328), stream=stream0)
        buf182 = buf163; del buf163  # reuse
        buf183 = empty((1, 256), device='cuda', dtype=torch.float32)
        buf185 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___3___attn_qkv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_12.run(buf179, buf180, buf181, primals_271, primals_272, buf182, buf183, buf185, primals_271, primals_272, 256, 13, grid=grid(256), stream=stream0)
        del primals_271
        del primals_272
        buf186 = empty((8, 4, 196, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf178, buf182, buf183, primals_64, primals_65, buf186, 100352, grid=grid(100352), stream=stream0)
        buf187 = empty((8, 4, 16, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_14.run(buf178, buf182, buf183, primals_64, primals_65, buf187, 512, 196, grid=grid(512, 196), stream=stream0)
        buf188 = buf140; del buf140  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf186, (32, 196, 16), (3136, 16, 1), 0), reinterpret_tensor(buf187, (32, 16, 196), (3136, 196, 1), 0), out=buf188)
        buf191 = empty((8, 4, 196, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_6, attn_7, getitem_15, mul_3], Original ATen: [aten._softmax, aten.add, aten.index, aten.mul]
        triton_per_fused__softmax_add_index_mul_15.run(buf188, primals_212, primals_4, buf191, 6272, 196, grid=grid(6272), stream=stream0)
        del buf188
        del primals_4
        buf192 = empty((8, 4, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_7], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf178, buf182, buf183, primals_64, primals_65, buf192, 200704, grid=grid(200704), stream=stream0)
        del primals_65
        buf193 = empty((32, 196, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf191, (32, 196, 196), (38416, 196, 1), 0), reinterpret_tensor(buf192, (32, 196, 32), (6272, 32, 1), 0), out=buf193)
        buf194 = empty((8, 196, 128), device='cuda', dtype=torch.float32)
        buf195 = empty((1568, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___3___attn_proj_act, x_40, x_41], Original ATen: [aten._unsafe_view, aten.clone, aten.hardswish, aten.view]
        triton_poi_fused__unsafe_view_clone_hardswish_view_17.run(buf193, buf194, buf195, 200704, grid=grid(200704), stream=stream0)
        buf196 = reinterpret_tensor(buf193, (1568, 128), (128, 1), 0); del buf193  # reuse
        # Source Nodes: [x_41], Original ATen: [aten.mm]
        extern_kernels.mm(buf195, reinterpret_tensor(primals_66, (128, 128), (1, 128), 0), out=buf196)
        buf197 = buf171; del buf171  # reuse
        buf198 = buf170; del buf170  # reuse
        buf199 = buf169; del buf169  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___3___attn_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf196, buf197, buf198, buf199, 1664, 121, grid=grid(1664), stream=stream0)
        buf200 = buf173; del buf173  # reuse
        buf201 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf203 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___3___attn_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_19.run(buf197, buf198, buf199, primals_274, primals_275, buf200, buf201, buf203, primals_274, primals_275, 128, 13, grid=grid(128), stream=stream0)
        del primals_274
        del primals_275
        buf204 = buf176; del buf176  # reuse
        buf205 = empty((1568, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_43, x_44], Original ATen: [aten._unsafe_view, aten.add, aten.clone]
        triton_poi_fused__unsafe_view_add_clone_22.run(buf204, buf196, buf200, buf201, primals_67, primals_68, buf205, 200704, grid=grid(200704), stream=stream0)
        del primals_68
        buf206 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44], Original ATen: [aten.mm]
        extern_kernels.mm(buf205, reinterpret_tensor(primals_69, (128, 256), (1, 128), 0), out=buf206)
        buf207 = buf181; del buf181  # reuse
        buf208 = buf180; del buf180  # reuse
        buf209 = buf179; del buf179  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___3___mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf206, buf207, buf208, buf209, 3328, 121, grid=grid(3328), stream=stream0)
        buf210 = buf183; del buf183  # reuse
        buf211 = empty((1, 256), device='cuda', dtype=torch.float32)
        buf213 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___3___mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_12.run(buf207, buf208, buf209, primals_277, primals_278, buf210, buf211, buf213, primals_277, primals_278, 256, 13, grid=grid(256), stream=stream0)
        del buf207
        del buf208
        del buf209
        del primals_277
        del primals_278
        buf214 = empty((8, 196, 256), device='cuda', dtype=torch.float32)
        buf215 = empty((1568, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___3___mlp_ln1_bn, x_45, x_46, x_48], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish, aten.view]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_view_21.run(buf206, buf210, buf211, primals_70, primals_71, buf214, buf215, 401408, grid=grid(401408), stream=stream0)
        del primals_71
        buf216 = empty((1568, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_48], Original ATen: [aten.mm]
        extern_kernels.mm(buf215, reinterpret_tensor(primals_72, (256, 128), (1, 256), 0), out=buf216)
        buf217 = buf199; del buf199  # reuse
        buf218 = buf198; del buf198  # reuse
        buf219 = buf197; del buf197  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___3___mlp_ln2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf216, buf217, buf218, buf219, 1664, 121, grid=grid(1664), stream=stream0)
        buf220 = buf201; del buf201  # reuse
        buf221 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf223 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___3___mlp_ln2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_19.run(buf217, buf218, buf219, primals_280, primals_281, buf220, buf221, buf223, primals_280, primals_281, 128, 13, grid=grid(128), stream=stream0)
        del buf217
        del buf218
        del buf219
        del primals_280
        del primals_281
        buf224 = buf204; del buf204  # reuse
        buf225 = empty((1568, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_51, x_52], Original ATen: [aten._unsafe_view, aten.add, aten.clone]
        triton_poi_fused__unsafe_view_add_clone_22.run(buf224, buf216, buf220, buf221, primals_73, primals_74, buf225, 200704, grid=grid(200704), stream=stream0)
        del primals_74
        buf226 = empty((1568, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_52], Original ATen: [aten.mm]
        extern_kernels.mm(buf225, reinterpret_tensor(primals_75, (128, 640), (1, 128), 0), out=buf226)
        buf227 = empty_strided((1, 640, 13), (8320, 1, 640), device='cuda', dtype=torch.float32)
        buf228 = empty_strided((1, 640, 13), (8320, 1, 640), device='cuda', dtype=torch.float32)
        buf229 = empty_strided((1, 640, 13), (8320, 1, 640), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stages___1___downsample_attn_downsample_kv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_23.run(buf226, buf227, buf228, buf229, 8320, 121, grid=grid(8320), stream=stream0)
        buf230 = empty((1, 640), device='cuda', dtype=torch.float32)
        buf231 = empty((1, 640), device='cuda', dtype=torch.float32)
        buf233 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stages___1___downsample_attn_downsample_kv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_24.run(buf227, buf228, buf229, primals_283, primals_284, buf230, buf231, buf233, primals_283, primals_284, 640, 13, grid=grid(640), stream=stream0)
        del buf227
        del buf228
        del buf229
        del primals_283
        del primals_284
        buf234 = empty((392, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_55], Original ATen: [aten.view]
        triton_poi_fused_view_25.run(buf224, buf234, 50176, grid=grid(50176), stream=stream0)
        buf235 = empty((392, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_55], Original ATen: [aten.mm]
        extern_kernels.mm(buf234, reinterpret_tensor(primals_78, (128, 128), (1, 128), 0), out=buf235)
        buf236 = empty_strided((1, 128, 4), (512, 1, 128), device='cuda', dtype=torch.float32)
        buf237 = empty_strided((1, 128, 4), (512, 1, 128), device='cuda', dtype=torch.float32)
        buf238 = empty_strided((1, 128, 4), (512, 1, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stages___1___downsample_attn_downsample_q_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_26.run(buf235, buf236, buf237, buf238, 512, 98, grid=grid(512), stream=stream0)
        buf239 = buf221; del buf221  # reuse
        buf240 = empty((1, 128), device='cuda', dtype=torch.float32)
        buf242 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stages___1___downsample_attn_downsample_q_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf236, buf237, buf238, primals_286, primals_287, buf239, buf240, buf242, primals_286, primals_287, 128, 4, grid=grid(128), stream=stream0)
        del primals_286
        del primals_287
        buf243 = empty((8, 8, 49, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_28.run(buf235, buf239, buf240, primals_79, primals_80, buf243, 50176, grid=grid(50176), stream=stream0)
        del buf240
        del primals_80
        buf244 = reinterpret_tensor(buf224, (8, 8, 16, 196), (25088, 3136, 196, 1), 0); del buf224  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.clone]
        triton_poi_fused_clone_29.run(buf226, buf230, buf231, primals_76, primals_77, buf244, 1024, 196, grid=grid(1024, 196), stream=stream0)
        buf245 = empty((64, 49, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf243, (64, 49, 16), (784, 16, 1), 0), reinterpret_tensor(buf244, (64, 16, 196), (3136, 196, 1), 0), out=buf245)
        buf248 = empty((8, 8, 49, 196), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_8, attn_9, getitem_19, mul_4], Original ATen: [aten._softmax, aten.add, aten.index, aten.mul]
        triton_per_fused__softmax_add_index_mul_30.run(buf245, primals_213, primals_5, buf248, 3136, 196, grid=grid(3136), stream=stream0)
        del buf245
        del primals_5
        buf249 = empty((8, 8, 196, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_31.run(buf226, buf230, buf231, primals_76, primals_77, buf249, 802816, grid=grid(802816), stream=stream0)
        del buf231
        del primals_77
        buf250 = empty((64, 49, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf248, (64, 49, 196), (9604, 196, 1), 0), reinterpret_tensor(buf249, (64, 196, 64), (12544, 64, 1), 0), out=buf250)
        buf251 = empty((8, 49, 512), device='cuda', dtype=torch.float32)
        buf252 = empty((392, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stages___1___downsample_attn_downsample_proj_act, x_56, x_57], Original ATen: [aten._unsafe_view, aten.clone, aten.hardswish, aten.view]
        triton_poi_fused__unsafe_view_clone_hardswish_view_32.run(buf250, buf251, buf252, 200704, grid=grid(200704), stream=stream0)
        buf253 = empty((392, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_57], Original ATen: [aten.mm]
        extern_kernels.mm(buf252, reinterpret_tensor(primals_81, (512, 256), (1, 512), 0), out=buf253)
        buf254 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        buf255 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        buf256 = empty_strided((1, 256, 4), (1024, 1, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stages___1___downsample_attn_downsample_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf253, buf254, buf255, buf256, 1024, 98, grid=grid(1024), stream=stream0)
        buf257 = buf211; del buf211  # reuse
        buf258 = empty((1, 256), device='cuda', dtype=torch.float32)
        buf260 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stages___1___downsample_attn_downsample_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_34.run(buf254, buf255, buf256, primals_289, primals_290, buf257, buf258, buf260, primals_289, primals_290, 256, 4, grid=grid(256), stream=stream0)
        del primals_289
        del primals_290
        buf261 = empty((392, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stages___1___downsample_attn_downsample_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_35.run(buf253, buf257, buf258, primals_82, primals_83, buf261, 100352, grid=grid(100352), stream=stream0)
        del primals_83
        buf262 = reinterpret_tensor(buf250, (392, 512), (512, 1), 0); del buf250  # reuse
        # Source Nodes: [x_60], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf261, (392, 256), (256, 1), 0), reinterpret_tensor(primals_84, (256, 512), (1, 256), 0), out=buf262)
        buf263 = empty_strided((1, 512, 4), (2048, 1, 512), device='cuda', dtype=torch.float32)
        buf264 = empty_strided((1, 512, 4), (2048, 1, 512), device='cuda', dtype=torch.float32)
        buf265 = empty_strided((1, 512, 4), (2048, 1, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stages___1___downsample_mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf262, buf263, buf264, buf265, 2048, 98, grid=grid(2048), stream=stream0)
        buf266 = reinterpret_tensor(buf238, (1, 512), (512, 1), 0); del buf238  # reuse
        buf267 = reinterpret_tensor(buf237, (1, 512), (512, 1), 0); del buf237  # reuse
        buf269 = reinterpret_tensor(buf236, (512, ), (1, ), 0); del buf236  # reuse
        # Source Nodes: [getattr_l__mod___stages___1___downsample_mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_37.run(buf263, buf264, buf265, primals_292, primals_293, buf266, buf267, buf269, primals_292, primals_293, 512, 4, grid=grid(512), stream=stream0)
        del primals_292
        del primals_293
        buf270 = empty((8, 49, 512), device='cuda', dtype=torch.float32)
        buf271 = empty((392, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stages___1___downsample_mlp_ln1_bn, x_61, x_62, x_64], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish, aten.view]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_view_38.run(buf262, buf266, buf267, primals_85, primals_86, buf270, buf271, 200704, grid=grid(200704), stream=stream0)
        del primals_86
        buf272 = empty((392, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_64], Original ATen: [aten.mm]
        extern_kernels.mm(buf271, reinterpret_tensor(primals_87, (512, 256), (1, 512), 0), out=buf272)
        buf273 = buf256; del buf256  # reuse
        buf274 = buf255; del buf255  # reuse
        buf275 = buf254; del buf254  # reuse
        # Source Nodes: [getattr_l__mod___stages___1___downsample_mlp_ln2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf272, buf273, buf274, buf275, 1024, 98, grid=grid(1024), stream=stream0)
        buf276 = buf258; del buf258  # reuse
        buf277 = empty((1, 256), device='cuda', dtype=torch.float32)
        buf279 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stages___1___downsample_mlp_ln2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_34.run(buf273, buf274, buf275, primals_295, primals_296, buf276, buf277, buf279, primals_295, primals_296, 256, 4, grid=grid(256), stream=stream0)
        del primals_295
        del primals_296
        buf280 = empty((8, 49, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_67], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf261, buf272, buf276, buf277, primals_88, primals_89, buf280, 100352, grid=grid(100352), stream=stream0)
        del primals_89
        buf281 = empty((392, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (392, 256), (256, 1), 0), reinterpret_tensor(primals_90, (256, 512), (1, 256), 0), out=buf281)
        buf282 = buf265; del buf265  # reuse
        buf283 = buf264; del buf264  # reuse
        buf284 = buf263; del buf263  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___0___attn_qkv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf281, buf282, buf283, buf284, 2048, 98, grid=grid(2048), stream=stream0)
        buf285 = buf267; del buf267  # reuse
        buf286 = empty((1, 512), device='cuda', dtype=torch.float32)
        buf288 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___0___attn_qkv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_37.run(buf282, buf283, buf284, primals_298, primals_299, buf285, buf286, buf288, primals_298, primals_299, 512, 4, grid=grid(512), stream=stream0)
        del primals_298
        del primals_299
        buf289 = empty((8, 8, 49, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf281, buf285, buf286, primals_91, primals_92, buf289, 50176, grid=grid(50176), stream=stream0)
        buf290 = empty((8, 8, 16, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_10], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf281, buf285, buf286, primals_91, primals_92, buf290, 1024, 49, grid=grid(1024, 49), stream=stream0)
        buf291 = empty((64, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf289, (64, 49, 16), (784, 16, 1), 0), reinterpret_tensor(buf290, (64, 16, 49), (784, 49, 1), 0), out=buf291)
        buf294 = empty((8, 8, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_10, attn_11, getitem_23, mul_5], Original ATen: [aten._softmax, aten.add, aten.index, aten.mul]
        triton_per_fused__softmax_add_index_mul_42.run(buf291, primals_214, primals_6, buf294, 3136, 49, grid=grid(3136), stream=stream0)
        del primals_6
        buf295 = empty((8, 8, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_11], Original ATen: [aten.clone]
        triton_poi_fused_clone_43.run(buf281, buf285, buf286, primals_91, primals_92, buf295, 100352, grid=grid(100352), stream=stream0)
        del primals_92
        buf296 = empty((64, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf294, (64, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf295, (64, 49, 32), (1568, 32, 1), 0), out=buf296)
        buf297 = empty((8, 49, 256), device='cuda', dtype=torch.float32)
        buf298 = empty((392, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___0___attn_proj_act, x_69, x_70], Original ATen: [aten._unsafe_view, aten.clone, aten.hardswish, aten.view]
        triton_poi_fused__unsafe_view_clone_hardswish_view_44.run(buf296, buf297, buf298, 100352, grid=grid(100352), stream=stream0)
        buf299 = reinterpret_tensor(buf296, (392, 256), (256, 1), 0); del buf296  # reuse
        # Source Nodes: [x_70], Original ATen: [aten.mm]
        extern_kernels.mm(buf298, reinterpret_tensor(primals_93, (256, 256), (1, 256), 0), out=buf299)
        buf300 = buf275; del buf275  # reuse
        buf301 = buf274; del buf274  # reuse
        buf302 = buf273; del buf273  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___0___attn_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf299, buf300, buf301, buf302, 1024, 98, grid=grid(1024), stream=stream0)
        buf303 = buf277; del buf277  # reuse
        buf304 = empty((1, 256), device='cuda', dtype=torch.float32)
        buf306 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___0___attn_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_34.run(buf300, buf301, buf302, primals_301, primals_302, buf303, buf304, buf306, primals_301, primals_302, 256, 4, grid=grid(256), stream=stream0)
        del primals_301
        del primals_302
        buf307 = empty((8, 49, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_72], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf280, buf299, buf303, buf304, primals_94, primals_95, buf307, 100352, grid=grid(100352), stream=stream0)
        del primals_95
        buf308 = empty((392, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_73], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf307, (392, 256), (256, 1), 0), reinterpret_tensor(primals_96, (256, 512), (1, 256), 0), out=buf308)
        buf309 = buf284; del buf284  # reuse
        buf310 = buf283; del buf283  # reuse
        buf311 = buf282; del buf282  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___0___mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf308, buf309, buf310, buf311, 2048, 98, grid=grid(2048), stream=stream0)
        buf312 = buf286; del buf286  # reuse
        buf313 = empty((1, 512), device='cuda', dtype=torch.float32)
        buf315 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___0___mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_37.run(buf309, buf310, buf311, primals_304, primals_305, buf312, buf313, buf315, primals_304, primals_305, 512, 4, grid=grid(512), stream=stream0)
        del primals_304
        del primals_305
        buf316 = empty((8, 49, 512), device='cuda', dtype=torch.float32)
        buf317 = empty((392, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___0___mlp_ln1_bn, x_74, x_75, x_77], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish, aten.view]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_view_38.run(buf308, buf312, buf313, primals_97, primals_98, buf316, buf317, 200704, grid=grid(200704), stream=stream0)
        del primals_98
        buf318 = empty((392, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_77], Original ATen: [aten.mm]
        extern_kernels.mm(buf317, reinterpret_tensor(primals_99, (512, 256), (1, 512), 0), out=buf318)
        buf319 = buf302; del buf302  # reuse
        buf320 = buf301; del buf301  # reuse
        buf321 = buf300; del buf300  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___0___mlp_ln2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf318, buf319, buf320, buf321, 1024, 98, grid=grid(1024), stream=stream0)
        buf322 = buf304; del buf304  # reuse
        buf323 = empty((1, 256), device='cuda', dtype=torch.float32)
        buf325 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___0___mlp_ln2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_34.run(buf319, buf320, buf321, primals_307, primals_308, buf322, buf323, buf325, primals_307, primals_308, 256, 4, grid=grid(256), stream=stream0)
        del primals_307
        del primals_308
        buf326 = empty((8, 49, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_79], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf307, buf318, buf322, buf323, primals_100, primals_101, buf326, 100352, grid=grid(100352), stream=stream0)
        del primals_101
        buf327 = empty((392, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_80], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf326, (392, 256), (256, 1), 0), reinterpret_tensor(primals_102, (256, 512), (1, 256), 0), out=buf327)
        buf328 = buf311; del buf311  # reuse
        buf329 = buf310; del buf310  # reuse
        buf330 = buf309; del buf309  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___1___attn_qkv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf327, buf328, buf329, buf330, 2048, 98, grid=grid(2048), stream=stream0)
        buf331 = buf313; del buf313  # reuse
        buf332 = empty((1, 512), device='cuda', dtype=torch.float32)
        buf334 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___1___attn_qkv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_37.run(buf328, buf329, buf330, primals_310, primals_311, buf331, buf332, buf334, primals_310, primals_311, 512, 4, grid=grid(512), stream=stream0)
        del primals_310
        del primals_311
        buf335 = empty((8, 8, 49, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf327, buf331, buf332, primals_103, primals_104, buf335, 50176, grid=grid(50176), stream=stream0)
        buf336 = empty((8, 8, 16, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf327, buf331, buf332, primals_103, primals_104, buf336, 1024, 49, grid=grid(1024, 49), stream=stream0)
        buf337 = buf291; del buf291  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf335, (64, 49, 16), (784, 16, 1), 0), reinterpret_tensor(buf336, (64, 16, 49), (784, 49, 1), 0), out=buf337)
        buf340 = empty((8, 8, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_12, attn_13, getitem_27, mul_6], Original ATen: [aten._softmax, aten.add, aten.index, aten.mul]
        triton_per_fused__softmax_add_index_mul_42.run(buf337, primals_215, primals_7, buf340, 3136, 49, grid=grid(3136), stream=stream0)
        del primals_7
        buf341 = empty((8, 8, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_13], Original ATen: [aten.clone]
        triton_poi_fused_clone_43.run(buf327, buf331, buf332, primals_103, primals_104, buf341, 100352, grid=grid(100352), stream=stream0)
        del primals_104
        buf342 = empty((64, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf340, (64, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf341, (64, 49, 32), (1568, 32, 1), 0), out=buf342)
        buf343 = empty((8, 49, 256), device='cuda', dtype=torch.float32)
        buf344 = empty((392, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___1___attn_proj_act, x_81, x_82], Original ATen: [aten._unsafe_view, aten.clone, aten.hardswish, aten.view]
        triton_poi_fused__unsafe_view_clone_hardswish_view_44.run(buf342, buf343, buf344, 100352, grid=grid(100352), stream=stream0)
        buf345 = reinterpret_tensor(buf342, (392, 256), (256, 1), 0); del buf342  # reuse
        # Source Nodes: [x_82], Original ATen: [aten.mm]
        extern_kernels.mm(buf344, reinterpret_tensor(primals_105, (256, 256), (1, 256), 0), out=buf345)
        buf346 = buf321; del buf321  # reuse
        buf347 = buf320; del buf320  # reuse
        buf348 = buf319; del buf319  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___1___attn_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf345, buf346, buf347, buf348, 1024, 98, grid=grid(1024), stream=stream0)
        buf349 = buf323; del buf323  # reuse
        buf350 = empty((1, 256), device='cuda', dtype=torch.float32)
        buf352 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___1___attn_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_34.run(buf346, buf347, buf348, primals_313, primals_314, buf349, buf350, buf352, primals_313, primals_314, 256, 4, grid=grid(256), stream=stream0)
        del primals_313
        del primals_314
        buf353 = empty((8, 49, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_84], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf326, buf345, buf349, buf350, primals_106, primals_107, buf353, 100352, grid=grid(100352), stream=stream0)
        del primals_107
        buf354 = empty((392, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_85], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf353, (392, 256), (256, 1), 0), reinterpret_tensor(primals_108, (256, 512), (1, 256), 0), out=buf354)
        buf355 = buf330; del buf330  # reuse
        buf356 = buf329; del buf329  # reuse
        buf357 = buf328; del buf328  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___1___mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf354, buf355, buf356, buf357, 2048, 98, grid=grid(2048), stream=stream0)
        buf358 = buf332; del buf332  # reuse
        buf359 = empty((1, 512), device='cuda', dtype=torch.float32)
        buf361 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___1___mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_37.run(buf355, buf356, buf357, primals_316, primals_317, buf358, buf359, buf361, primals_316, primals_317, 512, 4, grid=grid(512), stream=stream0)
        del primals_316
        del primals_317
        buf362 = empty((8, 49, 512), device='cuda', dtype=torch.float32)
        buf363 = empty((392, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___1___mlp_ln1_bn, x_86, x_87, x_89], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish, aten.view]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_view_38.run(buf354, buf358, buf359, primals_109, primals_110, buf362, buf363, 200704, grid=grid(200704), stream=stream0)
        del primals_110
        buf364 = empty((392, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89], Original ATen: [aten.mm]
        extern_kernels.mm(buf363, reinterpret_tensor(primals_111, (512, 256), (1, 512), 0), out=buf364)
        buf365 = buf348; del buf348  # reuse
        buf366 = buf347; del buf347  # reuse
        buf367 = buf346; del buf346  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___1___mlp_ln2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf364, buf365, buf366, buf367, 1024, 98, grid=grid(1024), stream=stream0)
        buf368 = buf350; del buf350  # reuse
        buf369 = empty((1, 256), device='cuda', dtype=torch.float32)
        buf371 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___1___mlp_ln2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_34.run(buf365, buf366, buf367, primals_319, primals_320, buf368, buf369, buf371, primals_319, primals_320, 256, 4, grid=grid(256), stream=stream0)
        del primals_319
        del primals_320
        buf372 = empty((8, 49, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_91], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf353, buf364, buf368, buf369, primals_112, primals_113, buf372, 100352, grid=grid(100352), stream=stream0)
        del primals_113
        buf373 = empty((392, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_92], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf372, (392, 256), (256, 1), 0), reinterpret_tensor(primals_114, (256, 512), (1, 256), 0), out=buf373)
        buf374 = buf357; del buf357  # reuse
        buf375 = buf356; del buf356  # reuse
        buf376 = buf355; del buf355  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___2___attn_qkv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf373, buf374, buf375, buf376, 2048, 98, grid=grid(2048), stream=stream0)
        buf377 = buf359; del buf359  # reuse
        buf378 = empty((1, 512), device='cuda', dtype=torch.float32)
        buf380 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___2___attn_qkv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_37.run(buf374, buf375, buf376, primals_322, primals_323, buf377, buf378, buf380, primals_322, primals_323, 512, 4, grid=grid(512), stream=stream0)
        del primals_322
        del primals_323
        buf381 = empty((8, 8, 49, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf373, buf377, buf378, primals_115, primals_116, buf381, 50176, grid=grid(50176), stream=stream0)
        buf382 = empty((8, 8, 16, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_14], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf373, buf377, buf378, primals_115, primals_116, buf382, 1024, 49, grid=grid(1024, 49), stream=stream0)
        buf383 = buf337; del buf337  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf381, (64, 49, 16), (784, 16, 1), 0), reinterpret_tensor(buf382, (64, 16, 49), (784, 49, 1), 0), out=buf383)
        buf386 = empty((8, 8, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_14, attn_15, getitem_31, mul_7], Original ATen: [aten._softmax, aten.add, aten.index, aten.mul]
        triton_per_fused__softmax_add_index_mul_42.run(buf383, primals_216, primals_8, buf386, 3136, 49, grid=grid(3136), stream=stream0)
        del primals_8
        buf387 = empty((8, 8, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_15], Original ATen: [aten.clone]
        triton_poi_fused_clone_43.run(buf373, buf377, buf378, primals_115, primals_116, buf387, 100352, grid=grid(100352), stream=stream0)
        del primals_116
        buf388 = empty((64, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf386, (64, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf387, (64, 49, 32), (1568, 32, 1), 0), out=buf388)
        buf389 = empty((8, 49, 256), device='cuda', dtype=torch.float32)
        buf390 = empty((392, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___2___attn_proj_act, x_93, x_94], Original ATen: [aten._unsafe_view, aten.clone, aten.hardswish, aten.view]
        triton_poi_fused__unsafe_view_clone_hardswish_view_44.run(buf388, buf389, buf390, 100352, grid=grid(100352), stream=stream0)
        buf391 = reinterpret_tensor(buf388, (392, 256), (256, 1), 0); del buf388  # reuse
        # Source Nodes: [x_94], Original ATen: [aten.mm]
        extern_kernels.mm(buf390, reinterpret_tensor(primals_117, (256, 256), (1, 256), 0), out=buf391)
        buf392 = buf367; del buf367  # reuse
        buf393 = buf366; del buf366  # reuse
        buf394 = buf365; del buf365  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___2___attn_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf391, buf392, buf393, buf394, 1024, 98, grid=grid(1024), stream=stream0)
        buf395 = buf369; del buf369  # reuse
        buf396 = empty((1, 256), device='cuda', dtype=torch.float32)
        buf398 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___2___attn_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_34.run(buf392, buf393, buf394, primals_325, primals_326, buf395, buf396, buf398, primals_325, primals_326, 256, 4, grid=grid(256), stream=stream0)
        del primals_325
        del primals_326
        buf399 = empty((8, 49, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_96], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf372, buf391, buf395, buf396, primals_118, primals_119, buf399, 100352, grid=grid(100352), stream=stream0)
        del primals_119
        buf400 = empty((392, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_97], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf399, (392, 256), (256, 1), 0), reinterpret_tensor(primals_120, (256, 512), (1, 256), 0), out=buf400)
        buf401 = buf376; del buf376  # reuse
        buf402 = buf375; del buf375  # reuse
        buf403 = buf374; del buf374  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___2___mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf400, buf401, buf402, buf403, 2048, 98, grid=grid(2048), stream=stream0)
        buf404 = buf378; del buf378  # reuse
        buf405 = empty((1, 512), device='cuda', dtype=torch.float32)
        buf407 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___2___mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_37.run(buf401, buf402, buf403, primals_328, primals_329, buf404, buf405, buf407, primals_328, primals_329, 512, 4, grid=grid(512), stream=stream0)
        del primals_328
        del primals_329
        buf408 = empty((8, 49, 512), device='cuda', dtype=torch.float32)
        buf409 = empty((392, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___2___mlp_ln1_bn, x_101, x_98, x_99], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish, aten.view]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_view_38.run(buf400, buf404, buf405, primals_121, primals_122, buf408, buf409, 200704, grid=grid(200704), stream=stream0)
        del primals_122
        buf410 = empty((392, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_101], Original ATen: [aten.mm]
        extern_kernels.mm(buf409, reinterpret_tensor(primals_123, (512, 256), (1, 512), 0), out=buf410)
        buf411 = buf394; del buf394  # reuse
        buf412 = buf393; del buf393  # reuse
        buf413 = buf392; del buf392  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___2___mlp_ln2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf410, buf411, buf412, buf413, 1024, 98, grid=grid(1024), stream=stream0)
        buf414 = buf396; del buf396  # reuse
        buf415 = empty((1, 256), device='cuda', dtype=torch.float32)
        buf417 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___2___mlp_ln2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_34.run(buf411, buf412, buf413, primals_331, primals_332, buf414, buf415, buf417, primals_331, primals_332, 256, 4, grid=grid(256), stream=stream0)
        del primals_331
        del primals_332
        buf418 = empty((8, 49, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_103], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf399, buf410, buf414, buf415, primals_124, primals_125, buf418, 100352, grid=grid(100352), stream=stream0)
        del primals_125
        buf419 = empty((392, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_104], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf418, (392, 256), (256, 1), 0), reinterpret_tensor(primals_126, (256, 512), (1, 256), 0), out=buf419)
        buf420 = buf403; del buf403  # reuse
        buf421 = buf402; del buf402  # reuse
        buf422 = buf401; del buf401  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___3___attn_qkv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf419, buf420, buf421, buf422, 2048, 98, grid=grid(2048), stream=stream0)
        buf423 = buf405; del buf405  # reuse
        buf424 = empty((1, 512), device='cuda', dtype=torch.float32)
        buf426 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___3___attn_qkv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_37.run(buf420, buf421, buf422, primals_334, primals_335, buf423, buf424, buf426, primals_334, primals_335, 512, 4, grid=grid(512), stream=stream0)
        del primals_334
        del primals_335
        buf427 = empty((8, 8, 49, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_40.run(buf419, buf423, buf424, primals_127, primals_128, buf427, 50176, grid=grid(50176), stream=stream0)
        buf428 = empty((8, 8, 16, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_41.run(buf419, buf423, buf424, primals_127, primals_128, buf428, 1024, 49, grid=grid(1024, 49), stream=stream0)
        buf429 = buf383; del buf383  # reuse
        # Source Nodes: [matmul_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf427, (64, 49, 16), (784, 16, 1), 0), reinterpret_tensor(buf428, (64, 16, 49), (784, 49, 1), 0), out=buf429)
        buf432 = empty((8, 8, 49, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_16, attn_17, getitem_35, mul_8], Original ATen: [aten._softmax, aten.add, aten.index, aten.mul]
        triton_per_fused__softmax_add_index_mul_42.run(buf429, primals_217, primals_9, buf432, 3136, 49, grid=grid(3136), stream=stream0)
        del buf429
        del primals_9
        buf433 = empty((8, 8, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_17], Original ATen: [aten.clone]
        triton_poi_fused_clone_43.run(buf419, buf423, buf424, primals_127, primals_128, buf433, 100352, grid=grid(100352), stream=stream0)
        del primals_128
        buf434 = empty((64, 49, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf432, (64, 49, 49), (2401, 49, 1), 0), reinterpret_tensor(buf433, (64, 49, 32), (1568, 32, 1), 0), out=buf434)
        buf435 = empty((8, 49, 256), device='cuda', dtype=torch.float32)
        buf436 = empty((392, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___3___attn_proj_act, x_105, x_106], Original ATen: [aten._unsafe_view, aten.clone, aten.hardswish, aten.view]
        triton_poi_fused__unsafe_view_clone_hardswish_view_44.run(buf434, buf435, buf436, 100352, grid=grid(100352), stream=stream0)
        buf437 = reinterpret_tensor(buf434, (392, 256), (256, 1), 0); del buf434  # reuse
        # Source Nodes: [x_106], Original ATen: [aten.mm]
        extern_kernels.mm(buf436, reinterpret_tensor(primals_129, (256, 256), (1, 256), 0), out=buf437)
        buf438 = buf413; del buf413  # reuse
        buf439 = buf412; del buf412  # reuse
        buf440 = buf411; del buf411  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___3___attn_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf437, buf438, buf439, buf440, 1024, 98, grid=grid(1024), stream=stream0)
        buf441 = buf415; del buf415  # reuse
        buf442 = empty((1, 256), device='cuda', dtype=torch.float32)
        buf444 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___3___attn_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_34.run(buf438, buf439, buf440, primals_337, primals_338, buf441, buf442, buf444, primals_337, primals_338, 256, 4, grid=grid(256), stream=stream0)
        del primals_337
        del primals_338
        buf445 = empty((8, 49, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_108], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf418, buf437, buf441, buf442, primals_130, primals_131, buf445, 100352, grid=grid(100352), stream=stream0)
        del primals_131
        buf446 = empty((392, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_109], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf445, (392, 256), (256, 1), 0), reinterpret_tensor(primals_132, (256, 512), (1, 256), 0), out=buf446)
        buf447 = buf422; del buf422  # reuse
        buf448 = buf421; del buf421  # reuse
        buf449 = buf420; del buf420  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___3___mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_36.run(buf446, buf447, buf448, buf449, 2048, 98, grid=grid(2048), stream=stream0)
        buf450 = buf424; del buf424  # reuse
        buf451 = empty((1, 512), device='cuda', dtype=torch.float32)
        buf453 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___3___mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_37.run(buf447, buf448, buf449, primals_340, primals_341, buf450, buf451, buf453, primals_340, primals_341, 512, 4, grid=grid(512), stream=stream0)
        del buf447
        del buf448
        del buf449
        del primals_340
        del primals_341
        buf454 = empty((8, 49, 512), device='cuda', dtype=torch.float32)
        buf455 = empty((392, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___3___mlp_ln1_bn, x_110, x_111, x_113], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish, aten.view]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_view_38.run(buf446, buf450, buf451, primals_133, primals_134, buf454, buf455, 200704, grid=grid(200704), stream=stream0)
        del buf451
        del primals_134
        buf456 = empty((392, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_113], Original ATen: [aten.mm]
        extern_kernels.mm(buf455, reinterpret_tensor(primals_135, (512, 256), (1, 512), 0), out=buf456)
        buf457 = buf440; del buf440  # reuse
        buf458 = buf439; del buf439  # reuse
        buf459 = buf438; del buf438  # reuse
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___3___mlp_ln2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf456, buf457, buf458, buf459, 1024, 98, grid=grid(1024), stream=stream0)
        buf460 = buf442; del buf442  # reuse
        buf461 = empty((1, 256), device='cuda', dtype=torch.float32)
        buf463 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___3___mlp_ln2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_34.run(buf457, buf458, buf459, primals_343, primals_344, buf460, buf461, buf463, primals_343, primals_344, 256, 4, grid=grid(256), stream=stream0)
        del buf457
        del buf458
        del buf459
        del primals_343
        del primals_344
        buf464 = empty((8, 49, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_116], Original ATen: [aten.add]
        triton_poi_fused_add_39.run(buf445, buf456, buf460, buf461, primals_136, primals_137, buf464, 100352, grid=grid(100352), stream=stream0)
        del primals_137
        buf465 = empty((392, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_117], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf464, (392, 256), (256, 1), 0), reinterpret_tensor(primals_138, (256, 1280), (1, 256), 0), out=buf465)
        buf466 = empty_strided((1, 1280, 4), (5120, 1, 1280), device='cuda', dtype=torch.float32)
        buf467 = empty_strided((1, 1280, 4), (5120, 1, 1280), device='cuda', dtype=torch.float32)
        buf468 = empty_strided((1, 1280, 4), (5120, 1, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stages___2___downsample_attn_downsample_kv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_45.run(buf465, buf466, buf467, buf468, 5120, 98, grid=grid(5120), stream=stream0)
        buf469 = empty((1, 1280), device='cuda', dtype=torch.float32)
        buf470 = empty((1, 1280), device='cuda', dtype=torch.float32)
        buf472 = empty((1280, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stages___2___downsample_attn_downsample_kv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_46.run(buf466, buf467, buf468, primals_346, primals_347, buf469, buf470, buf472, primals_346, primals_347, 1280, 4, grid=grid(1280), stream=stream0)
        del buf466
        del buf467
        del buf468
        del primals_346
        del primals_347
        buf473 = empty((128, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_120], Original ATen: [aten.view]
        triton_poi_fused_view_47.run(buf464, buf473, 32768, grid=grid(32768), stream=stream0)
        buf474 = empty((128, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_120], Original ATen: [aten.mm]
        extern_kernels.mm(buf473, reinterpret_tensor(primals_141, (256, 256), (1, 256), 0), out=buf474)
        buf475 = buf461; del buf461  # reuse
        buf476 = empty((1, 256), device='cuda', dtype=torch.float32)
        buf478 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stages___2___downsample_attn_downsample_q_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_48.run(buf474, primals_349, primals_350, buf475, buf476, buf478, primals_349, primals_350, 256, 128, grid=grid(256), stream=stream0)
        del primals_349
        del primals_350
        buf479 = empty((8, 16, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_49.run(buf474, buf475, buf476, primals_142, primals_143, buf479, 32768, grid=grid(32768), stream=stream0)
        del buf476
        del primals_143
        buf480 = empty((8, 16, 16, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_18], Original ATen: [aten.clone]
        triton_poi_fused_clone_50.run(buf465, buf469, buf470, primals_139, primals_140, buf480, 2048, 49, grid=grid(2048, 49), stream=stream0)
        buf481 = empty((128, 16, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf479, (128, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf480, (128, 16, 49), (784, 49, 1), 0), out=buf481)
        buf484 = empty((8, 16, 16, 49), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_18, attn_19, getitem_39, mul_9], Original ATen: [aten._softmax, aten.add, aten.index, aten.mul]
        triton_per_fused__softmax_add_index_mul_51.run(buf481, primals_218, primals_10, buf484, 2048, 49, grid=grid(2048), stream=stream0)
        del buf481
        del primals_10
        buf485 = empty((8, 16, 49, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_52.run(buf465, buf469, buf470, primals_139, primals_140, buf485, 401408, grid=grid(401408), stream=stream0)
        del buf470
        del primals_140
        buf486 = empty((128, 16, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf484, (128, 16, 49), (784, 49, 1), 0), reinterpret_tensor(buf485, (128, 49, 64), (3136, 64, 1), 0), out=buf486)
        buf487 = empty((8, 16, 1024), device='cuda', dtype=torch.float32)
        buf488 = empty((128, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stages___2___downsample_attn_downsample_proj_act, x_121, x_122], Original ATen: [aten._unsafe_view, aten.clone, aten.hardswish, aten.view]
        triton_poi_fused__unsafe_view_clone_hardswish_view_53.run(buf486, buf487, buf488, 131072, grid=grid(131072), stream=stream0)
        del buf486
        buf489 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_122], Original ATen: [aten.mm]
        extern_kernels.mm(buf488, reinterpret_tensor(primals_144, (1024, 384), (1, 1024), 0), out=buf489)
        buf490 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf491 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf493 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stages___2___downsample_attn_downsample_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf489, primals_352, primals_353, buf490, buf491, buf493, primals_352, primals_353, 384, 128, grid=grid(384), stream=stream0)
        del primals_352
        del primals_353
        buf494 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stages___2___downsample_attn_downsample_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_55.run(buf489, buf490, buf491, primals_145, primals_146, buf494, 49152, grid=grid(49152), stream=stream0)
        del primals_146
        buf495 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_125], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf494, (128, 384), (384, 1), 0), reinterpret_tensor(primals_147, (384, 768), (1, 384), 0), out=buf495)
        buf496 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf497 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf499 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stages___2___downsample_mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf495, primals_355, primals_356, buf496, buf497, buf499, primals_355, primals_356, 768, 128, grid=grid(768), stream=stream0)
        del primals_355
        del primals_356
        buf500 = empty((8, 16, 768), device='cuda', dtype=torch.float32)
        buf501 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stages___2___downsample_mlp_ln1_bn, x_126, x_127, x_129], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish, aten.view]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_view_57.run(buf495, buf496, buf497, primals_148, primals_149, buf500, buf501, 98304, grid=grid(98304), stream=stream0)
        del primals_149
        buf502 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_129], Original ATen: [aten.mm]
        extern_kernels.mm(buf501, reinterpret_tensor(primals_150, (768, 384), (1, 768), 0), out=buf502)
        buf503 = buf491; del buf491  # reuse
        buf504 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf506 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stages___2___downsample_mlp_ln2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf502, primals_358, primals_359, buf503, buf504, buf506, primals_358, primals_359, 384, 128, grid=grid(384), stream=stream0)
        del primals_358
        del primals_359
        buf507 = empty((8, 16, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_132], Original ATen: [aten.add]
        triton_poi_fused_add_58.run(buf494, buf502, buf503, buf504, primals_151, primals_152, buf507, 49152, grid=grid(49152), stream=stream0)
        del primals_152
        buf508 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_133], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf507, (128, 384), (384, 1), 0), reinterpret_tensor(primals_153, (384, 768), (1, 384), 0), out=buf508)
        buf509 = buf497; del buf497  # reuse
        buf510 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf512 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___0___attn_qkv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf508, primals_361, primals_362, buf509, buf510, buf512, primals_361, primals_362, 768, 128, grid=grid(768), stream=stream0)
        del primals_361
        del primals_362
        buf513 = empty((8, 12, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_59.run(buf508, buf509, buf510, primals_154, primals_155, buf513, 24576, grid=grid(24576), stream=stream0)
        buf514 = empty((8, 12, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_20], Original ATen: [aten.clone]
        triton_poi_fused_clone_60.run(buf508, buf509, buf510, primals_154, primals_155, buf514, 1536, 16, grid=grid(1536, 16), stream=stream0)
        buf515 = empty((96, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf513, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf514, (96, 16, 16), (256, 16, 1), 0), out=buf515)
        buf518 = empty((8, 12, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_20, attn_21, getitem_43, mul_10], Original ATen: [aten._softmax, aten.add, aten.index, aten.mul]
        triton_per_fused__softmax_add_index_mul_61.run(buf515, primals_219, primals_11, buf518, 1536, 16, grid=grid(1536), stream=stream0)
        del primals_11
        buf519 = empty((8, 12, 16, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_21], Original ATen: [aten.clone]
        triton_poi_fused_clone_62.run(buf508, buf509, buf510, primals_154, primals_155, buf519, 49152, grid=grid(49152), stream=stream0)
        del primals_155
        buf520 = empty((96, 16, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf518, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf519, (96, 16, 32), (512, 32, 1), 0), out=buf520)
        buf521 = empty((8, 16, 384), device='cuda', dtype=torch.float32)
        buf522 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___0___attn_proj_act, x_134, x_135], Original ATen: [aten._unsafe_view, aten.clone, aten.hardswish, aten.view]
        triton_poi_fused__unsafe_view_clone_hardswish_view_63.run(buf520, buf521, buf522, 49152, grid=grid(49152), stream=stream0)
        buf523 = reinterpret_tensor(buf520, (128, 384), (384, 1), 0); del buf520  # reuse
        # Source Nodes: [x_135], Original ATen: [aten.mm]
        extern_kernels.mm(buf522, reinterpret_tensor(primals_156, (384, 384), (1, 384), 0), out=buf523)
        buf524 = buf504; del buf504  # reuse
        buf525 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf527 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___0___attn_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf523, primals_364, primals_365, buf524, buf525, buf527, primals_364, primals_365, 384, 128, grid=grid(384), stream=stream0)
        del primals_364
        del primals_365
        buf528 = empty((8, 16, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_137], Original ATen: [aten.add]
        triton_poi_fused_add_58.run(buf507, buf523, buf524, buf525, primals_157, primals_158, buf528, 49152, grid=grid(49152), stream=stream0)
        del primals_158
        buf529 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_138], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf528, (128, 384), (384, 1), 0), reinterpret_tensor(primals_159, (384, 768), (1, 384), 0), out=buf529)
        buf530 = buf510; del buf510  # reuse
        buf531 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf533 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___0___mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf529, primals_367, primals_368, buf530, buf531, buf533, primals_367, primals_368, 768, 128, grid=grid(768), stream=stream0)
        del primals_367
        del primals_368
        buf534 = empty((8, 16, 768), device='cuda', dtype=torch.float32)
        buf535 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___0___mlp_ln1_bn, x_139, x_140, x_142], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish, aten.view]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_view_57.run(buf529, buf530, buf531, primals_160, primals_161, buf534, buf535, 98304, grid=grid(98304), stream=stream0)
        del primals_161
        buf536 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_142], Original ATen: [aten.mm]
        extern_kernels.mm(buf535, reinterpret_tensor(primals_162, (768, 384), (1, 768), 0), out=buf536)
        buf537 = buf525; del buf525  # reuse
        buf538 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf540 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___0___mlp_ln2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf536, primals_370, primals_371, buf537, buf538, buf540, primals_370, primals_371, 384, 128, grid=grid(384), stream=stream0)
        del primals_370
        del primals_371
        buf541 = empty((8, 16, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_144], Original ATen: [aten.add]
        triton_poi_fused_add_58.run(buf528, buf536, buf537, buf538, primals_163, primals_164, buf541, 49152, grid=grid(49152), stream=stream0)
        del primals_164
        buf542 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_145], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf541, (128, 384), (384, 1), 0), reinterpret_tensor(primals_165, (384, 768), (1, 384), 0), out=buf542)
        buf543 = buf531; del buf531  # reuse
        buf544 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf546 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___1___attn_qkv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf542, primals_373, primals_374, buf543, buf544, buf546, primals_373, primals_374, 768, 128, grid=grid(768), stream=stream0)
        del primals_373
        del primals_374
        buf547 = reinterpret_tensor(buf515, (8, 12, 16, 16), (3072, 256, 16, 1), 0); del buf515  # reuse
        # Source Nodes: [matmul_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_59.run(buf542, buf543, buf544, primals_166, primals_167, buf547, 24576, grid=grid(24576), stream=stream0)
        buf548 = empty((8, 12, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_60.run(buf542, buf543, buf544, primals_166, primals_167, buf548, 1536, 16, grid=grid(1536, 16), stream=stream0)
        buf549 = empty((96, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf547, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf548, (96, 16, 16), (256, 16, 1), 0), out=buf549)
        buf552 = empty((8, 12, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_22, attn_23, getitem_47, mul_11], Original ATen: [aten._softmax, aten.add, aten.index, aten.mul]
        triton_per_fused__softmax_add_index_mul_61.run(buf549, primals_220, primals_12, buf552, 1536, 16, grid=grid(1536), stream=stream0)
        del primals_12
        buf553 = empty((8, 12, 16, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_23], Original ATen: [aten.clone]
        triton_poi_fused_clone_62.run(buf542, buf543, buf544, primals_166, primals_167, buf553, 49152, grid=grid(49152), stream=stream0)
        del primals_167
        buf554 = empty((96, 16, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf552, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf553, (96, 16, 32), (512, 32, 1), 0), out=buf554)
        buf555 = empty((8, 16, 384), device='cuda', dtype=torch.float32)
        buf556 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___1___attn_proj_act, x_146, x_147], Original ATen: [aten._unsafe_view, aten.clone, aten.hardswish, aten.view]
        triton_poi_fused__unsafe_view_clone_hardswish_view_63.run(buf554, buf555, buf556, 49152, grid=grid(49152), stream=stream0)
        buf557 = reinterpret_tensor(buf554, (128, 384), (384, 1), 0); del buf554  # reuse
        # Source Nodes: [x_147], Original ATen: [aten.mm]
        extern_kernels.mm(buf556, reinterpret_tensor(primals_168, (384, 384), (1, 384), 0), out=buf557)
        buf558 = buf538; del buf538  # reuse
        buf559 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf561 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___1___attn_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf557, primals_376, primals_377, buf558, buf559, buf561, primals_376, primals_377, 384, 128, grid=grid(384), stream=stream0)
        del primals_376
        del primals_377
        buf562 = empty((8, 16, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_149], Original ATen: [aten.add]
        triton_poi_fused_add_58.run(buf541, buf557, buf558, buf559, primals_169, primals_170, buf562, 49152, grid=grid(49152), stream=stream0)
        del primals_170
        buf563 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_150], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf562, (128, 384), (384, 1), 0), reinterpret_tensor(primals_171, (384, 768), (1, 384), 0), out=buf563)
        buf564 = buf544; del buf544  # reuse
        buf565 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf567 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___1___mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf563, primals_379, primals_380, buf564, buf565, buf567, primals_379, primals_380, 768, 128, grid=grid(768), stream=stream0)
        del primals_379
        del primals_380
        buf568 = empty((8, 16, 768), device='cuda', dtype=torch.float32)
        buf569 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___1___mlp_ln1_bn, x_151, x_152, x_154], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish, aten.view]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_view_57.run(buf563, buf564, buf565, primals_172, primals_173, buf568, buf569, 98304, grid=grid(98304), stream=stream0)
        del primals_173
        buf570 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_154], Original ATen: [aten.mm]
        extern_kernels.mm(buf569, reinterpret_tensor(primals_174, (768, 384), (1, 768), 0), out=buf570)
        buf571 = buf559; del buf559  # reuse
        buf572 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf574 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___1___mlp_ln2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf570, primals_382, primals_383, buf571, buf572, buf574, primals_382, primals_383, 384, 128, grid=grid(384), stream=stream0)
        del primals_382
        del primals_383
        buf575 = empty((8, 16, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_156], Original ATen: [aten.add]
        triton_poi_fused_add_58.run(buf562, buf570, buf571, buf572, primals_175, primals_176, buf575, 49152, grid=grid(49152), stream=stream0)
        del primals_176
        buf576 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_157], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf575, (128, 384), (384, 1), 0), reinterpret_tensor(primals_177, (384, 768), (1, 384), 0), out=buf576)
        buf577 = buf565; del buf565  # reuse
        buf578 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf580 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___2___attn_qkv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf576, primals_385, primals_386, buf577, buf578, buf580, primals_385, primals_386, 768, 128, grid=grid(768), stream=stream0)
        del primals_385
        del primals_386
        buf581 = reinterpret_tensor(buf549, (8, 12, 16, 16), (3072, 256, 16, 1), 0); del buf549  # reuse
        # Source Nodes: [matmul_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_59.run(buf576, buf577, buf578, primals_178, primals_179, buf581, 24576, grid=grid(24576), stream=stream0)
        buf582 = empty((8, 12, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_24], Original ATen: [aten.clone]
        triton_poi_fused_clone_60.run(buf576, buf577, buf578, primals_178, primals_179, buf582, 1536, 16, grid=grid(1536, 16), stream=stream0)
        buf583 = empty((96, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf581, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf582, (96, 16, 16), (256, 16, 1), 0), out=buf583)
        buf586 = empty((8, 12, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_24, attn_25, getitem_51, mul_12], Original ATen: [aten._softmax, aten.add, aten.index, aten.mul]
        triton_per_fused__softmax_add_index_mul_61.run(buf583, primals_221, primals_13, buf586, 1536, 16, grid=grid(1536), stream=stream0)
        del primals_13
        buf587 = empty((8, 12, 16, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_25], Original ATen: [aten.clone]
        triton_poi_fused_clone_62.run(buf576, buf577, buf578, primals_178, primals_179, buf587, 49152, grid=grid(49152), stream=stream0)
        del primals_179
        buf588 = empty((96, 16, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf586, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf587, (96, 16, 32), (512, 32, 1), 0), out=buf588)
        buf589 = empty((8, 16, 384), device='cuda', dtype=torch.float32)
        buf590 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___2___attn_proj_act, x_158, x_159], Original ATen: [aten._unsafe_view, aten.clone, aten.hardswish, aten.view]
        triton_poi_fused__unsafe_view_clone_hardswish_view_63.run(buf588, buf589, buf590, 49152, grid=grid(49152), stream=stream0)
        buf591 = reinterpret_tensor(buf588, (128, 384), (384, 1), 0); del buf588  # reuse
        # Source Nodes: [x_159], Original ATen: [aten.mm]
        extern_kernels.mm(buf590, reinterpret_tensor(primals_180, (384, 384), (1, 384), 0), out=buf591)
        buf592 = buf572; del buf572  # reuse
        buf593 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf595 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___2___attn_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf591, primals_388, primals_389, buf592, buf593, buf595, primals_388, primals_389, 384, 128, grid=grid(384), stream=stream0)
        del primals_388
        del primals_389
        buf596 = empty((8, 16, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_161], Original ATen: [aten.add]
        triton_poi_fused_add_58.run(buf575, buf591, buf592, buf593, primals_181, primals_182, buf596, 49152, grid=grid(49152), stream=stream0)
        del primals_182
        buf597 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_162], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf596, (128, 384), (384, 1), 0), reinterpret_tensor(primals_183, (384, 768), (1, 384), 0), out=buf597)
        buf598 = buf578; del buf578  # reuse
        buf599 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf601 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___2___mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf597, primals_391, primals_392, buf598, buf599, buf601, primals_391, primals_392, 768, 128, grid=grid(768), stream=stream0)
        del primals_391
        del primals_392
        buf602 = empty((8, 16, 768), device='cuda', dtype=torch.float32)
        buf603 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___2___mlp_ln1_bn, x_163, x_164, x_166], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish, aten.view]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_view_57.run(buf597, buf598, buf599, primals_184, primals_185, buf602, buf603, 98304, grid=grid(98304), stream=stream0)
        del primals_185
        buf604 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_166], Original ATen: [aten.mm]
        extern_kernels.mm(buf603, reinterpret_tensor(primals_186, (768, 384), (1, 768), 0), out=buf604)
        buf605 = buf593; del buf593  # reuse
        buf606 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf608 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___2___mlp_ln2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf604, primals_394, primals_395, buf605, buf606, buf608, primals_394, primals_395, 384, 128, grid=grid(384), stream=stream0)
        del primals_394
        del primals_395
        buf609 = empty((8, 16, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_168], Original ATen: [aten.add]
        triton_poi_fused_add_58.run(buf596, buf604, buf605, buf606, primals_187, primals_188, buf609, 49152, grid=grid(49152), stream=stream0)
        del primals_188
        buf610 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_169], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf609, (128, 384), (384, 1), 0), reinterpret_tensor(primals_189, (384, 768), (1, 384), 0), out=buf610)
        buf611 = buf599; del buf599  # reuse
        buf612 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf614 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___3___attn_qkv_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf610, primals_397, primals_398, buf611, buf612, buf614, primals_397, primals_398, 768, 128, grid=grid(768), stream=stream0)
        del primals_397
        del primals_398
        buf615 = reinterpret_tensor(buf583, (8, 12, 16, 16), (3072, 256, 16, 1), 0); del buf583  # reuse
        # Source Nodes: [matmul_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_59.run(buf610, buf611, buf612, primals_190, primals_191, buf615, 24576, grid=grid(24576), stream=stream0)
        buf616 = empty((8, 12, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_26], Original ATen: [aten.clone]
        triton_poi_fused_clone_60.run(buf610, buf611, buf612, primals_190, primals_191, buf616, 1536, 16, grid=grid(1536, 16), stream=stream0)
        buf617 = empty((96, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf615, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf616, (96, 16, 16), (256, 16, 1), 0), out=buf617)
        buf620 = empty((8, 12, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [attn_26, attn_27, getitem_55, mul_13], Original ATen: [aten._softmax, aten.add, aten.index, aten.mul]
        triton_per_fused__softmax_add_index_mul_61.run(buf617, primals_222, primals_14, buf620, 1536, 16, grid=grid(1536), stream=stream0)
        del buf617
        del primals_14
        buf621 = empty((8, 12, 16, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_27], Original ATen: [aten.clone]
        triton_poi_fused_clone_62.run(buf610, buf611, buf612, primals_190, primals_191, buf621, 49152, grid=grid(49152), stream=stream0)
        del primals_191
        buf622 = empty((96, 16, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf620, (96, 16, 16), (256, 16, 1), 0), reinterpret_tensor(buf621, (96, 16, 32), (512, 32, 1), 0), out=buf622)
        buf623 = empty((8, 16, 384), device='cuda', dtype=torch.float32)
        buf624 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___3___attn_proj_act, x_170, x_171], Original ATen: [aten._unsafe_view, aten.clone, aten.hardswish, aten.view]
        triton_poi_fused__unsafe_view_clone_hardswish_view_63.run(buf622, buf623, buf624, 49152, grid=grid(49152), stream=stream0)
        buf625 = reinterpret_tensor(buf622, (128, 384), (384, 1), 0); del buf622  # reuse
        # Source Nodes: [x_171], Original ATen: [aten.mm]
        extern_kernels.mm(buf624, reinterpret_tensor(primals_192, (384, 384), (1, 384), 0), out=buf625)
        buf626 = buf606; del buf606  # reuse
        buf627 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf629 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___3___attn_proj_ln_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf625, primals_400, primals_401, buf626, buf627, buf629, primals_400, primals_401, 384, 128, grid=grid(384), stream=stream0)
        del primals_400
        del primals_401
        buf630 = empty((8, 16, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_173], Original ATen: [aten.add]
        triton_poi_fused_add_58.run(buf609, buf625, buf626, buf627, primals_193, primals_194, buf630, 49152, grid=grid(49152), stream=stream0)
        del primals_194
        buf631 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_174], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf630, (128, 384), (384, 1), 0), reinterpret_tensor(primals_195, (384, 768), (1, 384), 0), out=buf631)
        buf632 = buf612; del buf612  # reuse
        buf633 = empty((1, 768), device='cuda', dtype=torch.float32)
        buf635 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___3___mlp_ln1_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf631, primals_403, primals_404, buf632, buf633, buf635, primals_403, primals_404, 768, 128, grid=grid(768), stream=stream0)
        del primals_403
        del primals_404
        buf636 = empty((8, 16, 768), device='cuda', dtype=torch.float32)
        buf637 = empty((128, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___3___mlp_ln1_bn, x_175, x_176, x_178], Original ATen: [aten._native_batch_norm_legit_functional, aten.hardswish, aten.view]
        triton_poi_fused__native_batch_norm_legit_functional_hardswish_view_57.run(buf631, buf632, buf633, primals_196, primals_197, buf636, buf637, 98304, grid=grid(98304), stream=stream0)
        del buf633
        del primals_197
        buf638 = empty((128, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_178], Original ATen: [aten.mm]
        extern_kernels.mm(buf637, reinterpret_tensor(primals_198, (768, 384), (1, 768), 0), out=buf638)
        buf639 = buf627; del buf627  # reuse
        buf640 = empty((1, 384), device='cuda', dtype=torch.float32)
        buf642 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___3___mlp_ln2_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf638, primals_406, primals_407, buf639, buf640, buf642, primals_406, primals_407, 384, 128, grid=grid(384), stream=stream0)
        del primals_406
        del primals_407
        buf643 = empty((8, 384), device='cuda', dtype=torch.float32)
        buf644 = buf643; del buf643  # reuse
        # Source Nodes: [x_183, x_184], Original ATen: [aten.add, aten.mean]
        triton_per_fused_add_mean_64.run(buf644, buf630, buf638, buf639, buf640, primals_199, primals_200, 3072, 16, grid=grid(3072), stream=stream0)
        del primals_200
        buf645 = buf640; del buf640  # reuse
        buf646 = empty((1, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___head_bn, l__mod___head_dist_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_65.run(buf644, primals_409, primals_412, primals_410, primals_413, buf645, buf646, primals_409, primals_412, primals_410, primals_413, 384, 8, grid=grid(384), stream=stream0)
        del primals_409
        del primals_410
        del primals_412
        del primals_413
        buf648 = empty((8, 384), device='cuda', dtype=torch.float32)
        buf650 = empty((8, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___head_bn, l__mod___head_dist_bn], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_66.run(buf644, buf645, buf646, primals_201, primals_202, primals_205, primals_206, buf648, buf650, 3072, grid=grid(3072), stream=stream0)
        del buf645
        del buf646
        del primals_202
        del primals_206
        buf649 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf648, reinterpret_tensor(primals_203, (384, 1000), (1, 384), 0), out=buf649)
        buf651 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf650, reinterpret_tensor(primals_207, (384, 1000), (1, 384), 0), out=buf651)
        buf652 = buf649; del buf649  # reuse
        # Source Nodes: [add_40, pred], Original ATen: [aten.add, aten.div]
        triton_poi_fused_add_div_67.run(buf652, primals_204, buf651, primals_208, 8000, grid=grid(8000), stream=stream0)
        del buf651
        del primals_204
        del primals_208
        # Source Nodes: [l__mod___stem_conv1_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_225, primals_225, 1, grid=grid(1), stream=stream0)
        del primals_225
        # Source Nodes: [l__mod___stem_conv2_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_228, primals_228, 1, grid=grid(1), stream=stream0)
        del primals_228
        # Source Nodes: [l__mod___stem_conv3_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_231, primals_231, 1, grid=grid(1), stream=stream0)
        del primals_231
        # Source Nodes: [x], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_234, primals_234, 1, grid=grid(1), stream=stream0)
        del primals_234
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___attn_qkv_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_237, primals_237, 1, grid=grid(1), stream=stream0)
        del primals_237
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___attn_proj_ln_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_240, primals_240, 1, grid=grid(1), stream=stream0)
        del primals_240
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___mlp_ln1_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_243, primals_243, 1, grid=grid(1), stream=stream0)
        del primals_243
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___0___mlp_ln2_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_246, primals_246, 1, grid=grid(1), stream=stream0)
        del primals_246
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___1___attn_qkv_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_249, primals_249, 1, grid=grid(1), stream=stream0)
        del primals_249
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___1___attn_proj_ln_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_252, primals_252, 1, grid=grid(1), stream=stream0)
        del primals_252
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___1___mlp_ln1_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_255, primals_255, 1, grid=grid(1), stream=stream0)
        del primals_255
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___1___mlp_ln2_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_258, primals_258, 1, grid=grid(1), stream=stream0)
        del primals_258
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___2___attn_qkv_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_261, primals_261, 1, grid=grid(1), stream=stream0)
        del primals_261
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___2___attn_proj_ln_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_264, primals_264, 1, grid=grid(1), stream=stream0)
        del primals_264
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___2___mlp_ln1_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_267, primals_267, 1, grid=grid(1), stream=stream0)
        del primals_267
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___2___mlp_ln2_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_270, primals_270, 1, grid=grid(1), stream=stream0)
        del primals_270
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___3___attn_qkv_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_273, primals_273, 1, grid=grid(1), stream=stream0)
        del primals_273
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___3___attn_proj_ln_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_276, primals_276, 1, grid=grid(1), stream=stream0)
        del primals_276
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___3___mlp_ln1_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_279, primals_279, 1, grid=grid(1), stream=stream0)
        del primals_279
        # Source Nodes: [getattr_getattr_l__mod___stages___0___blocks___3___mlp_ln2_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_282, primals_282, 1, grid=grid(1), stream=stream0)
        del primals_282
        # Source Nodes: [getattr_l__mod___stages___1___downsample_attn_downsample_kv_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_285, primals_285, 1, grid=grid(1), stream=stream0)
        del primals_285
        # Source Nodes: [getattr_l__mod___stages___1___downsample_attn_downsample_q_ln_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_288, primals_288, 1, grid=grid(1), stream=stream0)
        del primals_288
        # Source Nodes: [getattr_l__mod___stages___1___downsample_attn_downsample_proj_ln_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_291, primals_291, 1, grid=grid(1), stream=stream0)
        del primals_291
        # Source Nodes: [getattr_l__mod___stages___1___downsample_mlp_ln1_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_294, primals_294, 1, grid=grid(1), stream=stream0)
        del primals_294
        # Source Nodes: [getattr_l__mod___stages___1___downsample_mlp_ln2_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_297, primals_297, 1, grid=grid(1), stream=stream0)
        del primals_297
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___0___attn_qkv_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_300, primals_300, 1, grid=grid(1), stream=stream0)
        del primals_300
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___0___attn_proj_ln_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_303, primals_303, 1, grid=grid(1), stream=stream0)
        del primals_303
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___0___mlp_ln1_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_306, primals_306, 1, grid=grid(1), stream=stream0)
        del primals_306
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___0___mlp_ln2_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_309, primals_309, 1, grid=grid(1), stream=stream0)
        del primals_309
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___1___attn_qkv_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_312, primals_312, 1, grid=grid(1), stream=stream0)
        del primals_312
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___1___attn_proj_ln_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_315, primals_315, 1, grid=grid(1), stream=stream0)
        del primals_315
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___1___mlp_ln1_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_318, primals_318, 1, grid=grid(1), stream=stream0)
        del primals_318
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___1___mlp_ln2_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_321, primals_321, 1, grid=grid(1), stream=stream0)
        del primals_321
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___2___attn_qkv_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_324, primals_324, 1, grid=grid(1), stream=stream0)
        del primals_324
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___2___attn_proj_ln_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_327, primals_327, 1, grid=grid(1), stream=stream0)
        del primals_327
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___2___mlp_ln1_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_330, primals_330, 1, grid=grid(1), stream=stream0)
        del primals_330
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___2___mlp_ln2_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_333, primals_333, 1, grid=grid(1), stream=stream0)
        del primals_333
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___3___attn_qkv_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_336, primals_336, 1, grid=grid(1), stream=stream0)
        del primals_336
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___3___attn_proj_ln_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_339, primals_339, 1, grid=grid(1), stream=stream0)
        del primals_339
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___3___mlp_ln1_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_342, primals_342, 1, grid=grid(1), stream=stream0)
        del primals_342
        # Source Nodes: [getattr_getattr_l__mod___stages___1___blocks___3___mlp_ln2_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_345, primals_345, 1, grid=grid(1), stream=stream0)
        del primals_345
        # Source Nodes: [getattr_l__mod___stages___2___downsample_attn_downsample_kv_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_348, primals_348, 1, grid=grid(1), stream=stream0)
        del primals_348
        # Source Nodes: [getattr_l__mod___stages___2___downsample_attn_downsample_q_ln_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_351, primals_351, 1, grid=grid(1), stream=stream0)
        del primals_351
        # Source Nodes: [getattr_l__mod___stages___2___downsample_attn_downsample_proj_ln_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_354, primals_354, 1, grid=grid(1), stream=stream0)
        del primals_354
        # Source Nodes: [getattr_l__mod___stages___2___downsample_mlp_ln1_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_357, primals_357, 1, grid=grid(1), stream=stream0)
        del primals_357
        # Source Nodes: [getattr_l__mod___stages___2___downsample_mlp_ln2_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_360, primals_360, 1, grid=grid(1), stream=stream0)
        del primals_360
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___0___attn_qkv_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_363, primals_363, 1, grid=grid(1), stream=stream0)
        del primals_363
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___0___attn_proj_ln_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_366, primals_366, 1, grid=grid(1), stream=stream0)
        del primals_366
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___0___mlp_ln1_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_369, primals_369, 1, grid=grid(1), stream=stream0)
        del primals_369
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___0___mlp_ln2_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_372, primals_372, 1, grid=grid(1), stream=stream0)
        del primals_372
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___1___attn_qkv_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_375, primals_375, 1, grid=grid(1), stream=stream0)
        del primals_375
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___1___attn_proj_ln_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_378, primals_378, 1, grid=grid(1), stream=stream0)
        del primals_378
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___1___mlp_ln1_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_381, primals_381, 1, grid=grid(1), stream=stream0)
        del primals_381
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___1___mlp_ln2_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_384, primals_384, 1, grid=grid(1), stream=stream0)
        del primals_384
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___2___attn_qkv_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_387, primals_387, 1, grid=grid(1), stream=stream0)
        del primals_387
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___2___attn_proj_ln_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_390, primals_390, 1, grid=grid(1), stream=stream0)
        del primals_390
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___2___mlp_ln1_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_393, primals_393, 1, grid=grid(1), stream=stream0)
        del primals_393
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___2___mlp_ln2_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_396, primals_396, 1, grid=grid(1), stream=stream0)
        del primals_396
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___3___attn_qkv_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_399, primals_399, 1, grid=grid(1), stream=stream0)
        del primals_399
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___3___attn_proj_ln_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_402, primals_402, 1, grid=grid(1), stream=stream0)
        del primals_402
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___3___mlp_ln1_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_405, primals_405, 1, grid=grid(1), stream=stream0)
        del primals_405
        # Source Nodes: [getattr_getattr_l__mod___stages___2___blocks___3___mlp_ln2_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_408, primals_408, 1, grid=grid(1), stream=stream0)
        del primals_408
        # Source Nodes: [l__mod___head_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_411, primals_411, 1, grid=grid(1), stream=stream0)
        del primals_411
        # Source Nodes: [l__mod___head_dist_bn], Original ATen: [aten.add]
        triton_poi_fused_add_68.run(primals_414, primals_414, 1, grid=grid(1), stream=stream0)
        del primals_414
        return (buf652, primals_15, primals_16, primals_18, primals_19, primals_21, primals_22, primals_24, primals_25, primals_28, primals_31, primals_34, primals_37, primals_40, primals_43, primals_46, primals_49, primals_52, primals_55, primals_58, primals_61, primals_64, primals_67, primals_70, primals_73, primals_76, primals_79, primals_82, primals_85, primals_88, primals_91, primals_94, primals_97, primals_100, primals_103, primals_106, primals_109, primals_112, primals_115, primals_118, primals_121, primals_124, primals_127, primals_130, primals_133, primals_136, primals_139, primals_142, primals_145, primals_148, primals_151, primals_154, primals_157, primals_160, primals_163, primals_166, primals_169, primals_172, primals_175, primals_178, primals_181, primals_184, primals_187, primals_190, primals_193, primals_196, primals_199, primals_201, primals_205, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_415, buf0, buf7, buf8, buf9, buf10, buf17, buf18, buf19, buf20, buf24, buf25, buf26, buf27, buf31, buf33, buf34, buf41, buf50, buf51, buf52, buf59, buf61, buf62, buf69, buf70, buf71, buf72, buf79, buf81, buf82, buf89, buf98, buf99, buf100, buf107, buf109, buf110, buf117, buf118, buf119, buf120, buf127, buf129, buf130, buf137, buf146, buf147, buf148, buf155, buf157, buf158, buf165, buf166, buf167, buf168, buf175, buf177, buf178, buf185, buf194, buf195, buf196, buf203, buf205, buf206, buf213, buf214, buf215, buf216, buf223, buf225, buf226, buf233, buf234, buf235, buf242, buf251, buf252, buf253, buf260, reinterpret_tensor(buf261, (392, 256), (256, 1), 0), buf262, buf269, buf270, buf271, buf272, buf279, reinterpret_tensor(buf280, (392, 256), (256, 1), 0), buf281, buf288, buf297, buf298, buf299, buf306, reinterpret_tensor(buf307, (392, 256), (256, 1), 0), buf308, buf315, buf316, buf317, buf318, buf325, reinterpret_tensor(buf326, (392, 256), (256, 1), 0), buf327, buf334, buf343, buf344, buf345, buf352, reinterpret_tensor(buf353, (392, 256), (256, 1), 0), buf354, buf361, buf362, buf363, buf364, buf371, reinterpret_tensor(buf372, (392, 256), (256, 1), 0), buf373, buf380, buf389, buf390, buf391, buf398, reinterpret_tensor(buf399, (392, 256), (256, 1), 0), buf400, buf407, buf408, buf409, buf410, buf417, reinterpret_tensor(buf418, (392, 256), (256, 1), 0), buf419, buf426, buf435, buf436, buf437, buf444, reinterpret_tensor(buf445, (392, 256), (256, 1), 0), buf446, buf453, buf454, buf455, buf456, buf463, reinterpret_tensor(buf464, (392, 256), (256, 1), 0), buf465, buf472, buf473, buf474, buf478, buf487, buf488, buf489, buf493, reinterpret_tensor(buf494, (128, 384), (384, 1), 0), buf495, buf499, buf500, buf501, buf502, buf506, reinterpret_tensor(buf507, (128, 384), (384, 1), 0), buf508, buf512, buf521, buf522, buf523, buf527, reinterpret_tensor(buf528, (128, 384), (384, 1), 0), buf529, buf533, buf534, buf535, buf536, buf540, reinterpret_tensor(buf541, (128, 384), (384, 1), 0), buf542, buf546, buf555, buf556, buf557, buf561, reinterpret_tensor(buf562, (128, 384), (384, 1), 0), buf563, buf567, buf568, buf569, buf570, buf574, reinterpret_tensor(buf575, (128, 384), (384, 1), 0), buf576, buf580, buf589, buf590, buf591, buf595, reinterpret_tensor(buf596, (128, 384), (384, 1), 0), buf597, buf601, buf602, buf603, buf604, buf608, reinterpret_tensor(buf609, (128, 384), (384, 1), 0), buf610, buf614, buf623, buf624, buf625, buf629, reinterpret_tensor(buf630, (128, 384), (384, 1), 0), buf631, buf635, buf636, buf637, buf638, buf642, buf644, buf648, buf650, reinterpret_tensor(primals_207, (1000, 384), (384, 1), 0), reinterpret_tensor(primals_203, (1000, 384), (384, 1), 0), reinterpret_tensor(buf639, (1, 384), (384, 1), 0), reinterpret_tensor(primals_198, (384, 768), (768, 1), 0), reinterpret_tensor(buf632, (1, 768), (768, 1), 0), reinterpret_tensor(primals_195, (768, 384), (384, 1), 0), reinterpret_tensor(buf626, (1, 384), (384, 1), 0), reinterpret_tensor(primals_192, (384, 384), (384, 1), 0), reinterpret_tensor(buf620, (96, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf621, (96, 32, 16), (512, 1, 32), 0), buf620, reinterpret_tensor(buf615, (96, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf616, (96, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf611, (1, 768), (768, 1), 0), reinterpret_tensor(primals_189, (768, 384), (384, 1), 0), reinterpret_tensor(buf605, (1, 384), (384, 1), 0), reinterpret_tensor(primals_186, (384, 768), (768, 1), 0), reinterpret_tensor(buf598, (1, 768), (768, 1), 0), reinterpret_tensor(primals_183, (768, 384), (384, 1), 0), reinterpret_tensor(buf592, (1, 384), (384, 1), 0), reinterpret_tensor(primals_180, (384, 384), (384, 1), 0), reinterpret_tensor(buf586, (96, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf587, (96, 32, 16), (512, 1, 32), 0), buf586, reinterpret_tensor(buf581, (96, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf582, (96, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf577, (1, 768), (768, 1), 0), reinterpret_tensor(primals_177, (768, 384), (384, 1), 0), reinterpret_tensor(buf571, (1, 384), (384, 1), 0), reinterpret_tensor(primals_174, (384, 768), (768, 1), 0), reinterpret_tensor(buf564, (1, 768), (768, 1), 0), reinterpret_tensor(primals_171, (768, 384), (384, 1), 0), reinterpret_tensor(buf558, (1, 384), (384, 1), 0), reinterpret_tensor(primals_168, (384, 384), (384, 1), 0), reinterpret_tensor(buf552, (96, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf553, (96, 32, 16), (512, 1, 32), 0), buf552, reinterpret_tensor(buf547, (96, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf548, (96, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf543, (1, 768), (768, 1), 0), reinterpret_tensor(primals_165, (768, 384), (384, 1), 0), reinterpret_tensor(buf537, (1, 384), (384, 1), 0), reinterpret_tensor(primals_162, (384, 768), (768, 1), 0), reinterpret_tensor(buf530, (1, 768), (768, 1), 0), reinterpret_tensor(primals_159, (768, 384), (384, 1), 0), reinterpret_tensor(buf524, (1, 384), (384, 1), 0), reinterpret_tensor(primals_156, (384, 384), (384, 1), 0), reinterpret_tensor(buf518, (96, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf519, (96, 32, 16), (512, 1, 32), 0), buf518, reinterpret_tensor(buf513, (96, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf514, (96, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf509, (1, 768), (768, 1), 0), reinterpret_tensor(primals_153, (768, 384), (384, 1), 0), reinterpret_tensor(buf503, (1, 384), (384, 1), 0), reinterpret_tensor(primals_150, (384, 768), (768, 1), 0), reinterpret_tensor(buf496, (1, 768), (768, 1), 0), reinterpret_tensor(primals_147, (768, 384), (384, 1), 0), reinterpret_tensor(buf490, (1, 384), (384, 1), 0), reinterpret_tensor(primals_144, (384, 1024), (1024, 1), 0), reinterpret_tensor(buf484, (128, 49, 16), (784, 1, 49), 0), reinterpret_tensor(buf485, (128, 64, 49), (3136, 1, 64), 0), buf484, reinterpret_tensor(buf479, (128, 16, 16), (256, 1, 16), 0), reinterpret_tensor(buf480, (128, 49, 16), (784, 1, 49), 0), reinterpret_tensor(buf475, (1, 256), (256, 1), 0), reinterpret_tensor(primals_141, (256, 256), (256, 1), 0), reinterpret_tensor(buf469, (1, 1280), (1280, 1), 0), reinterpret_tensor(primals_138, (1280, 256), (256, 1), 0), reinterpret_tensor(buf460, (1, 256), (256, 1), 0), reinterpret_tensor(primals_135, (256, 512), (512, 1), 0), reinterpret_tensor(buf450, (1, 512), (512, 1), 0), reinterpret_tensor(primals_132, (512, 256), (256, 1), 0), reinterpret_tensor(buf441, (1, 256), (256, 1), 0), reinterpret_tensor(primals_129, (256, 256), (256, 1), 0), reinterpret_tensor(buf432, (64, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf433, (64, 32, 49), (1568, 1, 32), 0), buf432, reinterpret_tensor(buf427, (64, 16, 49), (784, 1, 16), 0), reinterpret_tensor(buf428, (64, 49, 16), (784, 1, 49), 0), reinterpret_tensor(buf423, (1, 512), (512, 1), 0), reinterpret_tensor(primals_126, (512, 256), (256, 1), 0), reinterpret_tensor(buf414, (1, 256), (256, 1), 0), reinterpret_tensor(primals_123, (256, 512), (512, 1), 0), reinterpret_tensor(buf404, (1, 512), (512, 1), 0), reinterpret_tensor(primals_120, (512, 256), (256, 1), 0), reinterpret_tensor(buf395, (1, 256), (256, 1), 0), reinterpret_tensor(primals_117, (256, 256), (256, 1), 0), reinterpret_tensor(buf386, (64, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf387, (64, 32, 49), (1568, 1, 32), 0), buf386, reinterpret_tensor(buf381, (64, 16, 49), (784, 1, 16), 0), reinterpret_tensor(buf382, (64, 49, 16), (784, 1, 49), 0), reinterpret_tensor(buf377, (1, 512), (512, 1), 0), reinterpret_tensor(primals_114, (512, 256), (256, 1), 0), reinterpret_tensor(buf368, (1, 256), (256, 1), 0), reinterpret_tensor(primals_111, (256, 512), (512, 1), 0), reinterpret_tensor(buf358, (1, 512), (512, 1), 0), reinterpret_tensor(primals_108, (512, 256), (256, 1), 0), reinterpret_tensor(buf349, (1, 256), (256, 1), 0), reinterpret_tensor(primals_105, (256, 256), (256, 1), 0), reinterpret_tensor(buf340, (64, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf341, (64, 32, 49), (1568, 1, 32), 0), buf340, reinterpret_tensor(buf335, (64, 16, 49), (784, 1, 16), 0), reinterpret_tensor(buf336, (64, 49, 16), (784, 1, 49), 0), reinterpret_tensor(buf331, (1, 512), (512, 1), 0), reinterpret_tensor(primals_102, (512, 256), (256, 1), 0), reinterpret_tensor(buf322, (1, 256), (256, 1), 0), reinterpret_tensor(primals_99, (256, 512), (512, 1), 0), reinterpret_tensor(buf312, (1, 512), (512, 1), 0), reinterpret_tensor(primals_96, (512, 256), (256, 1), 0), reinterpret_tensor(buf303, (1, 256), (256, 1), 0), reinterpret_tensor(primals_93, (256, 256), (256, 1), 0), reinterpret_tensor(buf294, (64, 49, 49), (2401, 1, 49), 0), reinterpret_tensor(buf295, (64, 32, 49), (1568, 1, 32), 0), buf294, reinterpret_tensor(buf289, (64, 16, 49), (784, 1, 16), 0), reinterpret_tensor(buf290, (64, 49, 16), (784, 1, 49), 0), reinterpret_tensor(buf285, (1, 512), (512, 1), 0), reinterpret_tensor(primals_90, (512, 256), (256, 1), 0), reinterpret_tensor(buf276, (1, 256), (256, 1), 0), reinterpret_tensor(primals_87, (256, 512), (512, 1), 0), reinterpret_tensor(buf266, (1, 512), (512, 1), 0), reinterpret_tensor(primals_84, (512, 256), (256, 1), 0), reinterpret_tensor(buf257, (1, 256), (256, 1), 0), reinterpret_tensor(primals_81, (256, 512), (512, 1), 0), reinterpret_tensor(buf248, (64, 196, 49), (9604, 1, 196), 0), reinterpret_tensor(buf249, (64, 64, 196), (12544, 1, 64), 0), buf248, reinterpret_tensor(buf243, (64, 16, 49), (784, 1, 16), 0), reinterpret_tensor(buf244, (64, 196, 16), (3136, 1, 196), 0), reinterpret_tensor(buf239, (1, 128), (128, 1), 0), reinterpret_tensor(primals_78, (128, 128), (128, 1), 0), reinterpret_tensor(buf230, (1, 640), (640, 1), 0), reinterpret_tensor(primals_75, (640, 128), (128, 1), 0), reinterpret_tensor(buf220, (1, 128), (128, 1), 0), reinterpret_tensor(primals_72, (128, 256), (256, 1), 0), reinterpret_tensor(buf210, (1, 256), (256, 1), 0), reinterpret_tensor(primals_69, (256, 128), (128, 1), 0), reinterpret_tensor(buf200, (1, 128), (128, 1), 0), reinterpret_tensor(primals_66, (128, 128), (128, 1), 0), reinterpret_tensor(buf191, (32, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf192, (32, 32, 196), (6272, 1, 32), 0), buf191, reinterpret_tensor(buf186, (32, 16, 196), (3136, 1, 16), 0), reinterpret_tensor(buf187, (32, 196, 16), (3136, 1, 196), 0), reinterpret_tensor(buf182, (1, 256), (256, 1), 0), reinterpret_tensor(primals_63, (256, 128), (128, 1), 0), reinterpret_tensor(buf172, (1, 128), (128, 1), 0), reinterpret_tensor(primals_60, (128, 256), (256, 1), 0), reinterpret_tensor(buf162, (1, 256), (256, 1), 0), reinterpret_tensor(primals_57, (256, 128), (128, 1), 0), reinterpret_tensor(buf152, (1, 128), (128, 1), 0), reinterpret_tensor(primals_54, (128, 128), (128, 1), 0), reinterpret_tensor(buf143, (32, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf144, (32, 32, 196), (6272, 1, 32), 0), buf143, reinterpret_tensor(buf138, (32, 16, 196), (3136, 1, 16), 0), reinterpret_tensor(buf139, (32, 196, 16), (3136, 1, 196), 0), reinterpret_tensor(buf134, (1, 256), (256, 1), 0), reinterpret_tensor(primals_51, (256, 128), (128, 1), 0), reinterpret_tensor(buf124, (1, 128), (128, 1), 0), reinterpret_tensor(primals_48, (128, 256), (256, 1), 0), reinterpret_tensor(buf114, (1, 256), (256, 1), 0), reinterpret_tensor(primals_45, (256, 128), (128, 1), 0), reinterpret_tensor(buf104, (1, 128), (128, 1), 0), reinterpret_tensor(primals_42, (128, 128), (128, 1), 0), reinterpret_tensor(buf95, (32, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf96, (32, 32, 196), (6272, 1, 32), 0), buf95, reinterpret_tensor(buf90, (32, 16, 196), (3136, 1, 16), 0), reinterpret_tensor(buf91, (32, 196, 16), (3136, 1, 196), 0), reinterpret_tensor(buf86, (1, 256), (256, 1), 0), reinterpret_tensor(primals_39, (256, 128), (128, 1), 0), reinterpret_tensor(buf76, (1, 128), (128, 1), 0), reinterpret_tensor(primals_36, (128, 256), (256, 1), 0), reinterpret_tensor(buf66, (1, 256), (256, 1), 0), reinterpret_tensor(primals_33, (256, 128), (128, 1), 0), reinterpret_tensor(buf56, (1, 128), (128, 1), 0), reinterpret_tensor(primals_30, (128, 128), (128, 1), 0), reinterpret_tensor(buf47, (32, 196, 196), (38416, 1, 196), 0), reinterpret_tensor(buf48, (32, 32, 196), (6272, 1, 32), 0), buf47, reinterpret_tensor(buf42, (32, 16, 196), (3136, 1, 16), 0), reinterpret_tensor(buf43, (32, 196, 16), (3136, 1, 196), 0), reinterpret_tensor(buf38, (1, 256), (256, 1), 0), reinterpret_tensor(primals_27, (256, 128), (128, 1), 0), reinterpret_tensor(buf28, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf21, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf14, (1, 32, 1, 1), (32, 1, 1, 1), 0), reinterpret_tensor(buf4, (1, 16, 1, 1), (16, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((4, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((4, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((4, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((4, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((8, 196), (196, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((8, 49), (49, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((8, 49), (49, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((8, 49), (49, 1), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((8, 49), (49, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((16, 49), (49, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((12, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((12, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((12, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((12, 16), (16, 1), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((32, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((256, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((128, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((640, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((128, 128), (128, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((512, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((256, 512), (512, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((1280, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((256, 256), (256, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((384, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((384, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((768, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((384, 768), (768, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((1000, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    primals_223 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_226 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_229 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_232 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_235 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_238 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_241 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_244 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_247 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_250 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_253 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_256 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_259 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_262 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_265 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_268 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_271 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_274 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_277 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_280 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_283 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_286 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_289 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_292 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_295 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_298 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_301 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_304 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_307 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_310 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_313 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_316 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_319 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_322 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_325 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_328 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_331 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_334 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_337 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_340 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_343 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_346 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_349 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_352 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_355 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_358 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_361 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_364 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_367 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_370 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_373 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_376 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_379 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_382 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_385 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_388 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_391 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_394 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_397 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_400 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_403 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_406 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_409 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_412 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_415 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('levit_128', benchmark_compiled_module)
