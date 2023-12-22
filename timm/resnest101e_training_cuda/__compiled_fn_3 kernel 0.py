
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


# kernel path: /tmp/torchinductor_youkaichao/dj/cdjitocfoac3kp7c5yizcslesobal43usdyultmqk6n6mjwm5pts.py
# Source Nodes: [l__mod___conv1_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___conv1_1 => var_mean
triton_red_fused__native_batch_norm_legit_functional_0 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 16
    x1 = (xindex // 16)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (1048576*((r2 + (8192*x0)) // 16384)) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/w5/cw5mfbu6d7ir7bki5sl3nyquxltffl7m7liwggodyzhxjakdcduy.py
# Source Nodes: [l__mod___conv1_1], Original ATen: [aten._native_batch_norm_legit_functional]
# l__mod___conv1_1 => add_1, add_2, add_3, mul_1, mul_2, mul_3, mul_4, mul_5, rsqrt, squeeze_1, var_mean
triton_per_fused__native_batch_norm_legit_functional_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_1', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
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
    tmp2 = tl.load(in_ptr2 + (r1 + (16*x0)), rmask & xmask, other=0.0)
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
    tmp16 = 131072.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000076294527394
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


# kernel path: /tmp/torchinductor_youkaichao/7w/c7wulzinxbbhjvcwqxavykr4pifdvks2f3wi5mhkh2clphrqehqb.py
# Source Nodes: [l__mod___conv1_1, l__mod___conv1_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# l__mod___conv1_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
# l__mod___conv1_2 => relu
triton_poi_fused__native_batch_norm_legit_functional_relu_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 64
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 131072.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mq/cmqji6h6ytsnoykkdd5tkvns2qccjh6tm7xwiauippirahaqgox4.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => var_mean_2
triton_red_fused__native_batch_norm_legit_functional_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_3', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 32768
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
        tmp0 = tl.load(in_ptr0 + ((16384*x0) + (2097152*(r2 // 16384)) + (4194304*x1) + (r2 % 16384)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ra/craqrbkojzs6ul7ku77mqzfuv7ccnnwiqvsmjrhz23e65hqfo54r.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => add_11, add_12, add_13, mul_15, mul_16, mul_17, mul_18, mul_19, rsqrt_2, squeeze_7, var_mean_2
triton_per_fused__native_batch_norm_legit_functional_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_4', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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
    tmp16 = 131072.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.0000076294527394
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


# kernel path: /tmp/torchinductor_youkaichao/e3/ce3s6qhugp6xcip5kxlastjthy4osq3hd5faausjxz4n2loommqf.py
# Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_1 => add_11, add_14, mul_14, mul_20, rsqrt_2, sub_2, var_mean_2
# x_2 => relu_2
triton_poi_fused__native_batch_norm_legit_functional_relu_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16777216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 131072.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wa/cwajghhm2xuquetwoez2julygr5y3h5s6ndbq32ggdlk5nl362bk.py
# Source Nodes: [shortcut], Original ATen: [aten.max_pool2d_with_indices]
# shortcut => getitem_6, getitem_7
triton_poi_fused_max_pool2d_with_indices_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 64) % 64
    x0 = xindex % 64
    x3 = (xindex // 64)
    x4 = xindex
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-129) + (2*x0) + (256*x3)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-128) + (2*x0) + (256*x3)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-127) + (2*x0) + (256*x3)), tmp27, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + (2*x0) + (256*x3)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + ((2*x0) + (256*x3)), tmp41, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + (2*x0) + (256*x3)), tmp46, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 1 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (127 + (2*x0) + (256*x3)), tmp55, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (128 + (2*x0) + (256*x3)), tmp60, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (129 + (2*x0) + (256*x3)), tmp65, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tmp70 = tmp21 > tmp13
    tmp71 = (-128) + (2*x0) + (256*x1)
    tmp72 = (-129) + (2*x0) + (256*x1)
    tmp73 = tl.where(tmp70, tmp71, tmp72)
    tmp74 = tmp30 > tmp22
    tmp75 = (-127) + (2*x0) + (256*x1)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tmp77 = tmp39 > tmp31
    tmp78 = (-1) + (2*x0) + (256*x1)
    tmp79 = tl.where(tmp77, tmp78, tmp76)
    tmp80 = tmp44 > tmp40
    tmp81 = (2*x0) + (256*x1)
    tmp82 = tl.where(tmp80, tmp81, tmp79)
    tmp83 = tmp49 > tmp45
    tmp84 = 1 + (2*x0) + (256*x1)
    tmp85 = tl.where(tmp83, tmp84, tmp82)
    tmp86 = tmp58 > tmp50
    tmp87 = 127 + (2*x0) + (256*x1)
    tmp88 = tl.where(tmp86, tmp87, tmp85)
    tmp89 = tmp63 > tmp59
    tmp90 = 128 + (2*x0) + (256*x1)
    tmp91 = tl.where(tmp89, tmp90, tmp88)
    tmp92 = tmp68 > tmp64
    tmp93 = 129 + (2*x0) + (256*x1)
    tmp94 = tl.where(tmp92, tmp93, tmp91)
    tl.store(out_ptr0 + (x4), tmp69, None)
    tl.store(out_ptr1 + (x4), tmp94, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dn/cdns3wckwwpz3jbalunzzmnyfbvshggv2sauflrupsl34a5klfmh.py
# Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
# out_1 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((4096*x0) + (262144*(r2 // 4096)) + (524288*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/av/cavydzzogqhgxbmv6xzukc35mekajxaaqa3ykycfysskhm6bjjks.py
# Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
# out_1 => add_16, add_17, add_18, mul_22, mul_23, mul_24, mul_25, mul_26, rsqrt_3, squeeze_10, var_mean_3
triton_per_fused__native_batch_norm_legit_functional_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_8', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (64*r1)), rmask & xmask, other=0.0)
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
    tmp16 = 32768.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.000030518509476
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


# kernel path: /tmp/torchinductor_youkaichao/2k/c2kq55he7youclllxymkf3jmq2unqioajv2ft56c7bbc7y6rw4tu.py
# Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# out_1 => add_16, add_19, mul_21, mul_27, rsqrt_3, sub_3, var_mean_3
# out_2 => relu_3
triton_poi_fused__native_batch_norm_legit_functional_relu_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 64
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 32768.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wm/cwmoaozpyhfniwxfic5rbka7eh6cxncpxtmj36q7v7y74hlfeg45.py
# Source Nodes: [x_5], Original ATen: [aten._native_batch_norm_legit_functional]
# x_5 => var_mean_4
triton_red_fused__native_batch_norm_legit_functional_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_10', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
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
        tmp0 = tl.load(in_ptr0 + ((4096*x0) + (524288*(r2 // 4096)) + (1048576*x1) + (r2 % 4096)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/j7/cj77clo7lxlfsal2hpe6kfzxzfkqpd4ija6gcaxoouwozhudwkrn.py
# Source Nodes: [x_5], Original ATen: [aten._native_batch_norm_legit_functional]
# x_5 => add_21, add_22, add_23, mul_29, mul_30, mul_31, mul_32, mul_33, rsqrt_4, squeeze_13, var_mean_4
triton_per_fused__native_batch_norm_legit_functional_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_11', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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
    tmp16 = 32768.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp27 = 1.000030518509476
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


# kernel path: /tmp/torchinductor_youkaichao/kd/ckdakvpqhbw4dms2qybd4vvb635tbikvwxfrwdvdeg6bozqq6jx2.py
# Source Nodes: [x_5, x_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_5 => add_21, add_24, mul_28, mul_34, rsqrt_4, sub_4, var_mean_4
# x_7 => relu_4
triton_poi_fused__native_batch_norm_legit_functional_relu_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 32768.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/52/c52ijim2ieidcbv5f3zyd5z6cxvfl2q2mdboujowl6wnjwmn2oxb.py
# Source Nodes: [x_gap, x_gap_1], Original ATen: [aten.mean, aten.sum]
# x_gap => sum_1
# x_gap_1 => mean
triton_red_fused_mean_sum_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_sum_13', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 64
    x1 = (xindex // 64)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (4096*x0) + (524288*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (262144 + r2 + (4096*x0) + (524288*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp6 = 4096.0
    tmp7 = tmp4 / tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xc/cxc5tctwqft2bunkbq22djidu32pvuu33ijqijpsv74dtuvqrevl.py
# Source Nodes: [x_gap_2], Original ATen: [aten.convolution]
# x_gap_2 => convolution_5
triton_poi_fused_convolution_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jt/cjtiqjddn3ekg4p6e2fmglraybk2komreijfwzod3uz75horjm3n.py
# Source Nodes: [x_gap_3], Original ATen: [aten._native_batch_norm_legit_functional]
# x_gap_3 => add_27, add_28, mul_36, mul_37, mul_38, mul_39, mul_40, var_mean_5
triton_per_fused__native_batch_norm_legit_functional_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_15', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr3', 'out_ptr5']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr3, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (32*r1)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
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
    tmp23 = 8.0
    tmp24 = tmp16 / tmp23
    tmp25 = 1.1428571428571428
    tmp26 = tmp24 * tmp25
    tmp27 = tmp26 * tmp17
    tmp29 = tmp28 * tmp20
    tmp30 = tmp27 + tmp29
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr5 + (x0), tmp30, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hr/chryq6ozde3dweasbavp33cossnj6hi6svzoyql4zaksupq6xvux.py
# Source Nodes: [x_gap_3, x_gap_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_gap_3 => add_26, add_29, mul_35, mul_41, rsqrt_5, sub_5, var_mean_5
# x_gap_4 => relu_5
triton_poi_fused__native_batch_norm_legit_functional_relu_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 8.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sk/csks2zt6btiotprxarovhnqkd6c7ccgiehk3hntny2rbu6mrpska.py
# Source Nodes: [x_10], Original ATen: [aten._softmax]
# x_10 => amax, exp, sub_6
triton_poi_fused__softmax_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = xindex % 128
    x0 = xindex % 64
    x2 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x4), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (x0 + (128*x2)), xmask, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (64 + x0 + (128*x2)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (64 + x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 + tmp7
    tmp9 = triton_helpers.maximum(tmp5, tmp8)
    tmp10 = tmp2 - tmp9
    tmp11 = tl.exp(tmp10)
    tl.store(out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ys/cyspng3vdp5a5qouu2mudzh3isjoebgcuslrcvnuwetw7ahvexme.py
# Source Nodes: [x_10], Original ATen: [aten._softmax]
# x_10 => div, sum_2
triton_poi_fused__softmax_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 64
    x2 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x0 + (128*x2)), xmask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (64 + x0 + (128*x2)), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 / tmp3
    tl.store(out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a5/ca5mi7db3zpgk4uwn4hi4ha7f3v7kedvr2xsmcawvvj33inw4pzw.py
# Source Nodes: [mul, out_3], Original ATen: [aten.mul, aten.sum]
# mul => mul_42
# out_3 => sum_3
triton_poi_fused_mul_sum_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sum_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 262144)
    x3 = xindex % 262144
    x1 = (xindex // 4096) % 64
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (524288*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (128*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (262144 + x3 + (524288*x2)), None)
    tmp4 = tl.load(in_ptr1 + (64 + x1 + (128*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tl.store(out_ptr0 + (x4), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/r6/cr6drmo6aq6ucjj3zq2qtzn76lzwgswsayusmvts6tgfso2ryfgh.py
# Source Nodes: [out_9], Original ATen: [aten._native_batch_norm_legit_functional]
# out_9 => add_31, add_32, add_33, mul_44, mul_45, mul_46, mul_47, mul_48, rsqrt_6, squeeze_19, var_mean_6
triton_red_fused__native_batch_norm_legit_functional_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[256, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_20', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 32768
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
        r1 = rindex % 4096
        r2 = (rindex // 4096)
        tmp0 = tl.load(in_ptr0 + (r1 + (4096*x0) + (1048576*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp5 = 32768.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = 0.1
    tmp11 = tmp2 * tmp10
    tmp13 = 0.9
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = 1.000030518509476
    tmp17 = tmp6 * tmp16
    tmp18 = tmp17 * tmp10
    tmp20 = tmp19 * tmp13
    tmp21 = tmp18 + tmp20
    tl.store(out_ptr2 + (x0), tmp9, xmask)
    tl.store(out_ptr4 + (x0), tmp15, xmask)
    tl.store(out_ptr6 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/de/cde6ommhfxmkhfs7ie23hjjxvdjl7qhnzmhttpepxiftrnvfx5de.py
# Source Nodes: [out_10, out_9, shortcut_1, shortcut_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_10 => add_40
# out_9 => add_31, add_34, mul_43, mul_49, rsqrt_6, sub_7, var_mean_6
# shortcut_1 => add_36, add_39, mul_50, mul_56, rsqrt_7, sub_8, var_mean_7
# shortcut_2 => relu_6
triton_poi_fused__native_batch_norm_legit_functional_add_relu_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x3), None)
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 32768.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = triton_helpers.maximum(0, tmp26)
    tl.store(in_out_ptr0 + (x3), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dx/cdx57tsj53yhab34wbeawtfjagwmwylsqli6ny4gebpp52swi4or.py
# Source Nodes: [out_21, out_22, shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_21 => add_57, add_60, mul_79, mul_85, rsqrt_11, sub_13, var_mean_11
# out_22 => add_61
# shortcut_3 => relu_10
triton_poi_fused__native_batch_norm_legit_functional_add_relu_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x3), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 32768.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = triton_helpers.maximum(0, tmp15)
    tl.store(out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ko/ckovxibrnl7ihap2x2kcu6z23ruqlmmqc7qoxozetntox6pzovhc.py
# Source Nodes: [x_30, x_32], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_30 => add_89, add_92, mul_122, mul_128, rsqrt_17, sub_20, var_mean_17
# x_32 => relu_16
triton_poi_fused__native_batch_norm_legit_functional_relu_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8388608
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 4096) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 32768.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7q/c7q5pykcm325aup65afk5qwyaza7mw453ufkn55ygcdy2rhskbdj.py
# Source Nodes: [x_gap_15, x_gap_16], Original ATen: [aten.mean, aten.sum]
# x_gap_15 => sum_10
# x_gap_16 => mean_3
triton_red_fused_mean_sum_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_mean_sum_24', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 128
    x1 = (xindex // 128)
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (4096*x0) + (1048576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp1 = tl.load(in_ptr0 + (524288 + r2 + (4096*x0) + (1048576*x1)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tmp0 + tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask & xmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tmp6 = 4096.0
    tmp7 = tmp4 / tmp6
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp7, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5s/c5sofoitasgvuqcu2ah52uwwcu6iuuxnymjy4t5g2klqtdma6gya.py
# Source Nodes: [x_gap_17], Original ATen: [aten.convolution]
# x_gap_17 => convolution_21
triton_poi_fused_convolution_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_25', 'mutated_arg_names': ['in_out_ptr0']},
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
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lk/clkqfuwmpxfd25gqhip4bnrf2ukqhf3wtuxclufb2uquswsinkou.py
# Source Nodes: [x_gap_18], Original ATen: [aten._native_batch_norm_legit_functional]
# x_gap_18 => add_95, add_96, mul_130, mul_131, mul_132, mul_133, mul_134, var_mean_18
triton_per_fused__native_batch_norm_legit_functional_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[64, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_26', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr3', 'out_ptr5']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr3, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 64
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*r1)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
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
    tmp23 = 8.0
    tmp24 = tmp16 / tmp23
    tmp25 = 1.1428571428571428
    tmp26 = tmp24 * tmp25
    tmp27 = tmp26 * tmp17
    tmp29 = tmp28 * tmp20
    tmp30 = tmp27 + tmp29
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr5 + (x0), tmp30, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/es/ces4eqxnpu7wdghqjpjia75ci2dbywr5qlh3sbje322su55kajmu.py
# Source Nodes: [x_gap_18, x_gap_19], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_gap_18 => add_94, add_97, mul_129, mul_135, rsqrt_18, sub_21, var_mean_18
# x_gap_19 => relu_17
triton_poi_fused__native_batch_norm_legit_functional_relu_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 8.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sa/csahcvt7j6iqhdoz6dxdq6ao5tzchywgqru2b6b4oocpdwdscuxu.py
# Source Nodes: [x_35], Original ATen: [aten._softmax]
# x_35 => amax_3, exp_3, sub_22
triton_poi_fused__softmax_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = xindex % 256
    x0 = xindex % 128
    x2 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (x0 + (256*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (128 + x0 + (256*x2)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (128 + x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 + tmp7
    tmp9 = triton_helpers.maximum(tmp5, tmp8)
    tmp10 = tmp2 - tmp9
    tmp11 = tl.exp(tmp10)
    tl.store(out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yt/cytiyjspwpn5ujm4v2667logpaxrzqpyr7aphqetgtfmluynjwjv.py
# Source Nodes: [x_35], Original ATen: [aten._softmax]
# x_35 => div_3, sum_11
triton_poi_fused__softmax_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 128
    x2 = (xindex // 256)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x0 + (256*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (128 + x0 + (256*x2)), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 / tmp3
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xg/cxgpskhwpfwl5ebx5ucrzru67bwqkgkptvspt3ytophufo2nnyoe.py
# Source Nodes: [mul_3, out_39], Original ATen: [aten.mul, aten.sum]
# mul_3 => mul_136
# out_39 => sum_12
triton_poi_fused_mul_sum_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sum_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 524288)
    x3 = xindex % 524288
    x1 = (xindex // 4096) % 128
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (1048576*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (256*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (524288 + x3 + (1048576*x2)), None)
    tmp4 = tl.load(in_ptr1 + (128 + x1 + (256*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tl.store(out_ptr0 + (x4), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/a4/ca4bdxnynm6piguogwxs5x6gqbvugxbkthbngdjnkwmjwa6by3cw.py
# Source Nodes: [out_44], Original ATen: [aten.avg_pool2d]
# out_44 => avg_pool2d
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
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 32) % 32
    x0 = xindex % 32
    x3 = (xindex // 32)
    x4 = xindex
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-65) + (2*x0) + (128*x3)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-64) + (2*x0) + (128*x3)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-63) + (2*x0) + (128*x3)), tmp27, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + (2*x0) + (128*x3)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + ((2*x0) + (128*x3)), tmp41, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + (2*x0) + (128*x3)), tmp46, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (63 + (2*x0) + (128*x3)), tmp55, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (64 + (2*x0) + (128*x3)), tmp60, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (65 + (2*x0) + (128*x3)), tmp65, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 65, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x4), tmp145, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6q/c6qdhzcmm6agsawo2hjddvvz4pyzfc4zc7qx6pdm2iccci2trvfo.py
# Source Nodes: [out_46], Original ATen: [aten._native_batch_norm_legit_functional]
# out_46 => add_100, add_101, add_99, mul_138, mul_139, mul_140, mul_141, mul_142, rsqrt_19, squeeze_58, var_mean_19
triton_red_fused__native_batch_norm_legit_functional_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[512, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_32', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 8192
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
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (524288*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp5 = 8192.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = 0.1
    tmp11 = tmp2 * tmp10
    tmp13 = 0.9
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = 1.0001220852154804
    tmp17 = tmp6 * tmp16
    tmp18 = tmp17 * tmp10
    tmp20 = tmp19 * tmp13
    tmp21 = tmp18 + tmp20
    tl.store(out_ptr2 + (x0), tmp9, xmask)
    tl.store(out_ptr4 + (x0), tmp15, xmask)
    tl.store(out_ptr6 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cl/cclyyox5anpm2co4wx53rjfts74qwjwicax6yvj2hwcafjmjcfau.py
# Source Nodes: [getattr_l__mod___layer2___0___downsample_0], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer2___0___downsample_0 => avg_pool2d_1
triton_poi_fused_avg_pool2d_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_33', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ce/ccecqavjyd76ku5ttnpkof5aab6pzzehrav7gl6mki33l57a4psl.py
# Source Nodes: [out_46, out_47, shortcut_5, shortcut_6], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_46 => add_102, add_99, mul_137, mul_143, rsqrt_19, sub_23, var_mean_19
# out_47 => add_108
# shortcut_5 => add_104, add_107, mul_144, mul_150, rsqrt_20, sub_24, var_mean_20
# shortcut_6 => relu_18
triton_poi_fused__native_batch_norm_legit_functional_add_relu_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_34', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x3), None)
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 8192.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = triton_helpers.maximum(0, tmp26)
    tl.store(in_out_ptr0 + (x3), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fz/cfzoarqe27cyec4zd6mrdx2r44alfmfecfj2dcptmsb7iegstfxb.py
# Source Nodes: [out_50], Original ATen: [aten._native_batch_norm_legit_functional]
# out_50 => add_110, add_111, add_112, mul_152, mul_153, mul_154, mul_155, mul_156, rsqrt_21, squeeze_64, var_mean_21
triton_red_fused__native_batch_norm_legit_functional_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_35', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 8192
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
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp5 = 8192.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = 0.1
    tmp11 = tmp2 * tmp10
    tmp13 = 0.9
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = 1.0001220852154804
    tmp17 = tmp6 * tmp16
    tmp18 = tmp17 * tmp10
    tmp20 = tmp19 * tmp13
    tmp21 = tmp18 + tmp20
    tl.store(out_ptr2 + (x0), tmp9, xmask)
    tl.store(out_ptr4 + (x0), tmp15, xmask)
    tl.store(out_ptr6 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6i/c6ihjapvgnkvinhn2eucatsp7mxpi2ngp6ovlysdamt4iir34ikk.py
# Source Nodes: [out_50, out_51], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# out_50 => add_110, add_113, mul_151, mul_157, rsqrt_21, sub_25, var_mean_21
# out_51 => relu_19
triton_poi_fused__native_batch_norm_legit_functional_relu_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 8192.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/np/cnppxakrujdf6r6df4i5pmjcdmywlnpewpq25tkwuhmfp7cpfzy3.py
# Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
# x_38 => add_115, add_116, add_117, mul_159, mul_160, mul_161, mul_162, mul_163, rsqrt_22, squeeze_67, var_mean_22
triton_red_fused__native_batch_norm_legit_functional_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_37', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8192
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
        r1 = rindex % 1024
        r2 = (rindex // 1024)
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp5 = 8192.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = 0.1
    tmp11 = tmp2 * tmp10
    tmp13 = 0.9
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = 1.0001220852154804
    tmp17 = tmp6 * tmp16
    tmp18 = tmp17 * tmp10
    tmp20 = tmp19 * tmp13
    tmp21 = tmp18 + tmp20
    tl.store(out_ptr2 + (x0), tmp9, xmask)
    tl.store(out_ptr4 + (x0), tmp15, xmask)
    tl.store(out_ptr6 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xf/cxfkuzdrqnvw24hb3kosg75xd4nh2lz6s4uc4v6sasjpwnstleqt.py
# Source Nodes: [x_38, x_40], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_38 => add_115, add_118, mul_158, mul_164, rsqrt_22, sub_26, var_mean_22
# x_40 => relu_20
triton_poi_fused__native_batch_norm_legit_functional_relu_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 8192.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kr/ckrfjj2wxxblndeb276bvrtcvrul6le3kwfe33ctkcukkhh5yc64.py
# Source Nodes: [x_gap_20, x_gap_21], Original ATen: [aten.mean, aten.sum]
# x_gap_20 => sum_13
# x_gap_21 => mean_4
triton_per_fused_mean_sum_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_sum_39', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (1024*x0) + (262144*x1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (131072 + r2 + (1024*x0) + (262144*x1)), rmask & xmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask & xmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp7 = 1024.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ip/ciplkyojjzegxir7tetn7nvv3z6pf7zobvxhvk4qh663ltevn2ug.py
# Source Nodes: [mul_4, out_52], Original ATen: [aten.mul, aten.sum]
# mul_4 => mul_172
# out_52 => sum_15
triton_poi_fused_mul_sum_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sum_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 131072)
    x3 = xindex % 131072
    x1 = (xindex // 1024) % 128
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (262144*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (256*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (131072 + x3 + (262144*x2)), None)
    tmp4 = tl.load(in_ptr1 + (128 + x1 + (256*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tl.store(out_ptr0 + (x4), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rw/crwf4prlvczgeqfbfexuoosqfkbbiex4yj6n7u4mkixwti2jrf66.py
# Source Nodes: [out_58, out_59, shortcut_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_58 => add_125, add_128, mul_173, mul_179, rsqrt_24, sub_29, var_mean_24
# out_59 => add_129
# shortcut_7 => relu_22
triton_poi_fused__native_batch_norm_legit_functional_add_relu_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x3), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 8192.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = triton_helpers.maximum(0, tmp15)
    tl.store(out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ic/cicxcwmwv7imflawj4bqbb672ktmstjexhmr7q2omhtirgyvw3lj.py
# Source Nodes: [x_63, x_65], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_63 => add_178, add_181, mul_245, mul_251, rsqrt_34, sub_41, var_mean_34
# x_65 => relu_32
triton_poi_fused__native_batch_norm_legit_functional_relu_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 8192.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qc/cqcoefwqihak4j7sydabni2pqtvrfz4yxfj5s2s7j7wbyul5vryf.py
# Source Nodes: [x_gap_35, x_gap_36], Original ATen: [aten.mean, aten.sum]
# x_gap_35 => sum_22
# x_gap_36 => mean_7
triton_per_fused_mean_sum_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_sum_43', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (1024*x0) + (524288*x1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (262144 + r2 + (1024*x0) + (524288*x1)), rmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp7 = 1024.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rn/crnmhiddutgplcnfwu5k27ectc2az72uzcqqf3dumsgxfjoxopfs.py
# Source Nodes: [x_gap_37], Original ATen: [aten.convolution]
# x_gap_37 => convolution_42
triton_poi_fused_convolution_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_44', 'mutated_arg_names': ['in_out_ptr0']},
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
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kv/ckvua24vj3mb3zaqbkn3sgsig6qwhcj4yh2jbc5tp4hsp7ryg4wd.py
# Source Nodes: [x_gap_38], Original ATen: [aten._native_batch_norm_legit_functional]
# x_gap_38 => add_184, add_185, mul_253, mul_254, mul_255, mul_256, mul_257, var_mean_35
triton_per_fused__native_batch_norm_legit_functional_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[128, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_45', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr3', 'out_ptr5']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr3, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*r1)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
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
    tmp23 = 8.0
    tmp24 = tmp16 / tmp23
    tmp25 = 1.1428571428571428
    tmp26 = tmp24 * tmp25
    tmp27 = tmp26 * tmp17
    tmp29 = tmp28 * tmp20
    tmp30 = tmp27 + tmp29
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr5 + (x0), tmp30, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7l/c7lfakbzgudqhh6eaeyqglhn3p3hhyvdlalx6oyrbclaqsmlw7sb.py
# Source Nodes: [x_gap_38, x_gap_39], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_gap_38 => add_183, add_186, mul_252, mul_258, rsqrt_35, sub_42, var_mean_35
# x_gap_39 => relu_33
triton_poi_fused__native_batch_norm_legit_functional_relu_46 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 8.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f3/cf35fsuc4xxkxugzwn356j77l7omuqmnbgr5nhutybqr46mlyk22.py
# Source Nodes: [x_68], Original ATen: [aten._softmax]
# x_68 => amax_7, exp_7, sub_43
triton_poi_fused__softmax_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = xindex % 512
    x0 = xindex % 256
    x2 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (x0 + (512*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (256 + x0 + (512*x2)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (256 + x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 + tmp7
    tmp9 = triton_helpers.maximum(tmp5, tmp8)
    tmp10 = tmp2 - tmp9
    tmp11 = tl.exp(tmp10)
    tl.store(out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xr/cxreg5mnxw7gygxrjbi7isithnevhkkcsgualayconsutv35dfqx.py
# Source Nodes: [x_68], Original ATen: [aten._softmax]
# x_68 => div_7, sum_23
triton_poi_fused__softmax_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 256
    x2 = (xindex // 512)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x0 + (512*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (256 + x0 + (512*x2)), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 / tmp3
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ey/ceyicxhuyfdhefd5hoywk7bytvoawkbmkdo77gjeagajenyfw2jn.py
# Source Nodes: [mul_7, out_88], Original ATen: [aten.mul, aten.sum]
# mul_7 => mul_259
# out_88 => sum_24
triton_poi_fused_mul_sum_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sum_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 262144)
    x3 = xindex % 262144
    x1 = (xindex // 1024) % 256
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (524288*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (512*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (262144 + x3 + (524288*x2)), None)
    tmp4 = tl.load(in_ptr1 + (256 + x1 + (512*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tl.store(out_ptr0 + (x4), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/54/c54hba5zqxvmt56srcqd5dmllos3rjfpu5jqglywdnqtx5x7gsau.py
# Source Nodes: [out_93], Original ATen: [aten.avg_pool2d]
# out_93 => avg_pool2d_2
triton_poi_fused_avg_pool2d_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 16) % 16
    x0 = xindex % 16
    x3 = (xindex // 16)
    x4 = xindex
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 32, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-33) + (2*x0) + (64*x3)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-32) + (2*x0) + (64*x3)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-31) + (2*x0) + (64*x3)), tmp27, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + (2*x0) + (64*x3)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + ((2*x0) + (64*x3)), tmp41, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + (2*x0) + (64*x3)), tmp46, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (31 + (2*x0) + (64*x3)), tmp55, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (32 + (2*x0) + (64*x3)), tmp60, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (33 + (2*x0) + (64*x3)), tmp65, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 33, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x4), tmp145, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/va/cvak22ow3chifncohdkfvfqmeqicpypumscaiu3js66n2wa36vdn.py
# Source Nodes: [out_95], Original ATen: [aten._native_batch_norm_legit_functional]
# out_95 => add_188, add_189, add_190, mul_261, mul_262, mul_263, mul_264, mul_265, rsqrt_36, squeeze_109, var_mean_36
triton_red_fused__native_batch_norm_legit_functional_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_51', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 2048
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
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (262144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp5 = 2048.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = 0.1
    tmp11 = tmp2 * tmp10
    tmp13 = 0.9
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = 1.0004885197850513
    tmp17 = tmp6 * tmp16
    tmp18 = tmp17 * tmp10
    tmp20 = tmp19 * tmp13
    tmp21 = tmp18 + tmp20
    tl.store(out_ptr2 + (x0), tmp9, xmask)
    tl.store(out_ptr4 + (x0), tmp15, xmask)
    tl.store(out_ptr6 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/va/cvahl2bjsjrlx7zfva6toulgr7poyctt2px4emlgt3cp4fmpqjce.py
# Source Nodes: [getattr_l__mod___layer3___0___downsample_0], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer3___0___downsample_0 => avg_pool2d_3
triton_poi_fused_avg_pool2d_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_52', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/tx/ctxcr6bzz6k6ozi7bme6vcey3ykoal4i6xzfp7pvtt5f4rdk4oto.py
# Source Nodes: [out_95, out_96, shortcut_10, shortcut_11], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_95 => add_188, add_191, mul_260, mul_266, rsqrt_36, sub_44, var_mean_36
# out_96 => add_197
# shortcut_10 => add_193, add_196, mul_267, mul_273, rsqrt_37, sub_45, var_mean_37
# shortcut_11 => relu_34
triton_poi_fused__native_batch_norm_legit_functional_add_relu_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_53', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x3), None)
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = triton_helpers.maximum(0, tmp26)
    tl.store(in_out_ptr0 + (x3), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5w/c5witekgw23z6spaqojelhyqahjvl2ep43cy6bpfm5zsnfhniuuw.py
# Source Nodes: [out_99], Original ATen: [aten._native_batch_norm_legit_functional]
# out_99 => add_199, add_200, add_201, mul_275, mul_276, mul_277, mul_278, mul_279, rsqrt_38, squeeze_115, var_mean_38
triton_red_fused__native_batch_norm_legit_functional_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_54', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 2048
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
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (65536*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp5 = 2048.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = 0.1
    tmp11 = tmp2 * tmp10
    tmp13 = 0.9
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = 1.0004885197850513
    tmp17 = tmp6 * tmp16
    tmp18 = tmp17 * tmp10
    tmp20 = tmp19 * tmp13
    tmp21 = tmp18 + tmp20
    tl.store(out_ptr2 + (x0), tmp9, xmask)
    tl.store(out_ptr4 + (x0), tmp15, xmask)
    tl.store(out_ptr6 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4p/c4psgztjefkf2e7migmtrxoyfc26ytlnt6cemewcd7wxrilqynl5.py
# Source Nodes: [out_100, out_99], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# out_100 => relu_35
# out_99 => add_199, add_202, mul_274, mul_280, rsqrt_38, sub_46, var_mean_38
triton_poi_fused__native_batch_norm_legit_functional_relu_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dd/cdd2c7bqwno4ziu7zompqs6qzyqgufdqxa52g6ctda3kl5wshye4.py
# Source Nodes: [x_71], Original ATen: [aten._native_batch_norm_legit_functional]
# x_71 => add_204, add_205, add_206, mul_282, mul_283, mul_284, mul_285, mul_286, rsqrt_39, squeeze_118, var_mean_39
triton_red_fused__native_batch_norm_legit_functional_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_56', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2048
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
        r1 = rindex % 256
        r2 = (rindex // 256)
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (131072*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp5 = 2048.0
    tmp6 = tmp3 / tmp5
    tmp7 = 1e-05
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = 0.1
    tmp11 = tmp2 * tmp10
    tmp13 = 0.9
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = 1.0004885197850513
    tmp17 = tmp6 * tmp16
    tmp18 = tmp17 * tmp10
    tmp20 = tmp19 * tmp13
    tmp21 = tmp18 + tmp20
    tl.store(out_ptr2 + (x0), tmp9, xmask)
    tl.store(out_ptr4 + (x0), tmp15, xmask)
    tl.store(out_ptr6 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hb/chbh3gujfmgla4dqxmfnduqwqkfddsdgefw45r2e235bn5jfdy7g.py
# Source Nodes: [x_71, x_73], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_71 => add_204, add_207, mul_281, mul_287, rsqrt_39, sub_47, var_mean_39
# x_73 => relu_36
triton_poi_fused__native_batch_norm_legit_functional_relu_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vl/cvlieprq7aey7svxe3yfgylhmws6du34c2ejiiwn6c3zaiwuxxsb.py
# Source Nodes: [x_gap_40, x_gap_41], Original ATen: [aten.mean, aten.sum]
# x_gap_40 => sum_25
# x_gap_41 => mean_8
triton_per_fused_mean_sum_58 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_sum_58', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 256
    x1 = (xindex // 256)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (256*x0) + (131072*x1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (65536 + r2 + (256*x0) + (131072*x1)), rmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp7 = 256.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jp/cjpjqhipgkhpbalxrailvaeneanofodapi3q5hdqwsne4mj46agk.py
# Source Nodes: [mul_8, out_101], Original ATen: [aten.mul, aten.sum]
# mul_8 => mul_295
# out_101 => sum_27
triton_poi_fused_mul_sum_59 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sum_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 65536)
    x3 = xindex % 65536
    x1 = (xindex // 256) % 256
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (131072*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (512*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (65536 + x3 + (131072*x2)), None)
    tmp4 = tl.load(in_ptr1 + (256 + x1 + (512*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tl.store(out_ptr0 + (x4), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2k/c2kiyrp3p7fzjrrs43m2yz2coarmosmsdkxoialu24jhwugfiapi.py
# Source Nodes: [out_107, out_108, shortcut_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_107 => add_214, add_217, mul_296, mul_302, rsqrt_41, sub_50, var_mean_41
# out_108 => add_218
# shortcut_12 => relu_38
triton_poi_fused__native_batch_norm_legit_functional_add_relu_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x3), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = triton_helpers.maximum(0, tmp15)
    tl.store(out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/f4/cf4iuaihqorvpswrgfbozkeaxzwnzdg6zg4xhgycjgcunoeln666.py
# Source Nodes: [x_248, x_250], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_248 => add_666, add_669, mul_919, mul_925, rsqrt_127, sub_157, var_mean_127
# x_250 => relu_124
triton_poi_fused__native_batch_norm_legit_functional_relu_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_61', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 2048.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tz/ctz5qdq5wy6qlmjkzz2s3uyw2653mbg6rnkfh3xln2jiq6ymfix7.py
# Source Nodes: [x_gap_150, x_gap_151], Original ATen: [aten.mean, aten.sum]
# x_gap_150 => sum_91
# x_gap_151 => mean_30
triton_per_fused_mean_sum_62 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_sum_62', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel):
    xnumel = 4096
    XBLOCK: tl.constexpr = 1
    rnumel = 256
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (256*x0) + (262144*x1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (131072 + r2 + (256*x0) + (262144*x1)), rmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = triton_helpers.promote_to_tensor(tl.sum(tmp5, 0))
    tmp7 = 256.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/es/cesacbrjnlvlzdzwvmsvbsv3v6wcynb6vszqjagmbw5ti7lxfcxc.py
# Source Nodes: [x_gap_152], Original ATen: [aten.convolution]
# x_gap_152 => convolution_158
triton_poi_fused_convolution_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_63', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/u3/cu3qlvrpx56fwrqkhlga57aqguljvqoam5h4rihar2i6njyuvmg2.py
# Source Nodes: [x_gap_153], Original ATen: [aten._native_batch_norm_legit_functional]
# x_gap_153 => add_672, add_673, mul_927, mul_928, mul_929, mul_930, mul_931, var_mean_128
triton_per_fused__native_batch_norm_legit_functional_64 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 8],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_64', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr3', 'out_ptr5']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr3, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 256
    rnumel = 8
    RBLOCK: tl.constexpr = 8
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*r1)), rmask & xmask, other=0.0)
    tmp19 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
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
    tmp23 = 8.0
    tmp24 = tmp16 / tmp23
    tmp25 = 1.1428571428571428
    tmp26 = tmp24 * tmp25
    tmp27 = tmp26 * tmp17
    tmp29 = tmp28 * tmp20
    tmp30 = tmp27 + tmp29
    tl.store(out_ptr3 + (x0), tmp22, xmask)
    tl.store(out_ptr5 + (x0), tmp30, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kd/ckd6k4zqb6mz6knui3zeorvdhsx64z3plix5bzdis6vg4vp55zlg.py
# Source Nodes: [x_gap_153, x_gap_154], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_gap_153 => add_671, add_674, mul_926, mul_932, rsqrt_128, sub_158, var_mean_128
# x_gap_154 => relu_125
triton_poi_fused__native_batch_norm_legit_functional_relu_65 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
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
    tmp4 = 8.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x2), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kp/ckpfbniasipcwx5pa6ayslbr4rinwuhbdsvogwnx5h3szic2wt3o.py
# Source Nodes: [x_253], Original ATen: [aten._softmax]
# x_253 => amax_30, exp_30, sub_159
triton_poi_fused__softmax_66 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_66', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x4 = xindex % 1024
    x0 = xindex % 512
    x2 = (xindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr0 + (512 + x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr1 + (512 + x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp5 = tmp3 + tmp4
    tmp8 = tmp6 + tmp7
    tmp9 = triton_helpers.maximum(tmp5, tmp8)
    tmp10 = tmp2 - tmp9
    tmp11 = tl.exp(tmp10)
    tl.store(out_ptr0 + (x3), tmp11, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6e/c6e6ocextwfnetmq62oni3mtnoz2aebr2c62wpm4tt2gfcvsw7dq.py
# Source Nodes: [x_253], Original ATen: [aten._softmax]
# x_253 => div_30, sum_92
triton_poi_fused__softmax_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__softmax_67', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x0 = xindex % 512
    x2 = (xindex // 1024)
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr0 + (512 + x0 + (1024*x2)), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 / tmp3
    tl.store(out_ptr0 + (x3), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ut/cutyljj5e3fj73ou6saiishxgr7mzsxpodu26vketlzbb6jmlqmh.py
# Source Nodes: [mul_30, out_365], Original ATen: [aten.mul, aten.sum]
# mul_30 => mul_933
# out_365 => sum_93
triton_poi_fused_mul_sum_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sum_68', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 131072)
    x3 = xindex % 131072
    x1 = (xindex // 256) % 512
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (262144*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (131072 + x3 + (262144*x2)), None)
    tmp4 = tl.load(in_ptr1 + (512 + x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tl.store(out_ptr0 + (x4), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vj/cvjy24eznuirbqaluipofeaqauy3gqbl7jayqj6ppruhi3sve4hm.py
# Source Nodes: [out_370], Original ATen: [aten.avg_pool2d]
# out_370 => avg_pool2d_4
triton_poi_fused_avg_pool2d_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_69', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 8) % 8
    x0 = xindex % 8
    x3 = (xindex // 8)
    x4 = xindex
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 16, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-17) + (2*x0) + (32*x3)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-16) + (2*x0) + (32*x3)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-15) + (2*x0) + (32*x3)), tmp27, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + (2*x0) + (32*x3)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + ((2*x0) + (32*x3)), tmp41, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + (2*x0) + (32*x3)), tmp46, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (15 + (2*x0) + (32*x3)), tmp55, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (16 + (2*x0) + (32*x3)), tmp60, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (17 + (2*x0) + (32*x3)), tmp65, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 17, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x4), tmp145, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qg/cqg3rakup7a2ugoqnkcve2hrgeht7tmpdutziffjfhrkrfuettsh.py
# Source Nodes: [out_372], Original ATen: [aten._native_batch_norm_legit_functional]
# out_372 => add_676, add_677, add_678, mul_935, mul_936, mul_937, mul_938, mul_939, rsqrt_129, squeeze_388, var_mean_129
triton_per_fused__native_batch_norm_legit_functional_70 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_70', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel):
    xnumel = 2048
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 64
    r2 = (rindex // 64)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (131072*r2)), rmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 512, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = 512.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = 0.1
    tmp23 = tmp10 * tmp22
    tmp25 = 0.9
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 + tmp26
    tmp28 = 1.0019569471624266
    tmp29 = tmp18 * tmp28
    tmp30 = tmp29 * tmp22
    tmp32 = tmp31 * tmp25
    tmp33 = tmp30 + tmp32
    tl.store(out_ptr2 + (x0), tmp21, None)
    tl.store(out_ptr4 + (x0), tmp27, None)
    tl.store(out_ptr6 + (x0), tmp33, None)
    tl.store(out_ptr0 + (x0), tmp10, None)
    tl.store(out_ptr1 + (x0), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bs/cbssn3nvnxzpl6znypsdsj4vvoalcmz4nwuydz72nrtqtrnkhden.py
# Source Nodes: [getattr_l__mod___layer4___0___downsample_0], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer4___0___downsample_0 => avg_pool2d_5
triton_poi_fused_avg_pool2d_71 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_71', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
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


# kernel path: /tmp/torchinductor_youkaichao/2a/c2aqn53cugjzeoebjyfkjftyryvjb2awldwab6wexjtzpcum3eik.py
# Source Nodes: [out_372, out_373, shortcut_34, shortcut_35], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_372 => add_676, add_679, mul_934, mul_940, rsqrt_129, sub_160, var_mean_129
# out_373 => add_685
# shortcut_34 => add_681, add_684, mul_941, mul_947, rsqrt_130, sub_161, var_mean_130
# shortcut_35 => relu_126
triton_poi_fused__native_batch_norm_legit_functional_add_relu_72 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_72', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 2048
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x3), None)
    tmp15 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp16 = tmp14 - tmp15
    tmp18 = tmp17 / tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp16 * tmp20
    tmp23 = tmp21 * tmp22
    tmp25 = tmp23 + tmp24
    tmp26 = tmp13 + tmp25
    tmp27 = triton_helpers.maximum(0, tmp26)
    tl.store(in_out_ptr0 + (x3), tmp27, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6q/c6qaviv74hy5x7syessigt4jsafroevdb4vj6wxwtoa7a32oiac7.py
# Source Nodes: [out_376], Original ATen: [aten._native_batch_norm_legit_functional]
# out_376 => add_687, add_688, add_689, mul_949, mul_950, mul_951, mul_952, mul_953, rsqrt_131, squeeze_394, var_mean_131
triton_per_fused__native_batch_norm_legit_functional_73 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_73', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel):
    xnumel = 512
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 64
    r2 = (rindex // 64)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (32768*r2)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
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
    tmp17 = 512.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = 0.1
    tmp23 = tmp10 * tmp22
    tmp25 = 0.9
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 + tmp26
    tmp28 = 1.0019569471624266
    tmp29 = tmp18 * tmp28
    tmp30 = tmp29 * tmp22
    tmp32 = tmp31 * tmp25
    tmp33 = tmp30 + tmp32
    tl.store(out_ptr2 + (x0), tmp21, xmask)
    tl.store(out_ptr4 + (x0), tmp27, xmask)
    tl.store(out_ptr6 + (x0), tmp33, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qe/cqeb7hejutk7jos75yoavciottah7rkq367ng6v2qnh6lu6bzqht.py
# Source Nodes: [out_376, out_377], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# out_376 => add_687, add_690, mul_948, mul_954, rsqrt_131, sub_162, var_mean_131
# out_377 => relu_127
triton_poi_fused__native_batch_norm_legit_functional_relu_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_74', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2x/c2xl4h254go2y6vqr63rdrkmne7zqqqoainjqc35xgfrtfy3j4rf.py
# Source Nodes: [x_256], Original ATen: [aten._native_batch_norm_legit_functional]
# x_256 => add_692, add_693, add_694, mul_956, mul_957, mul_958, mul_959, mul_960, rsqrt_132, squeeze_397, var_mean_132
triton_per_fused__native_batch_norm_legit_functional_75 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_75', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel):
    xnumel = 1024
    XBLOCK: tl.constexpr = 1
    rnumel = 512
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 64
    r2 = (rindex // 64)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (65536*r2)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
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
    tmp17 = 512.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = 0.1
    tmp23 = tmp10 * tmp22
    tmp25 = 0.9
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 + tmp26
    tmp28 = 1.0019569471624266
    tmp29 = tmp18 * tmp28
    tmp30 = tmp29 * tmp22
    tmp32 = tmp31 * tmp25
    tmp33 = tmp30 + tmp32
    tl.store(out_ptr2 + (x0), tmp21, xmask)
    tl.store(out_ptr4 + (x0), tmp27, xmask)
    tl.store(out_ptr6 + (x0), tmp33, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rn/crnpxrhdjw7jhcnes4bksbbionbp3h4smad6ml3nfhymahcvn6cz.py
# Source Nodes: [x_256, x_258], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_256 => add_692, add_695, mul_955, mul_961, rsqrt_132, sub_163, var_mean_132
# x_258 => relu_128
triton_poi_fused__native_batch_norm_legit_functional_relu_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_76', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6c/c6c3jc4iyqjm4tufn3xuxv7zuhknwl3v2h4io53gafz3n3jrwysi.py
# Source Nodes: [x_gap_155, x_gap_156], Original ATen: [aten.mean, aten.sum]
# x_gap_155 => sum_94
# x_gap_156 => mean_31
triton_per_fused_mean_sum_77 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_sum_77', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 512
    x1 = (xindex // 512)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (r2 + (64*x0) + (65536*x1)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (32768 + r2 + (64*x0) + (65536*x1)), rmask, other=0.0)
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp5 = tl.where(rmask, tmp3, 0)
    tmp6 = tl.sum(tmp5, 1)[:, None]
    tmp7 = 64.0
    tmp8 = tmp6 / tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ed/cedbqu5nwhc7po6cddcezetr2kzrjuuydzp4uludt3j3w3njussw.py
# Source Nodes: [mul_31, out_378], Original ATen: [aten.mul, aten.sum]
# mul_31 => mul_969
# out_378 => sum_96
triton_poi_fused_mul_sum_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_sum_78', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 32768)
    x3 = xindex % 32768
    x1 = (xindex // 64) % 512
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x3 + (65536*x2)), None)
    tmp1 = tl.load(in_ptr1 + (x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (32768 + x3 + (65536*x2)), None)
    tmp4 = tl.load(in_ptr1 + (512 + x1 + (1024*x2)), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp5 = tmp3 * tmp4
    tmp6 = tmp2 + tmp5
    tl.store(out_ptr0 + (x4), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nj/cnjy2cwzbnzq3uqaawxbduphwxuympqn4ij62pveydcyv7egqcfv.py
# Source Nodes: [out_384, out_385, shortcut_36], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_384 => add_702, add_705, mul_970, mul_976, rsqrt_134, sub_166, var_mean_134
# out_385 => add_706
# shortcut_36 => relu_130
triton_poi_fused__native_batch_norm_legit_functional_add_relu_79 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1048576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 2048
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x3), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = triton_helpers.maximum(0, tmp15)
    tl.store(out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6p/c6pamt5kqlykeid74h2ks7qcap2fcheavmbadhrnqxqfzukltvyp.py
# Source Nodes: [out_396, out_397, x_272, x_273, x_275], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.mean, aten.relu, aten.threshold_backward, aten.view]
# out_396 => add_723, add_726, mul_1005, mul_999, rsqrt_138, sub_171, var_mean_138
# out_397 => add_727
# x_272 => relu_134
# x_273 => mean_33
# x_275 => view_198
triton_per_fused__native_batch_norm_legit_functional_add_mean_relu_threshold_backward_view_80 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_add_mean_relu_threshold_backward_view_80', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + (r2 + (64*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (r2 + (64*x3)), rmask, other=0.0)
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = triton_helpers.maximum(0, tmp15)
    tmp17 = 0.0
    tmp18 = tmp16 <= tmp17
    tmp19 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp21 = tl.where(rmask, tmp19, 0)
    tmp22 = tl.sum(tmp21, 1)[:, None]
    tmp23 = 64.0
    tmp24 = tmp22 / tmp23
    tl.store(out_ptr1 + (r2 + (64*x3)), tmp18, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp24, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ox/coxjuj4deglod7tiqaex74ohbnsdihje6uk7aaubvyhlkkebafg7.py
# Source Nodes: [l__mod___conv1_1], Original ATen: [aten.add]
# l__mod___conv1_1 => add
triton_poi_fused_add_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_81', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897, primals_898, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, primals_907, primals_908, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_919, primals_920, primals_921, primals_922, primals_923, primals_924, primals_925, primals_926, primals_927, primals_928, primals_929, primals_930, primals_931, primals_932, primals_933, primals_934, primals_935, primals_936 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_10, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_11, (64, ), (1, ))
    assert_size_stride(primals_12, (64, ), (1, ))
    assert_size_stride(primals_13, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_16, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_17, (32, ), (1, ))
    assert_size_stride(primals_18, (32, ), (1, ))
    assert_size_stride(primals_19, (32, ), (1, ))
    assert_size_stride(primals_20, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_22, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_24, (256, ), (1, ))
    assert_size_stride(primals_25, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_26, (256, ), (1, ))
    assert_size_stride(primals_27, (256, ), (1, ))
    assert_size_stride(primals_28, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_29, (64, ), (1, ))
    assert_size_stride(primals_30, (64, ), (1, ))
    assert_size_stride(primals_31, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_32, (128, ), (1, ))
    assert_size_stride(primals_33, (128, ), (1, ))
    assert_size_stride(primals_34, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_35, (32, ), (1, ))
    assert_size_stride(primals_36, (32, ), (1, ))
    assert_size_stride(primals_37, (32, ), (1, ))
    assert_size_stride(primals_38, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_40, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_41, (256, ), (1, ))
    assert_size_stride(primals_42, (256, ), (1, ))
    assert_size_stride(primals_43, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_44, (64, ), (1, ))
    assert_size_stride(primals_45, (64, ), (1, ))
    assert_size_stride(primals_46, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_48, (128, ), (1, ))
    assert_size_stride(primals_49, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_50, (32, ), (1, ))
    assert_size_stride(primals_51, (32, ), (1, ))
    assert_size_stride(primals_52, (32, ), (1, ))
    assert_size_stride(primals_53, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_54, (128, ), (1, ))
    assert_size_stride(primals_55, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_56, (256, ), (1, ))
    assert_size_stride(primals_57, (256, ), (1, ))
    assert_size_stride(primals_58, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_59, (128, ), (1, ))
    assert_size_stride(primals_60, (128, ), (1, ))
    assert_size_stride(primals_61, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_62, (256, ), (1, ))
    assert_size_stride(primals_63, (256, ), (1, ))
    assert_size_stride(primals_64, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_65, (64, ), (1, ))
    assert_size_stride(primals_66, (64, ), (1, ))
    assert_size_stride(primals_67, (64, ), (1, ))
    assert_size_stride(primals_68, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_69, (256, ), (1, ))
    assert_size_stride(primals_70, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_71, (512, ), (1, ))
    assert_size_stride(primals_72, (512, ), (1, ))
    assert_size_stride(primals_73, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_74, (512, ), (1, ))
    assert_size_stride(primals_75, (512, ), (1, ))
    assert_size_stride(primals_76, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_77, (128, ), (1, ))
    assert_size_stride(primals_78, (128, ), (1, ))
    assert_size_stride(primals_79, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_80, (256, ), (1, ))
    assert_size_stride(primals_81, (256, ), (1, ))
    assert_size_stride(primals_82, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_83, (64, ), (1, ))
    assert_size_stride(primals_84, (64, ), (1, ))
    assert_size_stride(primals_85, (64, ), (1, ))
    assert_size_stride(primals_86, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_87, (256, ), (1, ))
    assert_size_stride(primals_88, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_89, (512, ), (1, ))
    assert_size_stride(primals_90, (512, ), (1, ))
    assert_size_stride(primals_91, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_92, (128, ), (1, ))
    assert_size_stride(primals_93, (128, ), (1, ))
    assert_size_stride(primals_94, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_95, (256, ), (1, ))
    assert_size_stride(primals_96, (256, ), (1, ))
    assert_size_stride(primals_97, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_98, (64, ), (1, ))
    assert_size_stride(primals_99, (64, ), (1, ))
    assert_size_stride(primals_100, (64, ), (1, ))
    assert_size_stride(primals_101, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_102, (256, ), (1, ))
    assert_size_stride(primals_103, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_104, (512, ), (1, ))
    assert_size_stride(primals_105, (512, ), (1, ))
    assert_size_stride(primals_106, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_107, (128, ), (1, ))
    assert_size_stride(primals_108, (128, ), (1, ))
    assert_size_stride(primals_109, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_110, (256, ), (1, ))
    assert_size_stride(primals_111, (256, ), (1, ))
    assert_size_stride(primals_112, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_113, (64, ), (1, ))
    assert_size_stride(primals_114, (64, ), (1, ))
    assert_size_stride(primals_115, (64, ), (1, ))
    assert_size_stride(primals_116, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_117, (256, ), (1, ))
    assert_size_stride(primals_118, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_119, (512, ), (1, ))
    assert_size_stride(primals_120, (512, ), (1, ))
    assert_size_stride(primals_121, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_122, (256, ), (1, ))
    assert_size_stride(primals_123, (256, ), (1, ))
    assert_size_stride(primals_124, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_125, (512, ), (1, ))
    assert_size_stride(primals_126, (512, ), (1, ))
    assert_size_stride(primals_127, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_128, (128, ), (1, ))
    assert_size_stride(primals_129, (128, ), (1, ))
    assert_size_stride(primals_130, (128, ), (1, ))
    assert_size_stride(primals_131, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_132, (512, ), (1, ))
    assert_size_stride(primals_133, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_134, (1024, ), (1, ))
    assert_size_stride(primals_135, (1024, ), (1, ))
    assert_size_stride(primals_136, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_137, (1024, ), (1, ))
    assert_size_stride(primals_138, (1024, ), (1, ))
    assert_size_stride(primals_139, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_140, (256, ), (1, ))
    assert_size_stride(primals_141, (256, ), (1, ))
    assert_size_stride(primals_142, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_143, (512, ), (1, ))
    assert_size_stride(primals_144, (512, ), (1, ))
    assert_size_stride(primals_145, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_146, (128, ), (1, ))
    assert_size_stride(primals_147, (128, ), (1, ))
    assert_size_stride(primals_148, (128, ), (1, ))
    assert_size_stride(primals_149, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_150, (512, ), (1, ))
    assert_size_stride(primals_151, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_152, (1024, ), (1, ))
    assert_size_stride(primals_153, (1024, ), (1, ))
    assert_size_stride(primals_154, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_155, (256, ), (1, ))
    assert_size_stride(primals_156, (256, ), (1, ))
    assert_size_stride(primals_157, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_158, (512, ), (1, ))
    assert_size_stride(primals_159, (512, ), (1, ))
    assert_size_stride(primals_160, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_161, (128, ), (1, ))
    assert_size_stride(primals_162, (128, ), (1, ))
    assert_size_stride(primals_163, (128, ), (1, ))
    assert_size_stride(primals_164, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_165, (512, ), (1, ))
    assert_size_stride(primals_166, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_167, (1024, ), (1, ))
    assert_size_stride(primals_168, (1024, ), (1, ))
    assert_size_stride(primals_169, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_170, (256, ), (1, ))
    assert_size_stride(primals_171, (256, ), (1, ))
    assert_size_stride(primals_172, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_173, (512, ), (1, ))
    assert_size_stride(primals_174, (512, ), (1, ))
    assert_size_stride(primals_175, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_176, (128, ), (1, ))
    assert_size_stride(primals_177, (128, ), (1, ))
    assert_size_stride(primals_178, (128, ), (1, ))
    assert_size_stride(primals_179, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_180, (512, ), (1, ))
    assert_size_stride(primals_181, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_182, (1024, ), (1, ))
    assert_size_stride(primals_183, (1024, ), (1, ))
    assert_size_stride(primals_184, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_185, (256, ), (1, ))
    assert_size_stride(primals_186, (256, ), (1, ))
    assert_size_stride(primals_187, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_188, (512, ), (1, ))
    assert_size_stride(primals_189, (512, ), (1, ))
    assert_size_stride(primals_190, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_191, (128, ), (1, ))
    assert_size_stride(primals_192, (128, ), (1, ))
    assert_size_stride(primals_193, (128, ), (1, ))
    assert_size_stride(primals_194, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_195, (512, ), (1, ))
    assert_size_stride(primals_196, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_197, (1024, ), (1, ))
    assert_size_stride(primals_198, (1024, ), (1, ))
    assert_size_stride(primals_199, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_200, (256, ), (1, ))
    assert_size_stride(primals_201, (256, ), (1, ))
    assert_size_stride(primals_202, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_203, (512, ), (1, ))
    assert_size_stride(primals_204, (512, ), (1, ))
    assert_size_stride(primals_205, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_206, (128, ), (1, ))
    assert_size_stride(primals_207, (128, ), (1, ))
    assert_size_stride(primals_208, (128, ), (1, ))
    assert_size_stride(primals_209, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_210, (512, ), (1, ))
    assert_size_stride(primals_211, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_212, (1024, ), (1, ))
    assert_size_stride(primals_213, (1024, ), (1, ))
    assert_size_stride(primals_214, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_215, (256, ), (1, ))
    assert_size_stride(primals_216, (256, ), (1, ))
    assert_size_stride(primals_217, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_218, (512, ), (1, ))
    assert_size_stride(primals_219, (512, ), (1, ))
    assert_size_stride(primals_220, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_221, (128, ), (1, ))
    assert_size_stride(primals_222, (128, ), (1, ))
    assert_size_stride(primals_223, (128, ), (1, ))
    assert_size_stride(primals_224, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_225, (512, ), (1, ))
    assert_size_stride(primals_226, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_227, (1024, ), (1, ))
    assert_size_stride(primals_228, (1024, ), (1, ))
    assert_size_stride(primals_229, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_230, (256, ), (1, ))
    assert_size_stride(primals_231, (256, ), (1, ))
    assert_size_stride(primals_232, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_233, (512, ), (1, ))
    assert_size_stride(primals_234, (512, ), (1, ))
    assert_size_stride(primals_235, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_236, (128, ), (1, ))
    assert_size_stride(primals_237, (128, ), (1, ))
    assert_size_stride(primals_238, (128, ), (1, ))
    assert_size_stride(primals_239, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_240, (512, ), (1, ))
    assert_size_stride(primals_241, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_242, (1024, ), (1, ))
    assert_size_stride(primals_243, (1024, ), (1, ))
    assert_size_stride(primals_244, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_245, (256, ), (1, ))
    assert_size_stride(primals_246, (256, ), (1, ))
    assert_size_stride(primals_247, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_248, (512, ), (1, ))
    assert_size_stride(primals_249, (512, ), (1, ))
    assert_size_stride(primals_250, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_251, (128, ), (1, ))
    assert_size_stride(primals_252, (128, ), (1, ))
    assert_size_stride(primals_253, (128, ), (1, ))
    assert_size_stride(primals_254, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_255, (512, ), (1, ))
    assert_size_stride(primals_256, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_257, (1024, ), (1, ))
    assert_size_stride(primals_258, (1024, ), (1, ))
    assert_size_stride(primals_259, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_260, (256, ), (1, ))
    assert_size_stride(primals_261, (256, ), (1, ))
    assert_size_stride(primals_262, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_263, (512, ), (1, ))
    assert_size_stride(primals_264, (512, ), (1, ))
    assert_size_stride(primals_265, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_266, (128, ), (1, ))
    assert_size_stride(primals_267, (128, ), (1, ))
    assert_size_stride(primals_268, (128, ), (1, ))
    assert_size_stride(primals_269, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_270, (512, ), (1, ))
    assert_size_stride(primals_271, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_272, (1024, ), (1, ))
    assert_size_stride(primals_273, (1024, ), (1, ))
    assert_size_stride(primals_274, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_275, (256, ), (1, ))
    assert_size_stride(primals_276, (256, ), (1, ))
    assert_size_stride(primals_277, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_278, (512, ), (1, ))
    assert_size_stride(primals_279, (512, ), (1, ))
    assert_size_stride(primals_280, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_281, (128, ), (1, ))
    assert_size_stride(primals_282, (128, ), (1, ))
    assert_size_stride(primals_283, (128, ), (1, ))
    assert_size_stride(primals_284, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_285, (512, ), (1, ))
    assert_size_stride(primals_286, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_287, (1024, ), (1, ))
    assert_size_stride(primals_288, (1024, ), (1, ))
    assert_size_stride(primals_289, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_290, (256, ), (1, ))
    assert_size_stride(primals_291, (256, ), (1, ))
    assert_size_stride(primals_292, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_293, (512, ), (1, ))
    assert_size_stride(primals_294, (512, ), (1, ))
    assert_size_stride(primals_295, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_296, (128, ), (1, ))
    assert_size_stride(primals_297, (128, ), (1, ))
    assert_size_stride(primals_298, (128, ), (1, ))
    assert_size_stride(primals_299, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_300, (512, ), (1, ))
    assert_size_stride(primals_301, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_302, (1024, ), (1, ))
    assert_size_stride(primals_303, (1024, ), (1, ))
    assert_size_stride(primals_304, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_305, (256, ), (1, ))
    assert_size_stride(primals_306, (256, ), (1, ))
    assert_size_stride(primals_307, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_308, (512, ), (1, ))
    assert_size_stride(primals_309, (512, ), (1, ))
    assert_size_stride(primals_310, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_311, (128, ), (1, ))
    assert_size_stride(primals_312, (128, ), (1, ))
    assert_size_stride(primals_313, (128, ), (1, ))
    assert_size_stride(primals_314, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_315, (512, ), (1, ))
    assert_size_stride(primals_316, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_317, (1024, ), (1, ))
    assert_size_stride(primals_318, (1024, ), (1, ))
    assert_size_stride(primals_319, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_320, (256, ), (1, ))
    assert_size_stride(primals_321, (256, ), (1, ))
    assert_size_stride(primals_322, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_323, (512, ), (1, ))
    assert_size_stride(primals_324, (512, ), (1, ))
    assert_size_stride(primals_325, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_326, (128, ), (1, ))
    assert_size_stride(primals_327, (128, ), (1, ))
    assert_size_stride(primals_328, (128, ), (1, ))
    assert_size_stride(primals_329, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_330, (512, ), (1, ))
    assert_size_stride(primals_331, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_332, (1024, ), (1, ))
    assert_size_stride(primals_333, (1024, ), (1, ))
    assert_size_stride(primals_334, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_335, (256, ), (1, ))
    assert_size_stride(primals_336, (256, ), (1, ))
    assert_size_stride(primals_337, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_338, (512, ), (1, ))
    assert_size_stride(primals_339, (512, ), (1, ))
    assert_size_stride(primals_340, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_341, (128, ), (1, ))
    assert_size_stride(primals_342, (128, ), (1, ))
    assert_size_stride(primals_343, (128, ), (1, ))
    assert_size_stride(primals_344, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_345, (512, ), (1, ))
    assert_size_stride(primals_346, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_347, (1024, ), (1, ))
    assert_size_stride(primals_348, (1024, ), (1, ))
    assert_size_stride(primals_349, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_350, (256, ), (1, ))
    assert_size_stride(primals_351, (256, ), (1, ))
    assert_size_stride(primals_352, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_353, (512, ), (1, ))
    assert_size_stride(primals_354, (512, ), (1, ))
    assert_size_stride(primals_355, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_356, (128, ), (1, ))
    assert_size_stride(primals_357, (128, ), (1, ))
    assert_size_stride(primals_358, (128, ), (1, ))
    assert_size_stride(primals_359, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_360, (512, ), (1, ))
    assert_size_stride(primals_361, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_362, (1024, ), (1, ))
    assert_size_stride(primals_363, (1024, ), (1, ))
    assert_size_stride(primals_364, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_365, (256, ), (1, ))
    assert_size_stride(primals_366, (256, ), (1, ))
    assert_size_stride(primals_367, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_368, (512, ), (1, ))
    assert_size_stride(primals_369, (512, ), (1, ))
    assert_size_stride(primals_370, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_371, (128, ), (1, ))
    assert_size_stride(primals_372, (128, ), (1, ))
    assert_size_stride(primals_373, (128, ), (1, ))
    assert_size_stride(primals_374, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_375, (512, ), (1, ))
    assert_size_stride(primals_376, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_377, (1024, ), (1, ))
    assert_size_stride(primals_378, (1024, ), (1, ))
    assert_size_stride(primals_379, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_380, (256, ), (1, ))
    assert_size_stride(primals_381, (256, ), (1, ))
    assert_size_stride(primals_382, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_383, (512, ), (1, ))
    assert_size_stride(primals_384, (512, ), (1, ))
    assert_size_stride(primals_385, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_386, (128, ), (1, ))
    assert_size_stride(primals_387, (128, ), (1, ))
    assert_size_stride(primals_388, (128, ), (1, ))
    assert_size_stride(primals_389, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_390, (512, ), (1, ))
    assert_size_stride(primals_391, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_392, (1024, ), (1, ))
    assert_size_stride(primals_393, (1024, ), (1, ))
    assert_size_stride(primals_394, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_395, (256, ), (1, ))
    assert_size_stride(primals_396, (256, ), (1, ))
    assert_size_stride(primals_397, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_398, (512, ), (1, ))
    assert_size_stride(primals_399, (512, ), (1, ))
    assert_size_stride(primals_400, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_401, (128, ), (1, ))
    assert_size_stride(primals_402, (128, ), (1, ))
    assert_size_stride(primals_403, (128, ), (1, ))
    assert_size_stride(primals_404, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_405, (512, ), (1, ))
    assert_size_stride(primals_406, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_407, (1024, ), (1, ))
    assert_size_stride(primals_408, (1024, ), (1, ))
    assert_size_stride(primals_409, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_410, (256, ), (1, ))
    assert_size_stride(primals_411, (256, ), (1, ))
    assert_size_stride(primals_412, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_413, (512, ), (1, ))
    assert_size_stride(primals_414, (512, ), (1, ))
    assert_size_stride(primals_415, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_416, (128, ), (1, ))
    assert_size_stride(primals_417, (128, ), (1, ))
    assert_size_stride(primals_418, (128, ), (1, ))
    assert_size_stride(primals_419, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_420, (512, ), (1, ))
    assert_size_stride(primals_421, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_422, (1024, ), (1, ))
    assert_size_stride(primals_423, (1024, ), (1, ))
    assert_size_stride(primals_424, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_425, (256, ), (1, ))
    assert_size_stride(primals_426, (256, ), (1, ))
    assert_size_stride(primals_427, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_428, (512, ), (1, ))
    assert_size_stride(primals_429, (512, ), (1, ))
    assert_size_stride(primals_430, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_431, (128, ), (1, ))
    assert_size_stride(primals_432, (128, ), (1, ))
    assert_size_stride(primals_433, (128, ), (1, ))
    assert_size_stride(primals_434, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_435, (512, ), (1, ))
    assert_size_stride(primals_436, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_437, (1024, ), (1, ))
    assert_size_stride(primals_438, (1024, ), (1, ))
    assert_size_stride(primals_439, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_440, (256, ), (1, ))
    assert_size_stride(primals_441, (256, ), (1, ))
    assert_size_stride(primals_442, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_443, (512, ), (1, ))
    assert_size_stride(primals_444, (512, ), (1, ))
    assert_size_stride(primals_445, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_446, (128, ), (1, ))
    assert_size_stride(primals_447, (128, ), (1, ))
    assert_size_stride(primals_448, (128, ), (1, ))
    assert_size_stride(primals_449, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_450, (512, ), (1, ))
    assert_size_stride(primals_451, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_452, (1024, ), (1, ))
    assert_size_stride(primals_453, (1024, ), (1, ))
    assert_size_stride(primals_454, (256, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_455, (256, ), (1, ))
    assert_size_stride(primals_456, (256, ), (1, ))
    assert_size_stride(primals_457, (512, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_458, (512, ), (1, ))
    assert_size_stride(primals_459, (512, ), (1, ))
    assert_size_stride(primals_460, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_461, (128, ), (1, ))
    assert_size_stride(primals_462, (128, ), (1, ))
    assert_size_stride(primals_463, (128, ), (1, ))
    assert_size_stride(primals_464, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_465, (512, ), (1, ))
    assert_size_stride(primals_466, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_467, (1024, ), (1, ))
    assert_size_stride(primals_468, (1024, ), (1, ))
    assert_size_stride(primals_469, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_470, (512, ), (1, ))
    assert_size_stride(primals_471, (512, ), (1, ))
    assert_size_stride(primals_472, (1024, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_473, (1024, ), (1, ))
    assert_size_stride(primals_474, (1024, ), (1, ))
    assert_size_stride(primals_475, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_476, (256, ), (1, ))
    assert_size_stride(primals_477, (256, ), (1, ))
    assert_size_stride(primals_478, (256, ), (1, ))
    assert_size_stride(primals_479, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_480, (1024, ), (1, ))
    assert_size_stride(primals_481, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_482, (2048, ), (1, ))
    assert_size_stride(primals_483, (2048, ), (1, ))
    assert_size_stride(primals_484, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_485, (2048, ), (1, ))
    assert_size_stride(primals_486, (2048, ), (1, ))
    assert_size_stride(primals_487, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_488, (512, ), (1, ))
    assert_size_stride(primals_489, (512, ), (1, ))
    assert_size_stride(primals_490, (1024, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_491, (1024, ), (1, ))
    assert_size_stride(primals_492, (1024, ), (1, ))
    assert_size_stride(primals_493, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_494, (256, ), (1, ))
    assert_size_stride(primals_495, (256, ), (1, ))
    assert_size_stride(primals_496, (256, ), (1, ))
    assert_size_stride(primals_497, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_498, (1024, ), (1, ))
    assert_size_stride(primals_499, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_500, (2048, ), (1, ))
    assert_size_stride(primals_501, (2048, ), (1, ))
    assert_size_stride(primals_502, (512, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_503, (512, ), (1, ))
    assert_size_stride(primals_504, (512, ), (1, ))
    assert_size_stride(primals_505, (1024, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_506, (1024, ), (1, ))
    assert_size_stride(primals_507, (1024, ), (1, ))
    assert_size_stride(primals_508, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_509, (256, ), (1, ))
    assert_size_stride(primals_510, (256, ), (1, ))
    assert_size_stride(primals_511, (256, ), (1, ))
    assert_size_stride(primals_512, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_513, (1024, ), (1, ))
    assert_size_stride(primals_514, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_515, (2048, ), (1, ))
    assert_size_stride(primals_516, (2048, ), (1, ))
    assert_size_stride(primals_517, (1000, 2048), (2048, 1))
    assert_size_stride(primals_518, (1000, ), (1, ))
    assert_size_stride(primals_519, (64, ), (1, ))
    assert_size_stride(primals_520, (64, ), (1, ))
    assert_size_stride(primals_521, (), ())
    assert_size_stride(primals_522, (64, ), (1, ))
    assert_size_stride(primals_523, (64, ), (1, ))
    assert_size_stride(primals_524, (), ())
    assert_size_stride(primals_525, (128, ), (1, ))
    assert_size_stride(primals_526, (128, ), (1, ))
    assert_size_stride(primals_527, (), ())
    assert_size_stride(primals_528, (64, ), (1, ))
    assert_size_stride(primals_529, (64, ), (1, ))
    assert_size_stride(primals_530, (), ())
    assert_size_stride(primals_531, (128, ), (1, ))
    assert_size_stride(primals_532, (128, ), (1, ))
    assert_size_stride(primals_533, (), ())
    assert_size_stride(primals_534, (32, ), (1, ))
    assert_size_stride(primals_535, (32, ), (1, ))
    assert_size_stride(primals_536, (), ())
    assert_size_stride(primals_537, (256, ), (1, ))
    assert_size_stride(primals_538, (256, ), (1, ))
    assert_size_stride(primals_539, (), ())
    assert_size_stride(primals_540, (256, ), (1, ))
    assert_size_stride(primals_541, (256, ), (1, ))
    assert_size_stride(primals_542, (), ())
    assert_size_stride(primals_543, (64, ), (1, ))
    assert_size_stride(primals_544, (64, ), (1, ))
    assert_size_stride(primals_545, (), ())
    assert_size_stride(primals_546, (128, ), (1, ))
    assert_size_stride(primals_547, (128, ), (1, ))
    assert_size_stride(primals_548, (), ())
    assert_size_stride(primals_549, (32, ), (1, ))
    assert_size_stride(primals_550, (32, ), (1, ))
    assert_size_stride(primals_551, (), ())
    assert_size_stride(primals_552, (256, ), (1, ))
    assert_size_stride(primals_553, (256, ), (1, ))
    assert_size_stride(primals_554, (), ())
    assert_size_stride(primals_555, (64, ), (1, ))
    assert_size_stride(primals_556, (64, ), (1, ))
    assert_size_stride(primals_557, (), ())
    assert_size_stride(primals_558, (128, ), (1, ))
    assert_size_stride(primals_559, (128, ), (1, ))
    assert_size_stride(primals_560, (), ())
    assert_size_stride(primals_561, (32, ), (1, ))
    assert_size_stride(primals_562, (32, ), (1, ))
    assert_size_stride(primals_563, (), ())
    assert_size_stride(primals_564, (256, ), (1, ))
    assert_size_stride(primals_565, (256, ), (1, ))
    assert_size_stride(primals_566, (), ())
    assert_size_stride(primals_567, (128, ), (1, ))
    assert_size_stride(primals_568, (128, ), (1, ))
    assert_size_stride(primals_569, (), ())
    assert_size_stride(primals_570, (256, ), (1, ))
    assert_size_stride(primals_571, (256, ), (1, ))
    assert_size_stride(primals_572, (), ())
    assert_size_stride(primals_573, (64, ), (1, ))
    assert_size_stride(primals_574, (64, ), (1, ))
    assert_size_stride(primals_575, (), ())
    assert_size_stride(primals_576, (512, ), (1, ))
    assert_size_stride(primals_577, (512, ), (1, ))
    assert_size_stride(primals_578, (), ())
    assert_size_stride(primals_579, (512, ), (1, ))
    assert_size_stride(primals_580, (512, ), (1, ))
    assert_size_stride(primals_581, (), ())
    assert_size_stride(primals_582, (128, ), (1, ))
    assert_size_stride(primals_583, (128, ), (1, ))
    assert_size_stride(primals_584, (), ())
    assert_size_stride(primals_585, (256, ), (1, ))
    assert_size_stride(primals_586, (256, ), (1, ))
    assert_size_stride(primals_587, (), ())
    assert_size_stride(primals_588, (64, ), (1, ))
    assert_size_stride(primals_589, (64, ), (1, ))
    assert_size_stride(primals_590, (), ())
    assert_size_stride(primals_591, (512, ), (1, ))
    assert_size_stride(primals_592, (512, ), (1, ))
    assert_size_stride(primals_593, (), ())
    assert_size_stride(primals_594, (128, ), (1, ))
    assert_size_stride(primals_595, (128, ), (1, ))
    assert_size_stride(primals_596, (), ())
    assert_size_stride(primals_597, (256, ), (1, ))
    assert_size_stride(primals_598, (256, ), (1, ))
    assert_size_stride(primals_599, (), ())
    assert_size_stride(primals_600, (64, ), (1, ))
    assert_size_stride(primals_601, (64, ), (1, ))
    assert_size_stride(primals_602, (), ())
    assert_size_stride(primals_603, (512, ), (1, ))
    assert_size_stride(primals_604, (512, ), (1, ))
    assert_size_stride(primals_605, (), ())
    assert_size_stride(primals_606, (128, ), (1, ))
    assert_size_stride(primals_607, (128, ), (1, ))
    assert_size_stride(primals_608, (), ())
    assert_size_stride(primals_609, (256, ), (1, ))
    assert_size_stride(primals_610, (256, ), (1, ))
    assert_size_stride(primals_611, (), ())
    assert_size_stride(primals_612, (64, ), (1, ))
    assert_size_stride(primals_613, (64, ), (1, ))
    assert_size_stride(primals_614, (), ())
    assert_size_stride(primals_615, (512, ), (1, ))
    assert_size_stride(primals_616, (512, ), (1, ))
    assert_size_stride(primals_617, (), ())
    assert_size_stride(primals_618, (256, ), (1, ))
    assert_size_stride(primals_619, (256, ), (1, ))
    assert_size_stride(primals_620, (), ())
    assert_size_stride(primals_621, (512, ), (1, ))
    assert_size_stride(primals_622, (512, ), (1, ))
    assert_size_stride(primals_623, (), ())
    assert_size_stride(primals_624, (128, ), (1, ))
    assert_size_stride(primals_625, (128, ), (1, ))
    assert_size_stride(primals_626, (), ())
    assert_size_stride(primals_627, (1024, ), (1, ))
    assert_size_stride(primals_628, (1024, ), (1, ))
    assert_size_stride(primals_629, (), ())
    assert_size_stride(primals_630, (1024, ), (1, ))
    assert_size_stride(primals_631, (1024, ), (1, ))
    assert_size_stride(primals_632, (), ())
    assert_size_stride(primals_633, (256, ), (1, ))
    assert_size_stride(primals_634, (256, ), (1, ))
    assert_size_stride(primals_635, (), ())
    assert_size_stride(primals_636, (512, ), (1, ))
    assert_size_stride(primals_637, (512, ), (1, ))
    assert_size_stride(primals_638, (), ())
    assert_size_stride(primals_639, (128, ), (1, ))
    assert_size_stride(primals_640, (128, ), (1, ))
    assert_size_stride(primals_641, (), ())
    assert_size_stride(primals_642, (1024, ), (1, ))
    assert_size_stride(primals_643, (1024, ), (1, ))
    assert_size_stride(primals_644, (), ())
    assert_size_stride(primals_645, (256, ), (1, ))
    assert_size_stride(primals_646, (256, ), (1, ))
    assert_size_stride(primals_647, (), ())
    assert_size_stride(primals_648, (512, ), (1, ))
    assert_size_stride(primals_649, (512, ), (1, ))
    assert_size_stride(primals_650, (), ())
    assert_size_stride(primals_651, (128, ), (1, ))
    assert_size_stride(primals_652, (128, ), (1, ))
    assert_size_stride(primals_653, (), ())
    assert_size_stride(primals_654, (1024, ), (1, ))
    assert_size_stride(primals_655, (1024, ), (1, ))
    assert_size_stride(primals_656, (), ())
    assert_size_stride(primals_657, (256, ), (1, ))
    assert_size_stride(primals_658, (256, ), (1, ))
    assert_size_stride(primals_659, (), ())
    assert_size_stride(primals_660, (512, ), (1, ))
    assert_size_stride(primals_661, (512, ), (1, ))
    assert_size_stride(primals_662, (), ())
    assert_size_stride(primals_663, (128, ), (1, ))
    assert_size_stride(primals_664, (128, ), (1, ))
    assert_size_stride(primals_665, (), ())
    assert_size_stride(primals_666, (1024, ), (1, ))
    assert_size_stride(primals_667, (1024, ), (1, ))
    assert_size_stride(primals_668, (), ())
    assert_size_stride(primals_669, (256, ), (1, ))
    assert_size_stride(primals_670, (256, ), (1, ))
    assert_size_stride(primals_671, (), ())
    assert_size_stride(primals_672, (512, ), (1, ))
    assert_size_stride(primals_673, (512, ), (1, ))
    assert_size_stride(primals_674, (), ())
    assert_size_stride(primals_675, (128, ), (1, ))
    assert_size_stride(primals_676, (128, ), (1, ))
    assert_size_stride(primals_677, (), ())
    assert_size_stride(primals_678, (1024, ), (1, ))
    assert_size_stride(primals_679, (1024, ), (1, ))
    assert_size_stride(primals_680, (), ())
    assert_size_stride(primals_681, (256, ), (1, ))
    assert_size_stride(primals_682, (256, ), (1, ))
    assert_size_stride(primals_683, (), ())
    assert_size_stride(primals_684, (512, ), (1, ))
    assert_size_stride(primals_685, (512, ), (1, ))
    assert_size_stride(primals_686, (), ())
    assert_size_stride(primals_687, (128, ), (1, ))
    assert_size_stride(primals_688, (128, ), (1, ))
    assert_size_stride(primals_689, (), ())
    assert_size_stride(primals_690, (1024, ), (1, ))
    assert_size_stride(primals_691, (1024, ), (1, ))
    assert_size_stride(primals_692, (), ())
    assert_size_stride(primals_693, (256, ), (1, ))
    assert_size_stride(primals_694, (256, ), (1, ))
    assert_size_stride(primals_695, (), ())
    assert_size_stride(primals_696, (512, ), (1, ))
    assert_size_stride(primals_697, (512, ), (1, ))
    assert_size_stride(primals_698, (), ())
    assert_size_stride(primals_699, (128, ), (1, ))
    assert_size_stride(primals_700, (128, ), (1, ))
    assert_size_stride(primals_701, (), ())
    assert_size_stride(primals_702, (1024, ), (1, ))
    assert_size_stride(primals_703, (1024, ), (1, ))
    assert_size_stride(primals_704, (), ())
    assert_size_stride(primals_705, (256, ), (1, ))
    assert_size_stride(primals_706, (256, ), (1, ))
    assert_size_stride(primals_707, (), ())
    assert_size_stride(primals_708, (512, ), (1, ))
    assert_size_stride(primals_709, (512, ), (1, ))
    assert_size_stride(primals_710, (), ())
    assert_size_stride(primals_711, (128, ), (1, ))
    assert_size_stride(primals_712, (128, ), (1, ))
    assert_size_stride(primals_713, (), ())
    assert_size_stride(primals_714, (1024, ), (1, ))
    assert_size_stride(primals_715, (1024, ), (1, ))
    assert_size_stride(primals_716, (), ())
    assert_size_stride(primals_717, (256, ), (1, ))
    assert_size_stride(primals_718, (256, ), (1, ))
    assert_size_stride(primals_719, (), ())
    assert_size_stride(primals_720, (512, ), (1, ))
    assert_size_stride(primals_721, (512, ), (1, ))
    assert_size_stride(primals_722, (), ())
    assert_size_stride(primals_723, (128, ), (1, ))
    assert_size_stride(primals_724, (128, ), (1, ))
    assert_size_stride(primals_725, (), ())
    assert_size_stride(primals_726, (1024, ), (1, ))
    assert_size_stride(primals_727, (1024, ), (1, ))
    assert_size_stride(primals_728, (), ())
    assert_size_stride(primals_729, (256, ), (1, ))
    assert_size_stride(primals_730, (256, ), (1, ))
    assert_size_stride(primals_731, (), ())
    assert_size_stride(primals_732, (512, ), (1, ))
    assert_size_stride(primals_733, (512, ), (1, ))
    assert_size_stride(primals_734, (), ())
    assert_size_stride(primals_735, (128, ), (1, ))
    assert_size_stride(primals_736, (128, ), (1, ))
    assert_size_stride(primals_737, (), ())
    assert_size_stride(primals_738, (1024, ), (1, ))
    assert_size_stride(primals_739, (1024, ), (1, ))
    assert_size_stride(primals_740, (), ())
    assert_size_stride(primals_741, (256, ), (1, ))
    assert_size_stride(primals_742, (256, ), (1, ))
    assert_size_stride(primals_743, (), ())
    assert_size_stride(primals_744, (512, ), (1, ))
    assert_size_stride(primals_745, (512, ), (1, ))
    assert_size_stride(primals_746, (), ())
    assert_size_stride(primals_747, (128, ), (1, ))
    assert_size_stride(primals_748, (128, ), (1, ))
    assert_size_stride(primals_749, (), ())
    assert_size_stride(primals_750, (1024, ), (1, ))
    assert_size_stride(primals_751, (1024, ), (1, ))
    assert_size_stride(primals_752, (), ())
    assert_size_stride(primals_753, (256, ), (1, ))
    assert_size_stride(primals_754, (256, ), (1, ))
    assert_size_stride(primals_755, (), ())
    assert_size_stride(primals_756, (512, ), (1, ))
    assert_size_stride(primals_757, (512, ), (1, ))
    assert_size_stride(primals_758, (), ())
    assert_size_stride(primals_759, (128, ), (1, ))
    assert_size_stride(primals_760, (128, ), (1, ))
    assert_size_stride(primals_761, (), ())
    assert_size_stride(primals_762, (1024, ), (1, ))
    assert_size_stride(primals_763, (1024, ), (1, ))
    assert_size_stride(primals_764, (), ())
    assert_size_stride(primals_765, (256, ), (1, ))
    assert_size_stride(primals_766, (256, ), (1, ))
    assert_size_stride(primals_767, (), ())
    assert_size_stride(primals_768, (512, ), (1, ))
    assert_size_stride(primals_769, (512, ), (1, ))
    assert_size_stride(primals_770, (), ())
    assert_size_stride(primals_771, (128, ), (1, ))
    assert_size_stride(primals_772, (128, ), (1, ))
    assert_size_stride(primals_773, (), ())
    assert_size_stride(primals_774, (1024, ), (1, ))
    assert_size_stride(primals_775, (1024, ), (1, ))
    assert_size_stride(primals_776, (), ())
    assert_size_stride(primals_777, (256, ), (1, ))
    assert_size_stride(primals_778, (256, ), (1, ))
    assert_size_stride(primals_779, (), ())
    assert_size_stride(primals_780, (512, ), (1, ))
    assert_size_stride(primals_781, (512, ), (1, ))
    assert_size_stride(primals_782, (), ())
    assert_size_stride(primals_783, (128, ), (1, ))
    assert_size_stride(primals_784, (128, ), (1, ))
    assert_size_stride(primals_785, (), ())
    assert_size_stride(primals_786, (1024, ), (1, ))
    assert_size_stride(primals_787, (1024, ), (1, ))
    assert_size_stride(primals_788, (), ())
    assert_size_stride(primals_789, (256, ), (1, ))
    assert_size_stride(primals_790, (256, ), (1, ))
    assert_size_stride(primals_791, (), ())
    assert_size_stride(primals_792, (512, ), (1, ))
    assert_size_stride(primals_793, (512, ), (1, ))
    assert_size_stride(primals_794, (), ())
    assert_size_stride(primals_795, (128, ), (1, ))
    assert_size_stride(primals_796, (128, ), (1, ))
    assert_size_stride(primals_797, (), ())
    assert_size_stride(primals_798, (1024, ), (1, ))
    assert_size_stride(primals_799, (1024, ), (1, ))
    assert_size_stride(primals_800, (), ())
    assert_size_stride(primals_801, (256, ), (1, ))
    assert_size_stride(primals_802, (256, ), (1, ))
    assert_size_stride(primals_803, (), ())
    assert_size_stride(primals_804, (512, ), (1, ))
    assert_size_stride(primals_805, (512, ), (1, ))
    assert_size_stride(primals_806, (), ())
    assert_size_stride(primals_807, (128, ), (1, ))
    assert_size_stride(primals_808, (128, ), (1, ))
    assert_size_stride(primals_809, (), ())
    assert_size_stride(primals_810, (1024, ), (1, ))
    assert_size_stride(primals_811, (1024, ), (1, ))
    assert_size_stride(primals_812, (), ())
    assert_size_stride(primals_813, (256, ), (1, ))
    assert_size_stride(primals_814, (256, ), (1, ))
    assert_size_stride(primals_815, (), ())
    assert_size_stride(primals_816, (512, ), (1, ))
    assert_size_stride(primals_817, (512, ), (1, ))
    assert_size_stride(primals_818, (), ())
    assert_size_stride(primals_819, (128, ), (1, ))
    assert_size_stride(primals_820, (128, ), (1, ))
    assert_size_stride(primals_821, (), ())
    assert_size_stride(primals_822, (1024, ), (1, ))
    assert_size_stride(primals_823, (1024, ), (1, ))
    assert_size_stride(primals_824, (), ())
    assert_size_stride(primals_825, (256, ), (1, ))
    assert_size_stride(primals_826, (256, ), (1, ))
    assert_size_stride(primals_827, (), ())
    assert_size_stride(primals_828, (512, ), (1, ))
    assert_size_stride(primals_829, (512, ), (1, ))
    assert_size_stride(primals_830, (), ())
    assert_size_stride(primals_831, (128, ), (1, ))
    assert_size_stride(primals_832, (128, ), (1, ))
    assert_size_stride(primals_833, (), ())
    assert_size_stride(primals_834, (1024, ), (1, ))
    assert_size_stride(primals_835, (1024, ), (1, ))
    assert_size_stride(primals_836, (), ())
    assert_size_stride(primals_837, (256, ), (1, ))
    assert_size_stride(primals_838, (256, ), (1, ))
    assert_size_stride(primals_839, (), ())
    assert_size_stride(primals_840, (512, ), (1, ))
    assert_size_stride(primals_841, (512, ), (1, ))
    assert_size_stride(primals_842, (), ())
    assert_size_stride(primals_843, (128, ), (1, ))
    assert_size_stride(primals_844, (128, ), (1, ))
    assert_size_stride(primals_845, (), ())
    assert_size_stride(primals_846, (1024, ), (1, ))
    assert_size_stride(primals_847, (1024, ), (1, ))
    assert_size_stride(primals_848, (), ())
    assert_size_stride(primals_849, (256, ), (1, ))
    assert_size_stride(primals_850, (256, ), (1, ))
    assert_size_stride(primals_851, (), ())
    assert_size_stride(primals_852, (512, ), (1, ))
    assert_size_stride(primals_853, (512, ), (1, ))
    assert_size_stride(primals_854, (), ())
    assert_size_stride(primals_855, (128, ), (1, ))
    assert_size_stride(primals_856, (128, ), (1, ))
    assert_size_stride(primals_857, (), ())
    assert_size_stride(primals_858, (1024, ), (1, ))
    assert_size_stride(primals_859, (1024, ), (1, ))
    assert_size_stride(primals_860, (), ())
    assert_size_stride(primals_861, (256, ), (1, ))
    assert_size_stride(primals_862, (256, ), (1, ))
    assert_size_stride(primals_863, (), ())
    assert_size_stride(primals_864, (512, ), (1, ))
    assert_size_stride(primals_865, (512, ), (1, ))
    assert_size_stride(primals_866, (), ())
    assert_size_stride(primals_867, (128, ), (1, ))
    assert_size_stride(primals_868, (128, ), (1, ))
    assert_size_stride(primals_869, (), ())
    assert_size_stride(primals_870, (1024, ), (1, ))
    assert_size_stride(primals_871, (1024, ), (1, ))
    assert_size_stride(primals_872, (), ())
    assert_size_stride(primals_873, (256, ), (1, ))
    assert_size_stride(primals_874, (256, ), (1, ))
    assert_size_stride(primals_875, (), ())
    assert_size_stride(primals_876, (512, ), (1, ))
    assert_size_stride(primals_877, (512, ), (1, ))
    assert_size_stride(primals_878, (), ())
    assert_size_stride(primals_879, (128, ), (1, ))
    assert_size_stride(primals_880, (128, ), (1, ))
    assert_size_stride(primals_881, (), ())
    assert_size_stride(primals_882, (1024, ), (1, ))
    assert_size_stride(primals_883, (1024, ), (1, ))
    assert_size_stride(primals_884, (), ())
    assert_size_stride(primals_885, (256, ), (1, ))
    assert_size_stride(primals_886, (256, ), (1, ))
    assert_size_stride(primals_887, (), ())
    assert_size_stride(primals_888, (512, ), (1, ))
    assert_size_stride(primals_889, (512, ), (1, ))
    assert_size_stride(primals_890, (), ())
    assert_size_stride(primals_891, (128, ), (1, ))
    assert_size_stride(primals_892, (128, ), (1, ))
    assert_size_stride(primals_893, (), ())
    assert_size_stride(primals_894, (1024, ), (1, ))
    assert_size_stride(primals_895, (1024, ), (1, ))
    assert_size_stride(primals_896, (), ())
    assert_size_stride(primals_897, (512, ), (1, ))
    assert_size_stride(primals_898, (512, ), (1, ))
    assert_size_stride(primals_899, (), ())
    assert_size_stride(primals_900, (1024, ), (1, ))
    assert_size_stride(primals_901, (1024, ), (1, ))
    assert_size_stride(primals_902, (), ())
    assert_size_stride(primals_903, (256, ), (1, ))
    assert_size_stride(primals_904, (256, ), (1, ))
    assert_size_stride(primals_905, (), ())
    assert_size_stride(primals_906, (2048, ), (1, ))
    assert_size_stride(primals_907, (2048, ), (1, ))
    assert_size_stride(primals_908, (), ())
    assert_size_stride(primals_909, (2048, ), (1, ))
    assert_size_stride(primals_910, (2048, ), (1, ))
    assert_size_stride(primals_911, (), ())
    assert_size_stride(primals_912, (512, ), (1, ))
    assert_size_stride(primals_913, (512, ), (1, ))
    assert_size_stride(primals_914, (), ())
    assert_size_stride(primals_915, (1024, ), (1, ))
    assert_size_stride(primals_916, (1024, ), (1, ))
    assert_size_stride(primals_917, (), ())
    assert_size_stride(primals_918, (256, ), (1, ))
    assert_size_stride(primals_919, (256, ), (1, ))
    assert_size_stride(primals_920, (), ())
    assert_size_stride(primals_921, (2048, ), (1, ))
    assert_size_stride(primals_922, (2048, ), (1, ))
    assert_size_stride(primals_923, (), ())
    assert_size_stride(primals_924, (512, ), (1, ))
    assert_size_stride(primals_925, (512, ), (1, ))
    assert_size_stride(primals_926, (), ())
    assert_size_stride(primals_927, (1024, ), (1, ))
    assert_size_stride(primals_928, (1024, ), (1, ))
    assert_size_stride(primals_929, (), ())
    assert_size_stride(primals_930, (256, ), (1, ))
    assert_size_stride(primals_931, (256, ), (1, ))
    assert_size_stride(primals_932, (), ())
    assert_size_stride(primals_933, (2048, ), (1, ))
    assert_size_stride(primals_934, (2048, ), (1, ))
    assert_size_stride(primals_935, (), ())
    assert_size_stride(primals_936, (8, 3, 256, 256), (196608, 65536, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [l__mod___conv1_0], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_936, primals_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 64, 128, 128), (1048576, 16384, 128, 1))
        buf1 = empty_strided((1, 64, 1, 1, 16), (1024, 16, 1024, 1024, 1), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((1, 64, 1, 1, 16), (1024, 16, 1024, 1024, 1), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((1, 64, 1, 1, 16), (1024, 16, 1024, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___conv1_1], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_cuda_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_0.run(buf0, buf1, buf2, buf3, 1024, 8192, grid=grid(1024), stream=stream0)
        buf4 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf7 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___conv1_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_1.run(buf1, buf2, buf3, primals_519, primals_520, buf4, buf5, buf7, primals_519, primals_520, 64, 16, grid=grid(64), stream=stream0)
        del primals_519
        del primals_520
        buf8 = empty((8, 64, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___conv1_1, l__mod___conv1_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_2.run(buf0, buf4, buf5, primals_2, primals_3, buf8, 8388608, grid=grid(8388608), stream=stream0)
        del primals_3
        # Source Nodes: [l__mod___conv1_3], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf8, primals_4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 64, 128, 128), (1048576, 16384, 128, 1))
        buf10 = buf3; del buf3  # reuse
        buf11 = buf2; del buf2  # reuse
        buf12 = buf1; del buf1  # reuse
        # Source Nodes: [l__mod___conv1_4], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_0.run(buf9, buf10, buf11, buf12, 1024, 8192, grid=grid(1024), stream=stream0)
        buf13 = buf5; del buf5  # reuse
        buf14 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf16 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___conv1_4], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_1.run(buf10, buf11, buf12, primals_522, primals_523, buf13, buf14, buf16, primals_522, primals_523, 64, 16, grid=grid(64), stream=stream0)
        del primals_522
        del primals_523
        buf17 = empty((8, 64, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___conv1_4, l__mod___conv1_5], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_2.run(buf9, buf13, buf14, primals_5, primals_6, buf17, 8388608, grid=grid(8388608), stream=stream0)
        del primals_6
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (8, 128, 128, 128), (2097152, 16384, 128, 1))
        buf19 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        buf20 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        buf21 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf18, buf19, buf20, buf21, 512, 32768, grid=grid(512), stream=stream0)
        buf22 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf23 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf25 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_4.run(buf19, buf20, buf21, primals_525, primals_526, buf22, buf23, buf25, primals_525, primals_526, 128, 4, grid=grid(128), stream=stream0)
        del primals_525
        del primals_526
        buf26 = empty((8, 128, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf18, buf22, buf23, primals_8, primals_9, buf26, 16777216, grid=grid(16777216), stream=stream0)
        del primals_9
        buf27 = empty((8, 128, 64, 64), device='cuda', dtype=torch.float32)
        buf28 = empty((8, 128, 64, 64), device='cuda', dtype=torch.int64)
        # Source Nodes: [shortcut], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_6.run(buf26, buf27, buf28, 4194304, grid=grid(4194304), stream=stream0)
        # Source Nodes: [out], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf27, primals_10, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf30 = empty_strided((1, 64, 1, 1, 4), (256, 1, 256, 256, 64), device='cuda', dtype=torch.float32)
        buf31 = empty_strided((1, 64, 1, 1, 4), (256, 1, 256, 256, 64), device='cuda', dtype=torch.float32)
        buf32 = empty_strided((1, 64, 1, 1, 4), (256, 1, 256, 256, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf29, buf30, buf31, buf32, 256, 8192, grid=grid(256), stream=stream0)
        buf33 = buf14; del buf14  # reuse
        buf34 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf36 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_8.run(buf30, buf31, buf32, primals_528, primals_529, buf33, buf34, buf36, primals_528, primals_529, 64, 4, grid=grid(64), stream=stream0)
        del primals_528
        del primals_529
        buf37 = empty((8, 64, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_9.run(buf29, buf33, buf34, primals_11, primals_12, buf37, 2097152, grid=grid(2097152), stream=stream0)
        del primals_12
        # Source Nodes: [x_4], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf38, (8, 128, 64, 64), (524288, 4096, 64, 1))
        buf39 = buf21; del buf21  # reuse
        buf40 = buf20; del buf20  # reuse
        buf41 = buf19; del buf19  # reuse
        # Source Nodes: [x_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_10.run(buf38, buf39, buf40, buf41, 512, 8192, grid=grid(512), stream=stream0)
        buf42 = buf23; del buf23  # reuse
        buf43 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf45 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_11.run(buf39, buf40, buf41, primals_531, primals_532, buf42, buf43, buf45, primals_531, primals_532, 128, 4, grid=grid(128), stream=stream0)
        del primals_531
        del primals_532
        buf46 = empty((8, 128, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_5, x_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_12.run(buf38, buf42, buf43, primals_14, primals_15, buf46, 4194304, grid=grid(4194304), stream=stream0)
        del primals_15
        buf47 = reinterpret_tensor(buf41, (8, 64, 1, 1), (64, 1, 512, 512), 0); del buf41  # reuse
        buf48 = reinterpret_tensor(buf47, (8, 64, 1, 1), (64, 1, 1, 1), 0); del buf47  # reuse
        # Source Nodes: [x_gap, x_gap_1], Original ATen: [aten.mean, aten.sum]
        triton_red_fused_mean_sum_13.run(buf48, buf46, 512, 4096, grid=grid(512), stream=stream0)
        # Source Nodes: [x_gap_2], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (8, 32, 1, 1), (32, 1, 1, 1))
        buf50 = buf49; del buf49  # reuse
        # Source Nodes: [x_gap_2], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf50, primals_17, 256, grid=grid(256), stream=stream0)
        del primals_17
        buf51 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf52 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_3], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf50, primals_534, primals_535, buf51, buf52, primals_534, primals_535, 32, 8, grid=grid(32), stream=stream0)
        del primals_534
        del primals_535
        buf54 = reinterpret_tensor(buf32, (8, 32, 1, 1), (32, 1, 1, 1), 0); del buf32  # reuse
        # Source Nodes: [x_gap_3, x_gap_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_16.run(buf50, buf51, buf52, primals_18, primals_19, buf54, 256, grid=grid(256), stream=stream0)
        del primals_19
        # Source Nodes: [x_attn], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, primals_20, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (8, 128, 1, 1), (128, 1, 1, 1))
        buf56 = reinterpret_tensor(buf12, (8, 2, 1, 64), (128, 64, 1024, 1), 0); del buf12  # reuse
        # Source Nodes: [x_10], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_17.run(buf55, primals_21, buf56, 1024, grid=grid(1024), stream=stream0)
        del primals_21
        buf57 = reinterpret_tensor(buf55, (8, 2, 1, 64), (128, 64, 64, 1), 0); del buf55  # reuse
        # Source Nodes: [x_10], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_18.run(buf56, buf57, 1024, grid=grid(1024), stream=stream0)
        buf58 = empty((8, 64, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul, out_3], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_19.run(buf46, buf57, buf58, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [out_8], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        buf60 = reinterpret_tensor(buf31, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf31  # reuse
        buf61 = reinterpret_tensor(buf30, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf30  # reuse
        buf63 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_9], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf59, primals_537, primals_538, buf60, buf61, buf63, primals_537, primals_538, 256, 32768, grid=grid(256), stream=stream0)
        del primals_537
        del primals_538
        # Source Nodes: [getattr_l__mod___layer1___0___downsample_1], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf27, primals_25, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        buf65 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf66 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf68 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf64, primals_540, primals_541, buf65, buf66, buf68, primals_540, primals_541, 256, 32768, grid=grid(256), stream=stream0)
        del primals_540
        del primals_541
        buf69 = empty((8, 256, 64, 64), device='cuda', dtype=torch.float32)
        buf70 = buf69; del buf69  # reuse
        # Source Nodes: [out_10, out_9, shortcut_1, shortcut_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_21.run(buf70, buf59, buf60, buf61, primals_23, primals_24, buf64, buf65, buf66, primals_26, primals_27, 8388608, grid=grid(8388608), stream=stream0)
        del primals_24
        del primals_27
        # Source Nodes: [out_12], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_28, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf72 = reinterpret_tensor(buf66, (1, 64, 1, 1, 4), (256, 1, 256, 256, 64), 0); del buf66  # reuse
        buf73 = reinterpret_tensor(buf61, (1, 64, 1, 1, 4), (256, 1, 256, 256, 64), 0); del buf61  # reuse
        buf74 = empty_strided((1, 64, 1, 1, 4), (256, 1, 256, 256, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf71, buf72, buf73, buf74, 256, 8192, grid=grid(256), stream=stream0)
        buf75 = buf34; del buf34  # reuse
        buf76 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf78 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_8.run(buf72, buf73, buf74, primals_543, primals_544, buf75, buf76, buf78, primals_543, primals_544, 64, 4, grid=grid(64), stream=stream0)
        del primals_543
        del primals_544
        buf79 = empty((8, 64, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_13, out_14], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_9.run(buf71, buf75, buf76, primals_29, primals_30, buf79, 2097152, grid=grid(2097152), stream=stream0)
        del primals_30
        # Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf80, (8, 128, 64, 64), (524288, 4096, 64, 1))
        buf81 = buf40; del buf40  # reuse
        buf82 = buf39; del buf39  # reuse
        buf83 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_10.run(buf80, buf81, buf82, buf83, 512, 8192, grid=grid(512), stream=stream0)
        buf84 = buf43; del buf43  # reuse
        buf85 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf87 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_11.run(buf81, buf82, buf83, primals_546, primals_547, buf84, buf85, buf87, primals_546, primals_547, 128, 4, grid=grid(128), stream=stream0)
        del primals_546
        del primals_547
        buf88 = empty((8, 128, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13, x_15], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_12.run(buf80, buf84, buf85, primals_32, primals_33, buf88, 4194304, grid=grid(4194304), stream=stream0)
        del primals_33
        buf89 = reinterpret_tensor(buf83, (8, 64, 1, 1), (64, 1, 512, 512), 0); del buf83  # reuse
        buf90 = reinterpret_tensor(buf89, (8, 64, 1, 1), (64, 1, 1, 1), 0); del buf89  # reuse
        # Source Nodes: [x_gap_5, x_gap_6], Original ATen: [aten.mean, aten.sum]
        triton_red_fused_mean_sum_13.run(buf90, buf88, 512, 4096, grid=grid(512), stream=stream0)
        # Source Nodes: [x_gap_7], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (8, 32, 1, 1), (32, 1, 1, 1))
        buf92 = buf91; del buf91  # reuse
        # Source Nodes: [x_gap_7], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf92, primals_35, 256, grid=grid(256), stream=stream0)
        del primals_35
        buf93 = buf52; del buf52  # reuse
        buf94 = buf51; del buf51  # reuse
        # Source Nodes: [x_gap_8], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf92, primals_549, primals_550, buf93, buf94, primals_549, primals_550, 32, 8, grid=grid(32), stream=stream0)
        del primals_549
        del primals_550
        buf96 = reinterpret_tensor(buf74, (8, 32, 1, 1), (32, 1, 1, 1), 0); del buf74  # reuse
        # Source Nodes: [x_gap_8, x_gap_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_16.run(buf92, buf93, buf94, primals_36, primals_37, buf96, 256, grid=grid(256), stream=stream0)
        del primals_37
        # Source Nodes: [x_attn_2], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, primals_38, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (8, 128, 1, 1), (128, 1, 1, 1))
        buf98 = buf56; del buf56  # reuse
        # Source Nodes: [x_18], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_17.run(buf97, primals_39, buf98, 1024, grid=grid(1024), stream=stream0)
        del primals_39
        buf99 = reinterpret_tensor(buf97, (8, 2, 1, 64), (128, 64, 64, 1), 0); del buf97  # reuse
        # Source Nodes: [x_18], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_18.run(buf98, buf99, 1024, grid=grid(1024), stream=stream0)
        buf100 = empty((8, 64, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_1, out_15], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_19.run(buf88, buf99, buf100, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [out_20], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_40, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        buf102 = reinterpret_tensor(buf73, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf73  # reuse
        buf103 = reinterpret_tensor(buf72, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf72  # reuse
        buf105 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf101, primals_552, primals_553, buf102, buf103, buf105, primals_552, primals_553, 256, 32768, grid=grid(256), stream=stream0)
        del primals_552
        del primals_553
        buf106 = empty((8, 256, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_21, out_22, shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf101, buf102, buf103, primals_41, primals_42, buf70, buf106, 8388608, grid=grid(8388608), stream=stream0)
        del primals_42
        # Source Nodes: [out_24], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_43, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf108 = reinterpret_tensor(buf103, (1, 64, 1, 1, 4), (256, 1, 256, 256, 64), 0); del buf103  # reuse
        buf109 = empty_strided((1, 64, 1, 1, 4), (256, 1, 256, 256, 64), device='cuda', dtype=torch.float32)
        buf110 = empty_strided((1, 64, 1, 1, 4), (256, 1, 256, 256, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_25], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf107, buf108, buf109, buf110, 256, 8192, grid=grid(256), stream=stream0)
        buf111 = buf76; del buf76  # reuse
        buf112 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf114 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_25], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_8.run(buf108, buf109, buf110, primals_555, primals_556, buf111, buf112, buf114, primals_555, primals_556, 64, 4, grid=grid(64), stream=stream0)
        del primals_555
        del primals_556
        buf115 = empty((8, 64, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_25, out_26], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_9.run(buf107, buf111, buf112, primals_44, primals_45, buf115, 2097152, grid=grid(2097152), stream=stream0)
        del primals_45
        # Source Nodes: [x_20], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(buf115, primals_46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf116, (8, 128, 64, 64), (524288, 4096, 64, 1))
        buf117 = buf82; del buf82  # reuse
        buf118 = buf81; del buf81  # reuse
        buf119 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_10.run(buf116, buf117, buf118, buf119, 512, 8192, grid=grid(512), stream=stream0)
        buf120 = buf85; del buf85  # reuse
        buf121 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf123 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_11.run(buf117, buf118, buf119, primals_558, primals_559, buf120, buf121, buf123, primals_558, primals_559, 128, 4, grid=grid(128), stream=stream0)
        del primals_558
        del primals_559
        buf124 = empty((8, 128, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_21, x_23], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_12.run(buf116, buf120, buf121, primals_47, primals_48, buf124, 4194304, grid=grid(4194304), stream=stream0)
        del primals_48
        buf125 = reinterpret_tensor(buf119, (8, 64, 1, 1), (64, 1, 512, 512), 0); del buf119  # reuse
        buf126 = reinterpret_tensor(buf125, (8, 64, 1, 1), (64, 1, 1, 1), 0); del buf125  # reuse
        # Source Nodes: [x_gap_10, x_gap_11], Original ATen: [aten.mean, aten.sum]
        triton_red_fused_mean_sum_13.run(buf126, buf124, 512, 4096, grid=grid(512), stream=stream0)
        # Source Nodes: [x_gap_12], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf126, primals_49, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (8, 32, 1, 1), (32, 1, 1, 1))
        buf128 = buf127; del buf127  # reuse
        # Source Nodes: [x_gap_12], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_14.run(buf128, primals_50, 256, grid=grid(256), stream=stream0)
        del primals_50
        buf129 = buf94; del buf94  # reuse
        buf130 = buf93; del buf93  # reuse
        # Source Nodes: [x_gap_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_15.run(buf128, primals_561, primals_562, buf129, buf130, primals_561, primals_562, 32, 8, grid=grid(32), stream=stream0)
        del primals_561
        del primals_562
        buf132 = reinterpret_tensor(buf110, (8, 32, 1, 1), (32, 1, 1, 1), 0); del buf110  # reuse
        # Source Nodes: [x_gap_13, x_gap_14], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_16.run(buf128, buf129, buf130, primals_51, primals_52, buf132, 256, grid=grid(256), stream=stream0)
        del buf129
        del buf130
        del primals_52
        # Source Nodes: [x_attn_4], Original ATen: [aten.convolution]
        buf133 = extern_kernels.convolution(buf132, primals_53, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (8, 128, 1, 1), (128, 1, 1, 1))
        buf134 = buf98; del buf98  # reuse
        # Source Nodes: [x_26], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_17.run(buf133, primals_54, buf134, 1024, grid=grid(1024), stream=stream0)
        del primals_54
        buf135 = reinterpret_tensor(buf133, (8, 2, 1, 64), (128, 64, 64, 1), 0); del buf133  # reuse
        # Source Nodes: [x_26], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_18.run(buf134, buf135, 1024, grid=grid(1024), stream=stream0)
        buf136 = empty((8, 64, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_2, out_27], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_19.run(buf124, buf135, buf136, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [out_32], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf136, primals_55, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        buf138 = reinterpret_tensor(buf109, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf109  # reuse
        buf139 = reinterpret_tensor(buf108, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf108  # reuse
        buf141 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_33], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf137, primals_564, primals_565, buf138, buf139, buf141, primals_564, primals_565, 256, 32768, grid=grid(256), stream=stream0)
        del primals_564
        del primals_565
        buf142 = empty((8, 256, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_33, out_34, shortcut_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf137, buf138, buf139, primals_56, primals_57, buf106, buf142, 8388608, grid=grid(8388608), stream=stream0)
        del primals_57
        # Source Nodes: [out_36], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, primals_58, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (8, 128, 64, 64), (524288, 4096, 64, 1))
        buf144 = buf118; del buf118  # reuse
        buf145 = buf117; del buf117  # reuse
        buf146 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_10.run(buf143, buf144, buf145, buf146, 512, 8192, grid=grid(512), stream=stream0)
        buf147 = buf121; del buf121  # reuse
        buf148 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf150 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_11.run(buf144, buf145, buf146, primals_567, primals_568, buf147, buf148, buf150, primals_567, primals_568, 128, 4, grid=grid(128), stream=stream0)
        del primals_567
        del primals_568
        buf151 = empty((8, 128, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_37, out_38], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_12.run(buf143, buf147, buf148, primals_59, primals_60, buf151, 4194304, grid=grid(4194304), stream=stream0)
        del primals_60
        # Source Nodes: [x_29], Original ATen: [aten.convolution]
        buf152 = extern_kernels.convolution(buf151, primals_61, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf152, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        buf153 = buf139; del buf139  # reuse
        buf154 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf156 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_30], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf152, primals_570, primals_571, buf153, buf154, buf156, primals_570, primals_571, 256, 32768, grid=grid(256), stream=stream0)
        del primals_570
        del primals_571
        buf157 = empty((8, 256, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_30, x_32], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_23.run(buf152, buf153, buf154, primals_62, primals_63, buf157, 8388608, grid=grid(8388608), stream=stream0)
        del primals_63
        buf158 = reinterpret_tensor(buf134, (8, 128, 1, 1), (128, 1, 1024, 1024), 0); del buf134  # reuse
        buf159 = reinterpret_tensor(buf158, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf158  # reuse
        # Source Nodes: [x_gap_15, x_gap_16], Original ATen: [aten.mean, aten.sum]
        triton_red_fused_mean_sum_24.run(buf159, buf157, 1024, 4096, grid=grid(1024), stream=stream0)
        # Source Nodes: [x_gap_17], Original ATen: [aten.convolution]
        buf160 = extern_kernels.convolution(buf159, primals_64, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf160, (8, 64, 1, 1), (64, 1, 1, 1))
        buf161 = buf160; del buf160  # reuse
        # Source Nodes: [x_gap_17], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf161, primals_65, 512, grid=grid(512), stream=stream0)
        del primals_65
        buf162 = buf112; del buf112  # reuse
        buf163 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_18], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf161, primals_573, primals_574, buf162, buf163, primals_573, primals_574, 64, 8, grid=grid(64), stream=stream0)
        del primals_573
        del primals_574
        buf165 = reinterpret_tensor(buf146, (8, 64, 1, 1), (64, 1, 1, 1), 0); del buf146  # reuse
        # Source Nodes: [x_gap_18, x_gap_19], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_27.run(buf161, buf162, buf163, primals_66, primals_67, buf165, 512, grid=grid(512), stream=stream0)
        del primals_67
        # Source Nodes: [x_attn_6], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf165, primals_68, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (8, 256, 1, 1), (256, 1, 1, 1))
        buf167 = empty_strided((8, 2, 1, 128), (256, 128, 2048, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_35], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf166, primals_69, buf167, 2048, grid=grid(2048), stream=stream0)
        del primals_69
        buf168 = reinterpret_tensor(buf166, (8, 2, 1, 128), (256, 128, 128, 1), 0); del buf166  # reuse
        # Source Nodes: [x_35], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_29.run(buf167, buf168, 2048, grid=grid(2048), stream=stream0)
        buf169 = empty((8, 128, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_3, out_39], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_30.run(buf157, buf168, buf169, 4194304, grid=grid(4194304), stream=stream0)
        buf170 = empty((8, 128, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_44], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_31.run(buf169, buf170, 1048576, grid=grid(1048576), stream=stream0)
        # Source Nodes: [out_45], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf170, primals_70, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (8, 512, 32, 32), (524288, 1024, 32, 1))
        buf172 = reinterpret_tensor(buf145, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf145  # reuse
        buf173 = reinterpret_tensor(buf144, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf144  # reuse
        buf175 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_46], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_32.run(buf171, primals_576, primals_577, buf172, buf173, buf175, primals_576, primals_577, 512, 8192, grid=grid(512), stream=stream0)
        del primals_576
        del primals_577
        buf176 = empty((8, 256, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___layer2___0___downsample_0], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_33.run(buf142, buf176, 2097152, grid=grid(2097152), stream=stream0)
        # Source Nodes: [getattr_l__mod___layer2___0___downsample_1], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(buf176, primals_73, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (8, 512, 32, 32), (524288, 1024, 32, 1))
        buf178 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf179 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf181 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_32.run(buf177, primals_579, primals_580, buf178, buf179, buf181, primals_579, primals_580, 512, 8192, grid=grid(512), stream=stream0)
        del primals_579
        del primals_580
        buf182 = empty((8, 512, 32, 32), device='cuda', dtype=torch.float32)
        buf183 = buf182; del buf182  # reuse
        # Source Nodes: [out_46, out_47, shortcut_5, shortcut_6], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_34.run(buf183, buf171, buf172, buf173, primals_71, primals_72, buf177, buf178, buf179, primals_74, primals_75, 4194304, grid=grid(4194304), stream=stream0)
        del primals_72
        del primals_75
        # Source Nodes: [out_49], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf183, primals_76, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf185 = buf148; del buf148  # reuse
        buf186 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf188 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_50], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf184, primals_582, primals_583, buf185, buf186, buf188, primals_582, primals_583, 128, 8192, grid=grid(128), stream=stream0)
        del primals_582
        del primals_583
        buf189 = empty((8, 128, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_50, out_51], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_36.run(buf184, buf185, buf186, primals_77, primals_78, buf189, 1048576, grid=grid(1048576), stream=stream0)
        del primals_78
        # Source Nodes: [x_37], Original ATen: [aten.convolution]
        buf190 = extern_kernels.convolution(buf189, primals_79, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf190, (8, 256, 32, 32), (262144, 1024, 32, 1))
        buf191 = buf154; del buf154  # reuse
        buf192 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf194 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf190, primals_585, primals_586, buf191, buf192, buf194, primals_585, primals_586, 256, 8192, grid=grid(256), stream=stream0)
        del primals_585
        del primals_586
        buf195 = empty((8, 256, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_38, x_40], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_38.run(buf190, buf191, buf192, primals_80, primals_81, buf195, 2097152, grid=grid(2097152), stream=stream0)
        del primals_81
        buf196 = reinterpret_tensor(buf11, (8, 128, 1, 1), (128, 1, 1024, 1024), 0); del buf11  # reuse
        buf197 = reinterpret_tensor(buf196, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf196  # reuse
        # Source Nodes: [x_gap_20, x_gap_21], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_39.run(buf197, buf195, 1024, 1024, grid=grid(1024), stream=stream0)
        # Source Nodes: [x_gap_22], Original ATen: [aten.convolution]
        buf198 = extern_kernels.convolution(buf197, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (8, 64, 1, 1), (64, 1, 1, 1))
        buf199 = buf198; del buf198  # reuse
        # Source Nodes: [x_gap_22], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf199, primals_83, 512, grid=grid(512), stream=stream0)
        del primals_83
        buf200 = buf163; del buf163  # reuse
        buf201 = buf162; del buf162  # reuse
        # Source Nodes: [x_gap_23], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf199, primals_588, primals_589, buf200, buf201, primals_588, primals_589, 64, 8, grid=grid(64), stream=stream0)
        del primals_588
        del primals_589
        buf203 = reinterpret_tensor(buf179, (8, 64, 1, 1), (64, 1, 1, 1), 0); del buf179  # reuse
        # Source Nodes: [x_gap_23, x_gap_24], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_27.run(buf199, buf200, buf201, primals_84, primals_85, buf203, 512, grid=grid(512), stream=stream0)
        del primals_85
        # Source Nodes: [x_attn_8], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf203, primals_86, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf204, (8, 256, 1, 1), (256, 1, 1, 1))
        buf205 = buf167; del buf167  # reuse
        # Source Nodes: [x_43], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf204, primals_87, buf205, 2048, grid=grid(2048), stream=stream0)
        del primals_87
        buf206 = reinterpret_tensor(buf204, (8, 2, 1, 128), (256, 128, 128, 1), 0); del buf204  # reuse
        # Source Nodes: [x_43], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_29.run(buf205, buf206, 2048, grid=grid(2048), stream=stream0)
        buf207 = empty((8, 128, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_4, out_52], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_40.run(buf195, buf206, buf207, 1048576, grid=grid(1048576), stream=stream0)
        # Source Nodes: [out_57], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, primals_88, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (8, 512, 32, 32), (524288, 1024, 32, 1))
        buf209 = buf173; del buf173  # reuse
        buf210 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf212 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_58], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_32.run(buf208, primals_591, primals_592, buf209, buf210, buf212, primals_591, primals_592, 512, 8192, grid=grid(512), stream=stream0)
        del primals_591
        del primals_592
        buf213 = empty((8, 512, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_58, out_59, shortcut_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_41.run(buf208, buf209, buf210, primals_89, primals_90, buf183, buf213, 4194304, grid=grid(4194304), stream=stream0)
        del primals_90
        # Source Nodes: [out_61], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf213, primals_91, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf215 = buf186; del buf186  # reuse
        buf216 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf218 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_62], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf214, primals_594, primals_595, buf215, buf216, buf218, primals_594, primals_595, 128, 8192, grid=grid(128), stream=stream0)
        del primals_594
        del primals_595
        buf219 = empty((8, 128, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_62, out_63], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_36.run(buf214, buf215, buf216, primals_92, primals_93, buf219, 1048576, grid=grid(1048576), stream=stream0)
        del primals_93
        # Source Nodes: [x_45], Original ATen: [aten.convolution]
        buf220 = extern_kernels.convolution(buf219, primals_94, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf220, (8, 256, 32, 32), (262144, 1024, 32, 1))
        buf221 = buf192; del buf192  # reuse
        buf222 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf224 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_46], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf220, primals_597, primals_598, buf221, buf222, buf224, primals_597, primals_598, 256, 8192, grid=grid(256), stream=stream0)
        del primals_597
        del primals_598
        buf225 = empty((8, 256, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_46, x_48], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_38.run(buf220, buf221, buf222, primals_95, primals_96, buf225, 2097152, grid=grid(2097152), stream=stream0)
        del primals_96
        buf226 = reinterpret_tensor(buf10, (8, 128, 1, 1), (128, 1, 1024, 1024), 0); del buf10  # reuse
        buf227 = reinterpret_tensor(buf226, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf226  # reuse
        # Source Nodes: [x_gap_25, x_gap_26], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_39.run(buf227, buf225, 1024, 1024, grid=grid(1024), stream=stream0)
        # Source Nodes: [x_gap_27], Original ATen: [aten.convolution]
        buf228 = extern_kernels.convolution(buf227, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (8, 64, 1, 1), (64, 1, 1, 1))
        buf229 = buf228; del buf228  # reuse
        # Source Nodes: [x_gap_27], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf229, primals_98, 512, grid=grid(512), stream=stream0)
        del primals_98
        buf230 = buf201; del buf201  # reuse
        buf231 = buf200; del buf200  # reuse
        # Source Nodes: [x_gap_28], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf229, primals_600, primals_601, buf230, buf231, primals_600, primals_601, 64, 8, grid=grid(64), stream=stream0)
        del primals_600
        del primals_601
        buf233 = reinterpret_tensor(buf210, (8, 64, 1, 1), (64, 1, 1, 1), 0); del buf210  # reuse
        # Source Nodes: [x_gap_28, x_gap_29], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_27.run(buf229, buf230, buf231, primals_99, primals_100, buf233, 512, grid=grid(512), stream=stream0)
        del primals_100
        # Source Nodes: [x_attn_10], Original ATen: [aten.convolution]
        buf234 = extern_kernels.convolution(buf233, primals_101, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf234, (8, 256, 1, 1), (256, 1, 1, 1))
        buf235 = buf205; del buf205  # reuse
        # Source Nodes: [x_51], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf234, primals_102, buf235, 2048, grid=grid(2048), stream=stream0)
        del primals_102
        buf236 = reinterpret_tensor(buf234, (8, 2, 1, 128), (256, 128, 128, 1), 0); del buf234  # reuse
        # Source Nodes: [x_51], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_29.run(buf235, buf236, 2048, grid=grid(2048), stream=stream0)
        buf237 = empty((8, 128, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_5, out_64], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_40.run(buf225, buf236, buf237, 1048576, grid=grid(1048576), stream=stream0)
        # Source Nodes: [out_69], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf237, primals_103, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf238, (8, 512, 32, 32), (524288, 1024, 32, 1))
        buf239 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf240 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf242 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_70], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_32.run(buf238, primals_603, primals_604, buf239, buf240, buf242, primals_603, primals_604, 512, 8192, grid=grid(512), stream=stream0)
        del primals_603
        del primals_604
        buf243 = empty((8, 512, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_70, out_71, shortcut_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_41.run(buf238, buf239, buf240, primals_104, primals_105, buf213, buf243, 4194304, grid=grid(4194304), stream=stream0)
        del primals_105
        # Source Nodes: [out_73], Original ATen: [aten.convolution]
        buf244 = extern_kernels.convolution(buf243, primals_106, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf244, (8, 128, 32, 32), (131072, 1024, 32, 1))
        buf245 = buf216; del buf216  # reuse
        buf246 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf248 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_74], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf244, primals_606, primals_607, buf245, buf246, buf248, primals_606, primals_607, 128, 8192, grid=grid(128), stream=stream0)
        del primals_606
        del primals_607
        buf249 = empty((8, 128, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_74, out_75], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_36.run(buf244, buf245, buf246, primals_107, primals_108, buf249, 1048576, grid=grid(1048576), stream=stream0)
        del primals_108
        # Source Nodes: [x_53], Original ATen: [aten.convolution]
        buf250 = extern_kernels.convolution(buf249, primals_109, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf250, (8, 256, 32, 32), (262144, 1024, 32, 1))
        buf251 = buf222; del buf222  # reuse
        buf252 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf254 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf250, primals_609, primals_610, buf251, buf252, buf254, primals_609, primals_610, 256, 8192, grid=grid(256), stream=stream0)
        del primals_609
        del primals_610
        buf255 = empty((8, 256, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54, x_56], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_38.run(buf250, buf251, buf252, primals_110, primals_111, buf255, 2097152, grid=grid(2097152), stream=stream0)
        del primals_111
        buf256 = empty_strided((8, 128, 1, 1), (128, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf257 = reinterpret_tensor(buf256, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf256  # reuse
        # Source Nodes: [x_gap_30, x_gap_31], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_39.run(buf257, buf255, 1024, 1024, grid=grid(1024), stream=stream0)
        # Source Nodes: [x_gap_32], Original ATen: [aten.convolution]
        buf258 = extern_kernels.convolution(buf257, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf258, (8, 64, 1, 1), (64, 1, 1, 1))
        buf259 = buf258; del buf258  # reuse
        # Source Nodes: [x_gap_32], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf259, primals_113, 512, grid=grid(512), stream=stream0)
        del primals_113
        buf260 = buf231; del buf231  # reuse
        buf261 = buf230; del buf230  # reuse
        # Source Nodes: [x_gap_33], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_26.run(buf259, primals_612, primals_613, buf260, buf261, primals_612, primals_613, 64, 8, grid=grid(64), stream=stream0)
        del primals_612
        del primals_613
        buf263 = reinterpret_tensor(buf240, (8, 64, 1, 1), (64, 1, 1, 1), 0); del buf240  # reuse
        # Source Nodes: [x_gap_33, x_gap_34], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_27.run(buf259, buf260, buf261, primals_114, primals_115, buf263, 512, grid=grid(512), stream=stream0)
        del buf260
        del buf261
        del primals_115
        # Source Nodes: [x_attn_12], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf263, primals_116, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (8, 256, 1, 1), (256, 1, 1, 1))
        buf265 = buf235; del buf235  # reuse
        # Source Nodes: [x_59], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_28.run(buf264, primals_117, buf265, 2048, grid=grid(2048), stream=stream0)
        del primals_117
        buf266 = reinterpret_tensor(buf264, (8, 2, 1, 128), (256, 128, 128, 1), 0); del buf264  # reuse
        # Source Nodes: [x_59], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_29.run(buf265, buf266, 2048, grid=grid(2048), stream=stream0)
        buf267 = empty((8, 128, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_6, out_76], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_40.run(buf255, buf266, buf267, 1048576, grid=grid(1048576), stream=stream0)
        # Source Nodes: [out_81], Original ATen: [aten.convolution]
        buf268 = extern_kernels.convolution(buf267, primals_118, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (8, 512, 32, 32), (524288, 1024, 32, 1))
        buf269 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf270 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf272 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_82], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_32.run(buf268, primals_615, primals_616, buf269, buf270, buf272, primals_615, primals_616, 512, 8192, grid=grid(512), stream=stream0)
        del primals_615
        del primals_616
        buf273 = empty((8, 512, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_82, out_83, shortcut_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_41.run(buf268, buf269, buf270, primals_119, primals_120, buf243, buf273, 4194304, grid=grid(4194304), stream=stream0)
        del primals_120
        # Source Nodes: [out_85], Original ATen: [aten.convolution]
        buf274 = extern_kernels.convolution(buf273, primals_121, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf274, (8, 256, 32, 32), (262144, 1024, 32, 1))
        buf275 = buf252; del buf252  # reuse
        buf276 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf278 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_86], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_37.run(buf274, primals_618, primals_619, buf275, buf276, buf278, primals_618, primals_619, 256, 8192, grid=grid(256), stream=stream0)
        del primals_618
        del primals_619
        buf279 = empty((8, 256, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_86, out_87], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_38.run(buf274, buf275, buf276, primals_122, primals_123, buf279, 2097152, grid=grid(2097152), stream=stream0)
        del primals_123
        # Source Nodes: [x_62], Original ATen: [aten.convolution]
        buf280 = extern_kernels.convolution(buf279, primals_124, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf280, (8, 512, 32, 32), (524288, 1024, 32, 1))
        buf281 = buf270; del buf270  # reuse
        buf282 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf284 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_63], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_32.run(buf280, primals_621, primals_622, buf281, buf282, buf284, primals_621, primals_622, 512, 8192, grid=grid(512), stream=stream0)
        del primals_621
        del primals_622
        buf285 = empty((8, 512, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_63, x_65], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_42.run(buf280, buf281, buf282, primals_125, primals_126, buf285, 4194304, grid=grid(4194304), stream=stream0)
        del primals_126
        buf286 = reinterpret_tensor(buf265, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf265  # reuse
        buf287 = reinterpret_tensor(buf286, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf286  # reuse
        # Source Nodes: [x_gap_35, x_gap_36], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_43.run(buf287, buf285, 2048, 1024, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_37], Original ATen: [aten.convolution]
        buf288 = extern_kernels.convolution(buf287, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf288, (8, 128, 1, 1), (128, 1, 1, 1))
        buf289 = buf288; del buf288  # reuse
        # Source Nodes: [x_gap_37], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf289, primals_128, 1024, grid=grid(1024), stream=stream0)
        del primals_128
        buf290 = buf246; del buf246  # reuse
        buf291 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_38], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf289, primals_624, primals_625, buf290, buf291, primals_624, primals_625, 128, 8, grid=grid(128), stream=stream0)
        del primals_624
        del primals_625
        buf293 = empty((8, 128, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_38, x_gap_39], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf289, buf290, buf291, primals_129, primals_130, buf293, 1024, grid=grid(1024), stream=stream0)
        del primals_130
        # Source Nodes: [x_attn_14], Original ATen: [aten.convolution]
        buf294 = extern_kernels.convolution(buf293, primals_131, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf294, (8, 512, 1, 1), (512, 1, 1, 1))
        buf295 = empty_strided((8, 2, 1, 256), (512, 256, 4096, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_47.run(buf294, primals_132, buf295, 4096, grid=grid(4096), stream=stream0)
        del primals_132
        buf296 = reinterpret_tensor(buf294, (8, 2, 1, 256), (512, 256, 256, 1), 0); del buf294  # reuse
        # Source Nodes: [x_68], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_48.run(buf295, buf296, 4096, grid=grid(4096), stream=stream0)
        buf297 = empty((8, 256, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_7, out_88], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_49.run(buf285, buf296, buf297, 2097152, grid=grid(2097152), stream=stream0)
        buf298 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_93], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_50.run(buf297, buf298, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [out_94], Original ATen: [aten.convolution]
        buf299 = extern_kernels.convolution(buf298, primals_133, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf299, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf300 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf301 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf303 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_95], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf299, primals_627, primals_628, buf300, buf301, buf303, primals_627, primals_628, 1024, 2048, grid=grid(1024), stream=stream0)
        del primals_627
        del primals_628
        buf304 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___layer3___0___downsample_0], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_52.run(buf273, buf304, 1048576, grid=grid(1048576), stream=stream0)
        # Source Nodes: [getattr_l__mod___layer3___0___downsample_1], Original ATen: [aten.convolution]
        buf305 = extern_kernels.convolution(buf304, primals_136, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf305, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf306 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf307 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf309 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_10], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf305, primals_630, primals_631, buf306, buf307, buf309, primals_630, primals_631, 1024, 2048, grid=grid(1024), stream=stream0)
        del primals_630
        del primals_631
        buf310 = empty((8, 1024, 16, 16), device='cuda', dtype=torch.float32)
        buf311 = buf310; del buf310  # reuse
        # Source Nodes: [out_95, out_96, shortcut_10, shortcut_11], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_53.run(buf311, buf299, buf300, buf301, primals_134, primals_135, buf305, buf306, buf307, primals_137, primals_138, 2097152, grid=grid(2097152), stream=stream0)
        del primals_135
        del primals_138
        # Source Nodes: [out_98], Original ATen: [aten.convolution]
        buf312 = extern_kernels.convolution(buf311, primals_139, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf312, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf313 = buf276; del buf276  # reuse
        buf314 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf316 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_99], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf312, primals_633, primals_634, buf313, buf314, buf316, primals_633, primals_634, 256, 2048, grid=grid(256), stream=stream0)
        del primals_633
        del primals_634
        buf317 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_100, out_99], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_55.run(buf312, buf313, buf314, primals_140, primals_141, buf317, 524288, grid=grid(524288), stream=stream0)
        del primals_141
        # Source Nodes: [x_70], Original ATen: [aten.convolution]
        buf318 = extern_kernels.convolution(buf317, primals_142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf318, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf319 = buf282; del buf282  # reuse
        buf320 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf322 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_71], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf318, primals_636, primals_637, buf319, buf320, buf322, primals_636, primals_637, 512, 2048, grid=grid(512), stream=stream0)
        del primals_636
        del primals_637
        buf323 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_71, x_73], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf318, buf319, buf320, primals_143, primals_144, buf323, 1048576, grid=grid(1048576), stream=stream0)
        del primals_144
        buf324 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf325 = reinterpret_tensor(buf324, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf324  # reuse
        # Source Nodes: [x_gap_40, x_gap_41], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_58.run(buf325, buf323, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_42], Original ATen: [aten.convolution]
        buf326 = extern_kernels.convolution(buf325, primals_145, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf326, (8, 128, 1, 1), (128, 1, 1, 1))
        buf327 = buf326; del buf326  # reuse
        # Source Nodes: [x_gap_42], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf327, primals_146, 1024, grid=grid(1024), stream=stream0)
        del primals_146
        buf328 = buf291; del buf291  # reuse
        buf329 = buf290; del buf290  # reuse
        # Source Nodes: [x_gap_43], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf327, primals_639, primals_640, buf328, buf329, primals_639, primals_640, 128, 8, grid=grid(128), stream=stream0)
        del primals_639
        del primals_640
        buf331 = reinterpret_tensor(buf307, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf307  # reuse
        # Source Nodes: [x_gap_43, x_gap_44], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf327, buf328, buf329, primals_147, primals_148, buf331, 1024, grid=grid(1024), stream=stream0)
        del primals_148
        # Source Nodes: [x_attn_16], Original ATen: [aten.convolution]
        buf332 = extern_kernels.convolution(buf331, primals_149, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf332, (8, 512, 1, 1), (512, 1, 1, 1))
        buf333 = buf295; del buf295  # reuse
        # Source Nodes: [x_76], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_47.run(buf332, primals_150, buf333, 4096, grid=grid(4096), stream=stream0)
        del primals_150
        buf334 = reinterpret_tensor(buf332, (8, 2, 1, 256), (512, 256, 256, 1), 0); del buf332  # reuse
        # Source Nodes: [x_76], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_48.run(buf333, buf334, 4096, grid=grid(4096), stream=stream0)
        buf335 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_8, out_101], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_59.run(buf323, buf334, buf335, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [out_106], Original ATen: [aten.convolution]
        buf336 = extern_kernels.convolution(buf335, primals_151, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf337 = buf301; del buf301  # reuse
        buf338 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf340 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_107], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf336, primals_642, primals_643, buf337, buf338, buf340, primals_642, primals_643, 1024, 2048, grid=grid(1024), stream=stream0)
        del primals_642
        del primals_643
        buf341 = empty((8, 1024, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_107, out_108, shortcut_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf336, buf337, buf338, primals_152, primals_153, buf311, buf341, 2097152, grid=grid(2097152), stream=stream0)
        del primals_153
        # Source Nodes: [out_110], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, primals_154, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf343 = buf314; del buf314  # reuse
        buf344 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf346 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_111], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf342, primals_645, primals_646, buf343, buf344, buf346, primals_645, primals_646, 256, 2048, grid=grid(256), stream=stream0)
        del primals_645
        del primals_646
        buf347 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_111, out_112], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_55.run(buf342, buf343, buf344, primals_155, primals_156, buf347, 524288, grid=grid(524288), stream=stream0)
        del primals_156
        # Source Nodes: [x_78], Original ATen: [aten.convolution]
        buf348 = extern_kernels.convolution(buf347, primals_157, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf348, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf349 = buf320; del buf320  # reuse
        buf350 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf352 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_79], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf348, primals_648, primals_649, buf349, buf350, buf352, primals_648, primals_649, 512, 2048, grid=grid(512), stream=stream0)
        del primals_648
        del primals_649
        buf353 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_79, x_81], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf348, buf349, buf350, primals_158, primals_159, buf353, 1048576, grid=grid(1048576), stream=stream0)
        del primals_159
        buf354 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf355 = reinterpret_tensor(buf354, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf354  # reuse
        # Source Nodes: [x_gap_45, x_gap_46], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_58.run(buf355, buf353, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_47], Original ATen: [aten.convolution]
        buf356 = extern_kernels.convolution(buf355, primals_160, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf356, (8, 128, 1, 1), (128, 1, 1, 1))
        buf357 = buf356; del buf356  # reuse
        # Source Nodes: [x_gap_47], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf357, primals_161, 1024, grid=grid(1024), stream=stream0)
        del primals_161
        buf358 = buf329; del buf329  # reuse
        buf359 = buf328; del buf328  # reuse
        # Source Nodes: [x_gap_48], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf357, primals_651, primals_652, buf358, buf359, primals_651, primals_652, 128, 8, grid=grid(128), stream=stream0)
        del primals_651
        del primals_652
        buf361 = reinterpret_tensor(buf338, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf338  # reuse
        # Source Nodes: [x_gap_48, x_gap_49], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf357, buf358, buf359, primals_162, primals_163, buf361, 1024, grid=grid(1024), stream=stream0)
        del primals_163
        # Source Nodes: [x_attn_18], Original ATen: [aten.convolution]
        buf362 = extern_kernels.convolution(buf361, primals_164, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf362, (8, 512, 1, 1), (512, 1, 1, 1))
        buf363 = buf333; del buf333  # reuse
        # Source Nodes: [x_84], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_47.run(buf362, primals_165, buf363, 4096, grid=grid(4096), stream=stream0)
        del primals_165
        buf364 = reinterpret_tensor(buf362, (8, 2, 1, 256), (512, 256, 256, 1), 0); del buf362  # reuse
        # Source Nodes: [x_84], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_48.run(buf363, buf364, 4096, grid=grid(4096), stream=stream0)
        buf365 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_9, out_113], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_59.run(buf353, buf364, buf365, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [out_118], Original ATen: [aten.convolution]
        buf366 = extern_kernels.convolution(buf365, primals_166, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf366, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf367 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf368 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf370 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_119], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf366, primals_654, primals_655, buf367, buf368, buf370, primals_654, primals_655, 1024, 2048, grid=grid(1024), stream=stream0)
        del primals_654
        del primals_655
        buf371 = empty((8, 1024, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_119, out_120, shortcut_13], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf366, buf367, buf368, primals_167, primals_168, buf341, buf371, 2097152, grid=grid(2097152), stream=stream0)
        del primals_168
        # Source Nodes: [out_122], Original ATen: [aten.convolution]
        buf372 = extern_kernels.convolution(buf371, primals_169, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf372, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf373 = buf344; del buf344  # reuse
        buf374 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf376 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_123], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf372, primals_657, primals_658, buf373, buf374, buf376, primals_657, primals_658, 256, 2048, grid=grid(256), stream=stream0)
        del primals_657
        del primals_658
        buf377 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_123, out_124], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_55.run(buf372, buf373, buf374, primals_170, primals_171, buf377, 524288, grid=grid(524288), stream=stream0)
        del primals_171
        # Source Nodes: [x_86], Original ATen: [aten.convolution]
        buf378 = extern_kernels.convolution(buf377, primals_172, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf378, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf379 = buf350; del buf350  # reuse
        buf380 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf382 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_87], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf378, primals_660, primals_661, buf379, buf380, buf382, primals_660, primals_661, 512, 2048, grid=grid(512), stream=stream0)
        del primals_660
        del primals_661
        buf383 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_87, x_89], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf378, buf379, buf380, primals_173, primals_174, buf383, 1048576, grid=grid(1048576), stream=stream0)
        del primals_174
        buf384 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf385 = reinterpret_tensor(buf384, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf384  # reuse
        # Source Nodes: [x_gap_50, x_gap_51], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_58.run(buf385, buf383, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_52], Original ATen: [aten.convolution]
        buf386 = extern_kernels.convolution(buf385, primals_175, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf386, (8, 128, 1, 1), (128, 1, 1, 1))
        buf387 = buf386; del buf386  # reuse
        # Source Nodes: [x_gap_52], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf387, primals_176, 1024, grid=grid(1024), stream=stream0)
        del primals_176
        buf388 = buf359; del buf359  # reuse
        buf389 = buf358; del buf358  # reuse
        # Source Nodes: [x_gap_53], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf387, primals_663, primals_664, buf388, buf389, primals_663, primals_664, 128, 8, grid=grid(128), stream=stream0)
        del primals_663
        del primals_664
        buf391 = reinterpret_tensor(buf368, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf368  # reuse
        # Source Nodes: [x_gap_53, x_gap_54], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf387, buf388, buf389, primals_177, primals_178, buf391, 1024, grid=grid(1024), stream=stream0)
        del primals_178
        # Source Nodes: [x_attn_20], Original ATen: [aten.convolution]
        buf392 = extern_kernels.convolution(buf391, primals_179, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf392, (8, 512, 1, 1), (512, 1, 1, 1))
        buf393 = buf363; del buf363  # reuse
        # Source Nodes: [x_92], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_47.run(buf392, primals_180, buf393, 4096, grid=grid(4096), stream=stream0)
        del primals_180
        buf394 = reinterpret_tensor(buf392, (8, 2, 1, 256), (512, 256, 256, 1), 0); del buf392  # reuse
        # Source Nodes: [x_92], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_48.run(buf393, buf394, 4096, grid=grid(4096), stream=stream0)
        buf395 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_10, out_125], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_59.run(buf383, buf394, buf395, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [out_130], Original ATen: [aten.convolution]
        buf396 = extern_kernels.convolution(buf395, primals_181, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf396, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf397 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf398 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf400 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_131], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf396, primals_666, primals_667, buf397, buf398, buf400, primals_666, primals_667, 1024, 2048, grid=grid(1024), stream=stream0)
        del primals_666
        del primals_667
        buf401 = empty((8, 1024, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_131, out_132, shortcut_14], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf396, buf397, buf398, primals_182, primals_183, buf371, buf401, 2097152, grid=grid(2097152), stream=stream0)
        del primals_183
        # Source Nodes: [out_134], Original ATen: [aten.convolution]
        buf402 = extern_kernels.convolution(buf401, primals_184, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf402, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf403 = buf374; del buf374  # reuse
        buf404 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf406 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_135], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf402, primals_669, primals_670, buf403, buf404, buf406, primals_669, primals_670, 256, 2048, grid=grid(256), stream=stream0)
        del primals_669
        del primals_670
        buf407 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_135, out_136], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_55.run(buf402, buf403, buf404, primals_185, primals_186, buf407, 524288, grid=grid(524288), stream=stream0)
        del primals_186
        # Source Nodes: [x_94], Original ATen: [aten.convolution]
        buf408 = extern_kernels.convolution(buf407, primals_187, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf408, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf409 = buf380; del buf380  # reuse
        buf410 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf412 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_95], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf408, primals_672, primals_673, buf409, buf410, buf412, primals_672, primals_673, 512, 2048, grid=grid(512), stream=stream0)
        del primals_672
        del primals_673
        buf413 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_95, x_97], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf408, buf409, buf410, primals_188, primals_189, buf413, 1048576, grid=grid(1048576), stream=stream0)
        del primals_189
        buf414 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf415 = reinterpret_tensor(buf414, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf414  # reuse
        # Source Nodes: [x_gap_55, x_gap_56], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_58.run(buf415, buf413, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_57], Original ATen: [aten.convolution]
        buf416 = extern_kernels.convolution(buf415, primals_190, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf416, (8, 128, 1, 1), (128, 1, 1, 1))
        buf417 = buf416; del buf416  # reuse
        # Source Nodes: [x_gap_57], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf417, primals_191, 1024, grid=grid(1024), stream=stream0)
        del primals_191
        buf418 = buf389; del buf389  # reuse
        buf419 = buf388; del buf388  # reuse
        # Source Nodes: [x_gap_58], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf417, primals_675, primals_676, buf418, buf419, primals_675, primals_676, 128, 8, grid=grid(128), stream=stream0)
        del primals_675
        del primals_676
        buf421 = reinterpret_tensor(buf398, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf398  # reuse
        # Source Nodes: [x_gap_58, x_gap_59], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf417, buf418, buf419, primals_192, primals_193, buf421, 1024, grid=grid(1024), stream=stream0)
        del primals_193
        # Source Nodes: [x_attn_22], Original ATen: [aten.convolution]
        buf422 = extern_kernels.convolution(buf421, primals_194, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf422, (8, 512, 1, 1), (512, 1, 1, 1))
        buf423 = buf393; del buf393  # reuse
        # Source Nodes: [x_100], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_47.run(buf422, primals_195, buf423, 4096, grid=grid(4096), stream=stream0)
        del primals_195
        buf424 = reinterpret_tensor(buf422, (8, 2, 1, 256), (512, 256, 256, 1), 0); del buf422  # reuse
        # Source Nodes: [x_100], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_48.run(buf423, buf424, 4096, grid=grid(4096), stream=stream0)
        buf425 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_11, out_137], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_59.run(buf413, buf424, buf425, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [out_142], Original ATen: [aten.convolution]
        buf426 = extern_kernels.convolution(buf425, primals_196, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf426, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf427 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf428 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf430 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_143], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf426, primals_678, primals_679, buf427, buf428, buf430, primals_678, primals_679, 1024, 2048, grid=grid(1024), stream=stream0)
        del primals_678
        del primals_679
        buf431 = empty((8, 1024, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_143, out_144, shortcut_15], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf426, buf427, buf428, primals_197, primals_198, buf401, buf431, 2097152, grid=grid(2097152), stream=stream0)
        del primals_198
        # Source Nodes: [out_146], Original ATen: [aten.convolution]
        buf432 = extern_kernels.convolution(buf431, primals_199, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf432, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf433 = buf404; del buf404  # reuse
        buf434 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf436 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_147], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf432, primals_681, primals_682, buf433, buf434, buf436, primals_681, primals_682, 256, 2048, grid=grid(256), stream=stream0)
        del primals_681
        del primals_682
        buf437 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_147, out_148], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_55.run(buf432, buf433, buf434, primals_200, primals_201, buf437, 524288, grid=grid(524288), stream=stream0)
        del primals_201
        # Source Nodes: [x_102], Original ATen: [aten.convolution]
        buf438 = extern_kernels.convolution(buf437, primals_202, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf438, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf439 = buf410; del buf410  # reuse
        buf440 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf442 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_103], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf438, primals_684, primals_685, buf439, buf440, buf442, primals_684, primals_685, 512, 2048, grid=grid(512), stream=stream0)
        del primals_684
        del primals_685
        buf443 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_103, x_105], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf438, buf439, buf440, primals_203, primals_204, buf443, 1048576, grid=grid(1048576), stream=stream0)
        del primals_204
        buf444 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf445 = reinterpret_tensor(buf444, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf444  # reuse
        # Source Nodes: [x_gap_60, x_gap_61], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_58.run(buf445, buf443, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_62], Original ATen: [aten.convolution]
        buf446 = extern_kernels.convolution(buf445, primals_205, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf446, (8, 128, 1, 1), (128, 1, 1, 1))
        buf447 = buf446; del buf446  # reuse
        # Source Nodes: [x_gap_62], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf447, primals_206, 1024, grid=grid(1024), stream=stream0)
        del primals_206
        buf448 = buf419; del buf419  # reuse
        buf449 = buf418; del buf418  # reuse
        # Source Nodes: [x_gap_63], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf447, primals_687, primals_688, buf448, buf449, primals_687, primals_688, 128, 8, grid=grid(128), stream=stream0)
        del primals_687
        del primals_688
        buf451 = reinterpret_tensor(buf428, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf428  # reuse
        # Source Nodes: [x_gap_63, x_gap_64], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf447, buf448, buf449, primals_207, primals_208, buf451, 1024, grid=grid(1024), stream=stream0)
        del primals_208
        # Source Nodes: [x_attn_24], Original ATen: [aten.convolution]
        buf452 = extern_kernels.convolution(buf451, primals_209, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf452, (8, 512, 1, 1), (512, 1, 1, 1))
        buf453 = buf423; del buf423  # reuse
        # Source Nodes: [x_108], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_47.run(buf452, primals_210, buf453, 4096, grid=grid(4096), stream=stream0)
        del primals_210
        buf454 = reinterpret_tensor(buf452, (8, 2, 1, 256), (512, 256, 256, 1), 0); del buf452  # reuse
        # Source Nodes: [x_108], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_48.run(buf453, buf454, 4096, grid=grid(4096), stream=stream0)
        buf455 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_12, out_149], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_59.run(buf443, buf454, buf455, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [out_154], Original ATen: [aten.convolution]
        buf456 = extern_kernels.convolution(buf455, primals_211, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf456, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf457 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf458 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf460 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_155], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf456, primals_690, primals_691, buf457, buf458, buf460, primals_690, primals_691, 1024, 2048, grid=grid(1024), stream=stream0)
        del primals_690
        del primals_691
        buf461 = empty((8, 1024, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_155, out_156, shortcut_16], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf456, buf457, buf458, primals_212, primals_213, buf431, buf461, 2097152, grid=grid(2097152), stream=stream0)
        del primals_213
        # Source Nodes: [out_158], Original ATen: [aten.convolution]
        buf462 = extern_kernels.convolution(buf461, primals_214, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf462, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf463 = buf434; del buf434  # reuse
        buf464 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf466 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_159], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf462, primals_693, primals_694, buf463, buf464, buf466, primals_693, primals_694, 256, 2048, grid=grid(256), stream=stream0)
        del primals_693
        del primals_694
        buf467 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_159, out_160], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_55.run(buf462, buf463, buf464, primals_215, primals_216, buf467, 524288, grid=grid(524288), stream=stream0)
        del primals_216
        # Source Nodes: [x_110], Original ATen: [aten.convolution]
        buf468 = extern_kernels.convolution(buf467, primals_217, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf468, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf469 = buf440; del buf440  # reuse
        buf470 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf472 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_111], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf468, primals_696, primals_697, buf469, buf470, buf472, primals_696, primals_697, 512, 2048, grid=grid(512), stream=stream0)
        del primals_696
        del primals_697
        buf473 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_111, x_113], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf468, buf469, buf470, primals_218, primals_219, buf473, 1048576, grid=grid(1048576), stream=stream0)
        del primals_219
        buf474 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf475 = reinterpret_tensor(buf474, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf474  # reuse
        # Source Nodes: [x_gap_65, x_gap_66], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_58.run(buf475, buf473, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_67], Original ATen: [aten.convolution]
        buf476 = extern_kernels.convolution(buf475, primals_220, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf476, (8, 128, 1, 1), (128, 1, 1, 1))
        buf477 = buf476; del buf476  # reuse
        # Source Nodes: [x_gap_67], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf477, primals_221, 1024, grid=grid(1024), stream=stream0)
        del primals_221
        buf478 = buf449; del buf449  # reuse
        buf479 = buf448; del buf448  # reuse
        # Source Nodes: [x_gap_68], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf477, primals_699, primals_700, buf478, buf479, primals_699, primals_700, 128, 8, grid=grid(128), stream=stream0)
        del primals_699
        del primals_700
        buf481 = reinterpret_tensor(buf458, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf458  # reuse
        # Source Nodes: [x_gap_68, x_gap_69], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf477, buf478, buf479, primals_222, primals_223, buf481, 1024, grid=grid(1024), stream=stream0)
        del primals_223
        # Source Nodes: [x_attn_26], Original ATen: [aten.convolution]
        buf482 = extern_kernels.convolution(buf481, primals_224, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf482, (8, 512, 1, 1), (512, 1, 1, 1))
        buf483 = buf453; del buf453  # reuse
        # Source Nodes: [x_116], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_47.run(buf482, primals_225, buf483, 4096, grid=grid(4096), stream=stream0)
        del primals_225
        buf484 = reinterpret_tensor(buf482, (8, 2, 1, 256), (512, 256, 256, 1), 0); del buf482  # reuse
        # Source Nodes: [x_116], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_48.run(buf483, buf484, 4096, grid=grid(4096), stream=stream0)
        buf485 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_13, out_161], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_59.run(buf473, buf484, buf485, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [out_166], Original ATen: [aten.convolution]
        buf486 = extern_kernels.convolution(buf485, primals_226, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf486, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf487 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf488 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf490 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_167], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf486, primals_702, primals_703, buf487, buf488, buf490, primals_702, primals_703, 1024, 2048, grid=grid(1024), stream=stream0)
        del primals_702
        del primals_703
        buf491 = empty((8, 1024, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_167, out_168, shortcut_17], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf486, buf487, buf488, primals_227, primals_228, buf461, buf491, 2097152, grid=grid(2097152), stream=stream0)
        del primals_228
        # Source Nodes: [out_170], Original ATen: [aten.convolution]
        buf492 = extern_kernels.convolution(buf491, primals_229, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf492, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf493 = buf464; del buf464  # reuse
        buf494 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf496 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_171], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf492, primals_705, primals_706, buf493, buf494, buf496, primals_705, primals_706, 256, 2048, grid=grid(256), stream=stream0)
        del primals_705
        del primals_706
        buf497 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_171, out_172], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_55.run(buf492, buf493, buf494, primals_230, primals_231, buf497, 524288, grid=grid(524288), stream=stream0)
        del primals_231
        # Source Nodes: [x_118], Original ATen: [aten.convolution]
        buf498 = extern_kernels.convolution(buf497, primals_232, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf498, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf499 = buf470; del buf470  # reuse
        buf500 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf502 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_119], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf498, primals_708, primals_709, buf499, buf500, buf502, primals_708, primals_709, 512, 2048, grid=grid(512), stream=stream0)
        del primals_708
        del primals_709
        buf503 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_119, x_121], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf498, buf499, buf500, primals_233, primals_234, buf503, 1048576, grid=grid(1048576), stream=stream0)
        del primals_234
        buf504 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf505 = reinterpret_tensor(buf504, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf504  # reuse
        # Source Nodes: [x_gap_70, x_gap_71], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_58.run(buf505, buf503, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_72], Original ATen: [aten.convolution]
        buf506 = extern_kernels.convolution(buf505, primals_235, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf506, (8, 128, 1, 1), (128, 1, 1, 1))
        buf507 = buf506; del buf506  # reuse
        # Source Nodes: [x_gap_72], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf507, primals_236, 1024, grid=grid(1024), stream=stream0)
        del primals_236
        buf508 = buf479; del buf479  # reuse
        buf509 = buf478; del buf478  # reuse
        # Source Nodes: [x_gap_73], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf507, primals_711, primals_712, buf508, buf509, primals_711, primals_712, 128, 8, grid=grid(128), stream=stream0)
        del primals_711
        del primals_712
        buf511 = reinterpret_tensor(buf488, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf488  # reuse
        # Source Nodes: [x_gap_73, x_gap_74], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf507, buf508, buf509, primals_237, primals_238, buf511, 1024, grid=grid(1024), stream=stream0)
        del primals_238
        # Source Nodes: [x_attn_28], Original ATen: [aten.convolution]
        buf512 = extern_kernels.convolution(buf511, primals_239, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf512, (8, 512, 1, 1), (512, 1, 1, 1))
        buf513 = buf483; del buf483  # reuse
        # Source Nodes: [x_124], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_47.run(buf512, primals_240, buf513, 4096, grid=grid(4096), stream=stream0)
        del primals_240
        buf514 = reinterpret_tensor(buf512, (8, 2, 1, 256), (512, 256, 256, 1), 0); del buf512  # reuse
        # Source Nodes: [x_124], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_48.run(buf513, buf514, 4096, grid=grid(4096), stream=stream0)
        buf515 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_14, out_173], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_59.run(buf503, buf514, buf515, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [out_178], Original ATen: [aten.convolution]
        buf516 = extern_kernels.convolution(buf515, primals_241, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf516, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf517 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf518 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf520 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_179], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf516, primals_714, primals_715, buf517, buf518, buf520, primals_714, primals_715, 1024, 2048, grid=grid(1024), stream=stream0)
        del primals_714
        del primals_715
        buf521 = empty((8, 1024, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_179, out_180, shortcut_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf516, buf517, buf518, primals_242, primals_243, buf491, buf521, 2097152, grid=grid(2097152), stream=stream0)
        del primals_243
        # Source Nodes: [out_182], Original ATen: [aten.convolution]
        buf522 = extern_kernels.convolution(buf521, primals_244, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf522, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf523 = buf494; del buf494  # reuse
        buf524 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf526 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_183], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf522, primals_717, primals_718, buf523, buf524, buf526, primals_717, primals_718, 256, 2048, grid=grid(256), stream=stream0)
        del primals_717
        del primals_718
        buf527 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_183, out_184], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_55.run(buf522, buf523, buf524, primals_245, primals_246, buf527, 524288, grid=grid(524288), stream=stream0)
        del primals_246
        # Source Nodes: [x_126], Original ATen: [aten.convolution]
        buf528 = extern_kernels.convolution(buf527, primals_247, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf528, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf529 = buf500; del buf500  # reuse
        buf530 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf532 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_127], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf528, primals_720, primals_721, buf529, buf530, buf532, primals_720, primals_721, 512, 2048, grid=grid(512), stream=stream0)
        del primals_720
        del primals_721
        buf533 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_127, x_129], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf528, buf529, buf530, primals_248, primals_249, buf533, 1048576, grid=grid(1048576), stream=stream0)
        del primals_249
        buf534 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf535 = reinterpret_tensor(buf534, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf534  # reuse
        # Source Nodes: [x_gap_75, x_gap_76], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_58.run(buf535, buf533, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_77], Original ATen: [aten.convolution]
        buf536 = extern_kernels.convolution(buf535, primals_250, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf536, (8, 128, 1, 1), (128, 1, 1, 1))
        buf537 = buf536; del buf536  # reuse
        # Source Nodes: [x_gap_77], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf537, primals_251, 1024, grid=grid(1024), stream=stream0)
        del primals_251
        buf538 = buf509; del buf509  # reuse
        buf539 = buf508; del buf508  # reuse
        # Source Nodes: [x_gap_78], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf537, primals_723, primals_724, buf538, buf539, primals_723, primals_724, 128, 8, grid=grid(128), stream=stream0)
        del primals_723
        del primals_724
        buf541 = reinterpret_tensor(buf518, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf518  # reuse
        # Source Nodes: [x_gap_78, x_gap_79], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf537, buf538, buf539, primals_252, primals_253, buf541, 1024, grid=grid(1024), stream=stream0)
        del primals_253
        # Source Nodes: [x_attn_30], Original ATen: [aten.convolution]
        buf542 = extern_kernels.convolution(buf541, primals_254, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf542, (8, 512, 1, 1), (512, 1, 1, 1))
        buf543 = buf513; del buf513  # reuse
        # Source Nodes: [x_132], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_47.run(buf542, primals_255, buf543, 4096, grid=grid(4096), stream=stream0)
        del primals_255
        buf544 = reinterpret_tensor(buf542, (8, 2, 1, 256), (512, 256, 256, 1), 0); del buf542  # reuse
        # Source Nodes: [x_132], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_48.run(buf543, buf544, 4096, grid=grid(4096), stream=stream0)
        buf545 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_15, out_185], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_59.run(buf533, buf544, buf545, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [out_190], Original ATen: [aten.convolution]
        buf546 = extern_kernels.convolution(buf545, primals_256, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf546, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf547 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf548 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf550 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_191], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf546, primals_726, primals_727, buf547, buf548, buf550, primals_726, primals_727, 1024, 2048, grid=grid(1024), stream=stream0)
        del primals_726
        del primals_727
        buf551 = empty((8, 1024, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_191, out_192, shortcut_19], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf546, buf547, buf548, primals_257, primals_258, buf521, buf551, 2097152, grid=grid(2097152), stream=stream0)
        del primals_258
        # Source Nodes: [out_194], Original ATen: [aten.convolution]
        buf552 = extern_kernels.convolution(buf551, primals_259, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf552, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf553 = buf524; del buf524  # reuse
        buf554 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf556 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_195], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf552, primals_729, primals_730, buf553, buf554, buf556, primals_729, primals_730, 256, 2048, grid=grid(256), stream=stream0)
        del primals_729
        del primals_730
        buf557 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_195, out_196], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_55.run(buf552, buf553, buf554, primals_260, primals_261, buf557, 524288, grid=grid(524288), stream=stream0)
        del primals_261
        # Source Nodes: [x_134], Original ATen: [aten.convolution]
        buf558 = extern_kernels.convolution(buf557, primals_262, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf558, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf559 = buf530; del buf530  # reuse
        buf560 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf562 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_135], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf558, primals_732, primals_733, buf559, buf560, buf562, primals_732, primals_733, 512, 2048, grid=grid(512), stream=stream0)
        del primals_732
        del primals_733
        buf563 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_135, x_137], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf558, buf559, buf560, primals_263, primals_264, buf563, 1048576, grid=grid(1048576), stream=stream0)
        del primals_264
        buf564 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf565 = reinterpret_tensor(buf564, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf564  # reuse
        # Source Nodes: [x_gap_80, x_gap_81], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_58.run(buf565, buf563, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_82], Original ATen: [aten.convolution]
        buf566 = extern_kernels.convolution(buf565, primals_265, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf566, (8, 128, 1, 1), (128, 1, 1, 1))
        buf567 = buf566; del buf566  # reuse
        # Source Nodes: [x_gap_82], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf567, primals_266, 1024, grid=grid(1024), stream=stream0)
        del primals_266
        buf568 = buf539; del buf539  # reuse
        buf569 = buf538; del buf538  # reuse
        # Source Nodes: [x_gap_83], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf567, primals_735, primals_736, buf568, buf569, primals_735, primals_736, 128, 8, grid=grid(128), stream=stream0)
        del primals_735
        del primals_736
        buf571 = reinterpret_tensor(buf548, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf548  # reuse
        # Source Nodes: [x_gap_83, x_gap_84], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf567, buf568, buf569, primals_267, primals_268, buf571, 1024, grid=grid(1024), stream=stream0)
        del primals_268
        # Source Nodes: [x_attn_32], Original ATen: [aten.convolution]
        buf572 = extern_kernels.convolution(buf571, primals_269, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf572, (8, 512, 1, 1), (512, 1, 1, 1))
        buf573 = buf543; del buf543  # reuse
        # Source Nodes: [x_140], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_47.run(buf572, primals_270, buf573, 4096, grid=grid(4096), stream=stream0)
        del primals_270
        buf574 = reinterpret_tensor(buf572, (8, 2, 1, 256), (512, 256, 256, 1), 0); del buf572  # reuse
        # Source Nodes: [x_140], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_48.run(buf573, buf574, 4096, grid=grid(4096), stream=stream0)
        buf575 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_16, out_197], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_59.run(buf563, buf574, buf575, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [out_202], Original ATen: [aten.convolution]
        buf576 = extern_kernels.convolution(buf575, primals_271, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf576, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf577 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf578 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf580 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_203], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf576, primals_738, primals_739, buf577, buf578, buf580, primals_738, primals_739, 1024, 2048, grid=grid(1024), stream=stream0)
        del primals_738
        del primals_739
        buf581 = empty((8, 1024, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_203, out_204, shortcut_20], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf576, buf577, buf578, primals_272, primals_273, buf551, buf581, 2097152, grid=grid(2097152), stream=stream0)
        del primals_273
        # Source Nodes: [out_206], Original ATen: [aten.convolution]
        buf582 = extern_kernels.convolution(buf581, primals_274, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf582, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf583 = buf554; del buf554  # reuse
        buf584 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf586 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_207], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf582, primals_741, primals_742, buf583, buf584, buf586, primals_741, primals_742, 256, 2048, grid=grid(256), stream=stream0)
        del primals_741
        del primals_742
        buf587 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_207, out_208], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_55.run(buf582, buf583, buf584, primals_275, primals_276, buf587, 524288, grid=grid(524288), stream=stream0)
        del primals_276
        # Source Nodes: [x_142], Original ATen: [aten.convolution]
        buf588 = extern_kernels.convolution(buf587, primals_277, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf588, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf589 = buf560; del buf560  # reuse
        buf590 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf592 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_143], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf588, primals_744, primals_745, buf589, buf590, buf592, primals_744, primals_745, 512, 2048, grid=grid(512), stream=stream0)
        del primals_744
        del primals_745
        buf593 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_143, x_145], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf588, buf589, buf590, primals_278, primals_279, buf593, 1048576, grid=grid(1048576), stream=stream0)
        del primals_279
        buf594 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf595 = reinterpret_tensor(buf594, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf594  # reuse
        # Source Nodes: [x_gap_85, x_gap_86], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_58.run(buf595, buf593, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_87], Original ATen: [aten.convolution]
        buf596 = extern_kernels.convolution(buf595, primals_280, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf596, (8, 128, 1, 1), (128, 1, 1, 1))
        buf597 = buf596; del buf596  # reuse
        # Source Nodes: [x_gap_87], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf597, primals_281, 1024, grid=grid(1024), stream=stream0)
        del primals_281
        buf598 = buf569; del buf569  # reuse
        buf599 = buf568; del buf568  # reuse
        # Source Nodes: [x_gap_88], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf597, primals_747, primals_748, buf598, buf599, primals_747, primals_748, 128, 8, grid=grid(128), stream=stream0)
        del primals_747
        del primals_748
        buf601 = reinterpret_tensor(buf578, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf578  # reuse
        # Source Nodes: [x_gap_88, x_gap_89], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf597, buf598, buf599, primals_282, primals_283, buf601, 1024, grid=grid(1024), stream=stream0)
        del primals_283
        # Source Nodes: [x_attn_34], Original ATen: [aten.convolution]
        buf602 = extern_kernels.convolution(buf601, primals_284, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf602, (8, 512, 1, 1), (512, 1, 1, 1))
        buf603 = buf573; del buf573  # reuse
        # Source Nodes: [x_148], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_47.run(buf602, primals_285, buf603, 4096, grid=grid(4096), stream=stream0)
        del primals_285
        buf604 = reinterpret_tensor(buf602, (8, 2, 1, 256), (512, 256, 256, 1), 0); del buf602  # reuse
        # Source Nodes: [x_148], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_48.run(buf603, buf604, 4096, grid=grid(4096), stream=stream0)
        buf605 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_17, out_209], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_59.run(buf593, buf604, buf605, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [out_214], Original ATen: [aten.convolution]
        buf606 = extern_kernels.convolution(buf605, primals_286, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf606, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf607 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf608 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf610 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_215], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf606, primals_750, primals_751, buf607, buf608, buf610, primals_750, primals_751, 1024, 2048, grid=grid(1024), stream=stream0)
        del primals_750
        del primals_751
        buf611 = empty((8, 1024, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_215, out_216, shortcut_21], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf606, buf607, buf608, primals_287, primals_288, buf581, buf611, 2097152, grid=grid(2097152), stream=stream0)
        del primals_288
        # Source Nodes: [out_218], Original ATen: [aten.convolution]
        buf612 = extern_kernels.convolution(buf611, primals_289, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf612, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf613 = buf584; del buf584  # reuse
        buf614 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf616 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_219], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf612, primals_753, primals_754, buf613, buf614, buf616, primals_753, primals_754, 256, 2048, grid=grid(256), stream=stream0)
        del primals_753
        del primals_754
        buf617 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_219, out_220], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_55.run(buf612, buf613, buf614, primals_290, primals_291, buf617, 524288, grid=grid(524288), stream=stream0)
        del primals_291
        # Source Nodes: [x_150], Original ATen: [aten.convolution]
        buf618 = extern_kernels.convolution(buf617, primals_292, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf618, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf619 = buf590; del buf590  # reuse
        buf620 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf622 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf618, primals_756, primals_757, buf619, buf620, buf622, primals_756, primals_757, 512, 2048, grid=grid(512), stream=stream0)
        del primals_756
        del primals_757
        buf623 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_151, x_153], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf618, buf619, buf620, primals_293, primals_294, buf623, 1048576, grid=grid(1048576), stream=stream0)
        del primals_294
        buf624 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf625 = reinterpret_tensor(buf624, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf624  # reuse
        # Source Nodes: [x_gap_90, x_gap_91], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_58.run(buf625, buf623, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_92], Original ATen: [aten.convolution]
        buf626 = extern_kernels.convolution(buf625, primals_295, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf626, (8, 128, 1, 1), (128, 1, 1, 1))
        buf627 = buf626; del buf626  # reuse
        # Source Nodes: [x_gap_92], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf627, primals_296, 1024, grid=grid(1024), stream=stream0)
        del primals_296
        buf628 = buf599; del buf599  # reuse
        buf629 = buf598; del buf598  # reuse
        # Source Nodes: [x_gap_93], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf627, primals_759, primals_760, buf628, buf629, primals_759, primals_760, 128, 8, grid=grid(128), stream=stream0)
        del primals_759
        del primals_760
        buf631 = reinterpret_tensor(buf608, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf608  # reuse
        # Source Nodes: [x_gap_93, x_gap_94], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf627, buf628, buf629, primals_297, primals_298, buf631, 1024, grid=grid(1024), stream=stream0)
        del primals_298
        # Source Nodes: [x_attn_36], Original ATen: [aten.convolution]
        buf632 = extern_kernels.convolution(buf631, primals_299, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf632, (8, 512, 1, 1), (512, 1, 1, 1))
        buf633 = buf603; del buf603  # reuse
        # Source Nodes: [x_156], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_47.run(buf632, primals_300, buf633, 4096, grid=grid(4096), stream=stream0)
        del primals_300
        buf634 = reinterpret_tensor(buf632, (8, 2, 1, 256), (512, 256, 256, 1), 0); del buf632  # reuse
        # Source Nodes: [x_156], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_48.run(buf633, buf634, 4096, grid=grid(4096), stream=stream0)
        buf635 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_18, out_221], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_59.run(buf623, buf634, buf635, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [out_226], Original ATen: [aten.convolution]
        buf636 = extern_kernels.convolution(buf635, primals_301, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf636, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf637 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf638 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf640 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_227], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf636, primals_762, primals_763, buf637, buf638, buf640, primals_762, primals_763, 1024, 2048, grid=grid(1024), stream=stream0)
        del primals_762
        del primals_763
        buf641 = empty((8, 1024, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_227, out_228, shortcut_22], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf636, buf637, buf638, primals_302, primals_303, buf611, buf641, 2097152, grid=grid(2097152), stream=stream0)
        del primals_303
        # Source Nodes: [out_230], Original ATen: [aten.convolution]
        buf642 = extern_kernels.convolution(buf641, primals_304, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf642, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf643 = buf614; del buf614  # reuse
        buf644 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf646 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_231], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf642, primals_765, primals_766, buf643, buf644, buf646, primals_765, primals_766, 256, 2048, grid=grid(256), stream=stream0)
        del primals_765
        del primals_766
        buf647 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_231, out_232], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_55.run(buf642, buf643, buf644, primals_305, primals_306, buf647, 524288, grid=grid(524288), stream=stream0)
        del primals_306
        # Source Nodes: [x_158], Original ATen: [aten.convolution]
        buf648 = extern_kernels.convolution(buf647, primals_307, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf648, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf649 = buf620; del buf620  # reuse
        buf650 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf652 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_159], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf648, primals_768, primals_769, buf649, buf650, buf652, primals_768, primals_769, 512, 2048, grid=grid(512), stream=stream0)
        del primals_768
        del primals_769
        buf653 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_159, x_161], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf648, buf649, buf650, primals_308, primals_309, buf653, 1048576, grid=grid(1048576), stream=stream0)
        del primals_309
        buf654 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf655 = reinterpret_tensor(buf654, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf654  # reuse
        # Source Nodes: [x_gap_95, x_gap_96], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_58.run(buf655, buf653, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_97], Original ATen: [aten.convolution]
        buf656 = extern_kernels.convolution(buf655, primals_310, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf656, (8, 128, 1, 1), (128, 1, 1, 1))
        buf657 = buf656; del buf656  # reuse
        # Source Nodes: [x_gap_97], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf657, primals_311, 1024, grid=grid(1024), stream=stream0)
        del primals_311
        buf658 = buf629; del buf629  # reuse
        buf659 = buf628; del buf628  # reuse
        # Source Nodes: [x_gap_98], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf657, primals_771, primals_772, buf658, buf659, primals_771, primals_772, 128, 8, grid=grid(128), stream=stream0)
        del primals_771
        del primals_772
        buf661 = reinterpret_tensor(buf638, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf638  # reuse
        # Source Nodes: [x_gap_98, x_gap_99], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf657, buf658, buf659, primals_312, primals_313, buf661, 1024, grid=grid(1024), stream=stream0)
        del primals_313
        # Source Nodes: [x_attn_38], Original ATen: [aten.convolution]
        buf662 = extern_kernels.convolution(buf661, primals_314, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf662, (8, 512, 1, 1), (512, 1, 1, 1))
        buf663 = buf633; del buf633  # reuse
        # Source Nodes: [x_164], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_47.run(buf662, primals_315, buf663, 4096, grid=grid(4096), stream=stream0)
        del primals_315
        buf664 = reinterpret_tensor(buf662, (8, 2, 1, 256), (512, 256, 256, 1), 0); del buf662  # reuse
        # Source Nodes: [x_164], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_48.run(buf663, buf664, 4096, grid=grid(4096), stream=stream0)
        buf665 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_19, out_233], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_59.run(buf653, buf664, buf665, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [out_238], Original ATen: [aten.convolution]
        buf666 = extern_kernels.convolution(buf665, primals_316, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf666, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf667 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf668 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf670 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_239], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf666, primals_774, primals_775, buf667, buf668, buf670, primals_774, primals_775, 1024, 2048, grid=grid(1024), stream=stream0)
        del primals_774
        del primals_775
        buf671 = empty((8, 1024, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_239, out_240, shortcut_23], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf666, buf667, buf668, primals_317, primals_318, buf641, buf671, 2097152, grid=grid(2097152), stream=stream0)
        del primals_318
        # Source Nodes: [out_242], Original ATen: [aten.convolution]
        buf672 = extern_kernels.convolution(buf671, primals_319, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf672, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf673 = buf644; del buf644  # reuse
        buf674 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf676 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_243], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf672, primals_777, primals_778, buf673, buf674, buf676, primals_777, primals_778, 256, 2048, grid=grid(256), stream=stream0)
        del primals_777
        del primals_778
        buf677 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_243, out_244], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_55.run(buf672, buf673, buf674, primals_320, primals_321, buf677, 524288, grid=grid(524288), stream=stream0)
        del primals_321
        # Source Nodes: [x_166], Original ATen: [aten.convolution]
        buf678 = extern_kernels.convolution(buf677, primals_322, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf678, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf679 = buf650; del buf650  # reuse
        buf680 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf682 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_167], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf678, primals_780, primals_781, buf679, buf680, buf682, primals_780, primals_781, 512, 2048, grid=grid(512), stream=stream0)
        del primals_780
        del primals_781
        buf683 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_167, x_169], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf678, buf679, buf680, primals_323, primals_324, buf683, 1048576, grid=grid(1048576), stream=stream0)
        del primals_324
        buf684 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf685 = reinterpret_tensor(buf684, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf684  # reuse
        # Source Nodes: [x_gap_100, x_gap_101], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_58.run(buf685, buf683, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_102], Original ATen: [aten.convolution]
        buf686 = extern_kernels.convolution(buf685, primals_325, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf686, (8, 128, 1, 1), (128, 1, 1, 1))
        buf687 = buf686; del buf686  # reuse
        # Source Nodes: [x_gap_102], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf687, primals_326, 1024, grid=grid(1024), stream=stream0)
        del primals_326
        buf688 = buf659; del buf659  # reuse
        buf689 = buf658; del buf658  # reuse
        # Source Nodes: [x_gap_103], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf687, primals_783, primals_784, buf688, buf689, primals_783, primals_784, 128, 8, grid=grid(128), stream=stream0)
        del primals_783
        del primals_784
        buf691 = reinterpret_tensor(buf668, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf668  # reuse
        # Source Nodes: [x_gap_103, x_gap_104], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf687, buf688, buf689, primals_327, primals_328, buf691, 1024, grid=grid(1024), stream=stream0)
        del primals_328
        # Source Nodes: [x_attn_40], Original ATen: [aten.convolution]
        buf692 = extern_kernels.convolution(buf691, primals_329, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf692, (8, 512, 1, 1), (512, 1, 1, 1))
        buf693 = buf663; del buf663  # reuse
        # Source Nodes: [x_172], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_47.run(buf692, primals_330, buf693, 4096, grid=grid(4096), stream=stream0)
        del primals_330
        buf694 = reinterpret_tensor(buf692, (8, 2, 1, 256), (512, 256, 256, 1), 0); del buf692  # reuse
        # Source Nodes: [x_172], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_48.run(buf693, buf694, 4096, grid=grid(4096), stream=stream0)
        buf695 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_20, out_245], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_59.run(buf683, buf694, buf695, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [out_250], Original ATen: [aten.convolution]
        buf696 = extern_kernels.convolution(buf695, primals_331, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf696, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf697 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf698 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf700 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_251], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf696, primals_786, primals_787, buf697, buf698, buf700, primals_786, primals_787, 1024, 2048, grid=grid(1024), stream=stream0)
        del primals_786
        del primals_787
        buf701 = empty((8, 1024, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_251, out_252, shortcut_24], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf696, buf697, buf698, primals_332, primals_333, buf671, buf701, 2097152, grid=grid(2097152), stream=stream0)
        del primals_333
        # Source Nodes: [out_254], Original ATen: [aten.convolution]
        buf702 = extern_kernels.convolution(buf701, primals_334, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf702, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf703 = buf674; del buf674  # reuse
        buf704 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf706 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_255], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf702, primals_789, primals_790, buf703, buf704, buf706, primals_789, primals_790, 256, 2048, grid=grid(256), stream=stream0)
        del primals_789
        del primals_790
        buf707 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_255, out_256], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_55.run(buf702, buf703, buf704, primals_335, primals_336, buf707, 524288, grid=grid(524288), stream=stream0)
        del primals_336
        # Source Nodes: [x_174], Original ATen: [aten.convolution]
        buf708 = extern_kernels.convolution(buf707, primals_337, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf708, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf709 = buf680; del buf680  # reuse
        buf710 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf712 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_175], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf708, primals_792, primals_793, buf709, buf710, buf712, primals_792, primals_793, 512, 2048, grid=grid(512), stream=stream0)
        del primals_792
        del primals_793
        buf713 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_175, x_177], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf708, buf709, buf710, primals_338, primals_339, buf713, 1048576, grid=grid(1048576), stream=stream0)
        del primals_339
        buf714 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf715 = reinterpret_tensor(buf714, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf714  # reuse
        # Source Nodes: [x_gap_105, x_gap_106], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_58.run(buf715, buf713, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_107], Original ATen: [aten.convolution]
        buf716 = extern_kernels.convolution(buf715, primals_340, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf716, (8, 128, 1, 1), (128, 1, 1, 1))
        buf717 = buf716; del buf716  # reuse
        # Source Nodes: [x_gap_107], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf717, primals_341, 1024, grid=grid(1024), stream=stream0)
        del primals_341
        buf718 = buf689; del buf689  # reuse
        buf719 = buf688; del buf688  # reuse
        # Source Nodes: [x_gap_108], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf717, primals_795, primals_796, buf718, buf719, primals_795, primals_796, 128, 8, grid=grid(128), stream=stream0)
        del primals_795
        del primals_796
        buf721 = reinterpret_tensor(buf698, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf698  # reuse
        # Source Nodes: [x_gap_108, x_gap_109], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf717, buf718, buf719, primals_342, primals_343, buf721, 1024, grid=grid(1024), stream=stream0)
        del primals_343
        # Source Nodes: [x_attn_42], Original ATen: [aten.convolution]
        buf722 = extern_kernels.convolution(buf721, primals_344, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf722, (8, 512, 1, 1), (512, 1, 1, 1))
        buf723 = buf693; del buf693  # reuse
        # Source Nodes: [x_180], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_47.run(buf722, primals_345, buf723, 4096, grid=grid(4096), stream=stream0)
        del primals_345
        buf724 = reinterpret_tensor(buf722, (8, 2, 1, 256), (512, 256, 256, 1), 0); del buf722  # reuse
        # Source Nodes: [x_180], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_48.run(buf723, buf724, 4096, grid=grid(4096), stream=stream0)
        buf725 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_21, out_257], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_59.run(buf713, buf724, buf725, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [out_262], Original ATen: [aten.convolution]
        buf726 = extern_kernels.convolution(buf725, primals_346, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf726, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf727 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf728 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf730 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_263], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf726, primals_798, primals_799, buf727, buf728, buf730, primals_798, primals_799, 1024, 2048, grid=grid(1024), stream=stream0)
        del primals_798
        del primals_799
        buf731 = empty((8, 1024, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_263, out_264, shortcut_25], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf726, buf727, buf728, primals_347, primals_348, buf701, buf731, 2097152, grid=grid(2097152), stream=stream0)
        del primals_348
        # Source Nodes: [out_266], Original ATen: [aten.convolution]
        buf732 = extern_kernels.convolution(buf731, primals_349, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf732, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf733 = buf704; del buf704  # reuse
        buf734 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf736 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_267], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf732, primals_801, primals_802, buf733, buf734, buf736, primals_801, primals_802, 256, 2048, grid=grid(256), stream=stream0)
        del primals_801
        del primals_802
        buf737 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_267, out_268], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_55.run(buf732, buf733, buf734, primals_350, primals_351, buf737, 524288, grid=grid(524288), stream=stream0)
        del primals_351
        # Source Nodes: [x_182], Original ATen: [aten.convolution]
        buf738 = extern_kernels.convolution(buf737, primals_352, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf738, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf739 = buf710; del buf710  # reuse
        buf740 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf742 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_183], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf738, primals_804, primals_805, buf739, buf740, buf742, primals_804, primals_805, 512, 2048, grid=grid(512), stream=stream0)
        del primals_804
        del primals_805
        buf743 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_183, x_185], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf738, buf739, buf740, primals_353, primals_354, buf743, 1048576, grid=grid(1048576), stream=stream0)
        del primals_354
        buf744 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf745 = reinterpret_tensor(buf744, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf744  # reuse
        # Source Nodes: [x_gap_110, x_gap_111], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_58.run(buf745, buf743, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_112], Original ATen: [aten.convolution]
        buf746 = extern_kernels.convolution(buf745, primals_355, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf746, (8, 128, 1, 1), (128, 1, 1, 1))
        buf747 = buf746; del buf746  # reuse
        # Source Nodes: [x_gap_112], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf747, primals_356, 1024, grid=grid(1024), stream=stream0)
        del primals_356
        buf748 = buf719; del buf719  # reuse
        buf749 = buf718; del buf718  # reuse
        # Source Nodes: [x_gap_113], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf747, primals_807, primals_808, buf748, buf749, primals_807, primals_808, 128, 8, grid=grid(128), stream=stream0)
        del primals_807
        del primals_808
        buf751 = reinterpret_tensor(buf728, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf728  # reuse
        # Source Nodes: [x_gap_113, x_gap_114], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf747, buf748, buf749, primals_357, primals_358, buf751, 1024, grid=grid(1024), stream=stream0)
        del primals_358
        # Source Nodes: [x_attn_44], Original ATen: [aten.convolution]
        buf752 = extern_kernels.convolution(buf751, primals_359, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf752, (8, 512, 1, 1), (512, 1, 1, 1))
        buf753 = buf723; del buf723  # reuse
        # Source Nodes: [x_188], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_47.run(buf752, primals_360, buf753, 4096, grid=grid(4096), stream=stream0)
        del primals_360
        buf754 = reinterpret_tensor(buf752, (8, 2, 1, 256), (512, 256, 256, 1), 0); del buf752  # reuse
        # Source Nodes: [x_188], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_48.run(buf753, buf754, 4096, grid=grid(4096), stream=stream0)
        buf755 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_22, out_269], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_59.run(buf743, buf754, buf755, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [out_274], Original ATen: [aten.convolution]
        buf756 = extern_kernels.convolution(buf755, primals_361, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf756, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf757 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf758 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf760 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_275], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf756, primals_810, primals_811, buf757, buf758, buf760, primals_810, primals_811, 1024, 2048, grid=grid(1024), stream=stream0)
        del primals_810
        del primals_811
        buf761 = empty((8, 1024, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_275, out_276, shortcut_26], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf756, buf757, buf758, primals_362, primals_363, buf731, buf761, 2097152, grid=grid(2097152), stream=stream0)
        del primals_363
        # Source Nodes: [out_278], Original ATen: [aten.convolution]
        buf762 = extern_kernels.convolution(buf761, primals_364, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf762, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf763 = buf734; del buf734  # reuse
        buf764 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf766 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_279], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf762, primals_813, primals_814, buf763, buf764, buf766, primals_813, primals_814, 256, 2048, grid=grid(256), stream=stream0)
        del primals_813
        del primals_814
        buf767 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_279, out_280], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_55.run(buf762, buf763, buf764, primals_365, primals_366, buf767, 524288, grid=grid(524288), stream=stream0)
        del primals_366
        # Source Nodes: [x_190], Original ATen: [aten.convolution]
        buf768 = extern_kernels.convolution(buf767, primals_367, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf768, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf769 = buf740; del buf740  # reuse
        buf770 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf772 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_191], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf768, primals_816, primals_817, buf769, buf770, buf772, primals_816, primals_817, 512, 2048, grid=grid(512), stream=stream0)
        del primals_816
        del primals_817
        buf773 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_191, x_193], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf768, buf769, buf770, primals_368, primals_369, buf773, 1048576, grid=grid(1048576), stream=stream0)
        del primals_369
        buf774 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf775 = reinterpret_tensor(buf774, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf774  # reuse
        # Source Nodes: [x_gap_115, x_gap_116], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_58.run(buf775, buf773, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_117], Original ATen: [aten.convolution]
        buf776 = extern_kernels.convolution(buf775, primals_370, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf776, (8, 128, 1, 1), (128, 1, 1, 1))
        buf777 = buf776; del buf776  # reuse
        # Source Nodes: [x_gap_117], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf777, primals_371, 1024, grid=grid(1024), stream=stream0)
        del primals_371
        buf778 = buf749; del buf749  # reuse
        buf779 = buf748; del buf748  # reuse
        # Source Nodes: [x_gap_118], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf777, primals_819, primals_820, buf778, buf779, primals_819, primals_820, 128, 8, grid=grid(128), stream=stream0)
        del primals_819
        del primals_820
        buf781 = reinterpret_tensor(buf758, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf758  # reuse
        # Source Nodes: [x_gap_118, x_gap_119], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf777, buf778, buf779, primals_372, primals_373, buf781, 1024, grid=grid(1024), stream=stream0)
        del primals_373
        # Source Nodes: [x_attn_46], Original ATen: [aten.convolution]
        buf782 = extern_kernels.convolution(buf781, primals_374, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf782, (8, 512, 1, 1), (512, 1, 1, 1))
        buf783 = buf753; del buf753  # reuse
        # Source Nodes: [x_196], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_47.run(buf782, primals_375, buf783, 4096, grid=grid(4096), stream=stream0)
        del primals_375
        buf784 = reinterpret_tensor(buf782, (8, 2, 1, 256), (512, 256, 256, 1), 0); del buf782  # reuse
        # Source Nodes: [x_196], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_48.run(buf783, buf784, 4096, grid=grid(4096), stream=stream0)
        buf785 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_23, out_281], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_59.run(buf773, buf784, buf785, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [out_286], Original ATen: [aten.convolution]
        buf786 = extern_kernels.convolution(buf785, primals_376, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf786, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf787 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf788 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf790 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_287], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf786, primals_822, primals_823, buf787, buf788, buf790, primals_822, primals_823, 1024, 2048, grid=grid(1024), stream=stream0)
        del primals_822
        del primals_823
        buf791 = empty((8, 1024, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_287, out_288, shortcut_27], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf786, buf787, buf788, primals_377, primals_378, buf761, buf791, 2097152, grid=grid(2097152), stream=stream0)
        del primals_378
        # Source Nodes: [out_290], Original ATen: [aten.convolution]
        buf792 = extern_kernels.convolution(buf791, primals_379, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf792, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf793 = buf764; del buf764  # reuse
        buf794 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf796 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_291], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf792, primals_825, primals_826, buf793, buf794, buf796, primals_825, primals_826, 256, 2048, grid=grid(256), stream=stream0)
        del primals_825
        del primals_826
        buf797 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_291, out_292], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_55.run(buf792, buf793, buf794, primals_380, primals_381, buf797, 524288, grid=grid(524288), stream=stream0)
        del primals_381
        # Source Nodes: [x_198], Original ATen: [aten.convolution]
        buf798 = extern_kernels.convolution(buf797, primals_382, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf798, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf799 = buf770; del buf770  # reuse
        buf800 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf802 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_199], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf798, primals_828, primals_829, buf799, buf800, buf802, primals_828, primals_829, 512, 2048, grid=grid(512), stream=stream0)
        del primals_828
        del primals_829
        buf803 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_199, x_201], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf798, buf799, buf800, primals_383, primals_384, buf803, 1048576, grid=grid(1048576), stream=stream0)
        del primals_384
        buf804 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf805 = reinterpret_tensor(buf804, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf804  # reuse
        # Source Nodes: [x_gap_120, x_gap_121], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_58.run(buf805, buf803, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_122], Original ATen: [aten.convolution]
        buf806 = extern_kernels.convolution(buf805, primals_385, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf806, (8, 128, 1, 1), (128, 1, 1, 1))
        buf807 = buf806; del buf806  # reuse
        # Source Nodes: [x_gap_122], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf807, primals_386, 1024, grid=grid(1024), stream=stream0)
        del primals_386
        buf808 = buf779; del buf779  # reuse
        buf809 = buf778; del buf778  # reuse
        # Source Nodes: [x_gap_123], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf807, primals_831, primals_832, buf808, buf809, primals_831, primals_832, 128, 8, grid=grid(128), stream=stream0)
        del primals_831
        del primals_832
        buf811 = reinterpret_tensor(buf788, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf788  # reuse
        # Source Nodes: [x_gap_123, x_gap_124], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf807, buf808, buf809, primals_387, primals_388, buf811, 1024, grid=grid(1024), stream=stream0)
        del primals_388
        # Source Nodes: [x_attn_48], Original ATen: [aten.convolution]
        buf812 = extern_kernels.convolution(buf811, primals_389, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf812, (8, 512, 1, 1), (512, 1, 1, 1))
        buf813 = buf783; del buf783  # reuse
        # Source Nodes: [x_204], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_47.run(buf812, primals_390, buf813, 4096, grid=grid(4096), stream=stream0)
        del primals_390
        buf814 = reinterpret_tensor(buf812, (8, 2, 1, 256), (512, 256, 256, 1), 0); del buf812  # reuse
        # Source Nodes: [x_204], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_48.run(buf813, buf814, 4096, grid=grid(4096), stream=stream0)
        buf815 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_24, out_293], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_59.run(buf803, buf814, buf815, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [out_298], Original ATen: [aten.convolution]
        buf816 = extern_kernels.convolution(buf815, primals_391, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf816, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf817 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf818 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf820 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_299], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf816, primals_834, primals_835, buf817, buf818, buf820, primals_834, primals_835, 1024, 2048, grid=grid(1024), stream=stream0)
        del primals_834
        del primals_835
        buf821 = empty((8, 1024, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_299, out_300, shortcut_28], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf816, buf817, buf818, primals_392, primals_393, buf791, buf821, 2097152, grid=grid(2097152), stream=stream0)
        del primals_393
        # Source Nodes: [out_302], Original ATen: [aten.convolution]
        buf822 = extern_kernels.convolution(buf821, primals_394, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf822, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf823 = buf794; del buf794  # reuse
        buf824 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf826 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_303], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf822, primals_837, primals_838, buf823, buf824, buf826, primals_837, primals_838, 256, 2048, grid=grid(256), stream=stream0)
        del primals_837
        del primals_838
        buf827 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_303, out_304], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_55.run(buf822, buf823, buf824, primals_395, primals_396, buf827, 524288, grid=grid(524288), stream=stream0)
        del primals_396
        # Source Nodes: [x_206], Original ATen: [aten.convolution]
        buf828 = extern_kernels.convolution(buf827, primals_397, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf828, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf829 = buf800; del buf800  # reuse
        buf830 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf832 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_207], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf828, primals_840, primals_841, buf829, buf830, buf832, primals_840, primals_841, 512, 2048, grid=grid(512), stream=stream0)
        del primals_840
        del primals_841
        buf833 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_207, x_209], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf828, buf829, buf830, primals_398, primals_399, buf833, 1048576, grid=grid(1048576), stream=stream0)
        del primals_399
        buf834 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf835 = reinterpret_tensor(buf834, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf834  # reuse
        # Source Nodes: [x_gap_125, x_gap_126], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_58.run(buf835, buf833, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_127], Original ATen: [aten.convolution]
        buf836 = extern_kernels.convolution(buf835, primals_400, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf836, (8, 128, 1, 1), (128, 1, 1, 1))
        buf837 = buf836; del buf836  # reuse
        # Source Nodes: [x_gap_127], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf837, primals_401, 1024, grid=grid(1024), stream=stream0)
        del primals_401
        buf838 = buf809; del buf809  # reuse
        buf839 = buf808; del buf808  # reuse
        # Source Nodes: [x_gap_128], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf837, primals_843, primals_844, buf838, buf839, primals_843, primals_844, 128, 8, grid=grid(128), stream=stream0)
        del primals_843
        del primals_844
        buf841 = reinterpret_tensor(buf818, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf818  # reuse
        # Source Nodes: [x_gap_128, x_gap_129], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf837, buf838, buf839, primals_402, primals_403, buf841, 1024, grid=grid(1024), stream=stream0)
        del primals_403
        # Source Nodes: [x_attn_50], Original ATen: [aten.convolution]
        buf842 = extern_kernels.convolution(buf841, primals_404, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf842, (8, 512, 1, 1), (512, 1, 1, 1))
        buf843 = buf813; del buf813  # reuse
        # Source Nodes: [x_212], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_47.run(buf842, primals_405, buf843, 4096, grid=grid(4096), stream=stream0)
        del primals_405
        buf844 = reinterpret_tensor(buf842, (8, 2, 1, 256), (512, 256, 256, 1), 0); del buf842  # reuse
        # Source Nodes: [x_212], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_48.run(buf843, buf844, 4096, grid=grid(4096), stream=stream0)
        buf845 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_25, out_305], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_59.run(buf833, buf844, buf845, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [out_310], Original ATen: [aten.convolution]
        buf846 = extern_kernels.convolution(buf845, primals_406, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf846, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf847 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf848 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf850 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_311], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf846, primals_846, primals_847, buf847, buf848, buf850, primals_846, primals_847, 1024, 2048, grid=grid(1024), stream=stream0)
        del primals_846
        del primals_847
        buf851 = empty((8, 1024, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_311, out_312, shortcut_29], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf846, buf847, buf848, primals_407, primals_408, buf821, buf851, 2097152, grid=grid(2097152), stream=stream0)
        del primals_408
        # Source Nodes: [out_314], Original ATen: [aten.convolution]
        buf852 = extern_kernels.convolution(buf851, primals_409, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf852, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf853 = buf824; del buf824  # reuse
        buf854 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf856 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_315], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf852, primals_849, primals_850, buf853, buf854, buf856, primals_849, primals_850, 256, 2048, grid=grid(256), stream=stream0)
        del primals_849
        del primals_850
        buf857 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_315, out_316], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_55.run(buf852, buf853, buf854, primals_410, primals_411, buf857, 524288, grid=grid(524288), stream=stream0)
        del primals_411
        # Source Nodes: [x_214], Original ATen: [aten.convolution]
        buf858 = extern_kernels.convolution(buf857, primals_412, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf858, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf859 = buf830; del buf830  # reuse
        buf860 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf862 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_215], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf858, primals_852, primals_853, buf859, buf860, buf862, primals_852, primals_853, 512, 2048, grid=grid(512), stream=stream0)
        del primals_852
        del primals_853
        buf863 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_215, x_217], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf858, buf859, buf860, primals_413, primals_414, buf863, 1048576, grid=grid(1048576), stream=stream0)
        del primals_414
        buf864 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf865 = reinterpret_tensor(buf864, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf864  # reuse
        # Source Nodes: [x_gap_130, x_gap_131], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_58.run(buf865, buf863, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_132], Original ATen: [aten.convolution]
        buf866 = extern_kernels.convolution(buf865, primals_415, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf866, (8, 128, 1, 1), (128, 1, 1, 1))
        buf867 = buf866; del buf866  # reuse
        # Source Nodes: [x_gap_132], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf867, primals_416, 1024, grid=grid(1024), stream=stream0)
        del primals_416
        buf868 = buf839; del buf839  # reuse
        buf869 = buf838; del buf838  # reuse
        # Source Nodes: [x_gap_133], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf867, primals_855, primals_856, buf868, buf869, primals_855, primals_856, 128, 8, grid=grid(128), stream=stream0)
        del primals_855
        del primals_856
        buf871 = reinterpret_tensor(buf848, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf848  # reuse
        # Source Nodes: [x_gap_133, x_gap_134], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf867, buf868, buf869, primals_417, primals_418, buf871, 1024, grid=grid(1024), stream=stream0)
        del primals_418
        # Source Nodes: [x_attn_52], Original ATen: [aten.convolution]
        buf872 = extern_kernels.convolution(buf871, primals_419, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf872, (8, 512, 1, 1), (512, 1, 1, 1))
        buf873 = buf843; del buf843  # reuse
        # Source Nodes: [x_220], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_47.run(buf872, primals_420, buf873, 4096, grid=grid(4096), stream=stream0)
        del primals_420
        buf874 = reinterpret_tensor(buf872, (8, 2, 1, 256), (512, 256, 256, 1), 0); del buf872  # reuse
        # Source Nodes: [x_220], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_48.run(buf873, buf874, 4096, grid=grid(4096), stream=stream0)
        buf875 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_26, out_317], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_59.run(buf863, buf874, buf875, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [out_322], Original ATen: [aten.convolution]
        buf876 = extern_kernels.convolution(buf875, primals_421, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf876, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf877 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf878 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf880 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_323], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf876, primals_858, primals_859, buf877, buf878, buf880, primals_858, primals_859, 1024, 2048, grid=grid(1024), stream=stream0)
        del primals_858
        del primals_859
        buf881 = empty((8, 1024, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_323, out_324, shortcut_30], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf876, buf877, buf878, primals_422, primals_423, buf851, buf881, 2097152, grid=grid(2097152), stream=stream0)
        del primals_423
        # Source Nodes: [out_326], Original ATen: [aten.convolution]
        buf882 = extern_kernels.convolution(buf881, primals_424, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf882, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf883 = buf854; del buf854  # reuse
        buf884 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf886 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_327], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf882, primals_861, primals_862, buf883, buf884, buf886, primals_861, primals_862, 256, 2048, grid=grid(256), stream=stream0)
        del primals_861
        del primals_862
        buf887 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_327, out_328], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_55.run(buf882, buf883, buf884, primals_425, primals_426, buf887, 524288, grid=grid(524288), stream=stream0)
        del primals_426
        # Source Nodes: [x_222], Original ATen: [aten.convolution]
        buf888 = extern_kernels.convolution(buf887, primals_427, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf888, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf889 = buf860; del buf860  # reuse
        buf890 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf892 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_223], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf888, primals_864, primals_865, buf889, buf890, buf892, primals_864, primals_865, 512, 2048, grid=grid(512), stream=stream0)
        del primals_864
        del primals_865
        buf893 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_223, x_225], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf888, buf889, buf890, primals_428, primals_429, buf893, 1048576, grid=grid(1048576), stream=stream0)
        del primals_429
        buf894 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf895 = reinterpret_tensor(buf894, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf894  # reuse
        # Source Nodes: [x_gap_135, x_gap_136], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_58.run(buf895, buf893, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_137], Original ATen: [aten.convolution]
        buf896 = extern_kernels.convolution(buf895, primals_430, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf896, (8, 128, 1, 1), (128, 1, 1, 1))
        buf897 = buf896; del buf896  # reuse
        # Source Nodes: [x_gap_137], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf897, primals_431, 1024, grid=grid(1024), stream=stream0)
        del primals_431
        buf898 = buf869; del buf869  # reuse
        buf899 = buf868; del buf868  # reuse
        # Source Nodes: [x_gap_138], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf897, primals_867, primals_868, buf898, buf899, primals_867, primals_868, 128, 8, grid=grid(128), stream=stream0)
        del primals_867
        del primals_868
        buf901 = reinterpret_tensor(buf878, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf878  # reuse
        # Source Nodes: [x_gap_138, x_gap_139], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf897, buf898, buf899, primals_432, primals_433, buf901, 1024, grid=grid(1024), stream=stream0)
        del primals_433
        # Source Nodes: [x_attn_54], Original ATen: [aten.convolution]
        buf902 = extern_kernels.convolution(buf901, primals_434, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf902, (8, 512, 1, 1), (512, 1, 1, 1))
        buf903 = buf873; del buf873  # reuse
        # Source Nodes: [x_228], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_47.run(buf902, primals_435, buf903, 4096, grid=grid(4096), stream=stream0)
        del primals_435
        buf904 = reinterpret_tensor(buf902, (8, 2, 1, 256), (512, 256, 256, 1), 0); del buf902  # reuse
        # Source Nodes: [x_228], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_48.run(buf903, buf904, 4096, grid=grid(4096), stream=stream0)
        buf905 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_27, out_329], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_59.run(buf893, buf904, buf905, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [out_334], Original ATen: [aten.convolution]
        buf906 = extern_kernels.convolution(buf905, primals_436, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf906, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf907 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf908 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf910 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_335], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf906, primals_870, primals_871, buf907, buf908, buf910, primals_870, primals_871, 1024, 2048, grid=grid(1024), stream=stream0)
        del primals_870
        del primals_871
        buf911 = empty((8, 1024, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_335, out_336, shortcut_31], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf906, buf907, buf908, primals_437, primals_438, buf881, buf911, 2097152, grid=grid(2097152), stream=stream0)
        del primals_438
        # Source Nodes: [out_338], Original ATen: [aten.convolution]
        buf912 = extern_kernels.convolution(buf911, primals_439, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf912, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf913 = buf884; del buf884  # reuse
        buf914 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf916 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_339], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf912, primals_873, primals_874, buf913, buf914, buf916, primals_873, primals_874, 256, 2048, grid=grid(256), stream=stream0)
        del primals_873
        del primals_874
        buf917 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_339, out_340], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_55.run(buf912, buf913, buf914, primals_440, primals_441, buf917, 524288, grid=grid(524288), stream=stream0)
        del primals_441
        # Source Nodes: [x_230], Original ATen: [aten.convolution]
        buf918 = extern_kernels.convolution(buf917, primals_442, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf918, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf919 = buf890; del buf890  # reuse
        buf920 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf922 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_231], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf918, primals_876, primals_877, buf919, buf920, buf922, primals_876, primals_877, 512, 2048, grid=grid(512), stream=stream0)
        del primals_876
        del primals_877
        buf923 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_231, x_233], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf918, buf919, buf920, primals_443, primals_444, buf923, 1048576, grid=grid(1048576), stream=stream0)
        del primals_444
        buf924 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf925 = reinterpret_tensor(buf924, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf924  # reuse
        # Source Nodes: [x_gap_140, x_gap_141], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_58.run(buf925, buf923, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_142], Original ATen: [aten.convolution]
        buf926 = extern_kernels.convolution(buf925, primals_445, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf926, (8, 128, 1, 1), (128, 1, 1, 1))
        buf927 = buf926; del buf926  # reuse
        # Source Nodes: [x_gap_142], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf927, primals_446, 1024, grid=grid(1024), stream=stream0)
        del primals_446
        buf928 = buf899; del buf899  # reuse
        buf929 = buf898; del buf898  # reuse
        # Source Nodes: [x_gap_143], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf927, primals_879, primals_880, buf928, buf929, primals_879, primals_880, 128, 8, grid=grid(128), stream=stream0)
        del primals_879
        del primals_880
        buf931 = reinterpret_tensor(buf908, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf908  # reuse
        # Source Nodes: [x_gap_143, x_gap_144], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf927, buf928, buf929, primals_447, primals_448, buf931, 1024, grid=grid(1024), stream=stream0)
        del primals_448
        # Source Nodes: [x_attn_56], Original ATen: [aten.convolution]
        buf932 = extern_kernels.convolution(buf931, primals_449, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf932, (8, 512, 1, 1), (512, 1, 1, 1))
        buf933 = buf903; del buf903  # reuse
        # Source Nodes: [x_236], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_47.run(buf932, primals_450, buf933, 4096, grid=grid(4096), stream=stream0)
        del primals_450
        buf934 = reinterpret_tensor(buf932, (8, 2, 1, 256), (512, 256, 256, 1), 0); del buf932  # reuse
        # Source Nodes: [x_236], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_48.run(buf933, buf934, 4096, grid=grid(4096), stream=stream0)
        buf935 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_28, out_341], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_59.run(buf923, buf934, buf935, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [out_346], Original ATen: [aten.convolution]
        buf936 = extern_kernels.convolution(buf935, primals_451, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf936, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf937 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf938 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf940 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_347], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf936, primals_882, primals_883, buf937, buf938, buf940, primals_882, primals_883, 1024, 2048, grid=grid(1024), stream=stream0)
        del primals_882
        del primals_883
        buf941 = empty((8, 1024, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_347, out_348, shortcut_32], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf936, buf937, buf938, primals_452, primals_453, buf911, buf941, 2097152, grid=grid(2097152), stream=stream0)
        del primals_453
        # Source Nodes: [out_350], Original ATen: [aten.convolution]
        buf942 = extern_kernels.convolution(buf941, primals_454, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf942, (8, 256, 16, 16), (65536, 256, 16, 1))
        buf943 = buf914; del buf914  # reuse
        buf944 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf946 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_351], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_54.run(buf942, primals_885, primals_886, buf943, buf944, buf946, primals_885, primals_886, 256, 2048, grid=grid(256), stream=stream0)
        del primals_885
        del primals_886
        buf947 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_351, out_352], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_55.run(buf942, buf943, buf944, primals_455, primals_456, buf947, 524288, grid=grid(524288), stream=stream0)
        del primals_456
        # Source Nodes: [x_238], Original ATen: [aten.convolution]
        buf948 = extern_kernels.convolution(buf947, primals_457, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf948, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf949 = buf920; del buf920  # reuse
        buf950 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf952 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_239], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf948, primals_888, primals_889, buf949, buf950, buf952, primals_888, primals_889, 512, 2048, grid=grid(512), stream=stream0)
        del primals_888
        del primals_889
        buf953 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_239, x_241], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf948, buf949, buf950, primals_458, primals_459, buf953, 1048576, grid=grid(1048576), stream=stream0)
        del primals_459
        buf954 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf955 = reinterpret_tensor(buf954, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf954  # reuse
        # Source Nodes: [x_gap_145, x_gap_146], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_58.run(buf955, buf953, 2048, 256, grid=grid(2048), stream=stream0)
        # Source Nodes: [x_gap_147], Original ATen: [aten.convolution]
        buf956 = extern_kernels.convolution(buf955, primals_460, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf956, (8, 128, 1, 1), (128, 1, 1, 1))
        buf957 = buf956; del buf956  # reuse
        # Source Nodes: [x_gap_147], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_44.run(buf957, primals_461, 1024, grid=grid(1024), stream=stream0)
        del primals_461
        buf958 = buf929; del buf929  # reuse
        buf959 = buf928; del buf928  # reuse
        # Source Nodes: [x_gap_148], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf957, primals_891, primals_892, buf958, buf959, primals_891, primals_892, 128, 8, grid=grid(128), stream=stream0)
        del primals_891
        del primals_892
        buf961 = reinterpret_tensor(buf938, (8, 128, 1, 1), (128, 1, 1, 1), 0); del buf938  # reuse
        # Source Nodes: [x_gap_148, x_gap_149], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_46.run(buf957, buf958, buf959, primals_462, primals_463, buf961, 1024, grid=grid(1024), stream=stream0)
        del buf958
        del buf959
        del primals_463
        # Source Nodes: [x_attn_58], Original ATen: [aten.convolution]
        buf962 = extern_kernels.convolution(buf961, primals_464, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf962, (8, 512, 1, 1), (512, 1, 1, 1))
        buf963 = buf933; del buf933  # reuse
        # Source Nodes: [x_244], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_47.run(buf962, primals_465, buf963, 4096, grid=grid(4096), stream=stream0)
        del primals_465
        buf964 = reinterpret_tensor(buf962, (8, 2, 1, 256), (512, 256, 256, 1), 0); del buf962  # reuse
        # Source Nodes: [x_244], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_48.run(buf963, buf964, 4096, grid=grid(4096), stream=stream0)
        buf965 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_29, out_353], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_59.run(buf953, buf964, buf965, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [out_358], Original ATen: [aten.convolution]
        buf966 = extern_kernels.convolution(buf965, primals_466, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf966, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf967 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf968 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf970 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_359], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf966, primals_894, primals_895, buf967, buf968, buf970, primals_894, primals_895, 1024, 2048, grid=grid(1024), stream=stream0)
        del primals_894
        del primals_895
        buf971 = empty((8, 1024, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_359, out_360, shortcut_33], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_60.run(buf966, buf967, buf968, primals_467, primals_468, buf941, buf971, 2097152, grid=grid(2097152), stream=stream0)
        del primals_468
        # Source Nodes: [out_362], Original ATen: [aten.convolution]
        buf972 = extern_kernels.convolution(buf971, primals_469, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf972, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf973 = buf950; del buf950  # reuse
        buf974 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf976 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_363], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_56.run(buf972, primals_897, primals_898, buf973, buf974, buf976, primals_897, primals_898, 512, 2048, grid=grid(512), stream=stream0)
        del primals_897
        del primals_898
        buf977 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_363, out_364], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_57.run(buf972, buf973, buf974, primals_470, primals_471, buf977, 1048576, grid=grid(1048576), stream=stream0)
        del primals_471
        # Source Nodes: [x_247], Original ATen: [aten.convolution]
        buf978 = extern_kernels.convolution(buf977, primals_472, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf978, (8, 1024, 16, 16), (262144, 256, 16, 1))
        buf979 = buf968; del buf968  # reuse
        buf980 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf982 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_248], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf978, primals_900, primals_901, buf979, buf980, buf982, primals_900, primals_901, 1024, 2048, grid=grid(1024), stream=stream0)
        del primals_900
        del primals_901
        buf983 = empty((8, 1024, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_248, x_250], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_61.run(buf978, buf979, buf980, primals_473, primals_474, buf983, 2097152, grid=grid(2097152), stream=stream0)
        del primals_474
        buf984 = reinterpret_tensor(buf963, (8, 512, 1, 1), (512, 1, 4096, 4096), 0); del buf963  # reuse
        buf985 = reinterpret_tensor(buf984, (8, 512, 1, 1), (512, 1, 1, 1), 0); del buf984  # reuse
        # Source Nodes: [x_gap_150, x_gap_151], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_62.run(buf985, buf983, 4096, 256, grid=grid(4096), stream=stream0)
        # Source Nodes: [x_gap_152], Original ATen: [aten.convolution]
        buf986 = extern_kernels.convolution(buf985, primals_475, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf986, (8, 256, 1, 1), (256, 1, 1, 1))
        buf987 = buf986; del buf986  # reuse
        # Source Nodes: [x_gap_152], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_63.run(buf987, primals_476, 2048, grid=grid(2048), stream=stream0)
        del primals_476
        buf988 = buf944; del buf944  # reuse
        buf989 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_153], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_64.run(buf987, primals_903, primals_904, buf988, buf989, primals_903, primals_904, 256, 8, grid=grid(256), stream=stream0)
        del primals_903
        del primals_904
        buf991 = empty((8, 256, 1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_gap_153, x_gap_154], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_65.run(buf987, buf988, buf989, primals_477, primals_478, buf991, 2048, grid=grid(2048), stream=stream0)
        del primals_478
        # Source Nodes: [x_attn_60], Original ATen: [aten.convolution]
        buf992 = extern_kernels.convolution(buf991, primals_479, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf992, (8, 1024, 1, 1), (1024, 1, 1, 1))
        buf993 = empty_strided((8, 2, 1, 512), (1024, 512, 8192, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_253], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_66.run(buf992, primals_480, buf993, 8192, grid=grid(8192), stream=stream0)
        del primals_480
        buf994 = reinterpret_tensor(buf992, (8, 2, 1, 512), (1024, 512, 512, 1), 0); del buf992  # reuse
        # Source Nodes: [x_253], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_67.run(buf993, buf994, 8192, grid=grid(8192), stream=stream0)
        buf995 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_30, out_365], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_68.run(buf983, buf994, buf995, 1048576, grid=grid(1048576), stream=stream0)
        buf996 = empty((8, 512, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_370], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_69.run(buf995, buf996, 262144, grid=grid(262144), stream=stream0)
        # Source Nodes: [out_371], Original ATen: [aten.convolution]
        buf997 = extern_kernels.convolution(buf996, primals_481, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf997, (8, 2048, 8, 8), (131072, 64, 8, 1))
        buf998 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf999 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf1001 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_372], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_70.run(buf997, primals_906, primals_907, buf998, buf999, buf1001, primals_906, primals_907, 2048, 512, grid=grid(2048), stream=stream0)
        del primals_906
        del primals_907
        buf1002 = empty((8, 1024, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___layer4___0___downsample_0], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_71.run(buf971, buf1002, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [getattr_l__mod___layer4___0___downsample_1], Original ATen: [aten.convolution]
        buf1003 = extern_kernels.convolution(buf1002, primals_484, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1003, (8, 2048, 8, 8), (131072, 64, 8, 1))
        buf1004 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf1005 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf1007 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_70.run(buf1003, primals_909, primals_910, buf1004, buf1005, buf1007, primals_909, primals_910, 2048, 512, grid=grid(2048), stream=stream0)
        del primals_909
        del primals_910
        buf1008 = empty((8, 2048, 8, 8), device='cuda', dtype=torch.float32)
        buf1009 = buf1008; del buf1008  # reuse
        # Source Nodes: [out_372, out_373, shortcut_34, shortcut_35], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_72.run(buf1009, buf997, buf998, buf999, primals_482, primals_483, buf1003, buf1004, buf1005, primals_485, primals_486, 1048576, grid=grid(1048576), stream=stream0)
        del primals_483
        del primals_486
        # Source Nodes: [out_375], Original ATen: [aten.convolution]
        buf1010 = extern_kernels.convolution(buf1009, primals_487, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1010, (8, 512, 8, 8), (32768, 64, 8, 1))
        buf1011 = buf974; del buf974  # reuse
        buf1012 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf1014 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_376], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_73.run(buf1010, primals_912, primals_913, buf1011, buf1012, buf1014, primals_912, primals_913, 512, 512, grid=grid(512), stream=stream0)
        del primals_912
        del primals_913
        buf1015 = empty((8, 512, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_376, out_377], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_74.run(buf1010, buf1011, buf1012, primals_488, primals_489, buf1015, 262144, grid=grid(262144), stream=stream0)
        del primals_489
        # Source Nodes: [x_255], Original ATen: [aten.convolution]
        buf1016 = extern_kernels.convolution(buf1015, primals_490, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf1016, (8, 1024, 8, 8), (65536, 64, 8, 1))
        buf1017 = buf980; del buf980  # reuse
        buf1018 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1020 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_256], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_75.run(buf1016, primals_915, primals_916, buf1017, buf1018, buf1020, primals_915, primals_916, 1024, 512, grid=grid(1024), stream=stream0)
        del primals_915
        del primals_916
        buf1021 = empty((8, 1024, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_256, x_258], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_76.run(buf1016, buf1017, buf1018, primals_491, primals_492, buf1021, 524288, grid=grid(524288), stream=stream0)
        del primals_492
        buf1022 = empty_strided((8, 512, 1, 1), (512, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf1023 = reinterpret_tensor(buf1022, (8, 512, 1, 1), (512, 1, 1, 1), 0); del buf1022  # reuse
        # Source Nodes: [x_gap_155, x_gap_156], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_77.run(buf1023, buf1021, 4096, 64, grid=grid(4096), stream=stream0)
        # Source Nodes: [x_gap_157], Original ATen: [aten.convolution]
        buf1024 = extern_kernels.convolution(buf1023, primals_493, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1024, (8, 256, 1, 1), (256, 1, 1, 1))
        buf1025 = buf1024; del buf1024  # reuse
        # Source Nodes: [x_gap_157], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_63.run(buf1025, primals_494, 2048, grid=grid(2048), stream=stream0)
        del primals_494
        buf1026 = buf989; del buf989  # reuse
        buf1027 = buf988; del buf988  # reuse
        # Source Nodes: [x_gap_158], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_64.run(buf1025, primals_918, primals_919, buf1026, buf1027, primals_918, primals_919, 256, 8, grid=grid(256), stream=stream0)
        del primals_918
        del primals_919
        buf1029 = reinterpret_tensor(buf999, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf999  # reuse
        # Source Nodes: [x_gap_158, x_gap_159], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_65.run(buf1025, buf1026, buf1027, primals_495, primals_496, buf1029, 2048, grid=grid(2048), stream=stream0)
        del primals_496
        # Source Nodes: [x_attn_62], Original ATen: [aten.convolution]
        buf1030 = extern_kernels.convolution(buf1029, primals_497, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1030, (8, 1024, 1, 1), (1024, 1, 1, 1))
        buf1031 = buf993; del buf993  # reuse
        # Source Nodes: [x_261], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_66.run(buf1030, primals_498, buf1031, 8192, grid=grid(8192), stream=stream0)
        del primals_498
        buf1032 = reinterpret_tensor(buf1030, (8, 2, 1, 512), (1024, 512, 512, 1), 0); del buf1030  # reuse
        # Source Nodes: [x_261], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_67.run(buf1031, buf1032, 8192, grid=grid(8192), stream=stream0)
        buf1033 = empty((8, 512, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_31, out_378], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_78.run(buf1021, buf1032, buf1033, 262144, grid=grid(262144), stream=stream0)
        # Source Nodes: [out_383], Original ATen: [aten.convolution]
        buf1034 = extern_kernels.convolution(buf1033, primals_499, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1034, (8, 2048, 8, 8), (131072, 64, 8, 1))
        buf1035 = buf1005; del buf1005  # reuse
        buf1036 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf1038 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_384], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_70.run(buf1034, primals_921, primals_922, buf1035, buf1036, buf1038, primals_921, primals_922, 2048, 512, grid=grid(2048), stream=stream0)
        del primals_921
        del primals_922
        buf1039 = empty((8, 2048, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_384, out_385, shortcut_36], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_79.run(buf1034, buf1035, buf1036, primals_500, primals_501, buf1009, buf1039, 1048576, grid=grid(1048576), stream=stream0)
        del primals_501
        # Source Nodes: [out_387], Original ATen: [aten.convolution]
        buf1040 = extern_kernels.convolution(buf1039, primals_502, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1040, (8, 512, 8, 8), (32768, 64, 8, 1))
        buf1041 = buf1012; del buf1012  # reuse
        buf1042 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf1044 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_388], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_73.run(buf1040, primals_924, primals_925, buf1041, buf1042, buf1044, primals_924, primals_925, 512, 512, grid=grid(512), stream=stream0)
        del primals_924
        del primals_925
        buf1045 = empty((8, 512, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_388, out_389], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_74.run(buf1040, buf1041, buf1042, primals_503, primals_504, buf1045, 262144, grid=grid(262144), stream=stream0)
        del buf1042
        del primals_504
        # Source Nodes: [x_263], Original ATen: [aten.convolution]
        buf1046 = extern_kernels.convolution(buf1045, primals_505, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=2, bias=None)
        assert_size_stride(buf1046, (8, 1024, 8, 8), (65536, 64, 8, 1))
        buf1047 = buf1018; del buf1018  # reuse
        buf1048 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf1050 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_264], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_75.run(buf1046, primals_927, primals_928, buf1047, buf1048, buf1050, primals_927, primals_928, 1024, 512, grid=grid(1024), stream=stream0)
        del primals_927
        del primals_928
        buf1051 = empty((8, 1024, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_264, x_266], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_76.run(buf1046, buf1047, buf1048, primals_506, primals_507, buf1051, 524288, grid=grid(524288), stream=stream0)
        del buf1048
        del primals_507
        buf1052 = empty_strided((8, 512, 1, 1), (512, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf1053 = reinterpret_tensor(buf1052, (8, 512, 1, 1), (512, 1, 1, 1), 0); del buf1052  # reuse
        # Source Nodes: [x_gap_160, x_gap_161], Original ATen: [aten.mean, aten.sum]
        triton_per_fused_mean_sum_77.run(buf1053, buf1051, 4096, 64, grid=grid(4096), stream=stream0)
        # Source Nodes: [x_gap_162], Original ATen: [aten.convolution]
        buf1054 = extern_kernels.convolution(buf1053, primals_508, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1054, (8, 256, 1, 1), (256, 1, 1, 1))
        buf1055 = buf1054; del buf1054  # reuse
        # Source Nodes: [x_gap_162], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_63.run(buf1055, primals_509, 2048, grid=grid(2048), stream=stream0)
        del primals_509
        buf1056 = buf1027; del buf1027  # reuse
        buf1057 = buf1026; del buf1026  # reuse
        # Source Nodes: [x_gap_163], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_64.run(buf1055, primals_930, primals_931, buf1056, buf1057, primals_930, primals_931, 256, 8, grid=grid(256), stream=stream0)
        del primals_930
        del primals_931
        buf1059 = reinterpret_tensor(buf1036, (8, 256, 1, 1), (256, 1, 1, 1), 0); del buf1036  # reuse
        # Source Nodes: [x_gap_163, x_gap_164], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_65.run(buf1055, buf1056, buf1057, primals_510, primals_511, buf1059, 2048, grid=grid(2048), stream=stream0)
        del buf1056
        del buf1057
        del primals_511
        # Source Nodes: [x_attn_64], Original ATen: [aten.convolution]
        buf1060 = extern_kernels.convolution(buf1059, primals_512, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1060, (8, 1024, 1, 1), (1024, 1, 1, 1))
        buf1061 = buf1031; del buf1031  # reuse
        # Source Nodes: [x_269], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_66.run(buf1060, primals_513, buf1061, 8192, grid=grid(8192), stream=stream0)
        del primals_513
        buf1062 = reinterpret_tensor(buf1060, (8, 2, 1, 512), (1024, 512, 512, 1), 0); del buf1060  # reuse
        # Source Nodes: [x_269], Original ATen: [aten._softmax]
        triton_poi_fused__softmax_67.run(buf1061, buf1062, 8192, grid=grid(8192), stream=stream0)
        del buf1061
        buf1063 = empty((8, 512, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul_32, out_390], Original ATen: [aten.mul, aten.sum]
        triton_poi_fused_mul_sum_78.run(buf1051, buf1062, buf1063, 262144, grid=grid(262144), stream=stream0)
        # Source Nodes: [out_395], Original ATen: [aten.convolution]
        buf1064 = extern_kernels.convolution(buf1063, primals_514, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf1064, (8, 2048, 8, 8), (131072, 64, 8, 1))
        buf1065 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf1066 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf1068 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_396], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_70.run(buf1064, primals_933, primals_934, buf1065, buf1066, buf1068, primals_933, primals_934, 2048, 512, grid=grid(2048), stream=stream0)
        del primals_933
        del primals_934
        buf1073 = empty((8, 2048, 8, 8), device='cuda', dtype=torch.bool)
        buf1070 = empty_strided((8, 2048, 1, 1), (2048, 1, 16384, 16384), device='cuda', dtype=torch.float32)
        buf1071 = reinterpret_tensor(buf1070, (8, 2048), (2048, 1), 0); del buf1070  # reuse
        # Source Nodes: [out_396, out_397, x_272, x_273, x_275], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.mean, aten.relu, aten.threshold_backward, aten.view]
        triton_per_fused__native_batch_norm_legit_functional_add_mean_relu_threshold_backward_view_80.run(buf1071, buf1064, buf1065, buf1066, primals_515, primals_516, buf1039, buf1073, 16384, 64, grid=grid(16384), stream=stream0)
        del buf1066
        del primals_516
        buf1072 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_518, buf1071, reinterpret_tensor(primals_517, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf1072)
        del primals_518
        # Source Nodes: [l__mod___conv1_1], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_521, primals_521, 1, grid=grid(1), stream=stream0)
        del primals_521
        # Source Nodes: [l__mod___conv1_4], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_524, primals_524, 1, grid=grid(1), stream=stream0)
        del primals_524
        # Source Nodes: [x_1], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_527, primals_527, 1, grid=grid(1), stream=stream0)
        del primals_527
        # Source Nodes: [out_1], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_530, primals_530, 1, grid=grid(1), stream=stream0)
        del primals_530
        # Source Nodes: [x_5], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_533, primals_533, 1, grid=grid(1), stream=stream0)
        del primals_533
        # Source Nodes: [x_gap_3], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_536, primals_536, 1, grid=grid(1), stream=stream0)
        del primals_536
        # Source Nodes: [out_9], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_539, primals_539, 1, grid=grid(1), stream=stream0)
        del primals_539
        # Source Nodes: [shortcut_1], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_542, primals_542, 1, grid=grid(1), stream=stream0)
        del primals_542
        # Source Nodes: [out_13], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_545, primals_545, 1, grid=grid(1), stream=stream0)
        del primals_545
        # Source Nodes: [x_13], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_548, primals_548, 1, grid=grid(1), stream=stream0)
        del primals_548
        # Source Nodes: [x_gap_8], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_551, primals_551, 1, grid=grid(1), stream=stream0)
        del primals_551
        # Source Nodes: [out_21], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_554, primals_554, 1, grid=grid(1), stream=stream0)
        del primals_554
        # Source Nodes: [out_25], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_557, primals_557, 1, grid=grid(1), stream=stream0)
        del primals_557
        # Source Nodes: [x_21], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_560, primals_560, 1, grid=grid(1), stream=stream0)
        del primals_560
        # Source Nodes: [x_gap_13], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_563, primals_563, 1, grid=grid(1), stream=stream0)
        del primals_563
        # Source Nodes: [out_33], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_566, primals_566, 1, grid=grid(1), stream=stream0)
        del primals_566
        # Source Nodes: [out_37], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_569, primals_569, 1, grid=grid(1), stream=stream0)
        del primals_569
        # Source Nodes: [x_30], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_572, primals_572, 1, grid=grid(1), stream=stream0)
        del primals_572
        # Source Nodes: [x_gap_18], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_575, primals_575, 1, grid=grid(1), stream=stream0)
        del primals_575
        # Source Nodes: [out_46], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_578, primals_578, 1, grid=grid(1), stream=stream0)
        del primals_578
        # Source Nodes: [shortcut_5], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_581, primals_581, 1, grid=grid(1), stream=stream0)
        del primals_581
        # Source Nodes: [out_50], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_584, primals_584, 1, grid=grid(1), stream=stream0)
        del primals_584
        # Source Nodes: [x_38], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_587, primals_587, 1, grid=grid(1), stream=stream0)
        del primals_587
        # Source Nodes: [x_gap_23], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_590, primals_590, 1, grid=grid(1), stream=stream0)
        del primals_590
        # Source Nodes: [out_58], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_593, primals_593, 1, grid=grid(1), stream=stream0)
        del primals_593
        # Source Nodes: [out_62], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_596, primals_596, 1, grid=grid(1), stream=stream0)
        del primals_596
        # Source Nodes: [x_46], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_599, primals_599, 1, grid=grid(1), stream=stream0)
        del primals_599
        # Source Nodes: [x_gap_28], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_602, primals_602, 1, grid=grid(1), stream=stream0)
        del primals_602
        # Source Nodes: [out_70], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_605, primals_605, 1, grid=grid(1), stream=stream0)
        del primals_605
        # Source Nodes: [out_74], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_608, primals_608, 1, grid=grid(1), stream=stream0)
        del primals_608
        # Source Nodes: [x_54], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_611, primals_611, 1, grid=grid(1), stream=stream0)
        del primals_611
        # Source Nodes: [x_gap_33], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_614, primals_614, 1, grid=grid(1), stream=stream0)
        del primals_614
        # Source Nodes: [out_82], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_617, primals_617, 1, grid=grid(1), stream=stream0)
        del primals_617
        # Source Nodes: [out_86], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_620, primals_620, 1, grid=grid(1), stream=stream0)
        del primals_620
        # Source Nodes: [x_63], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_623, primals_623, 1, grid=grid(1), stream=stream0)
        del primals_623
        # Source Nodes: [x_gap_38], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_626, primals_626, 1, grid=grid(1), stream=stream0)
        del primals_626
        # Source Nodes: [out_95], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_629, primals_629, 1, grid=grid(1), stream=stream0)
        del primals_629
        # Source Nodes: [shortcut_10], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_632, primals_632, 1, grid=grid(1), stream=stream0)
        del primals_632
        # Source Nodes: [out_99], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_635, primals_635, 1, grid=grid(1), stream=stream0)
        del primals_635
        # Source Nodes: [x_71], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_638, primals_638, 1, grid=grid(1), stream=stream0)
        del primals_638
        # Source Nodes: [x_gap_43], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_641, primals_641, 1, grid=grid(1), stream=stream0)
        del primals_641
        # Source Nodes: [out_107], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_644, primals_644, 1, grid=grid(1), stream=stream0)
        del primals_644
        # Source Nodes: [out_111], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_647, primals_647, 1, grid=grid(1), stream=stream0)
        del primals_647
        # Source Nodes: [x_79], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_650, primals_650, 1, grid=grid(1), stream=stream0)
        del primals_650
        # Source Nodes: [x_gap_48], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_653, primals_653, 1, grid=grid(1), stream=stream0)
        del primals_653
        # Source Nodes: [out_119], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_656, primals_656, 1, grid=grid(1), stream=stream0)
        del primals_656
        # Source Nodes: [out_123], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_659, primals_659, 1, grid=grid(1), stream=stream0)
        del primals_659
        # Source Nodes: [x_87], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_662, primals_662, 1, grid=grid(1), stream=stream0)
        del primals_662
        # Source Nodes: [x_gap_53], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_665, primals_665, 1, grid=grid(1), stream=stream0)
        del primals_665
        # Source Nodes: [out_131], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_668, primals_668, 1, grid=grid(1), stream=stream0)
        del primals_668
        # Source Nodes: [out_135], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_671, primals_671, 1, grid=grid(1), stream=stream0)
        del primals_671
        # Source Nodes: [x_95], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_674, primals_674, 1, grid=grid(1), stream=stream0)
        del primals_674
        # Source Nodes: [x_gap_58], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_677, primals_677, 1, grid=grid(1), stream=stream0)
        del primals_677
        # Source Nodes: [out_143], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_680, primals_680, 1, grid=grid(1), stream=stream0)
        del primals_680
        # Source Nodes: [out_147], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_683, primals_683, 1, grid=grid(1), stream=stream0)
        del primals_683
        # Source Nodes: [x_103], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_686, primals_686, 1, grid=grid(1), stream=stream0)
        del primals_686
        # Source Nodes: [x_gap_63], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_689, primals_689, 1, grid=grid(1), stream=stream0)
        del primals_689
        # Source Nodes: [out_155], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_692, primals_692, 1, grid=grid(1), stream=stream0)
        del primals_692
        # Source Nodes: [out_159], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_695, primals_695, 1, grid=grid(1), stream=stream0)
        del primals_695
        # Source Nodes: [x_111], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_698, primals_698, 1, grid=grid(1), stream=stream0)
        del primals_698
        # Source Nodes: [x_gap_68], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_701, primals_701, 1, grid=grid(1), stream=stream0)
        del primals_701
        # Source Nodes: [out_167], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_704, primals_704, 1, grid=grid(1), stream=stream0)
        del primals_704
        # Source Nodes: [out_171], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_707, primals_707, 1, grid=grid(1), stream=stream0)
        del primals_707
        # Source Nodes: [x_119], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_710, primals_710, 1, grid=grid(1), stream=stream0)
        del primals_710
        # Source Nodes: [x_gap_73], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_713, primals_713, 1, grid=grid(1), stream=stream0)
        del primals_713
        # Source Nodes: [out_179], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_716, primals_716, 1, grid=grid(1), stream=stream0)
        del primals_716
        # Source Nodes: [out_183], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_719, primals_719, 1, grid=grid(1), stream=stream0)
        del primals_719
        # Source Nodes: [x_127], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_722, primals_722, 1, grid=grid(1), stream=stream0)
        del primals_722
        # Source Nodes: [x_gap_78], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_725, primals_725, 1, grid=grid(1), stream=stream0)
        del primals_725
        # Source Nodes: [out_191], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_728, primals_728, 1, grid=grid(1), stream=stream0)
        del primals_728
        # Source Nodes: [out_195], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_731, primals_731, 1, grid=grid(1), stream=stream0)
        del primals_731
        # Source Nodes: [x_135], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_734, primals_734, 1, grid=grid(1), stream=stream0)
        del primals_734
        # Source Nodes: [x_gap_83], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_737, primals_737, 1, grid=grid(1), stream=stream0)
        del primals_737
        # Source Nodes: [out_203], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_740, primals_740, 1, grid=grid(1), stream=stream0)
        del primals_740
        # Source Nodes: [out_207], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_743, primals_743, 1, grid=grid(1), stream=stream0)
        del primals_743
        # Source Nodes: [x_143], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_746, primals_746, 1, grid=grid(1), stream=stream0)
        del primals_746
        # Source Nodes: [x_gap_88], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_749, primals_749, 1, grid=grid(1), stream=stream0)
        del primals_749
        # Source Nodes: [out_215], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_752, primals_752, 1, grid=grid(1), stream=stream0)
        del primals_752
        # Source Nodes: [out_219], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_755, primals_755, 1, grid=grid(1), stream=stream0)
        del primals_755
        # Source Nodes: [x_151], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_758, primals_758, 1, grid=grid(1), stream=stream0)
        del primals_758
        # Source Nodes: [x_gap_93], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_761, primals_761, 1, grid=grid(1), stream=stream0)
        del primals_761
        # Source Nodes: [out_227], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_764, primals_764, 1, grid=grid(1), stream=stream0)
        del primals_764
        # Source Nodes: [out_231], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_767, primals_767, 1, grid=grid(1), stream=stream0)
        del primals_767
        # Source Nodes: [x_159], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_770, primals_770, 1, grid=grid(1), stream=stream0)
        del primals_770
        # Source Nodes: [x_gap_98], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_773, primals_773, 1, grid=grid(1), stream=stream0)
        del primals_773
        # Source Nodes: [out_239], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_776, primals_776, 1, grid=grid(1), stream=stream0)
        del primals_776
        # Source Nodes: [out_243], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_779, primals_779, 1, grid=grid(1), stream=stream0)
        del primals_779
        # Source Nodes: [x_167], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_782, primals_782, 1, grid=grid(1), stream=stream0)
        del primals_782
        # Source Nodes: [x_gap_103], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_785, primals_785, 1, grid=grid(1), stream=stream0)
        del primals_785
        # Source Nodes: [out_251], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_788, primals_788, 1, grid=grid(1), stream=stream0)
        del primals_788
        # Source Nodes: [out_255], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_791, primals_791, 1, grid=grid(1), stream=stream0)
        del primals_791
        # Source Nodes: [x_175], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_794, primals_794, 1, grid=grid(1), stream=stream0)
        del primals_794
        # Source Nodes: [x_gap_108], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_797, primals_797, 1, grid=grid(1), stream=stream0)
        del primals_797
        # Source Nodes: [out_263], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_800, primals_800, 1, grid=grid(1), stream=stream0)
        del primals_800
        # Source Nodes: [out_267], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_803, primals_803, 1, grid=grid(1), stream=stream0)
        del primals_803
        # Source Nodes: [x_183], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_806, primals_806, 1, grid=grid(1), stream=stream0)
        del primals_806
        # Source Nodes: [x_gap_113], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_809, primals_809, 1, grid=grid(1), stream=stream0)
        del primals_809
        # Source Nodes: [out_275], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_812, primals_812, 1, grid=grid(1), stream=stream0)
        del primals_812
        # Source Nodes: [out_279], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_815, primals_815, 1, grid=grid(1), stream=stream0)
        del primals_815
        # Source Nodes: [x_191], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_818, primals_818, 1, grid=grid(1), stream=stream0)
        del primals_818
        # Source Nodes: [x_gap_118], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_821, primals_821, 1, grid=grid(1), stream=stream0)
        del primals_821
        # Source Nodes: [out_287], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_824, primals_824, 1, grid=grid(1), stream=stream0)
        del primals_824
        # Source Nodes: [out_291], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_827, primals_827, 1, grid=grid(1), stream=stream0)
        del primals_827
        # Source Nodes: [x_199], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_830, primals_830, 1, grid=grid(1), stream=stream0)
        del primals_830
        # Source Nodes: [x_gap_123], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_833, primals_833, 1, grid=grid(1), stream=stream0)
        del primals_833
        # Source Nodes: [out_299], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_836, primals_836, 1, grid=grid(1), stream=stream0)
        del primals_836
        # Source Nodes: [out_303], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_839, primals_839, 1, grid=grid(1), stream=stream0)
        del primals_839
        # Source Nodes: [x_207], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_842, primals_842, 1, grid=grid(1), stream=stream0)
        del primals_842
        # Source Nodes: [x_gap_128], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_845, primals_845, 1, grid=grid(1), stream=stream0)
        del primals_845
        # Source Nodes: [out_311], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_848, primals_848, 1, grid=grid(1), stream=stream0)
        del primals_848
        # Source Nodes: [out_315], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_851, primals_851, 1, grid=grid(1), stream=stream0)
        del primals_851
        # Source Nodes: [x_215], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_854, primals_854, 1, grid=grid(1), stream=stream0)
        del primals_854
        # Source Nodes: [x_gap_133], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_857, primals_857, 1, grid=grid(1), stream=stream0)
        del primals_857
        # Source Nodes: [out_323], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_860, primals_860, 1, grid=grid(1), stream=stream0)
        del primals_860
        # Source Nodes: [out_327], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_863, primals_863, 1, grid=grid(1), stream=stream0)
        del primals_863
        # Source Nodes: [x_223], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_866, primals_866, 1, grid=grid(1), stream=stream0)
        del primals_866
        # Source Nodes: [x_gap_138], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_869, primals_869, 1, grid=grid(1), stream=stream0)
        del primals_869
        # Source Nodes: [out_335], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_872, primals_872, 1, grid=grid(1), stream=stream0)
        del primals_872
        # Source Nodes: [out_339], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_875, primals_875, 1, grid=grid(1), stream=stream0)
        del primals_875
        # Source Nodes: [x_231], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_878, primals_878, 1, grid=grid(1), stream=stream0)
        del primals_878
        # Source Nodes: [x_gap_143], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_881, primals_881, 1, grid=grid(1), stream=stream0)
        del primals_881
        # Source Nodes: [out_347], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_884, primals_884, 1, grid=grid(1), stream=stream0)
        del primals_884
        # Source Nodes: [out_351], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_887, primals_887, 1, grid=grid(1), stream=stream0)
        del primals_887
        # Source Nodes: [x_239], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_890, primals_890, 1, grid=grid(1), stream=stream0)
        del primals_890
        # Source Nodes: [x_gap_148], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_893, primals_893, 1, grid=grid(1), stream=stream0)
        del primals_893
        # Source Nodes: [out_359], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_896, primals_896, 1, grid=grid(1), stream=stream0)
        del primals_896
        # Source Nodes: [out_363], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_899, primals_899, 1, grid=grid(1), stream=stream0)
        del primals_899
        # Source Nodes: [x_248], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_902, primals_902, 1, grid=grid(1), stream=stream0)
        del primals_902
        # Source Nodes: [x_gap_153], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_905, primals_905, 1, grid=grid(1), stream=stream0)
        del primals_905
        # Source Nodes: [out_372], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_908, primals_908, 1, grid=grid(1), stream=stream0)
        del primals_908
        # Source Nodes: [shortcut_34], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_911, primals_911, 1, grid=grid(1), stream=stream0)
        del primals_911
        # Source Nodes: [out_376], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_914, primals_914, 1, grid=grid(1), stream=stream0)
        del primals_914
        # Source Nodes: [x_256], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_917, primals_917, 1, grid=grid(1), stream=stream0)
        del primals_917
        # Source Nodes: [x_gap_158], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_920, primals_920, 1, grid=grid(1), stream=stream0)
        del primals_920
        # Source Nodes: [out_384], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_923, primals_923, 1, grid=grid(1), stream=stream0)
        del primals_923
        # Source Nodes: [out_388], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_926, primals_926, 1, grid=grid(1), stream=stream0)
        del primals_926
        # Source Nodes: [x_264], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_929, primals_929, 1, grid=grid(1), stream=stream0)
        del primals_929
        # Source Nodes: [x_gap_163], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_932, primals_932, 1, grid=grid(1), stream=stream0)
        del primals_932
        # Source Nodes: [out_396], Original ATen: [aten.add]
        triton_poi_fused_add_81.run(primals_935, primals_935, 1, grid=grid(1), stream=stream0)
        del primals_935
        return (buf1072, primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_18, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_36, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_51, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_66, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_84, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_99, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_114, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_129, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_147, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_162, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_173, primals_175, primals_177, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_192, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_207, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_222, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_237, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_252, primals_254, primals_256, primals_257, primals_259, primals_260, primals_262, primals_263, primals_265, primals_267, primals_269, primals_271, primals_272, primals_274, primals_275, primals_277, primals_278, primals_280, primals_282, primals_284, primals_286, primals_287, primals_289, primals_290, primals_292, primals_293, primals_295, primals_297, primals_299, primals_301, primals_302, primals_304, primals_305, primals_307, primals_308, primals_310, primals_312, primals_314, primals_316, primals_317, primals_319, primals_320, primals_322, primals_323, primals_325, primals_327, primals_329, primals_331, primals_332, primals_334, primals_335, primals_337, primals_338, primals_340, primals_342, primals_344, primals_346, primals_347, primals_349, primals_350, primals_352, primals_353, primals_355, primals_357, primals_359, primals_361, primals_362, primals_364, primals_365, primals_367, primals_368, primals_370, primals_372, primals_374, primals_376, primals_377, primals_379, primals_380, primals_382, primals_383, primals_385, primals_387, primals_389, primals_391, primals_392, primals_394, primals_395, primals_397, primals_398, primals_400, primals_402, primals_404, primals_406, primals_407, primals_409, primals_410, primals_412, primals_413, primals_415, primals_417, primals_419, primals_421, primals_422, primals_424, primals_425, primals_427, primals_428, primals_430, primals_432, primals_434, primals_436, primals_437, primals_439, primals_440, primals_442, primals_443, primals_445, primals_447, primals_449, primals_451, primals_452, primals_454, primals_455, primals_457, primals_458, primals_460, primals_462, primals_464, primals_466, primals_467, primals_469, primals_470, primals_472, primals_473, primals_475, primals_477, primals_479, primals_481, primals_482, primals_484, primals_485, primals_487, primals_488, primals_490, primals_491, primals_493, primals_495, primals_497, primals_499, primals_500, primals_502, primals_503, primals_505, primals_506, primals_508, primals_510, primals_512, primals_514, primals_515, primals_936, buf0, buf7, buf8, buf9, buf16, buf17, buf18, buf25, buf26, buf27, buf28, buf29, buf36, buf37, buf38, buf45, buf46, buf48, buf50, buf54, buf57, buf58, buf59, buf63, buf64, buf68, buf70, buf71, buf78, buf79, buf80, buf87, buf88, buf90, buf92, buf96, buf99, buf100, buf101, buf105, buf106, buf107, buf114, buf115, buf116, buf123, buf124, buf126, buf128, buf132, buf135, buf136, buf137, buf141, buf142, buf143, buf150, buf151, buf152, buf156, buf157, buf159, buf161, buf165, buf168, buf169, buf170, buf171, buf175, buf176, buf177, buf181, buf183, buf184, buf188, buf189, buf190, buf194, buf195, buf197, buf199, buf203, buf206, buf207, buf208, buf212, buf213, buf214, buf218, buf219, buf220, buf224, buf225, buf227, buf229, buf233, buf236, buf237, buf238, buf242, buf243, buf244, buf248, buf249, buf250, buf254, buf255, buf257, buf259, buf263, buf266, buf267, buf268, buf272, buf273, buf274, buf278, buf279, buf280, buf284, buf285, buf287, buf289, buf293, buf296, buf297, buf298, buf299, buf303, buf304, buf305, buf309, buf311, buf312, buf316, buf317, buf318, buf322, buf323, buf325, buf327, buf331, buf334, buf335, buf336, buf340, buf341, buf342, buf346, buf347, buf348, buf352, buf353, buf355, buf357, buf361, buf364, buf365, buf366, buf370, buf371, buf372, buf376, buf377, buf378, buf382, buf383, buf385, buf387, buf391, buf394, buf395, buf396, buf400, buf401, buf402, buf406, buf407, buf408, buf412, buf413, buf415, buf417, buf421, buf424, buf425, buf426, buf430, buf431, buf432, buf436, buf437, buf438, buf442, buf443, buf445, buf447, buf451, buf454, buf455, buf456, buf460, buf461, buf462, buf466, buf467, buf468, buf472, buf473, buf475, buf477, buf481, buf484, buf485, buf486, buf490, buf491, buf492, buf496, buf497, buf498, buf502, buf503, buf505, buf507, buf511, buf514, buf515, buf516, buf520, buf521, buf522, buf526, buf527, buf528, buf532, buf533, buf535, buf537, buf541, buf544, buf545, buf546, buf550, buf551, buf552, buf556, buf557, buf558, buf562, buf563, buf565, buf567, buf571, buf574, buf575, buf576, buf580, buf581, buf582, buf586, buf587, buf588, buf592, buf593, buf595, buf597, buf601, buf604, buf605, buf606, buf610, buf611, buf612, buf616, buf617, buf618, buf622, buf623, buf625, buf627, buf631, buf634, buf635, buf636, buf640, buf641, buf642, buf646, buf647, buf648, buf652, buf653, buf655, buf657, buf661, buf664, buf665, buf666, buf670, buf671, buf672, buf676, buf677, buf678, buf682, buf683, buf685, buf687, buf691, buf694, buf695, buf696, buf700, buf701, buf702, buf706, buf707, buf708, buf712, buf713, buf715, buf717, buf721, buf724, buf725, buf726, buf730, buf731, buf732, buf736, buf737, buf738, buf742, buf743, buf745, buf747, buf751, buf754, buf755, buf756, buf760, buf761, buf762, buf766, buf767, buf768, buf772, buf773, buf775, buf777, buf781, buf784, buf785, buf786, buf790, buf791, buf792, buf796, buf797, buf798, buf802, buf803, buf805, buf807, buf811, buf814, buf815, buf816, buf820, buf821, buf822, buf826, buf827, buf828, buf832, buf833, buf835, buf837, buf841, buf844, buf845, buf846, buf850, buf851, buf852, buf856, buf857, buf858, buf862, buf863, buf865, buf867, buf871, buf874, buf875, buf876, buf880, buf881, buf882, buf886, buf887, buf888, buf892, buf893, buf895, buf897, buf901, buf904, buf905, buf906, buf910, buf911, buf912, buf916, buf917, buf918, buf922, buf923, buf925, buf927, buf931, buf934, buf935, buf936, buf940, buf941, buf942, buf946, buf947, buf948, buf952, buf953, buf955, buf957, buf961, buf964, buf965, buf966, buf970, buf971, buf972, buf976, buf977, buf978, buf982, buf983, buf985, buf987, buf991, buf994, buf995, buf996, buf997, buf1001, buf1002, buf1003, buf1007, buf1009, buf1010, buf1014, buf1015, buf1016, buf1020, buf1021, buf1023, buf1025, buf1029, buf1032, buf1033, buf1034, buf1038, buf1039, buf1040, buf1044, buf1045, buf1046, buf1050, buf1051, buf1053, buf1055, buf1059, buf1062, buf1063, buf1064, buf1068, buf1071, reinterpret_tensor(primals_517, (1000, 2048), (2048, 1), 0), buf1073, reinterpret_tensor(buf1065, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf1047, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf1041, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf1035, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf1017, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf1011, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf1004, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf998, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf979, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf973, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf967, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf949, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf943, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf937, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf919, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf913, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf907, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf889, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf883, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf877, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf859, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf853, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf847, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf829, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf823, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf817, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf799, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf793, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf787, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf769, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf763, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf757, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf739, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf733, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf727, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf709, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf703, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf697, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf679, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf673, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf667, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf649, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf643, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf637, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf619, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf613, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf607, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf589, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf583, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf577, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf559, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf553, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf547, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf529, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf523, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf517, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf499, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf493, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf487, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf469, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf463, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf457, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf439, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf433, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf427, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf409, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf403, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf397, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf379, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf373, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf367, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf349, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf343, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf337, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf319, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf313, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf306, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf300, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf281, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf275, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf269, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf251, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf245, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf239, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf221, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf215, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf209, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf191, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf185, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf178, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf172, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf153, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf147, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf138, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf120, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf111, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf102, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf84, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf75, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf65, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf60, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf42, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf33, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf22, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf13, (1, 64, 1, 1), (64, 1, 1, 1), 0), reinterpret_tensor(buf4, (1, 64, 1, 1), (64, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((256, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((512, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((1024, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((1024, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((512, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((1024, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_522 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_525 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_528 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_531 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_534 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_537 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_540 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_543 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_546 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_549 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_552 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_555 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_558 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_561 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_564 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_567 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_570 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_573 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_576 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_579 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_582 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_585 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_588 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_591 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_594 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_597 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_600 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_603 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_606 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_609 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_612 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_615 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_618 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_621 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_624 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_627 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_630 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_632 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_633 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_635 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_636 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_638 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_639 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_641 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_642 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_644 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_645 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_647 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_648 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_650 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_651 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_653 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_654 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_656 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_657 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_659 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_660 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_661 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_662 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_663 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_664 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_665 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_666 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_667 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_668 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_669 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_670 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_671 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_672 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_673 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_674 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_675 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_676 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_677 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_678 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_679 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_680 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_681 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_682 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_683 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_684 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_685 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_686 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_687 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_688 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_689 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_690 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_691 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_692 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_693 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_694 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_695 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_696 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_697 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_698 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_699 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_700 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_701 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_702 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_703 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_704 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_705 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_706 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_707 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_708 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_709 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_710 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_711 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_712 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_713 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_714 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_715 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_716 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_717 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_718 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_719 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_720 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_721 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_722 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_723 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_724 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_725 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_726 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_727 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_728 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_729 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_730 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_731 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_732 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_733 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_734 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_735 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_736 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_737 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_738 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_739 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_740 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_741 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_742 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_743 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_744 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_745 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_746 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_747 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_748 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_749 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_750 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_751 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_752 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_753 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_754 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_755 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_756 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_757 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_758 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_759 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_760 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_761 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_762 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_763 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_764 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_765 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_766 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_767 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_768 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_769 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_770 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_771 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_772 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_773 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_774 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_775 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_776 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_777 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_778 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_779 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_780 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_781 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_782 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_783 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_784 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_785 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_786 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_787 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_788 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_789 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_790 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_791 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_792 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_793 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_794 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_795 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_796 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_797 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_798 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_799 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_800 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_801 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_802 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_803 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_804 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_805 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_806 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_807 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_808 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_809 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_810 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_811 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_812 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_813 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_814 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_815 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_816 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_817 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_818 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_819 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_820 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_821 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_822 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_823 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_824 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_825 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_826 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_827 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_828 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_829 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_830 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_831 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_832 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_833 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_834 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_835 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_836 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_837 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_838 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_839 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_840 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_841 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_842 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_843 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_844 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_845 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_846 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_847 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_848 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_849 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_850 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_851 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_852 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_853 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_854 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_855 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_856 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_857 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_858 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_859 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_860 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_861 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_862 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_863 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_864 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_865 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_866 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_867 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_868 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_869 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_870 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_871 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_872 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_873 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_874 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_875 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_876 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_877 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_878 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_879 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_880 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_881 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_882 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_883 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_884 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_885 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_886 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_887 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_888 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_889 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_890 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_891 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_892 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_893 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_894 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_895 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_896 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_897 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_898 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_899 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_900 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_901 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_902 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_903 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_904 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_905 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_906 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_907 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_908 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_909 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_910 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_911 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_912 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_913 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_914 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_915 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_916 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_917 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_918 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_919 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_920 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_921 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_922 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_923 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_924 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_925 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_926 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_927 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_928 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_929 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_930 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_931 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_932 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_933 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_934 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_935 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_936 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728, primals_729, primals_730, primals_731, primals_732, primals_733, primals_734, primals_735, primals_736, primals_737, primals_738, primals_739, primals_740, primals_741, primals_742, primals_743, primals_744, primals_745, primals_746, primals_747, primals_748, primals_749, primals_750, primals_751, primals_752, primals_753, primals_754, primals_755, primals_756, primals_757, primals_758, primals_759, primals_760, primals_761, primals_762, primals_763, primals_764, primals_765, primals_766, primals_767, primals_768, primals_769, primals_770, primals_771, primals_772, primals_773, primals_774, primals_775, primals_776, primals_777, primals_778, primals_779, primals_780, primals_781, primals_782, primals_783, primals_784, primals_785, primals_786, primals_787, primals_788, primals_789, primals_790, primals_791, primals_792, primals_793, primals_794, primals_795, primals_796, primals_797, primals_798, primals_799, primals_800, primals_801, primals_802, primals_803, primals_804, primals_805, primals_806, primals_807, primals_808, primals_809, primals_810, primals_811, primals_812, primals_813, primals_814, primals_815, primals_816, primals_817, primals_818, primals_819, primals_820, primals_821, primals_822, primals_823, primals_824, primals_825, primals_826, primals_827, primals_828, primals_829, primals_830, primals_831, primals_832, primals_833, primals_834, primals_835, primals_836, primals_837, primals_838, primals_839, primals_840, primals_841, primals_842, primals_843, primals_844, primals_845, primals_846, primals_847, primals_848, primals_849, primals_850, primals_851, primals_852, primals_853, primals_854, primals_855, primals_856, primals_857, primals_858, primals_859, primals_860, primals_861, primals_862, primals_863, primals_864, primals_865, primals_866, primals_867, primals_868, primals_869, primals_870, primals_871, primals_872, primals_873, primals_874, primals_875, primals_876, primals_877, primals_878, primals_879, primals_880, primals_881, primals_882, primals_883, primals_884, primals_885, primals_886, primals_887, primals_888, primals_889, primals_890, primals_891, primals_892, primals_893, primals_894, primals_895, primals_896, primals_897, primals_898, primals_899, primals_900, primals_901, primals_902, primals_903, primals_904, primals_905, primals_906, primals_907, primals_908, primals_909, primals_910, primals_911, primals_912, primals_913, primals_914, primals_915, primals_916, primals_917, primals_918, primals_919, primals_920, primals_921, primals_922, primals_923, primals_924, primals_925, primals_926, primals_927, primals_928, primals_929, primals_930, primals_931, primals_932, primals_933, primals_934, primals_935, primals_936]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('resnest101e', benchmark_compiled_module)
