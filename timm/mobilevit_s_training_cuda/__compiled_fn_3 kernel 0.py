
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


# kernel path: /tmp/torchinductor_youkaichao/qf/cqfsu57o6o5s72uca7m26y3vbgd7n3vna5vui5qyzidy5dfa5qcd.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => var_mean
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
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
        tmp0 = tl.load(in_ptr0 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (262144*((r2 + (8192*x0)) // 16384)) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/ik/ciky3d33orzvfdcjlyq6voatz56fyv5pjvcjale76l44ptidrkhp.py
# Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
# x_1 => add_1, add_2, add_3, mul_1, mul_2, mul_3, mul_4, mul_5, rsqrt, squeeze_1, var_mean
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_1', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
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


# kernel path: /tmp/torchinductor_youkaichao/7c/c7cmnxjd7pq5gocjhb2fadhbcnsll6kz4n4aybypvvxzlrjazl3w.py
# Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# shortcut => mul_7, sigmoid
# x_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2097152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 16
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x3), tmp15, None)
    tl.store(out_ptr2 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/po/cpoa2txaucft7bgzzwdv2trixrknlxgf45tahao6e4xnckbe6x6i.py
# Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
# x_7 => var_mean_1
triton_red_fused__native_batch_norm_legit_functional_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_3', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/sj/csjgdizwtme2zenlvp7z2clj7gdlx7ipotzicapnmtz6rv667536.py
# Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
# x_7 => add_6, add_7, add_8, mul_10, mul_11, mul_12, mul_13, mul_9, rsqrt_1, squeeze_4, var_mean_1
triton_per_fused__native_batch_norm_legit_functional_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_4', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/45/c455xnad656ou75y44wccb5wy5ippmyq53tvhio3rgnnfrdskdw4.py
# Source Nodes: [x_11, x_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_11 => mul_15, sigmoid_1
# x_7 => add_6, add_9, mul_14, mul_8, rsqrt_1, sub_1, var_mean_1
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x3), tmp15, None)
    tl.store(out_ptr2 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uv/cuv6hlzwvgzkx5f342vgvxvrjj7p74wll5eflruynxxdbl5ql277.py
# Source Nodes: [x_21], Original ATen: [aten._native_batch_norm_legit_functional]
# x_21 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_6', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
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
        tmp0 = tl.load(in_ptr0 + ((128*(((r2 + (8192*x0)) // 128) % 128)) + (16384*x1) + (524288*((r2 + (8192*x0)) // 16384)) + (r2 % 128)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/3o/c3oyngq2x4pwqy4vbi25irvat2wm2geuqt7ndktcodbue4qtk6ua.py
# Source Nodes: [x_21], Original ATen: [aten._native_batch_norm_legit_functional]
# x_21 => add_16, add_17, add_18, mul_25, mul_26, mul_27, mul_28, mul_29, rsqrt_3, squeeze_10, var_mean_3
triton_per_fused__native_batch_norm_legit_functional_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[32, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_7', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 32
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


# kernel path: /tmp/torchinductor_youkaichao/mk/cmkvlbso5hvengrwhkznh2fthnhvhs7xyzft2ez3wileqzjkidtn.py
# Source Nodes: [x_21], Original ATen: [aten._native_batch_norm_legit_functional]
# x_21 => add_16, add_19, mul_24, mul_30, rsqrt_3, sub_3, var_mean_3
triton_poi_fused__native_batch_norm_legit_functional_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16384) % 32
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
    tl.store(out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hr/chrksxqd7o4bw2l3jk3akmsxmfvria5ek6fki24czgwm6i2q35u2.py
# Source Nodes: [x_29], Original ATen: [aten._native_batch_norm_legit_functional]
# x_29 => var_mean_4
triton_red_fused__native_batch_norm_legit_functional_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_9', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/ua/cua7kicouvgjhyerf657aybttcjifondo3dmc6s4farmxucusdac.py
# Source Nodes: [x_29], Original ATen: [aten._native_batch_norm_legit_functional]
# x_29 => add_21, add_22, add_23, mul_32, mul_33, mul_34, mul_35, mul_36, rsqrt_4, squeeze_13, var_mean_4
triton_per_fused__native_batch_norm_legit_functional_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_10', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/wz/cwznf7dv6caarml3ez7y4nqgcpggu64id7qkngr4kk2jpe6ydcpy.py
# Source Nodes: [x_29, x_33], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_29 => add_21, add_24, mul_31, mul_37, rsqrt_4, sub_4, var_mean_4
# x_33 => mul_38, sigmoid_3
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x3), tmp15, None)
    tl.store(out_ptr2 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dt/cdtaec3r54eb57w5rtwfcv65fq2j4dv73bbneredzyrmfbrpdvbn.py
# Source Nodes: [x_35], Original ATen: [aten._native_batch_norm_legit_functional]
# x_35 => var_mean_5
triton_red_fused__native_batch_norm_legit_functional_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_12', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/hp/chpxyapcm7iaohltgiiutvisdothgsumaaj4qcb7ylzwtdhk6csq.py
# Source Nodes: [x_35], Original ATen: [aten._native_batch_norm_legit_functional]
# x_35 => add_26, add_27, add_28, mul_40, mul_41, mul_42, mul_43, mul_44, rsqrt_5, squeeze_16, var_mean_5
triton_per_fused__native_batch_norm_legit_functional_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_13', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/nu/cnusgzxxcifg7qkkpkvi54oso4oufl7aiteckijcbrej6vai4jo6.py
# Source Nodes: [x_35, x_39], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_35 => add_26, add_29, mul_39, mul_45, rsqrt_5, sub_5, var_mean_5
# x_39 => mul_46, sigmoid_4
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x3), tmp15, None)
    tl.store(out_ptr2 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vb/cvbfn64ecfjmlxyuuf5f5476u4544st3qwy2wewtasgnnr7kjhve.py
# Source Nodes: [x_43], Original ATen: [aten._native_batch_norm_legit_functional]
# x_43 => var_mean_6
triton_red_fused__native_batch_norm_legit_functional_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_15', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/2p/c2p2fllw3ucea6oek4oilrr33q5bvh2lzaaylkx6ngby6ie65ca5.py
# Source Nodes: [x_43], Original ATen: [aten._native_batch_norm_legit_functional]
# x_43 => add_31, add_32, add_33, mul_48, mul_49, mul_50, mul_51, mul_52, rsqrt_6, squeeze_19, var_mean_6
triton_per_fused__native_batch_norm_legit_functional_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_16', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/rv/crv7qkzr6fyqeon67ltzwdr5uirkfo5k6gcpo3kplnk3fngli6ts.py
# Source Nodes: [x_43], Original ATen: [aten._native_batch_norm_legit_functional]
# x_43 => add_31, add_34, mul_47, mul_53, rsqrt_6, sub_6, var_mean_6
triton_poi_fused__native_batch_norm_legit_functional_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_17', 'mutated_arg_names': []},
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
    tl.store(out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pr/cprugk32wbcjzjtfw4zdrszvdwlcmeu3tta7cwdxc6vsjq36j7ta.py
# Source Nodes: [x_51], Original ATen: [aten._native_batch_norm_legit_functional]
# x_51 => add_36, add_37, add_38, mul_55, mul_56, mul_57, mul_58, mul_59, rsqrt_7, squeeze_22, var_mean_7
triton_red_fused__native_batch_norm_legit_functional_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_18', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/dx/cdxvmjtrhfe27typ52l4tf7twrqxx7mhsjtwmqy2j6a5wo7gzbx6.py
# Source Nodes: [x_51, x_55], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_51 => add_36, add_39, mul_54, mul_60, rsqrt_7, sub_7, var_mean_7
# x_55 => mul_61, sigmoid_5
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x3), tmp15, None)
    tl.store(out_ptr2 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ha/chahdvt454rfnqk2pus7eksyaq77lm5nzq3hwzfx46wkjgg4ecfm.py
# Source Nodes: [x_65, x_72], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
# x_65 => add_46, add_49, mul_70, mul_76, rsqrt_9, sub_9, var_mean_9
# x_72 => add_50
triton_poi_fused__native_batch_norm_legit_functional_add_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/63/c63xaqnmaszp2zdm2rwkwwwqc4tgiekwkuhq5yeq3d23ransurv2.py
# Source Nodes: [x_103], Original ATen: [aten._native_batch_norm_legit_functional]
# x_103 => add_73, add_74, add_75, mul_109, mul_110, mul_111, mul_112, mul_113, rsqrt_14, squeeze_43, var_mean_14
triton_red_fused__native_batch_norm_legit_functional_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_21', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/2o/c2ouvaipwr2ketcpeccnqrpcghfc76nm3ierowlcpznt5h37el63.py
# Source Nodes: [x_103, x_107], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_103 => add_73, add_76, mul_108, mul_114, rsqrt_14, sub_14, var_mean_14
# x_107 => mul_115, sigmoid_10
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x3), tmp15, None)
    tl.store(out_ptr2 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qj/cqjbdkhlptgn4zh36psl2yyie3fzczuff6tdiebc4vqfqqmtapvg.py
# Source Nodes: [x_111], Original ATen: [aten._native_batch_norm_legit_functional]
# x_111 => add_78, add_79, add_80, mul_117, mul_118, mul_119, mul_120, mul_121, rsqrt_15, squeeze_46, var_mean_15
triton_red_fused__native_batch_norm_legit_functional_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_23', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 96
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
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (98304*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/wa/cwaaw3ppcnjxpv4kvtcwscmzkrlhfucwucunvpyprcufn6uvaubz.py
# Source Nodes: [x_111], Original ATen: [aten._native_batch_norm_legit_functional]
# x_111 => add_78, add_81, mul_116, mul_122, rsqrt_15, sub_15, var_mean_15
triton_poi_fused__native_batch_norm_legit_functional_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 96
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
    tl.store(out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bb/cbbablb7kvbcjeqeebzqljk463v5xuh2nmf3nthllklnp5xsubf3.py
# Source Nodes: [x_119, x_123], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_119 => add_83, add_86, mul_123, mul_129, rsqrt_16, sub_16, var_mean_16
# x_123 => mul_130, sigmoid_11
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 96
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x3), tmp15, None)
    tl.store(out_ptr2 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ad/cadbhaoxfy64faqoyokpg6bgxjyja7g5l7h35ezvncswy77qtupi.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm1 => add_87, mul_131, rsqrt_17, sub_17, var_mean_17
triton_red_fused_native_layer_norm_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_26', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 144
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
        tmp0 = tl.load(in_ptr0 + ((2*((((4*x1) + (x0 % 4)) // 4) % 16)) + (32*((x0 % 4) // 2)) + (64*((((4*x1) + (1024*r2) + (147456*(x0 // 4)) + (x0 % 4)) // 64) % 18432)) + ((x0 % 4) % 2)), rmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr1 + (x3), tmp3, None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp5 = tl.load(in_ptr0 + ((2*((((4*x1) + (x0 % 4)) // 4) % 16)) + (32*((x0 % 4) // 2)) + (64*((((4*x1) + (1024*r2) + (147456*(x0 // 4)) + (x0 % 4)) // 64) % 18432)) + ((x0 % 4) % 2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp6 = tmp5 - tmp2
        tmp7 = 144.0
        tmp8 = tmp3 / tmp7
        tmp9 = 1e-05
        tmp10 = tmp8 + tmp9
        tmp11 = tl.math.rsqrt(tmp10)
        tmp12 = tmp6 * tmp11
        tl.store(out_ptr2 + (r2 + (144*x1) + (36864*x0)), tmp12, rmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qq/cqqwzmxjjuhopfuwowdod5mtlgqwfl3rbmdlqe66ih55duykenoq.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___attn_qkv, getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm, aten.view]
# getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___attn_qkv => view_3
# getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm1 => add_88, mul_132
triton_poi_fused_native_layer_norm_view_27 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_view_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1179648
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 144
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x2), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/37/c37fhewrbhbwqmjdy3mpyl35cwjolpgtgzkeocvxpnx3qbgq75ya.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm2, x_131], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm2 => add_90, rsqrt_18, var_mean_18
# x_131 => add_89
triton_red_fused_add_native_layer_norm_native_layer_norm_backward_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_native_layer_norm_backward_28', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 256
    x1 = (xindex // 256)
    x3 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((2*((((4*x0) + (x1 % 4)) // 4) % 16)) + (32*((x1 % 4) // 2)) + (64*((((4*x0) + (1024*r2) + (147456*(x1 // 4)) + (x1 % 4)) // 64) % 18432)) + ((x1 % 4) % 2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r2 + (144*x3)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight,
        )
        tmp6_mean = tl.where(rmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x3), tmp6, None)
    tl.store(out_ptr1 + (x3), tmp7, None)
    tmp9 = 144.0
    tmp10 = tmp7 / tmp9
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = tl.math.rsqrt(tmp12)
    tmp14 = tmp13 / tmp9
    tl.store(out_ptr2 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/js/cjsjwumzckjk3ox3qrtkco37fbrzf33566aozsquszp3gqlc4x4y.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm2, x_131, x_132], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
# getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm2 => add_90, add_91, mul_133, mul_134, rsqrt_18, sub_18, var_mean_18
# x_131 => add_89
# x_132 => view_9
triton_poi_fused_add_native_layer_norm_view_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 65536], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_view_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 36864
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex % 144
    x2 = (xindex // 144)
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*((((4*x2) + (y0 % 4)) // 4) % 16)) + (32*((y0 % 4) // 2)) + (64*((((4*x2) + (1024*x1) + (147456*(y0 // 4)) + (y0 % 4)) // 64) % 18432)) + ((y0 % 4) % 2)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3 + (36864*y0)), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (256*y0)), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (256*y0)), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 144.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x3 + (36864*y0)), tmp13, ymask)
    tl.store(out_ptr1 + (x3 + (36864*y0)), tmp17, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7u/c7unttm3jduzmi4ldqqexvsh6ldsyod727fl4jducoiocr3gmb7v.py
# Source Nodes: [x_133, x_136], Original ATen: [aten.silu, aten.view]
# x_133 => mul_135, sigmoid_12
# x_136 => view_11
triton_poi_fused_silu_view_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_view_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2359296
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vt/cvtu6g4ytbja7dq5olbmrvsk4fwx4ufte3sikeqp2eh5za6vlrxl.py
# Source Nodes: [x_131, x_138], Original ATen: [aten.add]
# x_131 => add_89
# x_138 => add_92
triton_poi_fused_add_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 65536], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_31', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 36864
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex % 144
    x2 = (xindex // 144)
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*((((4*x2) + (y0 % 4)) // 4) % 16)) + (32*((y0 % 4) // 2)) + (64*((((4*x2) + (1024*x1) + (147456*(y0 // 4)) + (y0 % 4)) // 64) % 18432)) + ((y0 % 4) % 2)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3 + (36864*y0)), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_out_ptr0 + (x3 + (36864*y0)), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3 + (36864*y0)), tmp8, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h7/ch77nfdrxh4yeo6yd3pgqovbvov2xyadj2k2cjtxcayuvchtakla.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___attn_qkv, getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___attn_qkv => view_13
# getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___norm1 => add_93, add_94, mul_136, mul_137, rsqrt_19, sub_19, var_mean_19
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_32', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (144*x0)), rmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 144, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 144.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp22 / tmp18
    tl.store(out_ptr2 + (r1 + (144*x0)), tmp23, rmask)
    tl.store(out_ptr3 + (r1 + (144*x0)), tmp27, rmask)
    tl.store(out_ptr4 + (x0), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/b5/cb5fyw6v4mjjfjucmdhvxyt3o2fzghxdmavyrzfm2a74mi7vk2im.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___norm2, x_143, x_144], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___norm2 => add_96, add_97, mul_138, mul_139, rsqrt_20, sub_20, var_mean_20
# x_143 => add_95
# x_144 => view_19
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_33', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (144*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (144*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 144, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 144.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (144*x0)), tmp27, rmask)
    tl.store(out_ptr3 + (r1 + (144*x0)), tmp31, rmask)
    tl.store(out_ptr4 + (x0), tmp32, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kg/ckgi4evknsub2zfopttzmsqopf4unbnyreak4l5pq33bx3iq6d5o.py
# Source Nodes: [x_143, x_151, x_152], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# x_143 => add_95
# x_151 => add_98
# x_152 => add_99, mul_141, rsqrt_21, sub_21, var_mean_21
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_34', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 144
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (144*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (144*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (144*x0)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 144, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 144.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp32 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (144*x0)), tmp8, rmask)
    tl.store(out_ptr2 + (r1 + (144*x0)), tmp31, rmask)
    tl.store(out_ptr3 + (x0), tmp32, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sy/csyjpezxds6dr5md4sn7kr567e4jueshwn2mtc2kejdgj7sn3uoo.py
# Source Nodes: [x_155], Original ATen: [aten._unsafe_view, aten.clone]
# x_155 => clone_21, view_25
triton_poi_fused__unsafe_view_clone_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1152
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 32
    x3 = (xindex // 32)
    y0 = yindex % 144
    y1 = (yindex // 144)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + ((144*((((2*(x3 % 2)) + (4*(x2 // 2)) + (64*((x2 + (32*x3)) // 64)) + (x2 % 2)) // 4) % 256)) + (36864*(((2*(x3 % 2)) + (x2 % 2)) % 4)) + (147456*((((2*(x3 % 2)) + (4*(x2 // 2)) + (64*((x2 + (32*x3)) // 64)) + (1024*y0) + (147456*y1) + (x2 % 2)) // 147456) % 8)) + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (64*((x2 + (32*x3)) // 64)) + (1024*y0) + (x2 % 2)) // 1024) % 144)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (64*((x2 + (32*x3)) // 64)) + (1024*y0) + (x2 % 2)) // 1024) % 144), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (64*((x2 + (32*x3)) // 64)) + (1024*y0) + (x2 % 2)) // 1024) % 144), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x5 + (1024*y4)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yj/cyjkx25r7cwn6alu7biaelnadbrgswvhlfdwcdwcwq4snjemsyu3.py
# Source Nodes: [x_157], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# x_157 => add_102, add_105, mul_143, mul_149, rsqrt_22, sub_22, var_mean_22
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 96
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = 1.0
    tmp16 = tmp15 - tmp14
    tmp17 = tmp13 * tmp16
    tmp18 = tmp17 + tmp15
    tmp19 = tmp14 * tmp18
    tl.store(out_ptr0 + (x3), tmp13, None)
    tl.store(out_ptr1 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/s5/cs5cw4oa7lfkk6cbzgunsu7cu2rbztcmqbxweigkq5g2huwuzl3f.py
# Source Nodes: [cat_5], Original ATen: [aten.cat]
# cat_5 => cat
triton_poi_fused_cat_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1572864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 1024) % 192
    x2 = (xindex // 196608)
    x3 = xindex % 196608
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 96, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (98304*x2)), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 192, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-98304) + x3 + (98304*x2)), tmp8, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp8, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp7, tmp15)
    tl.store(out_ptr0 + (x4), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/77/c776ftiqvstc5hxecadtnxedkb3d5pa5v2l7luqa7vnxodlnzpgt.py
# Source Nodes: [x_169], Original ATen: [aten._native_batch_norm_legit_functional]
# x_169 => add_112, add_113, add_114, mul_160, mul_161, mul_162, mul_163, mul_164, rsqrt_24, squeeze_58, var_mean_24
triton_red_fused__native_batch_norm_legit_functional_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_38', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
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
        tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0) + (393216*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/gu/cguupx6wdomkkt6prmq6e423bpawzimii374ej7atdu7qn3tyfms.py
# Source Nodes: [x_169, x_173], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_169 => add_112, add_115, mul_159, mul_165, rsqrt_24, sub_24, var_mean_24
# x_173 => mul_166, sigmoid_16
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 1024) % 384
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x3), tmp15, None)
    tl.store(out_ptr2 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/l7/cl7ttiswewrhhc5pyupnqv5usv3x3wcwexr4f3w5yuv2y77iz5go.py
# Source Nodes: [x_175], Original ATen: [aten._native_batch_norm_legit_functional]
# x_175 => add_117, add_118, add_119, mul_168, mul_169, mul_170, mul_171, mul_172, rsqrt_25, squeeze_61, var_mean_25
triton_red_fused__native_batch_norm_legit_functional_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_40', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 384
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
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (98304*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/66/c66rggpbiljomee6ia3uar4p3k6y2b6cjogdmjnstumyrb5fxgtq.py
# Source Nodes: [x_175, x_179], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_175 => add_117, add_120, mul_167, mul_173, rsqrt_25, sub_25, var_mean_25
# x_179 => mul_174, sigmoid_17
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 384
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x3), tmp15, None)
    tl.store(out_ptr2 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/z3/cz3ygz4556khyjgrzfqrjf2uivcjxyg4a6zhiuv3u77qsgdsyckf.py
# Source Nodes: [x_183], Original ATen: [aten._native_batch_norm_legit_functional]
# x_183 => add_122, add_123, add_124, mul_176, mul_177, mul_178, mul_179, mul_180, rsqrt_26, squeeze_64, var_mean_26
triton_red_fused__native_batch_norm_legit_functional_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_42', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 128
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
        tmp0 = tl.load(in_ptr0 + (r1 + (256*x0) + (32768*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/pt/cptaadjgr6yylbaqdtnqk6isfccejt5h57poh5ybwtxddgtiakvg.py
# Source Nodes: [x_183], Original ATen: [aten._native_batch_norm_legit_functional]
# x_183 => add_122, add_125, mul_175, mul_181, rsqrt_26, sub_26, var_mean_26
triton_poi_fused__native_batch_norm_legit_functional_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 128
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
    tl.store(out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/oc/cocams3fc2rosgcvwpbklpwxkn6gimbuuqiriyyyk5pyke3kieyn.py
# Source Nodes: [x_191, x_195], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_191 => add_127, add_130, mul_182, mul_188, rsqrt_27, sub_27, var_mean_27
# x_195 => mul_189, sigmoid_18
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 128
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x3), tmp15, None)
    tl.store(out_ptr2 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ly/cly6q4kzugbwyqxqeuo2owzohe777ntc4moltluhsxazda46b7od.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm1 => var_mean_28
triton_red_fused_native_layer_norm_45 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_45', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32) % 64
    x2 = (xindex // 2048)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + ((2*((((4*x1) + (x0 % 4)) // 4) % 8)) + (16*((x0 % 4) // 2)) + (32*((((4*x1) + (256*r3) + (24576*x2) + (49152*(x0 // 4)) + (x0 % 4)) // 32) % 12288)) + ((x0 % 4) % 2)), rmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x4), tmp2, None)
    tl.store(out_ptr1 + (x4), tmp3, None)
    tl.store(out_ptr2 + (x4), tmp4, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/t3/ct3yow3wtru7jithv6bkyjn3httu77pqv7zypasboehriueydmx5.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm1 => var_mean_28
triton_per_fused_native_layer_norm_46 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 2],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_46', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp0 = tl.load(in_ptr0 + (x3 + (2048*r2)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x3 + (2048*r2)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x3 + (2048*r2)), rmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp3, 0)
    tmp8 = tl.where(rmask, tmp4, 0)
    tmp9 = tl.where(rmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tl.store(out_ptr0 + (x1 + (64*x0)), tmp13, None)
    tl.store(out_ptr1 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ef/cefcl654db5j2gn53msoczvdr5zjq7rjsxhif4e7tmjifuymgaam.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___attn_qkv, getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm, aten.view]
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___attn_qkv => view_29
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm1 => add_131, add_132, mul_190, mul_191, rsqrt_28, sub_28, var_mean_28
triton_poi_fused_native_layer_norm_view_47 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_view_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 12288
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex % 192
    x2 = (xindex // 192)
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*((((4*x2) + (y0 % 4)) // 4) % 8)) + (16*((y0 % 4) // 2)) + (32*((((4*x2) + (256*x1) + (49152*(y0 // 4)) + (y0 % 4)) // 32) % 12288)) + ((y0 % 4) % 2)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (64*y0)), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (32*x2)), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 192.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3 + (12288*y0)), tmp9, ymask)
    tl.store(out_ptr1 + (x3 + (12288*y0)), tmp13, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5b/c5buk6ekjk4vztirenkvnzal7pq2bskcl5jpna6bppn2dkhp3bul.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm2, x_203], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm2 => var_mean_29
# x_203 => add_133
triton_red_fused_add_native_layer_norm_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_48', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2) % 64
    x2 = (xindex // 128)
    x4 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + ((2*((((4*x1) + (x2 % 4)) // 4) % 8)) + (16*((x2 % 4) // 2)) + (32*((((4*x1) + (256*r3) + (24576*x0) + (49152*(x2 // 4)) + (x2 % 4)) // 32) % 12288)) + ((x2 % 4) % 2)), rmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (96*x4)), rmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r3 + (96*x0)), rmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight,
        )
        tmp6_mean = tl.where(rmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp6, None)
    tl.store(out_ptr1 + (x4), tmp7, None)
    tl.store(out_ptr2 + (x4), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yx/cyxxeg4tq7zlvvqoirmldlfog77qgkmfkdr2xdppdjz545h5cjvt.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm2, x_203], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm2 => add_134, rsqrt_29, var_mean_29
# x_203 => add_133
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_49 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_49', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (2*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (2*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (2*x0)), rmask, other=0.0)
    tmp3 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp5 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp3, 0)
    tmp8 = tl.where(rmask, tmp4, 0)
    tmp9 = tl.where(rmask, tmp5, 0)
    tmp10, tmp11, tmp12 = triton_helpers.welford(tmp7, tmp8, tmp9, 1)
    tmp13 = tmp10[:, None]
    tmp14 = tmp11[:, None]
    tmp15 = tmp12[:, None]
    tmp16 = 192.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp20 / tmp16
    tl.store(out_ptr2 + (x0), tmp21, None)
    tl.store(out_ptr0 + (x0), tmp13, None)
    tl.store(out_ptr1 + (x0), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ah/cah7vj4wq4fahfvq3okqgro4brh3tjjvgd245mbekpxpgikyvzix.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm2, x_203, x_204], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm2 => add_134, add_135, mul_192, mul_193, rsqrt_29, sub_29, var_mean_29
# x_203 => add_133
# x_204 => view_35
triton_poi_fused_add_native_layer_norm_view_50 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_view_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 12288
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex % 192
    x2 = (xindex // 192)
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*((((4*x2) + (y0 % 4)) // 4) % 8)) + (16*((y0 % 4) // 2)) + (32*((((4*x2) + (256*x1) + (49152*(y0 // 4)) + (y0 % 4)) // 32) % 12288)) + ((y0 % 4) % 2)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3 + (12288*y0)), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (64*y0)), ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (64*y0)), ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 192.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x3 + (12288*y0)), tmp13, ymask)
    tl.store(out_ptr1 + (x3 + (12288*y0)), tmp17, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lv/clvpgpuveolqlnmqzhigurn4r4l3cm5zjbnu7s7vgrvj2zjlzohr.py
# Source Nodes: [x_205, x_208], Original ATen: [aten.silu, aten.view]
# x_205 => mul_194, sigmoid_19
# x_208 => view_37
triton_poi_fused_silu_view_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_view_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 786432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lu/clupshxsmbin7ebnaji5egp5hacp4rtnk6f4rjlbws4k3ebr2w2i.py
# Source Nodes: [x_203, x_210], Original ATen: [aten.add]
# x_203 => add_133
# x_210 => add_136
triton_poi_fused_add_52 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_52', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 12288
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex % 192
    x2 = (xindex // 192)
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*((((4*x2) + (y0 % 4)) // 4) % 8)) + (16*((y0 % 4) // 2)) + (32*((((4*x2) + (256*x1) + (49152*(y0 // 4)) + (y0 % 4)) // 32) % 12288)) + ((y0 % 4) % 2)), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x3 + (12288*y0)), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x3 + (12288*y0)), ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3 + (12288*y0)), tmp8, ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2l/c2lhxucjjusvpts2vrtg2vpicuhyhzzwvnytn7qabdokjpy4bq5q.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___attn_qkv, getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___attn_qkv => view_39
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___norm1 => add_137, add_138, mul_195, mul_196, rsqrt_30, sub_30, var_mean_30
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_53', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (192*x0)), rmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 192.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp22 / tmp18
    tl.store(out_ptr2 + (r1 + (192*x0)), tmp23, rmask)
    tl.store(out_ptr3 + (r1 + (192*x0)), tmp27, rmask)
    tl.store(out_ptr4 + (x0), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qi/cqiyxtki4vkwhbppgjkud6qsrru2xnivl5zk5crix45m55in3qa7.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___norm2, x_215, x_216], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___norm2 => add_140, add_141, mul_197, mul_198, rsqrt_31, sub_31, var_mean_31
# x_215 => add_139
# x_216 => view_45
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_54 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_54', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (192*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (192*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 192.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (192*x0)), tmp27, rmask)
    tl.store(out_ptr3 + (r1 + (192*x0)), tmp31, rmask)
    tl.store(out_ptr4 + (x0), tmp32, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3b/c3b7eosszuixeeua6nrkmtmw4s4qwm7ipjz4tjf2rhhfcufbp7a4.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___attn_qkv, getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___norm1, x_215, x_222], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___attn_qkv => view_49
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___norm1 => add_143, add_144, mul_200, mul_201, rsqrt_32, sub_32, var_mean_32
# x_215 => add_139
# x_222 => add_142
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_55 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_55', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (192*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (192*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (192*x0)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 192.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (192*x0)), tmp8, rmask)
    tl.store(out_ptr2 + (r1 + (192*x0)), tmp31, rmask)
    tl.store(out_ptr3 + (r1 + (192*x0)), tmp35, rmask)
    tl.store(out_ptr4 + (x0), tmp36, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gh/cghkl6vwbgkentk4mixhwpa35o3lsuqu32hsuvjcqmxbz5vqkx2e.py
# Source Nodes: [x_239, x_247, x_248], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# x_239 => add_151
# x_247 => add_154
# x_248 => add_155, mul_210, rsqrt_36, sub_36, var_mean_36
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_56', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 192
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (192*x0)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (192*x0)), rmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (192*x0)), rmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 192, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 192.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp32 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (192*x0)), tmp8, rmask)
    tl.store(out_ptr2 + (r1 + (192*x0)), tmp31, rmask)
    tl.store(out_ptr3 + (x0), tmp32, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cf/ccfos5joet7yblk6yat2odkdamcic2wq5qqgd7teddefxjp3p2of.py
# Source Nodes: [x_251], Original ATen: [aten._unsafe_view, aten.clone]
# x_251 => clone_42, view_71
triton_poi_fused__unsafe_view_clone_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 16
    x3 = (xindex // 16)
    y0 = yindex % 192
    y1 = (yindex // 192)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + ((192*((((2*(x3 % 2)) + (4*(x2 // 2)) + (32*((x2 + (16*x3)) // 32)) + (x2 % 2)) // 4) % 64)) + (12288*(((2*(x3 % 2)) + (x2 % 2)) % 4)) + (49152*((((2*(x3 % 2)) + (4*(x2 // 2)) + (32*((x2 + (16*x3)) // 32)) + (256*y0) + (49152*y1) + (x2 % 2)) // 49152) % 8)) + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (32*((x2 + (16*x3)) // 32)) + (256*y0) + (x2 % 2)) // 256) % 192)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (32*((x2 + (16*x3)) // 32)) + (256*y0) + (x2 % 2)) // 256) % 192), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (32*((x2 + (16*x3)) // 32)) + (256*y0) + (x2 % 2)) // 256) % 192), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x5 + (256*y4)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qd/cqdn6p2lpnl5sqnpx5ttwbwt4qzeekzvo2zkrrkxccubob5pjmju.py
# Source Nodes: [x_253], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# x_253 => add_158, add_161, mul_212, mul_218, rsqrt_37, sub_37, var_mean_37
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 256) % 128
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = 1.0
    tmp16 = tmp15 - tmp14
    tmp17 = tmp13 * tmp16
    tmp18 = tmp17 + tmp15
    tmp19 = tmp14 * tmp18
    tl.store(out_ptr0 + (x3), tmp13, None)
    tl.store(out_ptr1 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mq/cmqdx35b6ddt6ei7r4dvsw6m4hat43noascdkheev3bdtq7nsus5.py
# Source Nodes: [cat_4], Original ATen: [aten.cat]
# cat_4 => cat_1
triton_poi_fused_cat_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 524288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 256) % 256
    x2 = (xindex // 65536)
    x3 = xindex % 65536
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (32768*x2)), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 256, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-32768) + x3 + (32768*x2)), tmp8, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp8, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp7, tmp15)
    tl.store(out_ptr0 + (x4), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/oj/cojhbszj2oqhw6yb7z65jfemlkyeh3ts5wzgm3ros37ljiic4j6w.py
# Source Nodes: [x_265], Original ATen: [aten._native_batch_norm_legit_functional]
# x_265 => add_168, add_169, add_170, mul_229, mul_230, mul_231, mul_232, mul_233, rsqrt_39, squeeze_76, var_mean_39
triton_red_fused__native_batch_norm_legit_functional_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_60', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/4y/c4ysopttusmo2nam2f67tvldplfrwki3dwsorb5d5nm23g7tdetd.py
# Source Nodes: [x_265, x_269], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_265 => add_168, add_171, mul_228, mul_234, rsqrt_39, sub_39, var_mean_39
# x_269 => mul_235, sigmoid_25
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_61', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x3), tmp15, None)
    tl.store(out_ptr2 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/a2/ca2aqg2bb6hcsylwawe4ysxevjsaf3jv67blltp2kbpz6wrcbft6.py
# Source Nodes: [x_271], Original ATen: [aten._native_batch_norm_legit_functional]
# x_271 => add_173, add_174, add_175, mul_237, mul_238, mul_239, mul_240, mul_241, rsqrt_40, squeeze_79, var_mean_40
triton_per_fused__native_batch_norm_legit_functional_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_62', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/hj/chjgd4q5wwmm6olk2wfvpiktr7ge42mndo7yr7gxhbwvs3ladauj.py
# Source Nodes: [x_271, x_275], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_271 => add_173, add_176, mul_236, mul_242, rsqrt_40, sub_40, var_mean_40
# x_275 => mul_243, sigmoid_26
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_63', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x3), tmp15, None)
    tl.store(out_ptr2 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zg/czgzvdebbluf3lu6737vvslb7ui3faer63tso3qkfkkqjxybm754.py
# Source Nodes: [x_279], Original ATen: [aten._native_batch_norm_legit_functional]
# x_279 => add_178, add_179, add_180, mul_245, mul_246, mul_247, mul_248, mul_249, rsqrt_41, squeeze_82, var_mean_41
triton_per_fused__native_batch_norm_legit_functional_64 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_64', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel):
    xnumel = 160
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
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (10240*r2)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/dw/cdwwghklql7p6zzjdqyyje5fo5pqtqyjmlixprzjat44h42rwcmt.py
# Source Nodes: [x_279], Original ATen: [aten._native_batch_norm_legit_functional]
# x_279 => add_178, add_181, mul_244, mul_250, rsqrt_41, sub_41, var_mean_41
triton_poi_fused__native_batch_norm_legit_functional_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 81920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 160
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
    tl.store(out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hr/chr6exymoltamjks47dzytyyepd4gs4aozxlb6qkhrifdg5vgsr2.py
# Source Nodes: [x_287, x_291], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# x_287 => add_183, add_186, mul_251, mul_257, rsqrt_42, sub_42, var_mean_42
# x_291 => mul_258, sigmoid_27
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_66', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 81920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 160
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = tmp13 * tmp14
    tmp16 = 1.0
    tmp17 = tmp16 - tmp14
    tmp18 = tmp13 * tmp17
    tmp19 = tmp18 + tmp16
    tmp20 = tmp14 * tmp19
    tl.store(out_ptr1 + (x3), tmp15, None)
    tl.store(out_ptr2 + (x3), tmp20, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ag/cagv6haowhomeqicvrehpfic23v7rzt7xwjimuzgozdgcglp2hn2.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm1 => var_mean_43
triton_red_fused_native_layer_norm_67 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_native_layer_norm_67', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 32
    x1 = (xindex // 32) % 16
    x2 = (xindex // 512)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x4 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + ((2*((((4*x1) + (x0 % 4)) // 4) % 4)) + (8*((x0 % 4) // 2)) + (16*((((4*x1) + (64*r3) + (7680*x2) + (15360*(x0 // 4)) + (x0 % 4)) // 16) % 7680)) + ((x0 % 4) % 2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
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
    tl.store(out_ptr0 + (x4), tmp2, xmask)
    tl.store(out_ptr1 + (x4), tmp3, xmask)
    tl.store(out_ptr2 + (x4), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7y/c7y3qbcq7i62lqejw5m5dseyk364liqt6bbr6di427espffr4nof.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm1 => var_mean_43
triton_per_fused_native_layer_norm_68 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 2],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_68', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 32
    x1 = (xindex // 32)
    tmp0 = tl.load(in_ptr0 + (x3 + (512*r2)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x3 + (512*r2)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x3 + (512*r2)), rmask & xmask, other=0.0)
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
    tl.store(out_ptr0 + (x1 + (16*x0)), tmp13, xmask)
    tl.store(out_ptr1 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oh/cohra5bvktncokku3n3unmcudcwzuupalea73mbxtm7h43u36sqs.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___attn_qkv, getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm, aten.view]
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___attn_qkv => view_75
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm1 => add_187, add_188, mul_259, mul_260, rsqrt_43, sub_43, var_mean_43
triton_poi_fused_native_layer_norm_view_69 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_view_69', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 3840
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex % 240
    x2 = (xindex // 240)
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*((((4*x2) + (y0 % 4)) // 4) % 4)) + (8*((y0 % 4) // 2)) + (16*((((4*x2) + (64*x1) + (15360*(y0 // 4)) + (y0 % 4)) // 16) % 7680)) + ((y0 % 4) % 2)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2 + (16*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0 + (32*x2)), xmask & ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 240.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tl.store(out_ptr0 + (x3 + (3840*y0)), tmp9, xmask & ymask)
    tl.store(out_ptr1 + (x3 + (3840*y0)), tmp13, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ak/cakzgqmkffepr7xtwnmzixkpxqvw735pxlcfin5qvetpkkzv4y4u.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm2, x_299], Original ATen: [aten.add, aten.native_layer_norm]
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm2 => var_mean_44
# x_299 => add_189
triton_red_fused_add_native_layer_norm_70 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_native_layer_norm_70', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2
    x1 = (xindex // 2) % 16
    x2 = (xindex // 32)
    x4 = xindex
    tmp6_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp6_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + ((2*((((4*x1) + (x2 % 4)) // 4) % 4)) + (8*((x2 % 4) // 2)) + (16*((((4*x1) + (64*r3) + (7680*x0) + (15360*(x2 // 4)) + (x2 % 4)) // 16) % 7680)) + ((x2 % 4) % 2)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp1 = tl.load(in_ptr1 + (r3 + (120*x4)), rmask & xmask, eviction_policy='evict_first', other=0.0)
        tmp2 = tl.load(in_ptr2 + (r3 + (120*x0)), rmask & xmask, eviction_policy='evict_last', other=0.0)
        tmp3 = tmp1 + tmp2
        tmp4 = tmp0 + tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp6_mean_next, tmp6_m2_next, tmp6_weight_next = triton_helpers.welford_reduce(
            tmp5, tmp6_mean, tmp6_m2, tmp6_weight,
        )
        tmp6_mean = tl.where(rmask & xmask, tmp6_mean_next, tmp6_mean)
        tmp6_m2 = tl.where(rmask & xmask, tmp6_m2_next, tmp6_m2)
        tmp6_weight = tl.where(rmask & xmask, tmp6_weight_next, tmp6_weight)
    tmp6_tmp, tmp7_tmp, tmp8_tmp = triton_helpers.welford(
        tmp6_mean, tmp6_m2, tmp6_weight, 1
    )
    tmp6 = tmp6_tmp[:, None]
    tmp7 = tmp7_tmp[:, None]
    tmp8 = tmp8_tmp[:, None]
    tl.store(out_ptr0 + (x4), tmp6, xmask)
    tl.store(out_ptr1 + (x4), tmp7, xmask)
    tl.store(out_ptr2 + (x4), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w6/cw6sjjlg76sojhslmlwmmvqgocenmf3ree7ro353t6pl3mdjnkzx.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm2, x_299], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm2 => add_190, rsqrt_44, var_mean_44
# x_299 => add_189
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_71 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 2],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_71', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 2
    RBLOCK: tl.constexpr = 2
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (2*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (2*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1 + (2*x0)), rmask & xmask, other=0.0)
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
    tmp16 = 240.0
    tmp17 = tmp14 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp20 / tmp16
    tl.store(out_ptr2 + (x0), tmp21, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gk/cgkp2ptas4ia2yy5v7ptjbstlpasvsb3jedvvirqo2lnmqpqb6em.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm2, x_299, x_300], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm2 => add_190, add_191, mul_261, mul_262, rsqrt_44, sub_44, var_mean_44
# x_299 => add_189
# x_300 => view_81
triton_poi_fused_add_native_layer_norm_view_72 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_native_layer_norm_view_72', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 3840
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex % 240
    x2 = (xindex // 240)
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*((((4*x2) + (y0 % 4)) // 4) % 4)) + (8*((y0 % 4) // 2)) + (16*((((4*x2) + (64*x1) + (15360*(y0 // 4)) + (y0 % 4)) // 16) % 7680)) + ((y0 % 4) % 2)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x3 + (3840*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x2 + (16*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr4 + (x2 + (16*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp16 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp6 = tmp4 - tmp5
    tmp8 = 240.0
    tmp9 = tmp7 / tmp8
    tmp10 = 1e-05
    tmp11 = tmp9 + tmp10
    tmp12 = tl.math.rsqrt(tmp11)
    tmp13 = tmp6 * tmp12
    tmp15 = tmp13 * tmp14
    tmp17 = tmp15 + tmp16
    tl.store(out_ptr0 + (x3 + (3840*y0)), tmp13, xmask & ymask)
    tl.store(out_ptr1 + (x3 + (3840*y0)), tmp17, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xx/cxxgnv33u5drld5hflzycsmhicqk6bozbe7nmartvimgb7ut6dhr.py
# Source Nodes: [x_301, x_304], Original ATen: [aten.silu, aten.view]
# x_301 => mul_263, sigmoid_28
# x_304 => view_83
triton_poi_fused_silu_view_73 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_silu_view_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 245760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tl.sigmoid(tmp0)
    tmp2 = tmp0 * tmp1
    tl.store(out_ptr0 + (x0), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vf/cvfjmtxm53zozw2lj4vxpw6etjib4rirsugkejjne3ua52xlofcj.py
# Source Nodes: [x_299, x_306], Original ATen: [aten.add]
# x_299 => add_189
# x_306 => add_192
triton_poi_fused_add_74 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_74', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 3840
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex % 240
    x2 = (xindex // 240)
    y0 = yindex
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*((((4*x2) + (y0 % 4)) // 4) % 4)) + (8*((y0 % 4) // 2)) + (16*((((4*x2) + (64*x1) + (15360*(y0 // 4)) + (y0 % 4)) // 16) % 7680)) + ((y0 % 4) % 2)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_out_ptr0 + (x3 + (3840*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x3 + (3840*y0)), xmask & ymask, eviction_policy='evict_last')
    tmp6 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3 + (3840*y0)), tmp8, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p4/cp4iosmrmlmwjerr3kebxx7drvowqwokhg772qtalx6rb75zeoym.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___attn_qkv, getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___attn_qkv => view_85
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___norm1 => add_193, add_194, mul_264, mul_265, rsqrt_45, sub_45, var_mean_45
triton_per_fused_native_layer_norm_native_layer_norm_backward_view_75 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_native_layer_norm_native_layer_norm_backward_view_75', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 240
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp26 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = tl.full([XBLOCK, 1], 240, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = tl.sum(tmp15, 1)[:, None]
    tmp17 = tmp0 - tmp10
    tmp18 = 240.0
    tmp19 = tmp16 / tmp18
    tmp20 = 1e-05
    tmp21 = tmp19 + tmp20
    tmp22 = tl.math.rsqrt(tmp21)
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp22 / tmp18
    tl.store(out_ptr2 + (r1 + (240*x0)), tmp23, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (240*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/aw/cawiangbhfw7tncjuqwjp5eaxmqcnssp2ohpjzzubjechdajwpzn.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___norm2, x_311, x_312], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___norm2 => add_196, add_197, mul_266, mul_267, rsqrt_46, sub_46, var_mean_46
# x_311 => add_195
# x_312 => view_91
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_76 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_76', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 240
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp7 = tl.where(rmask & xmask, tmp5, 0)
    tmp8 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
    tmp10 = tl.where(rmask & xmask, tmp8, 0)
    tmp11 = tl.sum(tmp10, 1)[:, None]
    tmp12 = tl.full([XBLOCK, 1], 240, tl.int32)
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 / tmp13
    tmp15 = tmp5 - tmp14
    tmp16 = tmp15 * tmp15
    tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = tmp4 - tmp14
    tmp22 = 240.0
    tmp23 = tmp20 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp21 * tmp26
    tmp29 = tmp27 * tmp28
    tmp31 = tmp29 + tmp30
    tmp32 = tmp26 / tmp22
    tl.store(out_ptr2 + (r1 + (240*x0)), tmp27, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (240*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5n/c5nq6gjqni6bpfv7c7bqlsf6f4ehk7qodpdjl4lqnsdptsy74q7c.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___attn_qkv, getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___norm1, x_311, x_318], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___attn_qkv => view_95
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___norm1 => add_199, add_200, mul_269, mul_270, rsqrt_47, sub_47, var_mean_47
# x_311 => add_195
# x_318 => add_198
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_77 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_77', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 240
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp34 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 240, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 240.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp33 = tmp31 * tmp32
    tmp35 = tmp33 + tmp34
    tmp36 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (240*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (240*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (r1 + (240*x0)), tmp35, rmask & xmask)
    tl.store(out_ptr4 + (x0), tmp36, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/at/cattvod5zhimtvwgzu7rumaapuxv33bymipqebjv5ckgie53imgl.py
# Source Nodes: [x_323, x_331, x_332], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
# x_323 => add_201
# x_331 => add_204
# x_332 => add_205, mul_274, rsqrt_49, sub_49, var_mean_49
triton_per_fused_add_native_layer_norm_native_layer_norm_backward_78 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_native_layer_norm_native_layer_norm_backward_78', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 240
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp5 = tl.load(in_out_ptr0 + (r1 + (240*x0)), rmask & xmask, other=0.0)
    tmp6 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0.0)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 + tmp3
    tmp7 = tmp5 + tmp6
    tmp8 = tmp4 + tmp7
    tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
    tmp11 = tl.where(rmask & xmask, tmp9, 0)
    tmp12 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp14 = tl.where(rmask & xmask, tmp12, 0)
    tmp15 = tl.sum(tmp14, 1)[:, None]
    tmp16 = tl.full([XBLOCK, 1], 240, tl.int32)
    tmp17 = tmp16.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tmp19 = tmp9 - tmp18
    tmp20 = tmp19 * tmp19
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = tmp8 - tmp18
    tmp26 = 240.0
    tmp27 = tmp24 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp25 * tmp30
    tmp32 = tmp30 / tmp26
    tl.store(in_out_ptr0 + (r1 + (240*x0)), tmp8, rmask & xmask)
    tl.store(out_ptr2 + (r1 + (240*x0)), tmp31, rmask & xmask)
    tl.store(out_ptr3 + (x0), tmp32, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zt/cztolds4epstz4dk4cur7t6j66whwa4fjfk7w4lcfbauqo3nj5gu.py
# Source Nodes: [x_335], Original ATen: [aten._unsafe_view, aten.clone]
# x_335 => clone_60, view_107
triton_poi_fused__unsafe_view_clone_79 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__unsafe_view_clone_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 8
    x3 = (xindex // 8)
    y0 = yindex % 240
    y1 = (yindex // 240)
    x5 = xindex
    y4 = yindex
    tmp0 = tl.load(in_ptr0 + ((240*((((2*(x3 % 2)) + (4*(x2 // 2)) + (16*((x2 + (8*x3)) // 16)) + (x2 % 2)) // 4) % 16)) + (3840*(((2*(x3 % 2)) + (x2 % 2)) % 4)) + (15360*((((2*(x3 % 2)) + (4*(x2 // 2)) + (16*((x2 + (8*x3)) // 16)) + (64*y0) + (15360*y1) + (x2 % 2)) // 15360) % 8)) + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (16*((x2 + (8*x3)) // 16)) + (64*y0) + (x2 % 2)) // 64) % 240)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (16*((x2 + (8*x3)) // 16)) + (64*y0) + (x2 % 2)) // 64) % 240), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + ((((2*(x3 % 2)) + (4*(x2 // 2)) + (16*((x2 + (8*x3)) // 16)) + (64*y0) + (x2 % 2)) // 64) % 240), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp4 = tmp2 + tmp3
    tl.store(out_ptr0 + (x5 + (64*y4)), tmp4, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/py/cpyeld3osdn5fbzou2tk6grmu3cdn6xkd4vd5c6dnfyth4wjit63.py
# Source Nodes: [x_337], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
# x_337 => add_208, add_211, mul_276, mul_282, rsqrt_50, sub_50, var_mean_50
triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_80 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_80', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 81920
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 160
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
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = 1.0
    tmp16 = tmp15 - tmp14
    tmp17 = tmp13 * tmp16
    tmp18 = tmp17 + tmp15
    tmp19 = tmp14 * tmp18
    tl.store(out_ptr0 + (x3), tmp13, None)
    tl.store(out_ptr1 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/c2/cc2736uu77baqpqa2fipzrggfjujokoateqquoxnl7wqtpwehoph.py
# Source Nodes: [cat_3], Original ATen: [aten.cat]
# cat_3 => cat_2
triton_poi_fused_cat_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_81', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 64) % 320
    x2 = (xindex // 20480)
    x3 = xindex % 20480
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 160, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (10240*x2)), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 320, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-10240) + x3 + (10240*x2)), tmp8, other=0.0)
    tmp12 = tl.sigmoid(tmp11)
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp8, tmp13, tmp14)
    tmp16 = tl.where(tmp4, tmp7, tmp15)
    tl.store(out_ptr0 + (x4), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/z2/cz2vm7p3j3rsbt5wobl5ksfxoowejk5lsgpyxelwsv45tzpo5dtv.py
# Source Nodes: [x_350], Original ATen: [aten._native_batch_norm_legit_functional]
# x_350 => add_218, add_219, add_220, mul_293, mul_294, mul_295, mul_296, mul_297, rsqrt_52, squeeze_94, var_mean_52
triton_per_fused__native_batch_norm_legit_functional_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_82', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel):
    xnumel = 640
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
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0) + (40960*r2)), rmask & xmask, other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/fk/cfkphqf2g4yo46op7imn3dgnpac4bzxb6peb6djmc2m53b4lu2b5.py
# Source Nodes: [x_350, x_355, x_356, x_358], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mean, aten.mul, aten.sigmoid, aten.silu, aten.sub, aten.view]
# x_350 => add_218, add_221, mul_292, mul_298, rsqrt_52, sub_52, var_mean_52
# x_355 => mul_299, sigmoid_33
# x_356 => mean
# x_358 => view_108
triton_per_fused__native_batch_norm_legit_functional_add_fill_mean_mul_sigmoid_silu_sub_view_83 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_add_fill_mean_mul_sigmoid_silu_sub_view_83', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5120
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 640
    tmp0 = tl.load(in_ptr0 + (r2 + (64*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 512.0
    tmp5 = tmp3 / tmp4
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sigmoid(tmp13)
    tmp15 = 1.0
    tmp16 = tmp15 - tmp14
    tmp17 = tmp13 * tmp16
    tmp18 = tmp17 + tmp15
    tmp19 = tmp14 * tmp18
    tmp20 = tmp13 * tmp14
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp23 = tl.where(rmask & xmask, tmp21, 0)
    tmp24 = tl.sum(tmp23, 1)[:, None]
    tmp25 = 64.0
    tmp26 = tmp24 / tmp25
    tl.store(out_ptr1 + (r2 + (64*x3)), tmp19, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp26, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u4/cu4zoatnquv7yok5ysqfijeehagbjofrlpmpyfecq2hmy557yjiy.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm1 => add_187, rsqrt_43, var_mean_43
triton_poi_fused_native_layer_norm_native_layer_norm_backward_84 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_native_layer_norm_backward_84', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (32*x1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = 240.0
    tmp2 = tmp0 / tmp1
    tmp3 = 1e-05
    tmp4 = tmp2 + tmp3
    tmp5 = tl.math.rsqrt(tmp4)
    tmp6 = tmp5 / tmp1
    tl.store(out_ptr0 + (x1 + (16*y0)), tmp6, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3o/c3odc3us6t4sno43wvjwwpmzzubfdb4q26nva7xpz4c34sd6uq5x.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm1 => add_131, rsqrt_28, var_mean_28
triton_poi_fused_native_layer_norm_native_layer_norm_backward_85 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_native_layer_norm_backward_85', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (32*x1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = 192.0
    tmp2 = tmp0 / tmp1
    tmp3 = 1e-05
    tmp4 = tmp2 + tmp3
    tmp5 = tl.math.rsqrt(tmp4)
    tmp6 = tmp5 / tmp1
    tl.store(out_ptr0 + (x1 + (64*y0)), tmp6, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4a/c4avyg2myf3asoc73xy7pjebp3rk2nkow5n6ivnyeaylvtrzachg.py
# Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
# getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm1 => add_87, rsqrt_17, var_mean_17
triton_poi_fused_native_layer_norm_native_layer_norm_backward_86 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_layer_norm_native_layer_norm_backward_86', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32
    xnumel = 256
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (32*x1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = 144.0
    tmp2 = tmp0 / tmp1
    tmp3 = 1e-05
    tmp4 = tmp2 + tmp3
    tmp5 = tl.math.rsqrt(tmp4)
    tmp6 = tmp5 / tmp1
    tl.store(out_ptr0 + (x1 + (256*y0)), tmp6, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3h/c3h3b74a4ovrlodkpnxtkjl7plshap3nyt6xsnhjozj67hj35y2j.py
# Source Nodes: [add_], Original ATen: [aten.add]
# add_ => add
triton_poi_fused_add_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_87', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312 = args
    args.clear()
    assert_size_stride(primals_1, (16, ), (1, ))
    assert_size_stride(primals_2, (16, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (32, ), (1, ))
    assert_size_stride(primals_8, (32, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_10, (128, ), (1, ))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (64, ), (1, ))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_16, (256, ), (1, ))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_18, (256, ), (1, ))
    assert_size_stride(primals_19, (64, ), (1, ))
    assert_size_stride(primals_20, (64, ), (1, ))
    assert_size_stride(primals_21, (256, ), (1, ))
    assert_size_stride(primals_22, (256, ), (1, ))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_24, (256, ), (1, ))
    assert_size_stride(primals_25, (64, ), (1, ))
    assert_size_stride(primals_26, (64, ), (1, ))
    assert_size_stride(primals_27, (256, ), (1, ))
    assert_size_stride(primals_28, (256, ), (1, ))
    assert_size_stride(primals_29, (256, ), (1, ))
    assert_size_stride(primals_30, (256, ), (1, ))
    assert_size_stride(primals_31, (96, ), (1, ))
    assert_size_stride(primals_32, (96, ), (1, ))
    assert_size_stride(primals_33, (96, ), (1, ))
    assert_size_stride(primals_34, (96, ), (1, ))
    assert_size_stride(primals_35, (96, ), (1, ))
    assert_size_stride(primals_36, (96, ), (1, ))
    assert_size_stride(primals_37, (96, ), (1, ))
    assert_size_stride(primals_38, (96, ), (1, ))
    assert_size_stride(primals_39, (384, ), (1, ))
    assert_size_stride(primals_40, (384, ), (1, ))
    assert_size_stride(primals_41, (384, ), (1, ))
    assert_size_stride(primals_42, (384, ), (1, ))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_44, (128, ), (1, ))
    assert_size_stride(primals_45, (128, ), (1, ))
    assert_size_stride(primals_46, (128, ), (1, ))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_48, (128, ), (1, ))
    assert_size_stride(primals_49, (128, ), (1, ))
    assert_size_stride(primals_50, (128, ), (1, ))
    assert_size_stride(primals_51, (512, ), (1, ))
    assert_size_stride(primals_52, (512, ), (1, ))
    assert_size_stride(primals_53, (512, ), (1, ))
    assert_size_stride(primals_54, (512, ), (1, ))
    assert_size_stride(primals_55, (160, ), (1, ))
    assert_size_stride(primals_56, (160, ), (1, ))
    assert_size_stride(primals_57, (160, ), (1, ))
    assert_size_stride(primals_58, (160, ), (1, ))
    assert_size_stride(primals_59, (160, ), (1, ))
    assert_size_stride(primals_60, (160, ), (1, ))
    assert_size_stride(primals_61, (160, ), (1, ))
    assert_size_stride(primals_62, (160, ), (1, ))
    assert_size_stride(primals_63, (640, ), (1, ))
    assert_size_stride(primals_64, (640, ), (1, ))
    assert_size_stride(primals_65, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_66, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_67, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_68, (32, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_69, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_70, (128, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_71, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_72, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_73, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_74, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_75, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_76, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_77, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_78, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_79, (256, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_80, (96, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_81, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(primals_82, (144, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_83, (144, ), (1, ))
    assert_size_stride(primals_84, (144, ), (1, ))
    assert_size_stride(primals_85, (432, 144), (144, 1))
    assert_size_stride(primals_86, (432, ), (1, ))
    assert_size_stride(primals_87, (144, 144), (144, 1))
    assert_size_stride(primals_88, (144, ), (1, ))
    assert_size_stride(primals_89, (144, ), (1, ))
    assert_size_stride(primals_90, (144, ), (1, ))
    assert_size_stride(primals_91, (288, 144), (144, 1))
    assert_size_stride(primals_92, (288, ), (1, ))
    assert_size_stride(primals_93, (144, 288), (288, 1))
    assert_size_stride(primals_94, (144, ), (1, ))
    assert_size_stride(primals_95, (144, ), (1, ))
    assert_size_stride(primals_96, (144, ), (1, ))
    assert_size_stride(primals_97, (432, 144), (144, 1))
    assert_size_stride(primals_98, (432, ), (1, ))
    assert_size_stride(primals_99, (144, 144), (144, 1))
    assert_size_stride(primals_100, (144, ), (1, ))
    assert_size_stride(primals_101, (144, ), (1, ))
    assert_size_stride(primals_102, (144, ), (1, ))
    assert_size_stride(primals_103, (288, 144), (144, 1))
    assert_size_stride(primals_104, (288, ), (1, ))
    assert_size_stride(primals_105, (144, 288), (288, 1))
    assert_size_stride(primals_106, (144, ), (1, ))
    assert_size_stride(primals_107, (144, ), (1, ))
    assert_size_stride(primals_108, (144, ), (1, ))
    assert_size_stride(primals_109, (96, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(primals_110, (96, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_111, (384, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_112, (384, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_113, (128, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_114, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_115, (192, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_116, (192, ), (1, ))
    assert_size_stride(primals_117, (192, ), (1, ))
    assert_size_stride(primals_118, (576, 192), (192, 1))
    assert_size_stride(primals_119, (576, ), (1, ))
    assert_size_stride(primals_120, (192, 192), (192, 1))
    assert_size_stride(primals_121, (192, ), (1, ))
    assert_size_stride(primals_122, (192, ), (1, ))
    assert_size_stride(primals_123, (192, ), (1, ))
    assert_size_stride(primals_124, (384, 192), (192, 1))
    assert_size_stride(primals_125, (384, ), (1, ))
    assert_size_stride(primals_126, (192, 384), (384, 1))
    assert_size_stride(primals_127, (192, ), (1, ))
    assert_size_stride(primals_128, (192, ), (1, ))
    assert_size_stride(primals_129, (192, ), (1, ))
    assert_size_stride(primals_130, (576, 192), (192, 1))
    assert_size_stride(primals_131, (576, ), (1, ))
    assert_size_stride(primals_132, (192, 192), (192, 1))
    assert_size_stride(primals_133, (192, ), (1, ))
    assert_size_stride(primals_134, (192, ), (1, ))
    assert_size_stride(primals_135, (192, ), (1, ))
    assert_size_stride(primals_136, (384, 192), (192, 1))
    assert_size_stride(primals_137, (384, ), (1, ))
    assert_size_stride(primals_138, (192, 384), (384, 1))
    assert_size_stride(primals_139, (192, ), (1, ))
    assert_size_stride(primals_140, (192, ), (1, ))
    assert_size_stride(primals_141, (192, ), (1, ))
    assert_size_stride(primals_142, (576, 192), (192, 1))
    assert_size_stride(primals_143, (576, ), (1, ))
    assert_size_stride(primals_144, (192, 192), (192, 1))
    assert_size_stride(primals_145, (192, ), (1, ))
    assert_size_stride(primals_146, (192, ), (1, ))
    assert_size_stride(primals_147, (192, ), (1, ))
    assert_size_stride(primals_148, (384, 192), (192, 1))
    assert_size_stride(primals_149, (384, ), (1, ))
    assert_size_stride(primals_150, (192, 384), (384, 1))
    assert_size_stride(primals_151, (192, ), (1, ))
    assert_size_stride(primals_152, (192, ), (1, ))
    assert_size_stride(primals_153, (192, ), (1, ))
    assert_size_stride(primals_154, (576, 192), (192, 1))
    assert_size_stride(primals_155, (576, ), (1, ))
    assert_size_stride(primals_156, (192, 192), (192, 1))
    assert_size_stride(primals_157, (192, ), (1, ))
    assert_size_stride(primals_158, (192, ), (1, ))
    assert_size_stride(primals_159, (192, ), (1, ))
    assert_size_stride(primals_160, (384, 192), (192, 1))
    assert_size_stride(primals_161, (384, ), (1, ))
    assert_size_stride(primals_162, (192, 384), (384, 1))
    assert_size_stride(primals_163, (192, ), (1, ))
    assert_size_stride(primals_164, (192, ), (1, ))
    assert_size_stride(primals_165, (192, ), (1, ))
    assert_size_stride(primals_166, (128, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_167, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_168, (512, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_169, (512, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(primals_170, (160, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_171, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_172, (240, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_173, (240, ), (1, ))
    assert_size_stride(primals_174, (240, ), (1, ))
    assert_size_stride(primals_175, (720, 240), (240, 1))
    assert_size_stride(primals_176, (720, ), (1, ))
    assert_size_stride(primals_177, (240, 240), (240, 1))
    assert_size_stride(primals_178, (240, ), (1, ))
    assert_size_stride(primals_179, (240, ), (1, ))
    assert_size_stride(primals_180, (240, ), (1, ))
    assert_size_stride(primals_181, (480, 240), (240, 1))
    assert_size_stride(primals_182, (480, ), (1, ))
    assert_size_stride(primals_183, (240, 480), (480, 1))
    assert_size_stride(primals_184, (240, ), (1, ))
    assert_size_stride(primals_185, (240, ), (1, ))
    assert_size_stride(primals_186, (240, ), (1, ))
    assert_size_stride(primals_187, (720, 240), (240, 1))
    assert_size_stride(primals_188, (720, ), (1, ))
    assert_size_stride(primals_189, (240, 240), (240, 1))
    assert_size_stride(primals_190, (240, ), (1, ))
    assert_size_stride(primals_191, (240, ), (1, ))
    assert_size_stride(primals_192, (240, ), (1, ))
    assert_size_stride(primals_193, (480, 240), (240, 1))
    assert_size_stride(primals_194, (480, ), (1, ))
    assert_size_stride(primals_195, (240, 480), (480, 1))
    assert_size_stride(primals_196, (240, ), (1, ))
    assert_size_stride(primals_197, (240, ), (1, ))
    assert_size_stride(primals_198, (240, ), (1, ))
    assert_size_stride(primals_199, (720, 240), (240, 1))
    assert_size_stride(primals_200, (720, ), (1, ))
    assert_size_stride(primals_201, (240, 240), (240, 1))
    assert_size_stride(primals_202, (240, ), (1, ))
    assert_size_stride(primals_203, (240, ), (1, ))
    assert_size_stride(primals_204, (240, ), (1, ))
    assert_size_stride(primals_205, (480, 240), (240, 1))
    assert_size_stride(primals_206, (480, ), (1, ))
    assert_size_stride(primals_207, (240, 480), (480, 1))
    assert_size_stride(primals_208, (240, ), (1, ))
    assert_size_stride(primals_209, (240, ), (1, ))
    assert_size_stride(primals_210, (240, ), (1, ))
    assert_size_stride(primals_211, (160, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(primals_212, (160, 320, 3, 3), (2880, 9, 3, 1))
    assert_size_stride(primals_213, (640, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_214, (1000, 640), (640, 1))
    assert_size_stride(primals_215, (1000, ), (1, ))
    assert_size_stride(primals_216, (), ())
    assert_size_stride(primals_217, (16, ), (1, ))
    assert_size_stride(primals_218, (16, ), (1, ))
    assert_size_stride(primals_219, (), ())
    assert_size_stride(primals_220, (64, ), (1, ))
    assert_size_stride(primals_221, (64, ), (1, ))
    assert_size_stride(primals_222, (), ())
    assert_size_stride(primals_223, (64, ), (1, ))
    assert_size_stride(primals_224, (64, ), (1, ))
    assert_size_stride(primals_225, (), ())
    assert_size_stride(primals_226, (32, ), (1, ))
    assert_size_stride(primals_227, (32, ), (1, ))
    assert_size_stride(primals_228, (), ())
    assert_size_stride(primals_229, (128, ), (1, ))
    assert_size_stride(primals_230, (128, ), (1, ))
    assert_size_stride(primals_231, (), ())
    assert_size_stride(primals_232, (128, ), (1, ))
    assert_size_stride(primals_233, (128, ), (1, ))
    assert_size_stride(primals_234, (), ())
    assert_size_stride(primals_235, (64, ), (1, ))
    assert_size_stride(primals_236, (64, ), (1, ))
    assert_size_stride(primals_237, (), ())
    assert_size_stride(primals_238, (256, ), (1, ))
    assert_size_stride(primals_239, (256, ), (1, ))
    assert_size_stride(primals_240, (), ())
    assert_size_stride(primals_241, (256, ), (1, ))
    assert_size_stride(primals_242, (256, ), (1, ))
    assert_size_stride(primals_243, (), ())
    assert_size_stride(primals_244, (64, ), (1, ))
    assert_size_stride(primals_245, (64, ), (1, ))
    assert_size_stride(primals_246, (), ())
    assert_size_stride(primals_247, (256, ), (1, ))
    assert_size_stride(primals_248, (256, ), (1, ))
    assert_size_stride(primals_249, (), ())
    assert_size_stride(primals_250, (256, ), (1, ))
    assert_size_stride(primals_251, (256, ), (1, ))
    assert_size_stride(primals_252, (), ())
    assert_size_stride(primals_253, (64, ), (1, ))
    assert_size_stride(primals_254, (64, ), (1, ))
    assert_size_stride(primals_255, (), ())
    assert_size_stride(primals_256, (256, ), (1, ))
    assert_size_stride(primals_257, (256, ), (1, ))
    assert_size_stride(primals_258, (), ())
    assert_size_stride(primals_259, (256, ), (1, ))
    assert_size_stride(primals_260, (256, ), (1, ))
    assert_size_stride(primals_261, (), ())
    assert_size_stride(primals_262, (96, ), (1, ))
    assert_size_stride(primals_263, (96, ), (1, ))
    assert_size_stride(primals_264, (), ())
    assert_size_stride(primals_265, (96, ), (1, ))
    assert_size_stride(primals_266, (96, ), (1, ))
    assert_size_stride(primals_267, (), ())
    assert_size_stride(primals_268, (96, ), (1, ))
    assert_size_stride(primals_269, (96, ), (1, ))
    assert_size_stride(primals_270, (), ())
    assert_size_stride(primals_271, (96, ), (1, ))
    assert_size_stride(primals_272, (96, ), (1, ))
    assert_size_stride(primals_273, (), ())
    assert_size_stride(primals_274, (384, ), (1, ))
    assert_size_stride(primals_275, (384, ), (1, ))
    assert_size_stride(primals_276, (), ())
    assert_size_stride(primals_277, (384, ), (1, ))
    assert_size_stride(primals_278, (384, ), (1, ))
    assert_size_stride(primals_279, (), ())
    assert_size_stride(primals_280, (128, ), (1, ))
    assert_size_stride(primals_281, (128, ), (1, ))
    assert_size_stride(primals_282, (), ())
    assert_size_stride(primals_283, (128, ), (1, ))
    assert_size_stride(primals_284, (128, ), (1, ))
    assert_size_stride(primals_285, (), ())
    assert_size_stride(primals_286, (128, ), (1, ))
    assert_size_stride(primals_287, (128, ), (1, ))
    assert_size_stride(primals_288, (), ())
    assert_size_stride(primals_289, (128, ), (1, ))
    assert_size_stride(primals_290, (128, ), (1, ))
    assert_size_stride(primals_291, (), ())
    assert_size_stride(primals_292, (512, ), (1, ))
    assert_size_stride(primals_293, (512, ), (1, ))
    assert_size_stride(primals_294, (), ())
    assert_size_stride(primals_295, (512, ), (1, ))
    assert_size_stride(primals_296, (512, ), (1, ))
    assert_size_stride(primals_297, (), ())
    assert_size_stride(primals_298, (160, ), (1, ))
    assert_size_stride(primals_299, (160, ), (1, ))
    assert_size_stride(primals_300, (), ())
    assert_size_stride(primals_301, (160, ), (1, ))
    assert_size_stride(primals_302, (160, ), (1, ))
    assert_size_stride(primals_303, (), ())
    assert_size_stride(primals_304, (160, ), (1, ))
    assert_size_stride(primals_305, (160, ), (1, ))
    assert_size_stride(primals_306, (), ())
    assert_size_stride(primals_307, (160, ), (1, ))
    assert_size_stride(primals_308, (160, ), (1, ))
    assert_size_stride(primals_309, (), ())
    assert_size_stride(primals_310, (640, ), (1, ))
    assert_size_stride(primals_311, (640, ), (1, ))
    assert_size_stride(primals_312, (8, 3, 256, 256), (196608, 65536, 256, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_312, primals_65, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 16, 128, 128), (262144, 16384, 128, 1))
        buf1 = empty_strided((1, 16, 1, 1, 16), (256, 16, 256, 256, 1), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((1, 16, 1, 1, 16), (256, 16, 256, 256, 1), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((1, 16, 1, 1, 16), (256, 16, 256, 256, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_cuda_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_0.run(buf0, buf1, buf2, buf3, 256, 8192, grid=grid(256), stream=stream0)
        buf4 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((1, 16, 1, 1), (16, 1, 16, 16), device='cuda', dtype=torch.float32)
        buf7 = empty((16, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_1.run(buf1, buf2, buf3, primals_217, primals_218, buf4, buf5, buf7, primals_217, primals_218, 16, 16, grid=grid(16), stream=stream0)
        del primals_217
        del primals_218
        buf9 = empty((8, 16, 128, 128), device='cuda', dtype=torch.float32)
        buf510 = empty((8, 16, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_2.run(buf0, buf4, buf5, primals_1, primals_2, buf9, buf510, 2097152, grid=grid(2097152), stream=stream0)
        del buf5
        del primals_2
        # Source Nodes: [x_6], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_66, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (8, 64, 128, 128), (1048576, 16384, 128, 1))
        buf11 = empty_strided((1, 64, 1, 1, 16), (1024, 16, 1024, 1024, 1), device='cuda', dtype=torch.float32)
        buf12 = empty_strided((1, 64, 1, 1, 16), (1024, 16, 1024, 1024, 1), device='cuda', dtype=torch.float32)
        buf13 = empty_strided((1, 64, 1, 1, 16), (1024, 16, 1024, 1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf10, buf11, buf12, buf13, 1024, 8192, grid=grid(1024), stream=stream0)
        buf14 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf15 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf17 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_7], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_4.run(buf11, buf12, buf13, primals_220, primals_221, buf14, buf15, buf17, primals_220, primals_221, 64, 16, grid=grid(64), stream=stream0)
        del primals_220
        del primals_221
        buf19 = empty((8, 64, 128, 128), device='cuda', dtype=torch.float32)
        buf509 = empty((8, 64, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11, x_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_5.run(buf10, buf14, buf15, primals_3, primals_4, buf19, buf509, 8388608, grid=grid(8388608), stream=stream0)
        del primals_4
        # Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_67, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf20, (8, 64, 128, 128), (1048576, 16384, 128, 1))
        buf21 = buf13; del buf13  # reuse
        buf22 = buf12; del buf12  # reuse
        buf23 = buf11; del buf11  # reuse
        # Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_3.run(buf20, buf21, buf22, buf23, 1024, 8192, grid=grid(1024), stream=stream0)
        buf24 = buf15; del buf15  # reuse
        buf25 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf27 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_4.run(buf21, buf22, buf23, primals_223, primals_224, buf24, buf25, buf27, primals_223, primals_224, 64, 16, grid=grid(64), stream=stream0)
        del primals_223
        del primals_224
        buf29 = empty((8, 64, 128, 128), device='cuda', dtype=torch.float32)
        buf508 = empty((8, 64, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13, x_17], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_5.run(buf20, buf24, buf25, primals_5, primals_6, buf29, buf508, 8388608, grid=grid(8388608), stream=stream0)
        del primals_6
        # Source Nodes: [x_20], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_68, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 32, 128, 128), (524288, 16384, 128, 1))
        buf31 = empty_strided((1, 32, 1, 1, 16), (512, 16, 512, 512, 1), device='cuda', dtype=torch.float32)
        buf32 = empty_strided((1, 32, 1, 1, 16), (512, 16, 512, 512, 1), device='cuda', dtype=torch.float32)
        buf33 = empty_strided((1, 32, 1, 1, 16), (512, 16, 512, 512, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_6.run(buf30, buf31, buf32, buf33, 512, 8192, grid=grid(512), stream=stream0)
        buf34 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf35 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf37 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_7.run(buf31, buf32, buf33, primals_226, primals_227, buf34, buf35, buf37, primals_226, primals_227, 32, 16, grid=grid(32), stream=stream0)
        del primals_226
        del primals_227
        buf38 = empty((8, 32, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_8.run(buf30, buf34, buf35, primals_7, primals_8, buf38, 4194304, grid=grid(4194304), stream=stream0)
        del buf35
        del primals_8
        # Source Nodes: [x_28], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, primals_69, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (8, 128, 128, 128), (2097152, 16384, 128, 1))
        buf40 = reinterpret_tensor(buf33, (1, 128, 1, 1, 4), (512, 1, 512, 512, 128), 0); del buf33  # reuse
        buf41 = reinterpret_tensor(buf32, (1, 128, 1, 1, 4), (512, 1, 512, 512, 128), 0); del buf32  # reuse
        buf42 = reinterpret_tensor(buf31, (1, 128, 1, 1, 4), (512, 1, 512, 512, 128), 0); del buf31  # reuse
        # Source Nodes: [x_29], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_9.run(buf39, buf40, buf41, buf42, 512, 32768, grid=grid(512), stream=stream0)
        buf43 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf44 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf46 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_29], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_10.run(buf40, buf41, buf42, primals_229, primals_230, buf43, buf44, buf46, primals_229, primals_230, 128, 4, grid=grid(128), stream=stream0)
        del primals_229
        del primals_230
        buf48 = empty((8, 128, 128, 128), device='cuda', dtype=torch.float32)
        buf507 = empty((8, 128, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_29, x_33], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_11.run(buf39, buf43, buf44, primals_9, primals_10, buf48, buf507, 16777216, grid=grid(16777216), stream=stream0)
        del primals_10
        # Source Nodes: [x_34], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, primals_70, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf49, (8, 128, 64, 64), (524288, 4096, 64, 1))
        buf50 = buf42; del buf42  # reuse
        buf51 = buf41; del buf41  # reuse
        buf52 = buf40; del buf40  # reuse
        # Source Nodes: [x_35], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_12.run(buf49, buf50, buf51, buf52, 512, 8192, grid=grid(512), stream=stream0)
        buf53 = buf44; del buf44  # reuse
        buf54 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf56 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_35], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_13.run(buf50, buf51, buf52, primals_232, primals_233, buf53, buf54, buf56, primals_232, primals_233, 128, 4, grid=grid(128), stream=stream0)
        del primals_232
        del primals_233
        buf58 = empty((8, 128, 64, 64), device='cuda', dtype=torch.float32)
        buf506 = empty((8, 128, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_35, x_39], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_14.run(buf49, buf53, buf54, primals_11, primals_12, buf58, buf506, 4194304, grid=grid(4194304), stream=stream0)
        del primals_12
        # Source Nodes: [x_42], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, primals_71, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf60 = reinterpret_tensor(buf3, (1, 64, 1, 1, 4), (256, 1, 256, 256, 64), 0); del buf3  # reuse
        buf61 = reinterpret_tensor(buf2, (1, 64, 1, 1, 4), (256, 1, 256, 256, 64), 0); del buf2  # reuse
        buf62 = reinterpret_tensor(buf1, (1, 64, 1, 1, 4), (256, 1, 256, 256, 64), 0); del buf1  # reuse
        # Source Nodes: [x_43], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_15.run(buf59, buf60, buf61, buf62, 256, 8192, grid=grid(256), stream=stream0)
        buf63 = buf25; del buf25  # reuse
        buf64 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf66 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_43], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf60, buf61, buf62, primals_235, primals_236, buf63, buf64, buf66, primals_235, primals_236, 64, 4, grid=grid(64), stream=stream0)
        del primals_235
        del primals_236
        buf67 = empty((8, 64, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_43], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_17.run(buf59, buf63, buf64, primals_13, primals_14, buf67, 2097152, grid=grid(2097152), stream=stream0)
        del primals_14
        # Source Nodes: [x_50], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_72, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        buf69 = reinterpret_tensor(buf62, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf62  # reuse
        buf70 = reinterpret_tensor(buf61, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf61  # reuse
        buf72 = reinterpret_tensor(buf60, (256, ), (1, ), 0); del buf60  # reuse
        # Source Nodes: [x_51], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf68, primals_238, primals_239, buf69, buf70, buf72, primals_238, primals_239, 256, 32768, grid=grid(256), stream=stream0)
        del primals_238
        del primals_239
        buf74 = empty((8, 256, 64, 64), device='cuda', dtype=torch.float32)
        buf505 = empty((8, 256, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_51, x_55], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_19.run(buf68, buf69, buf70, primals_15, primals_16, buf74, buf505, 8388608, grid=grid(8388608), stream=stream0)
        del primals_16
        # Source Nodes: [x_56], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, primals_73, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf75, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        buf76 = buf70; del buf70  # reuse
        buf77 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf79 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf75, primals_241, primals_242, buf76, buf77, buf79, primals_241, primals_242, 256, 32768, grid=grid(256), stream=stream0)
        del primals_241
        del primals_242
        buf81 = empty((8, 256, 64, 64), device='cuda', dtype=torch.float32)
        buf504 = empty((8, 256, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_57, x_61], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_19.run(buf75, buf76, buf77, primals_17, primals_18, buf81, buf504, 8388608, grid=grid(8388608), stream=stream0)
        del primals_18
        # Source Nodes: [x_64], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, primals_74, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf83 = reinterpret_tensor(buf77, (1, 64, 1, 1, 4), (256, 1, 256, 256, 64), 0); del buf77  # reuse
        buf84 = empty_strided((1, 64, 1, 1, 4), (256, 1, 256, 256, 64), device='cuda', dtype=torch.float32)
        buf85 = empty_strided((1, 64, 1, 1, 4), (256, 1, 256, 256, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_65], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_15.run(buf82, buf83, buf84, buf85, 256, 8192, grid=grid(256), stream=stream0)
        buf86 = buf64; del buf64  # reuse
        buf87 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf89 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_65], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf83, buf84, buf85, primals_244, primals_245, buf86, buf87, buf89, primals_244, primals_245, 64, 4, grid=grid(64), stream=stream0)
        del primals_244
        del primals_245
        buf90 = empty((8, 64, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_65, x_72], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_20.run(buf82, buf86, buf87, primals_19, primals_20, buf67, buf90, 2097152, grid=grid(2097152), stream=stream0)
        del primals_20
        # Source Nodes: [x_73], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, primals_75, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        buf92 = reinterpret_tensor(buf85, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf85  # reuse
        buf93 = reinterpret_tensor(buf84, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf84  # reuse
        buf95 = reinterpret_tensor(buf83, (256, ), (1, ), 0); del buf83  # reuse
        # Source Nodes: [x_74], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf91, primals_247, primals_248, buf92, buf93, buf95, primals_247, primals_248, 256, 32768, grid=grid(256), stream=stream0)
        del primals_247
        del primals_248
        buf97 = empty((8, 256, 64, 64), device='cuda', dtype=torch.float32)
        buf503 = empty((8, 256, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74, x_78], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_19.run(buf91, buf92, buf93, primals_21, primals_22, buf97, buf503, 8388608, grid=grid(8388608), stream=stream0)
        del primals_22
        # Source Nodes: [x_79], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, primals_76, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf98, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        buf99 = buf93; del buf93  # reuse
        buf100 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf102 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_80], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf98, primals_250, primals_251, buf99, buf100, buf102, primals_250, primals_251, 256, 32768, grid=grid(256), stream=stream0)
        del primals_250
        del primals_251
        buf104 = empty((8, 256, 64, 64), device='cuda', dtype=torch.float32)
        buf502 = empty((8, 256, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_80, x_84], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_19.run(buf98, buf99, buf100, primals_23, primals_24, buf104, buf502, 8388608, grid=grid(8388608), stream=stream0)
        del primals_24
        # Source Nodes: [x_87], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, primals_77, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (8, 64, 64, 64), (262144, 4096, 64, 1))
        buf106 = reinterpret_tensor(buf100, (1, 64, 1, 1, 4), (256, 1, 256, 256, 64), 0); del buf100  # reuse
        buf107 = empty_strided((1, 64, 1, 1, 4), (256, 1, 256, 256, 64), device='cuda', dtype=torch.float32)
        buf108 = empty_strided((1, 64, 1, 1, 4), (256, 1, 256, 256, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_88], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_15.run(buf105, buf106, buf107, buf108, 256, 8192, grid=grid(256), stream=stream0)
        buf109 = buf87; del buf87  # reuse
        buf110 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf112 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_88], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_16.run(buf106, buf107, buf108, primals_253, primals_254, buf109, buf110, buf112, primals_253, primals_254, 64, 4, grid=grid(64), stream=stream0)
        del primals_253
        del primals_254
        buf113 = empty((8, 64, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_88, x_95], Original ATen: [aten._native_batch_norm_legit_functional, aten.add]
        triton_poi_fused__native_batch_norm_legit_functional_add_20.run(buf105, buf109, buf110, primals_25, primals_26, buf90, buf113, 2097152, grid=grid(2097152), stream=stream0)
        del buf110
        del primals_26
        # Source Nodes: [x_96], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf113, primals_78, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (8, 256, 64, 64), (1048576, 4096, 64, 1))
        buf115 = reinterpret_tensor(buf108, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf108  # reuse
        buf116 = reinterpret_tensor(buf107, (1, 256, 1, 1), (256, 1, 256, 256), 0); del buf107  # reuse
        buf118 = reinterpret_tensor(buf106, (256, ), (1, ), 0); del buf106  # reuse
        # Source Nodes: [x_97], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf114, primals_256, primals_257, buf115, buf116, buf118, primals_256, primals_257, 256, 32768, grid=grid(256), stream=stream0)
        del primals_256
        del primals_257
        buf120 = empty((8, 256, 64, 64), device='cuda', dtype=torch.float32)
        buf501 = empty((8, 256, 64, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_101, x_97], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_19.run(buf114, buf115, buf116, primals_27, primals_28, buf120, buf501, 8388608, grid=grid(8388608), stream=stream0)
        del primals_28
        # Source Nodes: [x_102], Original ATen: [aten.convolution]
        buf121 = extern_kernels.convolution(buf120, primals_79, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf121, (8, 256, 32, 32), (262144, 1024, 32, 1))
        buf122 = buf116; del buf116  # reuse
        buf123 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf125 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_103], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf121, primals_259, primals_260, buf122, buf123, buf125, primals_259, primals_260, 256, 8192, grid=grid(256), stream=stream0)
        del primals_259
        del primals_260
        buf127 = empty((8, 256, 32, 32), device='cuda', dtype=torch.float32)
        buf500 = empty((8, 256, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_103, x_107], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_22.run(buf121, buf122, buf123, primals_29, primals_30, buf127, buf500, 2097152, grid=grid(2097152), stream=stream0)
        del buf123
        del primals_30
        # Source Nodes: [x_110], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, primals_80, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (8, 96, 32, 32), (98304, 1024, 32, 1))
        buf129 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf130 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf132 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_111], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_23.run(buf128, primals_262, primals_263, buf129, buf130, buf132, primals_262, primals_263, 96, 8192, grid=grid(96), stream=stream0)
        del primals_262
        del primals_263
        buf133 = empty((8, 96, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_111], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_24.run(buf128, buf129, buf130, primals_31, primals_32, buf133, 786432, grid=grid(786432), stream=stream0)
        del primals_32
        # Source Nodes: [x_118], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, primals_81, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (8, 96, 32, 32), (98304, 1024, 32, 1))
        buf135 = buf130; del buf130  # reuse
        buf136 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf138 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_119], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_23.run(buf134, primals_265, primals_266, buf135, buf136, buf138, primals_265, primals_266, 96, 8192, grid=grid(96), stream=stream0)
        del primals_265
        del primals_266
        buf140 = empty((8, 96, 32, 32), device='cuda', dtype=torch.float32)
        buf499 = empty((8, 96, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_119, x_123], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_25.run(buf134, buf135, buf136, primals_33, primals_34, buf140, buf499, 786432, grid=grid(786432), stream=stream0)
        del primals_34
        # Source Nodes: [x_124], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (8, 144, 32, 32), (147456, 1024, 32, 1))
        buf143 = empty_strided((32, 256, 1), (1, 32, 8192), device='cuda', dtype=torch.float32)
        buf145 = empty((32, 256, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_26.run(buf141, buf143, buf145, 8192, 144, grid=grid(8192), stream=stream0)
        buf146 = empty((8192, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___attn_qkv, getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_27.run(buf145, primals_83, primals_84, buf146, 1179648, grid=grid(1179648), stream=stream0)
        del primals_84
        buf147 = empty((8192, 432), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_86, buf146, reinterpret_tensor(primals_85, (144, 432), (1, 144), 0), alpha=1, beta=1, out=buf147)
        del primals_86
        # Source Nodes: [x_127], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf148 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf147, (32, 4, 256, 36), (110592, 36, 432, 1), 0), reinterpret_tensor(buf147, (32, 4, 256, 36), (110592, 36, 432, 1), 144), reinterpret_tensor(buf147, (32, 4, 256, 36), (110592, 36, 432, 1), 288), None, True)
        buf149 = buf148[0]
        buf150 = buf148[1]
        buf151 = buf148[2]
        buf152 = buf148[3]
        del buf148
        buf153 = empty((8192, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf149, (8192, 144), (144, 1), 0), reinterpret_tensor(primals_87, (144, 144), (1, 144), 0), out=buf153)
        buf154 = empty_strided((32, 256, 1), (256, 1, 8192), device='cuda', dtype=torch.float32)
        buf155 = empty_strided((32, 256, 1), (256, 1, 8192), device='cuda', dtype=torch.float32)
        buf497 = empty((32, 256, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm2, x_131], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_red_fused_add_native_layer_norm_native_layer_norm_backward_28.run(buf141, buf153, primals_88, buf154, buf155, buf497, 8192, 144, grid=grid(8192), stream=stream0)
        buf157 = empty((32, 256, 144), device='cuda', dtype=torch.float32)
        buf158 = empty((8192, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm2, x_131, x_132], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_29.run(buf141, buf153, primals_88, buf154, buf155, primals_89, primals_90, buf157, buf158, 32, 36864, grid=grid(32, 36864), stream=stream0)
        del primals_90
        buf159 = empty((8192, 288), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_132], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_92, buf158, reinterpret_tensor(primals_91, (144, 288), (1, 144), 0), alpha=1, beta=1, out=buf159)
        del primals_92
        buf160 = empty((8192, 288), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_133, x_136], Original ATen: [aten.silu, aten.view]
        triton_poi_fused_silu_view_30.run(buf159, buf160, 2359296, grid=grid(2359296), stream=stream0)
        buf161 = empty((8192, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf160, reinterpret_tensor(primals_93, (288, 144), (1, 288), 0), out=buf161)
        buf162 = reinterpret_tensor(buf161, (32, 256, 144), (36864, 144, 1), 0); del buf161  # reuse
        # Source Nodes: [x_131, x_138], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(buf162, buf141, buf153, primals_88, primals_94, 32, 36864, grid=grid(32, 36864), stream=stream0)
        del primals_88
        del primals_94
        buf166 = reinterpret_tensor(buf153, (32, 256, 144), (36864, 144, 1), 0); del buf153  # reuse
        buf167 = reinterpret_tensor(buf141, (8192, 144), (144, 1), 0); del buf141  # reuse
        buf496 = reinterpret_tensor(buf155, (32, 256, 1), (256, 1, 1), 0); del buf155  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___attn_qkv, getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_32.run(buf162, primals_95, primals_96, buf166, buf167, buf496, 8192, 144, grid=grid(8192), stream=stream0)
        del primals_96
        buf168 = empty((8192, 432), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_98, buf167, reinterpret_tensor(primals_97, (144, 432), (1, 144), 0), alpha=1, beta=1, out=buf168)
        del primals_98
        # Source Nodes: [x_139], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf169 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf168, (32, 4, 256, 36), (110592, 36, 432, 1), 0), reinterpret_tensor(buf168, (32, 4, 256, 36), (110592, 36, 432, 1), 144), reinterpret_tensor(buf168, (32, 4, 256, 36), (110592, 36, 432, 1), 288), None, True)
        buf170 = buf169[0]
        buf171 = buf169[1]
        buf172 = buf169[2]
        buf173 = buf169[3]
        del buf169
        buf174 = empty((8192, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf170, (8192, 144), (144, 1), 0), reinterpret_tensor(primals_99, (144, 144), (1, 144), 0), out=buf174)
        buf178 = empty((32, 256, 144), device='cuda', dtype=torch.float32)
        buf179 = empty((8192, 144), device='cuda', dtype=torch.float32)
        buf495 = reinterpret_tensor(buf154, (32, 256, 1), (256, 1, 1), 0); del buf154  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___1___norm2, x_143, x_144], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_33.run(buf162, buf174, primals_100, primals_101, primals_102, buf178, buf179, buf495, 8192, 144, grid=grid(8192), stream=stream0)
        del primals_102
        buf180 = empty((8192, 288), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_144], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_104, buf179, reinterpret_tensor(primals_103, (144, 288), (1, 144), 0), alpha=1, beta=1, out=buf180)
        del primals_104
        buf181 = empty((8192, 288), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_145, x_148], Original ATen: [aten.silu, aten.view]
        triton_poi_fused_silu_view_30.run(buf180, buf181, 2359296, grid=grid(2359296), stream=stream0)
        buf182 = empty((8192, 144), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf181, reinterpret_tensor(primals_105, (288, 144), (1, 288), 0), out=buf182)
        buf183 = reinterpret_tensor(buf182, (32, 256, 144), (36864, 144, 1), 0); del buf182  # reuse
        buf187 = empty((32, 256, 144), device='cuda', dtype=torch.float32)
        buf494 = empty((32, 256, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_143, x_151, x_152], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_34.run(buf183, buf162, buf174, primals_100, primals_106, buf187, buf494, 8192, 144, grid=grid(8192), stream=stream0)
        del primals_100
        del primals_106
        buf188 = reinterpret_tensor(buf183, (8, 144, 32, 32), (147456, 1024, 32, 1), 0); del buf183  # reuse
        # Source Nodes: [x_155], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_35.run(buf187, primals_107, primals_108, buf188, 1152, 1024, grid=grid(1152, 1024), stream=stream0)
        del primals_108
        # Source Nodes: [x_156], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf188, primals_109, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (8, 96, 32, 32), (98304, 1024, 32, 1))
        buf190 = buf136; del buf136  # reuse
        buf191 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf193 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_157], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_23.run(buf189, primals_268, primals_269, buf190, buf191, buf193, primals_268, primals_269, 96, 8192, grid=grid(96), stream=stream0)
        del primals_268
        del primals_269
        buf194 = empty((8, 96, 32, 32), device='cuda', dtype=torch.float32)
        buf493 = empty((8, 96, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_157], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_36.run(buf189, buf190, buf191, primals_35, primals_36, buf194, buf493, 786432, grid=grid(786432), stream=stream0)
        del primals_36
        buf195 = empty((8, 192, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_5], Original ATen: [aten.cat]
        triton_poi_fused_cat_37.run(buf133, buf194, buf195, 1572864, grid=grid(1572864), stream=stream0)
        # Source Nodes: [x_162], Original ATen: [aten.convolution]
        buf196 = extern_kernels.convolution(buf195, primals_110, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf196, (8, 96, 32, 32), (98304, 1024, 32, 1))
        buf197 = buf191; del buf191  # reuse
        buf198 = empty_strided((1, 96, 1, 1), (96, 1, 96, 96), device='cuda', dtype=torch.float32)
        buf200 = empty((96, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_163], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_23.run(buf196, primals_271, primals_272, buf197, buf198, buf200, primals_271, primals_272, 96, 8192, grid=grid(96), stream=stream0)
        del primals_271
        del primals_272
        buf202 = buf194; del buf194  # reuse
        buf492 = empty((8, 96, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_6, x_163], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_25.run(buf196, buf197, buf198, primals_37, primals_38, buf202, buf492, 786432, grid=grid(786432), stream=stream0)
        del buf198
        del primals_38
        # Source Nodes: [x_168], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, primals_111, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (8, 384, 32, 32), (393216, 1024, 32, 1))
        buf204 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf205 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf207 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_169], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf203, primals_274, primals_275, buf204, buf205, buf207, primals_274, primals_275, 384, 8192, grid=grid(384), stream=stream0)
        del primals_274
        del primals_275
        buf209 = empty((8, 384, 32, 32), device='cuda', dtype=torch.float32)
        buf491 = empty((8, 384, 32, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_169, x_173], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_39.run(buf203, buf204, buf205, primals_39, primals_40, buf209, buf491, 3145728, grid=grid(3145728), stream=stream0)
        del primals_40
        # Source Nodes: [x_174], Original ATen: [aten.convolution]
        buf210 = extern_kernels.convolution(buf209, primals_112, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=384, bias=None)
        assert_size_stride(buf210, (8, 384, 16, 16), (98304, 256, 16, 1))
        buf211 = buf205; del buf205  # reuse
        buf212 = empty_strided((1, 384, 1, 1), (384, 1, 384, 384), device='cuda', dtype=torch.float32)
        buf214 = empty((384, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_175], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_40.run(buf210, primals_277, primals_278, buf211, buf212, buf214, primals_277, primals_278, 384, 2048, grid=grid(384), stream=stream0)
        del primals_277
        del primals_278
        buf216 = empty((8, 384, 16, 16), device='cuda', dtype=torch.float32)
        buf490 = empty((8, 384, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_175, x_179], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_41.run(buf210, buf211, buf212, primals_41, primals_42, buf216, buf490, 786432, grid=grid(786432), stream=stream0)
        del buf212
        del primals_42
        # Source Nodes: [x_182], Original ATen: [aten.convolution]
        buf217 = extern_kernels.convolution(buf216, primals_113, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (8, 128, 16, 16), (32768, 256, 16, 1))
        buf218 = buf54; del buf54  # reuse
        buf219 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf221 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_183], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf217, primals_280, primals_281, buf218, buf219, buf221, primals_280, primals_281, 128, 2048, grid=grid(128), stream=stream0)
        del primals_280
        del primals_281
        buf222 = empty((8, 128, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_183], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_43.run(buf217, buf218, buf219, primals_43, primals_44, buf222, 262144, grid=grid(262144), stream=stream0)
        del primals_44
        # Source Nodes: [x_190], Original ATen: [aten.convolution]
        buf223 = extern_kernels.convolution(buf222, primals_114, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf223, (8, 128, 16, 16), (32768, 256, 16, 1))
        buf224 = buf219; del buf219  # reuse
        buf225 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf227 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_191], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf223, primals_283, primals_284, buf224, buf225, buf227, primals_283, primals_284, 128, 2048, grid=grid(128), stream=stream0)
        del primals_283
        del primals_284
        buf229 = empty((8, 128, 16, 16), device='cuda', dtype=torch.float32)
        buf489 = empty((8, 128, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_191, x_195], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_44.run(buf223, buf224, buf225, primals_45, primals_46, buf229, buf489, 262144, grid=grid(262144), stream=stream0)
        del primals_46
        # Source Nodes: [x_196], Original ATen: [aten.convolution]
        buf230 = extern_kernels.convolution(buf229, primals_115, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf230, (8, 192, 16, 16), (49152, 256, 16, 1))
        buf231 = empty_strided((32, 64, 1, 2), (1, 32, 4096, 2048), device='cuda', dtype=torch.float32)
        buf232 = empty_strided((32, 64, 1, 2), (1, 32, 4096, 2048), device='cuda', dtype=torch.float32)
        buf233 = empty_strided((32, 64, 1, 2), (1, 32, 4096, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_45.run(buf230, buf231, buf232, buf233, 4096, 96, grid=grid(4096), stream=stream0)
        buf234 = empty_strided((32, 64, 1), (64, 1, 2048), device='cuda', dtype=torch.float32)
        buf235 = empty_strided((32, 64, 1), (1, 32, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_46.run(buf231, buf232, buf233, buf234, buf235, 2048, 2, grid=grid(2048), stream=stream0)
        buf237 = empty((32, 64, 192), device='cuda', dtype=torch.float32)
        buf238 = empty((2048, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___attn_qkv, getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_47.run(buf230, buf234, buf235, primals_116, primals_117, buf237, buf238, 32, 12288, grid=grid(32, 12288), stream=stream0)
        del primals_117
        buf239 = reinterpret_tensor(buf174, (2048, 576), (576, 1), 0); del buf174  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_119, buf238, reinterpret_tensor(primals_118, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf239)
        del primals_119
        # Source Nodes: [x_199], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf240 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf239, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf239, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf239, (32, 4, 64, 48), (36864, 48, 576, 1), 384), None, True)
        buf241 = buf240[0]
        buf242 = buf240[1]
        buf243 = buf240[2]
        buf244 = buf240[3]
        del buf240
        buf245 = empty((2048, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf241, (2048, 192), (192, 1), 0), reinterpret_tensor(primals_120, (192, 192), (1, 192), 0), out=buf245)
        buf246 = reinterpret_tensor(buf233, (32, 64, 1, 2), (128, 2, 4096, 1), 0); del buf233  # reuse
        buf247 = reinterpret_tensor(buf232, (32, 64, 1, 2), (128, 2, 4096, 1), 0); del buf232  # reuse
        buf248 = reinterpret_tensor(buf231, (32, 64, 1, 2), (128, 2, 4096, 1), 0); del buf231  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm2, x_203], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_48.run(buf230, buf245, primals_121, buf246, buf247, buf248, 4096, 96, grid=grid(4096), stream=stream0)
        buf249 = buf234; del buf234  # reuse
        buf250 = empty_strided((32, 64, 1), (64, 1, 2048), device='cuda', dtype=torch.float32)
        buf487 = empty((32, 64, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm2, x_203], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_49.run(buf246, buf247, buf248, buf249, buf250, buf487, 2048, 2, grid=grid(2048), stream=stream0)
        del buf246
        del buf247
        del buf248
        buf252 = empty((32, 64, 192), device='cuda', dtype=torch.float32)
        buf253 = empty((2048, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm2, x_203, x_204], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_50.run(buf230, buf245, primals_121, buf249, buf250, primals_122, primals_123, buf252, buf253, 32, 12288, grid=grid(32, 12288), stream=stream0)
        del primals_123
        buf254 = empty((2048, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_204], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_125, buf253, reinterpret_tensor(primals_124, (192, 384), (1, 192), 0), alpha=1, beta=1, out=buf254)
        del primals_125
        buf255 = empty((2048, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_205, x_208], Original ATen: [aten.silu, aten.view]
        triton_poi_fused_silu_view_51.run(buf254, buf255, 786432, grid=grid(786432), stream=stream0)
        buf256 = empty((2048, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf255, reinterpret_tensor(primals_126, (384, 192), (1, 384), 0), out=buf256)
        buf257 = reinterpret_tensor(buf245, (32, 64, 192), (12288, 192, 1), 0); del buf245  # reuse
        # Source Nodes: [x_203, x_210], Original ATen: [aten.add]
        triton_poi_fused_add_52.run(buf257, buf230, primals_121, buf256, primals_127, 32, 12288, grid=grid(32, 12288), stream=stream0)
        del primals_121
        del primals_127
        buf261 = reinterpret_tensor(buf256, (32, 64, 192), (12288, 192, 1), 0); del buf256  # reuse
        buf262 = reinterpret_tensor(buf230, (2048, 192), (192, 1), 0); del buf230  # reuse
        buf486 = reinterpret_tensor(buf250, (32, 64, 1), (64, 1, 1), 0); del buf250  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___attn_qkv, getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_53.run(buf257, primals_128, primals_129, buf261, buf262, buf486, 2048, 192, grid=grid(2048), stream=stream0)
        del primals_129
        buf263 = reinterpret_tensor(buf162, (2048, 576), (576, 1), 0); del buf162  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_131, buf262, reinterpret_tensor(primals_130, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf263)
        del primals_131
        # Source Nodes: [x_211], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf264 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf263, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf263, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf263, (32, 4, 64, 48), (36864, 48, 576, 1), 384), None, True)
        buf265 = buf264[0]
        buf266 = buf264[1]
        buf267 = buf264[2]
        buf268 = buf264[3]
        del buf264
        buf269 = empty((2048, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf265, (2048, 192), (192, 1), 0), reinterpret_tensor(primals_132, (192, 192), (1, 192), 0), out=buf269)
        buf273 = empty((32, 64, 192), device='cuda', dtype=torch.float32)
        buf274 = empty((2048, 192), device='cuda', dtype=torch.float32)
        buf485 = reinterpret_tensor(buf249, (32, 64, 1), (64, 1, 1), 0); del buf249  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___1___norm2, x_215, x_216], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_54.run(buf257, buf269, primals_133, primals_134, primals_135, buf273, buf274, buf485, 2048, 192, grid=grid(2048), stream=stream0)
        del primals_135
        buf275 = empty((2048, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_216], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_137, buf274, reinterpret_tensor(primals_136, (192, 384), (1, 192), 0), alpha=1, beta=1, out=buf275)
        del primals_137
        buf276 = empty((2048, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_217, x_220], Original ATen: [aten.silu, aten.view]
        triton_poi_fused_silu_view_51.run(buf275, buf276, 786432, grid=grid(786432), stream=stream0)
        buf277 = empty((2048, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf276, reinterpret_tensor(primals_138, (384, 192), (1, 384), 0), out=buf277)
        buf278 = reinterpret_tensor(buf277, (32, 64, 192), (12288, 192, 1), 0); del buf277  # reuse
        buf282 = empty((32, 64, 192), device='cuda', dtype=torch.float32)
        buf283 = empty((2048, 192), device='cuda', dtype=torch.float32)
        buf484 = empty((32, 64, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___attn_qkv, getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___norm1, x_215, x_222], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_55.run(buf278, buf257, buf269, primals_133, primals_139, primals_140, primals_141, buf282, buf283, buf484, 2048, 192, grid=grid(2048), stream=stream0)
        del primals_133
        del primals_139
        del primals_141
        buf284 = empty((2048, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_143, buf283, reinterpret_tensor(primals_142, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf284)
        del primals_143
        # Source Nodes: [x_223], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf285 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf284, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf284, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf284, (32, 4, 64, 48), (36864, 48, 576, 1), 384), None, True)
        buf286 = buf285[0]
        buf287 = buf285[1]
        buf288 = buf285[2]
        buf289 = buf285[3]
        del buf285
        buf290 = buf269; del buf269  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf286, (2048, 192), (192, 1), 0), reinterpret_tensor(primals_144, (192, 192), (1, 192), 0), out=buf290)
        buf294 = buf257; del buf257  # reuse
        buf295 = empty((2048, 192), device='cuda', dtype=torch.float32)
        buf483 = empty((32, 64, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___2___norm2, x_227, x_228], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_54.run(buf278, buf290, primals_145, primals_146, primals_147, buf294, buf295, buf483, 2048, 192, grid=grid(2048), stream=stream0)
        del primals_147
        buf296 = empty((2048, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_228], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_149, buf295, reinterpret_tensor(primals_148, (192, 384), (1, 192), 0), alpha=1, beta=1, out=buf296)
        del primals_149
        buf297 = empty((2048, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_229, x_232], Original ATen: [aten.silu, aten.view]
        triton_poi_fused_silu_view_51.run(buf296, buf297, 786432, grid=grid(786432), stream=stream0)
        buf298 = empty((2048, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf297, reinterpret_tensor(primals_150, (384, 192), (1, 384), 0), out=buf298)
        buf299 = reinterpret_tensor(buf298, (32, 64, 192), (12288, 192, 1), 0); del buf298  # reuse
        buf303 = empty((32, 64, 192), device='cuda', dtype=torch.float32)
        buf304 = empty((2048, 192), device='cuda', dtype=torch.float32)
        buf482 = empty((32, 64, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___attn_qkv, getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___norm1, x_227, x_234], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_55.run(buf299, buf278, buf290, primals_145, primals_151, primals_152, primals_153, buf303, buf304, buf482, 2048, 192, grid=grid(2048), stream=stream0)
        del primals_145
        del primals_151
        del primals_153
        buf305 = empty((2048, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_155, buf304, reinterpret_tensor(primals_154, (192, 576), (1, 192), 0), alpha=1, beta=1, out=buf305)
        del primals_155
        # Source Nodes: [x_235], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf306 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf305, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf305, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf305, (32, 4, 64, 48), (36864, 48, 576, 1), 384), None, True)
        buf307 = buf306[0]
        buf308 = buf306[1]
        buf309 = buf306[2]
        buf310 = buf306[3]
        del buf306
        buf311 = buf290; del buf290  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf307, (2048, 192), (192, 1), 0), reinterpret_tensor(primals_156, (192, 192), (1, 192), 0), out=buf311)
        buf315 = buf278; del buf278  # reuse
        buf316 = empty((2048, 192), device='cuda', dtype=torch.float32)
        buf481 = empty((32, 64, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___3___norm2, x_239, x_240], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_54.run(buf299, buf311, primals_157, primals_158, primals_159, buf315, buf316, buf481, 2048, 192, grid=grid(2048), stream=stream0)
        del primals_159
        buf317 = empty((2048, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_240], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_161, buf316, reinterpret_tensor(primals_160, (192, 384), (1, 192), 0), alpha=1, beta=1, out=buf317)
        del primals_161
        buf318 = empty((2048, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_241, x_244], Original ATen: [aten.silu, aten.view]
        triton_poi_fused_silu_view_51.run(buf317, buf318, 786432, grid=grid(786432), stream=stream0)
        buf319 = empty((2048, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf318, reinterpret_tensor(primals_162, (384, 192), (1, 384), 0), out=buf319)
        buf320 = reinterpret_tensor(buf319, (32, 64, 192), (12288, 192, 1), 0); del buf319  # reuse
        buf324 = empty((32, 64, 192), device='cuda', dtype=torch.float32)
        buf480 = empty((32, 64, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_239, x_247, x_248], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_56.run(buf320, buf299, buf311, primals_157, primals_163, buf324, buf480, 2048, 192, grid=grid(2048), stream=stream0)
        del buf299
        del buf311
        del primals_157
        del primals_163
        buf325 = reinterpret_tensor(buf320, (8, 192, 16, 16), (49152, 256, 16, 1), 0); del buf320  # reuse
        # Source Nodes: [x_251], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_57.run(buf324, primals_164, primals_165, buf325, 1536, 256, grid=grid(1536, 256), stream=stream0)
        del primals_165
        # Source Nodes: [x_252], Original ATen: [aten.convolution]
        buf326 = extern_kernels.convolution(buf325, primals_166, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf326, (8, 128, 16, 16), (32768, 256, 16, 1))
        buf327 = buf225; del buf225  # reuse
        buf328 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf330 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_253], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf326, primals_286, primals_287, buf327, buf328, buf330, primals_286, primals_287, 128, 2048, grid=grid(128), stream=stream0)
        del primals_286
        del primals_287
        buf331 = empty((8, 128, 16, 16), device='cuda', dtype=torch.float32)
        buf479 = empty((8, 128, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_253], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_58.run(buf326, buf327, buf328, primals_47, primals_48, buf331, buf479, 262144, grid=grid(262144), stream=stream0)
        del primals_48
        buf332 = empty((8, 256, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_4], Original ATen: [aten.cat]
        triton_poi_fused_cat_59.run(buf222, buf331, buf332, 524288, grid=grid(524288), stream=stream0)
        # Source Nodes: [x_258], Original ATen: [aten.convolution]
        buf333 = extern_kernels.convolution(buf332, primals_167, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf333, (8, 128, 16, 16), (32768, 256, 16, 1))
        buf334 = buf328; del buf328  # reuse
        buf335 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf337 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_259], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_42.run(buf333, primals_289, primals_290, buf334, buf335, buf337, primals_289, primals_290, 128, 2048, grid=grid(128), stream=stream0)
        del primals_289
        del primals_290
        buf339 = buf331; del buf331  # reuse
        buf478 = empty((8, 128, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_8, x_259], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_44.run(buf333, buf334, buf335, primals_49, primals_50, buf339, buf478, 262144, grid=grid(262144), stream=stream0)
        del buf335
        del primals_50
        # Source Nodes: [x_264], Original ATen: [aten.convolution]
        buf340 = extern_kernels.convolution(buf339, primals_168, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf340, (8, 512, 16, 16), (131072, 256, 16, 1))
        buf341 = reinterpret_tensor(buf52, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf52  # reuse
        buf342 = reinterpret_tensor(buf51, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf51  # reuse
        buf344 = reinterpret_tensor(buf50, (512, ), (1, ), 0); del buf50  # reuse
        # Source Nodes: [x_265], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_60.run(buf340, primals_292, primals_293, buf341, buf342, buf344, primals_292, primals_293, 512, 2048, grid=grid(512), stream=stream0)
        del primals_292
        del primals_293
        buf346 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        buf477 = empty((8, 512, 16, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_265, x_269], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_61.run(buf340, buf341, buf342, primals_51, primals_52, buf346, buf477, 1048576, grid=grid(1048576), stream=stream0)
        del primals_52
        # Source Nodes: [x_270], Original ATen: [aten.convolution]
        buf347 = extern_kernels.convolution(buf346, primals_169, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=512, bias=None)
        assert_size_stride(buf347, (8, 512, 8, 8), (32768, 64, 8, 1))
        buf348 = buf342; del buf342  # reuse
        buf349 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf351 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_271], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_62.run(buf347, primals_295, primals_296, buf348, buf349, buf351, primals_295, primals_296, 512, 512, grid=grid(512), stream=stream0)
        del primals_295
        del primals_296
        buf353 = empty((8, 512, 8, 8), device='cuda', dtype=torch.float32)
        buf476 = empty((8, 512, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_271, x_275], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_63.run(buf347, buf348, buf349, primals_53, primals_54, buf353, buf476, 262144, grid=grid(262144), stream=stream0)
        del primals_54
        # Source Nodes: [x_278], Original ATen: [aten.convolution]
        buf354 = extern_kernels.convolution(buf353, primals_170, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf354, (8, 160, 8, 8), (10240, 64, 8, 1))
        buf355 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf356 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf358 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_279], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_64.run(buf354, primals_298, primals_299, buf355, buf356, buf358, primals_298, primals_299, 160, 512, grid=grid(160), stream=stream0)
        del primals_298
        del primals_299
        buf359 = empty((8, 160, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_279], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_poi_fused__native_batch_norm_legit_functional_65.run(buf354, buf355, buf356, primals_55, primals_56, buf359, 81920, grid=grid(81920), stream=stream0)
        del primals_56
        # Source Nodes: [x_286], Original ATen: [aten.convolution]
        buf360 = extern_kernels.convolution(buf359, primals_171, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf360, (8, 160, 8, 8), (10240, 64, 8, 1))
        buf361 = buf356; del buf356  # reuse
        buf362 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf364 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_287], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_64.run(buf360, primals_301, primals_302, buf361, buf362, buf364, primals_301, primals_302, 160, 512, grid=grid(160), stream=stream0)
        del primals_301
        del primals_302
        buf366 = empty((8, 160, 8, 8), device='cuda', dtype=torch.float32)
        buf475 = empty((8, 160, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_287, x_291], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_66.run(buf360, buf361, buf362, primals_57, primals_58, buf366, buf475, 81920, grid=grid(81920), stream=stream0)
        del primals_58
        # Source Nodes: [x_292], Original ATen: [aten.convolution]
        buf367 = extern_kernels.convolution(buf366, primals_172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf367, (8, 240, 8, 8), (15360, 64, 8, 1))
        buf368 = reinterpret_tensor(buf23, (32, 16, 1, 2), (1, 32, 1024, 512), 0); del buf23  # reuse
        buf369 = reinterpret_tensor(buf22, (32, 16, 1, 2), (1, 32, 1024, 512), 0); del buf22  # reuse
        buf370 = reinterpret_tensor(buf21, (32, 16, 1, 2), (1, 32, 1024, 512), 0); del buf21  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm]
        triton_red_fused_native_layer_norm_67.run(buf367, buf368, buf369, buf370, 1024, 120, grid=grid(1024), stream=stream0)
        buf371 = reinterpret_tensor(buf349, (32, 16, 1), (16, 1, 512), 0); del buf349  # reuse
        buf372 = empty_strided((32, 16, 1), (1, 32, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm]
        triton_per_fused_native_layer_norm_68.run(buf368, buf369, buf370, buf371, buf372, 512, 2, grid=grid(512), stream=stream0)
        buf374 = empty((32, 16, 240), device='cuda', dtype=torch.float32)
        buf375 = empty((512, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___attn_qkv, getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm, aten.view]
        triton_poi_fused_native_layer_norm_view_69.run(buf367, buf371, buf372, primals_173, primals_174, buf374, buf375, 32, 3840, grid=grid(32, 3840), stream=stream0)
        del primals_174
        buf376 = empty((512, 720), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_176, buf375, reinterpret_tensor(primals_175, (240, 720), (1, 240), 0), alpha=1, beta=1, out=buf376)
        del primals_176
        # Source Nodes: [x_295], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf377 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf376, (32, 4, 16, 60), (11520, 60, 720, 1), 0), reinterpret_tensor(buf376, (32, 4, 16, 60), (11520, 60, 720, 1), 240), reinterpret_tensor(buf376, (32, 4, 16, 60), (11520, 60, 720, 1), 480), None, True)
        buf378 = buf377[0]
        buf379 = buf377[1]
        buf380 = buf377[2]
        buf381 = buf377[3]
        del buf377
        buf382 = empty((512, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf378, (512, 240), (240, 1), 0), reinterpret_tensor(primals_177, (240, 240), (1, 240), 0), out=buf382)
        buf383 = reinterpret_tensor(buf370, (32, 16, 1, 2), (32, 2, 1024, 1), 0); del buf370  # reuse
        buf384 = reinterpret_tensor(buf369, (32, 16, 1, 2), (32, 2, 1024, 1), 0); del buf369  # reuse
        buf385 = reinterpret_tensor(buf368, (32, 16, 1, 2), (32, 2, 1024, 1), 0); del buf368  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm2, x_299], Original ATen: [aten.add, aten.native_layer_norm]
        triton_red_fused_add_native_layer_norm_70.run(buf367, buf382, primals_178, buf383, buf384, buf385, 1024, 120, grid=grid(1024), stream=stream0)
        buf386 = buf371; del buf371  # reuse
        buf387 = empty_strided((32, 16, 1), (16, 1, 512), device='cuda', dtype=torch.float32)
        buf473 = empty((32, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm2, x_299], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_71.run(buf383, buf384, buf385, buf386, buf387, buf473, 512, 2, grid=grid(512), stream=stream0)
        del buf383
        del buf384
        del buf385
        buf389 = empty((32, 16, 240), device='cuda', dtype=torch.float32)
        buf390 = empty((512, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm2, x_299, x_300], Original ATen: [aten.add, aten.native_layer_norm, aten.view]
        triton_poi_fused_add_native_layer_norm_view_72.run(buf367, buf382, primals_178, buf386, buf387, primals_179, primals_180, buf389, buf390, 32, 3840, grid=grid(32, 3840), stream=stream0)
        del primals_180
        buf391 = empty((512, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_300], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_182, buf390, reinterpret_tensor(primals_181, (240, 480), (1, 240), 0), alpha=1, beta=1, out=buf391)
        del primals_182
        buf392 = empty((512, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_301, x_304], Original ATen: [aten.silu, aten.view]
        triton_poi_fused_silu_view_73.run(buf391, buf392, 245760, grid=grid(245760), stream=stream0)
        buf393 = empty((512, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf392, reinterpret_tensor(primals_183, (480, 240), (1, 480), 0), out=buf393)
        buf394 = reinterpret_tensor(buf382, (32, 16, 240), (3840, 240, 1), 0); del buf382  # reuse
        # Source Nodes: [x_299, x_306], Original ATen: [aten.add]
        triton_poi_fused_add_74.run(buf394, buf367, primals_178, buf393, primals_184, 32, 3840, grid=grid(32, 3840), stream=stream0)
        del primals_178
        del primals_184
        buf398 = reinterpret_tensor(buf393, (32, 16, 240), (3840, 240, 1), 0); del buf393  # reuse
        buf399 = reinterpret_tensor(buf367, (512, 240), (240, 1), 0); del buf367  # reuse
        buf472 = reinterpret_tensor(buf387, (32, 16, 1), (16, 1, 1), 0); del buf387  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___attn_qkv, getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_native_layer_norm_native_layer_norm_backward_view_75.run(buf394, primals_185, primals_186, buf398, buf399, buf472, 512, 240, grid=grid(512), stream=stream0)
        del primals_186
        buf400 = empty((512, 720), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_188, buf399, reinterpret_tensor(primals_187, (240, 720), (1, 240), 0), alpha=1, beta=1, out=buf400)
        del primals_188
        # Source Nodes: [x_307], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf401 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf400, (32, 4, 16, 60), (11520, 60, 720, 1), 0), reinterpret_tensor(buf400, (32, 4, 16, 60), (11520, 60, 720, 1), 240), reinterpret_tensor(buf400, (32, 4, 16, 60), (11520, 60, 720, 1), 480), None, True)
        buf402 = buf401[0]
        buf403 = buf401[1]
        buf404 = buf401[2]
        buf405 = buf401[3]
        del buf401
        buf406 = empty((512, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf402, (512, 240), (240, 1), 0), reinterpret_tensor(primals_189, (240, 240), (1, 240), 0), out=buf406)
        buf410 = empty((32, 16, 240), device='cuda', dtype=torch.float32)
        buf411 = empty((512, 240), device='cuda', dtype=torch.float32)
        buf471 = reinterpret_tensor(buf386, (32, 16, 1), (16, 1, 1), 0); del buf386  # reuse
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___1___norm2, x_311, x_312], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_76.run(buf394, buf406, primals_190, primals_191, primals_192, buf410, buf411, buf471, 512, 240, grid=grid(512), stream=stream0)
        del primals_192
        buf412 = empty((512, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_312], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_194, buf411, reinterpret_tensor(primals_193, (240, 480), (1, 240), 0), alpha=1, beta=1, out=buf412)
        del primals_194
        buf413 = empty((512, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_313, x_316], Original ATen: [aten.silu, aten.view]
        triton_poi_fused_silu_view_73.run(buf412, buf413, 245760, grid=grid(245760), stream=stream0)
        buf414 = empty((512, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf413, reinterpret_tensor(primals_195, (480, 240), (1, 480), 0), out=buf414)
        buf415 = reinterpret_tensor(buf414, (32, 16, 240), (3840, 240, 1), 0); del buf414  # reuse
        buf419 = empty((32, 16, 240), device='cuda', dtype=torch.float32)
        buf420 = empty((512, 240), device='cuda', dtype=torch.float32)
        buf470 = empty((32, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___attn_qkv, getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___norm1, x_311, x_318], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_77.run(buf415, buf394, buf406, primals_190, primals_196, primals_197, primals_198, buf419, buf420, buf470, 512, 240, grid=grid(512), stream=stream0)
        del primals_190
        del primals_196
        del primals_198
        buf421 = empty((512, 720), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___attn_qkv], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_200, buf420, reinterpret_tensor(primals_199, (240, 720), (1, 240), 0), alpha=1, beta=1, out=buf421)
        del primals_200
        # Source Nodes: [x_319], Original ATen: [aten._scaled_dot_product_efficient_attention]
        buf422 = aten._scaled_dot_product_efficient_attention(reinterpret_tensor(buf421, (32, 4, 16, 60), (11520, 60, 720, 1), 0), reinterpret_tensor(buf421, (32, 4, 16, 60), (11520, 60, 720, 1), 240), reinterpret_tensor(buf421, (32, 4, 16, 60), (11520, 60, 720, 1), 480), None, True)
        buf423 = buf422[0]
        buf424 = buf422[1]
        buf425 = buf422[2]
        buf426 = buf422[3]
        del buf422
        buf427 = buf406; del buf406  # reuse
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf423, (512, 240), (240, 1), 0), reinterpret_tensor(primals_201, (240, 240), (1, 240), 0), out=buf427)
        buf431 = buf394; del buf394  # reuse
        buf432 = empty((512, 240), device='cuda', dtype=torch.float32)
        buf469 = empty((32, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___2___norm2, x_323, x_324], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward, aten.view]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_view_76.run(buf415, buf427, primals_202, primals_203, primals_204, buf431, buf432, buf469, 512, 240, grid=grid(512), stream=stream0)
        del primals_204
        buf433 = empty((512, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_324], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_206, buf432, reinterpret_tensor(primals_205, (240, 480), (1, 240), 0), alpha=1, beta=1, out=buf433)
        del primals_206
        buf434 = empty((512, 480), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_325, x_328], Original ATen: [aten.silu, aten.view]
        triton_poi_fused_silu_view_73.run(buf433, buf434, 245760, grid=grid(245760), stream=stream0)
        buf435 = empty((512, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf434, reinterpret_tensor(primals_207, (480, 240), (1, 480), 0), out=buf435)
        buf436 = reinterpret_tensor(buf435, (32, 16, 240), (3840, 240, 1), 0); del buf435  # reuse
        buf440 = empty((32, 16, 240), device='cuda', dtype=torch.float32)
        buf468 = empty((32, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_323, x_331, x_332], Original ATen: [aten.add, aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_per_fused_add_native_layer_norm_native_layer_norm_backward_78.run(buf436, buf415, buf427, primals_202, primals_208, buf440, buf468, 512, 240, grid=grid(512), stream=stream0)
        del buf415
        del buf427
        del primals_202
        del primals_208
        buf441 = reinterpret_tensor(buf436, (8, 240, 8, 8), (15360, 64, 8, 1), 0); del buf436  # reuse
        # Source Nodes: [x_335], Original ATen: [aten._unsafe_view, aten.clone]
        triton_poi_fused__unsafe_view_clone_79.run(buf440, primals_209, primals_210, buf441, 1920, 64, grid=grid(1920, 64), stream=stream0)
        del primals_210
        # Source Nodes: [x_336], Original ATen: [aten.convolution]
        buf442 = extern_kernels.convolution(buf441, primals_211, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf442, (8, 160, 8, 8), (10240, 64, 8, 1))
        buf443 = buf362; del buf362  # reuse
        buf444 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf446 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_337], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_64.run(buf442, primals_304, primals_305, buf443, buf444, buf446, primals_304, primals_305, 160, 512, grid=grid(160), stream=stream0)
        del primals_304
        del primals_305
        buf447 = empty((8, 160, 8, 8), device='cuda', dtype=torch.float32)
        buf467 = empty((8, 160, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_337], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_sub_80.run(buf442, buf443, buf444, primals_59, primals_60, buf447, buf467, 81920, grid=grid(81920), stream=stream0)
        del primals_60
        buf448 = empty((8, 320, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_3], Original ATen: [aten.cat]
        triton_poi_fused_cat_81.run(buf359, buf447, buf448, 163840, grid=grid(163840), stream=stream0)
        # Source Nodes: [x_342], Original ATen: [aten.convolution]
        buf449 = extern_kernels.convolution(buf448, primals_212, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf449, (8, 160, 8, 8), (10240, 64, 8, 1))
        buf450 = buf444; del buf444  # reuse
        buf451 = empty_strided((1, 160, 1, 1), (160, 1, 160, 160), device='cuda', dtype=torch.float32)
        buf453 = empty((160, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_343], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_64.run(buf449, primals_307, primals_308, buf450, buf451, buf453, primals_307, primals_308, 160, 512, grid=grid(160), stream=stream0)
        del primals_307
        del primals_308
        buf455 = buf447; del buf447  # reuse
        buf466 = empty((8, 160, 8, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_343, x_348], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused__native_batch_norm_legit_functional_add_fill_mul_sigmoid_silu_sub_66.run(buf449, buf450, buf451, primals_61, primals_62, buf455, buf466, 81920, grid=grid(81920), stream=stream0)
        del buf451
        del primals_62
        # Source Nodes: [x_349], Original ATen: [aten.convolution]
        buf456 = extern_kernels.convolution(buf455, primals_213, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf456, (8, 640, 8, 8), (40960, 64, 8, 1))
        buf457 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf458 = empty_strided((1, 640, 1, 1), (640, 1, 640, 640), device='cuda', dtype=torch.float32)
        buf460 = empty((640, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_350], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_82.run(buf456, primals_310, primals_311, buf457, buf458, buf460, primals_310, primals_311, 640, 512, grid=grid(640), stream=stream0)
        del primals_310
        del primals_311
        buf465 = empty((8, 640, 8, 8), device='cuda', dtype=torch.float32)
        buf462 = empty_strided((8, 640, 1, 1), (640, 1, 5120, 5120), device='cuda', dtype=torch.float32)
        buf463 = reinterpret_tensor(buf462, (8, 640), (640, 1), 0); del buf462  # reuse
        # Source Nodes: [x_350, x_355, x_356, x_358], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.fill, aten.mean, aten.mul, aten.sigmoid, aten.silu, aten.sub, aten.view]
        triton_per_fused__native_batch_norm_legit_functional_add_fill_mean_mul_sigmoid_silu_sub_view_83.run(buf463, buf456, buf457, buf458, primals_63, primals_64, buf465, 5120, 64, grid=grid(5120), stream=stream0)
        del buf458
        del primals_64
        buf464 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_360], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_215, buf463, reinterpret_tensor(primals_214, (640, 1000), (1, 640), 0), alpha=1, beta=1, out=buf464)
        del primals_215
        buf474 = empty((32, 16, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___4_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_poi_fused_native_layer_norm_native_layer_norm_backward_84.run(buf372, buf474, 32, 16, grid=grid(32, 16), stream=stream0)
        del buf372
        buf488 = empty((32, 64, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___3_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_poi_fused_native_layer_norm_native_layer_norm_backward_85.run(buf235, buf488, 32, 64, grid=grid(32, 64), stream=stream0)
        del buf235
        buf498 = empty((32, 256, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_getattr_l__mod___stages___2_____1___transformer___0___norm1], Original ATen: [aten.native_layer_norm, aten.native_layer_norm_backward]
        triton_poi_fused_native_layer_norm_native_layer_norm_backward_86.run(buf143, buf498, 32, 256, grid=grid(32, 256), stream=stream0)
        del buf143
        # Source Nodes: [add_], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_216, primals_216, 1, grid=grid(1), stream=stream0)
        del primals_216
        # Source Nodes: [add__1], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_219, primals_219, 1, grid=grid(1), stream=stream0)
        del primals_219
        # Source Nodes: [add__2], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_222, primals_222, 1, grid=grid(1), stream=stream0)
        del primals_222
        # Source Nodes: [add__3], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_225, primals_225, 1, grid=grid(1), stream=stream0)
        del primals_225
        # Source Nodes: [add__4], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_228, primals_228, 1, grid=grid(1), stream=stream0)
        del primals_228
        # Source Nodes: [add__5], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_231, primals_231, 1, grid=grid(1), stream=stream0)
        del primals_231
        # Source Nodes: [add__6], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_234, primals_234, 1, grid=grid(1), stream=stream0)
        del primals_234
        # Source Nodes: [add__7], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_237, primals_237, 1, grid=grid(1), stream=stream0)
        del primals_237
        # Source Nodes: [add__8], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_240, primals_240, 1, grid=grid(1), stream=stream0)
        del primals_240
        # Source Nodes: [add__9], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_243, primals_243, 1, grid=grid(1), stream=stream0)
        del primals_243
        # Source Nodes: [add__10], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_246, primals_246, 1, grid=grid(1), stream=stream0)
        del primals_246
        # Source Nodes: [add__11], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_249, primals_249, 1, grid=grid(1), stream=stream0)
        del primals_249
        # Source Nodes: [add__12], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_252, primals_252, 1, grid=grid(1), stream=stream0)
        del primals_252
        # Source Nodes: [add__13], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_255, primals_255, 1, grid=grid(1), stream=stream0)
        del primals_255
        # Source Nodes: [add__14], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_258, primals_258, 1, grid=grid(1), stream=stream0)
        del primals_258
        # Source Nodes: [add__15], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_261, primals_261, 1, grid=grid(1), stream=stream0)
        del primals_261
        # Source Nodes: [add__16], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_264, primals_264, 1, grid=grid(1), stream=stream0)
        del primals_264
        # Source Nodes: [add__17], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_267, primals_267, 1, grid=grid(1), stream=stream0)
        del primals_267
        # Source Nodes: [add__18], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_270, primals_270, 1, grid=grid(1), stream=stream0)
        del primals_270
        # Source Nodes: [add__19], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_273, primals_273, 1, grid=grid(1), stream=stream0)
        del primals_273
        # Source Nodes: [add__20], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_276, primals_276, 1, grid=grid(1), stream=stream0)
        del primals_276
        # Source Nodes: [add__21], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_279, primals_279, 1, grid=grid(1), stream=stream0)
        del primals_279
        # Source Nodes: [add__22], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_282, primals_282, 1, grid=grid(1), stream=stream0)
        del primals_282
        # Source Nodes: [add__23], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_285, primals_285, 1, grid=grid(1), stream=stream0)
        del primals_285
        # Source Nodes: [add__24], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_288, primals_288, 1, grid=grid(1), stream=stream0)
        del primals_288
        # Source Nodes: [add__25], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_291, primals_291, 1, grid=grid(1), stream=stream0)
        del primals_291
        # Source Nodes: [add__26], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_294, primals_294, 1, grid=grid(1), stream=stream0)
        del primals_294
        # Source Nodes: [add__27], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_297, primals_297, 1, grid=grid(1), stream=stream0)
        del primals_297
        # Source Nodes: [add__28], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_300, primals_300, 1, grid=grid(1), stream=stream0)
        del primals_300
        # Source Nodes: [add__29], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_303, primals_303, 1, grid=grid(1), stream=stream0)
        del primals_303
        # Source Nodes: [add__30], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_306, primals_306, 1, grid=grid(1), stream=stream0)
        del primals_306
        # Source Nodes: [add__31], Original ATen: [aten.add]
        triton_poi_fused_add_87.run(primals_309, primals_309, 1, grid=grid(1), stream=stream0)
        del primals_309
        return (buf464, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_89, primals_95, primals_101, primals_107, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_122, primals_128, primals_134, primals_140, primals_146, primals_152, primals_158, primals_164, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_179, primals_185, primals_191, primals_197, primals_203, primals_209, primals_211, primals_212, primals_213, primals_312, buf0, buf7, buf9, buf10, buf17, buf19, buf20, buf27, buf29, buf30, buf37, buf38, buf39, buf46, buf48, buf49, buf56, buf58, buf59, buf66, buf67, buf68, buf72, buf74, buf75, buf79, buf81, buf82, buf89, buf90, buf91, buf95, buf97, buf98, buf102, buf104, buf105, buf112, buf113, buf114, buf118, buf120, buf121, buf125, buf127, buf128, buf132, buf133, buf134, buf138, buf140, buf145, buf146, reinterpret_tensor(buf147, (32, 4, 256, 36), (110592, 36, 432, 1), 0), reinterpret_tensor(buf147, (32, 4, 256, 36), (110592, 36, 432, 1), 144), reinterpret_tensor(buf147, (32, 4, 256, 36), (110592, 36, 432, 1), 288), buf150, buf151, buf152, reinterpret_tensor(buf149, (8192, 144), (144, 1), 0), buf157, buf158, buf159, buf160, buf166, buf167, reinterpret_tensor(buf168, (32, 4, 256, 36), (110592, 36, 432, 1), 0), reinterpret_tensor(buf168, (32, 4, 256, 36), (110592, 36, 432, 1), 144), reinterpret_tensor(buf168, (32, 4, 256, 36), (110592, 36, 432, 1), 288), buf171, buf172, buf173, reinterpret_tensor(buf170, (8192, 144), (144, 1), 0), buf178, buf179, buf180, buf181, buf187, buf188, buf189, buf193, buf195, buf196, buf200, buf202, buf203, buf207, buf209, buf210, buf214, buf216, buf217, buf221, buf222, buf223, buf227, buf229, buf237, buf238, reinterpret_tensor(buf239, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf239, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf239, (32, 4, 64, 48), (36864, 48, 576, 1), 384), buf242, buf243, buf244, reinterpret_tensor(buf241, (2048, 192), (192, 1), 0), buf252, buf253, buf254, buf255, buf261, buf262, reinterpret_tensor(buf263, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf263, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf263, (32, 4, 64, 48), (36864, 48, 576, 1), 384), buf266, buf267, buf268, reinterpret_tensor(buf265, (2048, 192), (192, 1), 0), buf273, buf274, buf275, buf276, buf282, buf283, reinterpret_tensor(buf284, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf284, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf284, (32, 4, 64, 48), (36864, 48, 576, 1), 384), buf287, buf288, buf289, reinterpret_tensor(buf286, (2048, 192), (192, 1), 0), buf294, buf295, buf296, buf297, buf303, buf304, reinterpret_tensor(buf305, (32, 4, 64, 48), (36864, 48, 576, 1), 0), reinterpret_tensor(buf305, (32, 4, 64, 48), (36864, 48, 576, 1), 192), reinterpret_tensor(buf305, (32, 4, 64, 48), (36864, 48, 576, 1), 384), buf308, buf309, buf310, reinterpret_tensor(buf307, (2048, 192), (192, 1), 0), buf315, buf316, buf317, buf318, buf324, buf325, buf326, buf330, buf332, buf333, buf337, buf339, buf340, buf344, buf346, buf347, buf351, buf353, buf354, buf358, buf359, buf360, buf364, buf366, buf374, buf375, reinterpret_tensor(buf376, (32, 4, 16, 60), (11520, 60, 720, 1), 0), reinterpret_tensor(buf376, (32, 4, 16, 60), (11520, 60, 720, 1), 240), reinterpret_tensor(buf376, (32, 4, 16, 60), (11520, 60, 720, 1), 480), buf379, buf380, buf381, reinterpret_tensor(buf378, (512, 240), (240, 1), 0), buf389, buf390, buf391, buf392, buf398, buf399, reinterpret_tensor(buf400, (32, 4, 16, 60), (11520, 60, 720, 1), 0), reinterpret_tensor(buf400, (32, 4, 16, 60), (11520, 60, 720, 1), 240), reinterpret_tensor(buf400, (32, 4, 16, 60), (11520, 60, 720, 1), 480), buf403, buf404, buf405, reinterpret_tensor(buf402, (512, 240), (240, 1), 0), buf410, buf411, buf412, buf413, buf419, buf420, reinterpret_tensor(buf421, (32, 4, 16, 60), (11520, 60, 720, 1), 0), reinterpret_tensor(buf421, (32, 4, 16, 60), (11520, 60, 720, 1), 240), reinterpret_tensor(buf421, (32, 4, 16, 60), (11520, 60, 720, 1), 480), buf424, buf425, buf426, reinterpret_tensor(buf423, (512, 240), (240, 1), 0), buf431, buf432, buf433, buf434, buf440, buf441, buf442, buf446, buf448, buf449, buf453, buf455, buf456, buf460, buf463, reinterpret_tensor(primals_214, (1000, 640), (640, 1), 0), buf465, reinterpret_tensor(buf457, (1, 640, 1, 1), (640, 1, 1, 1), 0), buf466, reinterpret_tensor(buf450, (1, 160, 1, 1), (160, 1, 1, 1), 0), buf467, reinterpret_tensor(buf443, (1, 160, 1, 1), (160, 1, 1, 1), 0), buf468, reinterpret_tensor(primals_207, (240, 480), (480, 1), 0), reinterpret_tensor(primals_205, (480, 240), (240, 1), 0), buf469, reinterpret_tensor(primals_201, (240, 240), (240, 1), 0), buf423, reinterpret_tensor(primals_199, (720, 240), (240, 1), 0), buf470, reinterpret_tensor(primals_195, (240, 480), (480, 1), 0), reinterpret_tensor(primals_193, (480, 240), (240, 1), 0), buf471, reinterpret_tensor(primals_189, (240, 240), (240, 1), 0), buf402, reinterpret_tensor(primals_187, (720, 240), (240, 1), 0), buf472, reinterpret_tensor(primals_183, (240, 480), (480, 1), 0), reinterpret_tensor(primals_181, (480, 240), (240, 1), 0), buf473, reinterpret_tensor(primals_177, (240, 240), (240, 1), 0), buf378, reinterpret_tensor(primals_175, (720, 240), (240, 1), 0), buf474, buf475, reinterpret_tensor(buf361, (1, 160, 1, 1), (160, 1, 1, 1), 0), reinterpret_tensor(buf355, (1, 160, 1, 1), (160, 1, 1, 1), 0), buf476, reinterpret_tensor(buf348, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf477, reinterpret_tensor(buf341, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf478, reinterpret_tensor(buf334, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf479, reinterpret_tensor(buf327, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf480, reinterpret_tensor(primals_162, (192, 384), (384, 1), 0), reinterpret_tensor(primals_160, (384, 192), (192, 1), 0), buf481, reinterpret_tensor(primals_156, (192, 192), (192, 1), 0), buf307, reinterpret_tensor(primals_154, (576, 192), (192, 1), 0), buf482, reinterpret_tensor(primals_150, (192, 384), (384, 1), 0), reinterpret_tensor(primals_148, (384, 192), (192, 1), 0), buf483, reinterpret_tensor(primals_144, (192, 192), (192, 1), 0), buf286, reinterpret_tensor(primals_142, (576, 192), (192, 1), 0), buf484, reinterpret_tensor(primals_138, (192, 384), (384, 1), 0), reinterpret_tensor(primals_136, (384, 192), (192, 1), 0), buf485, reinterpret_tensor(primals_132, (192, 192), (192, 1), 0), buf265, reinterpret_tensor(primals_130, (576, 192), (192, 1), 0), buf486, reinterpret_tensor(primals_126, (192, 384), (384, 1), 0), reinterpret_tensor(primals_124, (384, 192), (192, 1), 0), buf487, reinterpret_tensor(primals_120, (192, 192), (192, 1), 0), buf241, reinterpret_tensor(primals_118, (576, 192), (192, 1), 0), buf488, buf489, reinterpret_tensor(buf224, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf218, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf490, reinterpret_tensor(buf211, (1, 384, 1, 1), (384, 1, 1, 1), 0), buf491, reinterpret_tensor(buf204, (1, 384, 1, 1), (384, 1, 1, 1), 0), buf492, reinterpret_tensor(buf197, (1, 96, 1, 1), (96, 1, 1, 1), 0), buf493, reinterpret_tensor(buf190, (1, 96, 1, 1), (96, 1, 1, 1), 0), buf494, reinterpret_tensor(primals_105, (144, 288), (288, 1), 0), reinterpret_tensor(primals_103, (288, 144), (144, 1), 0), buf495, reinterpret_tensor(primals_99, (144, 144), (144, 1), 0), buf170, reinterpret_tensor(primals_97, (432, 144), (144, 1), 0), buf496, reinterpret_tensor(primals_93, (144, 288), (288, 1), 0), reinterpret_tensor(primals_91, (288, 144), (144, 1), 0), buf497, reinterpret_tensor(primals_87, (144, 144), (144, 1), 0), buf149, reinterpret_tensor(primals_85, (432, 144), (144, 1), 0), buf498, buf499, reinterpret_tensor(buf135, (1, 96, 1, 1), (96, 1, 1, 1), 0), reinterpret_tensor(buf129, (1, 96, 1, 1), (96, 1, 1, 1), 0), buf500, reinterpret_tensor(buf122, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf501, reinterpret_tensor(buf115, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf109, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf502, reinterpret_tensor(buf99, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf503, reinterpret_tensor(buf92, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf86, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf504, reinterpret_tensor(buf76, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf505, reinterpret_tensor(buf69, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf63, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf506, reinterpret_tensor(buf53, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf507, reinterpret_tensor(buf43, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf34, (1, 32, 1, 1), (32, 1, 1, 1), 0), buf508, reinterpret_tensor(buf24, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf509, reinterpret_tensor(buf14, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf510, reinterpret_tensor(buf4, (1, 16, 1, 1), (16, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((32, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((128, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((256, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((96, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((144, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((432, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((144, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((288, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((144, 288), (288, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((432, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((144, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((288, 144), (144, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((144, 288), (288, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((96, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((96, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((384, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((384, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((128, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((192, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((384, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((192, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((384, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((192, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((384, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((192, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((576, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((192, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((384, 192), (192, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((192, 384), (384, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((128, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((512, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((512, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((160, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((240, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((720, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((240, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((480, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((240, 480), (480, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((720, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((240, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((480, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((240, 480), (480, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((720, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((720, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((240, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((480, 240), (240, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((240, 480), (480, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((160, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((160, 320, 3, 3), (2880, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((640, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((1000, 640), (640, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_217 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_220 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_223 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_226 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_229 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_232 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_235 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_238 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_241 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_244 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_247 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_250 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_253 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_256 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_259 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_262 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_265 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_268 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_271 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_274 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_277 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_280 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_283 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_286 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_289 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_292 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_295 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_298 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_301 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_304 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_307 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_310 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((8, 3, 256, 256), (196608, 65536, 256, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mobilevit_s', benchmark_compiled_module)
