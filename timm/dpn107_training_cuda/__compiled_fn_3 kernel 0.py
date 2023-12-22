
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


# kernel path: /tmp/torchinductor_youkaichao/44/c4437c7pyxd3ucahdodyvxh6oaxynxf2cmh3qalqbkex76dj35wv.py
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
    size_hints=[512, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 25088
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
        tmp0 = tl.load(in_ptr0 + ((12544*x0) + (1605632*(r2 // 12544)) + (3211264*x1) + (r2 % 12544)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/nb/cnbw4vfwwkwcoqaxc7j2tjf3dzmm72fdbilv5vnh3qk2jjw5nnt7.py
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
    size_hints=[128, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_1', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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
    tmp16 = 100352.0
    tmp17 = tmp14 / tmp16
    tmp18 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/77/c77gtemdw4tenoyvd3huxqb5vlu5v4asvxdgnsm7ozrs7ruxkaj4.py
# Source Nodes: [x_1, x_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
# x_4 => relu
triton_poi_fused__native_batch_norm_legit_functional_relu_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 100352.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qd/cqdl3oi2wnkjk62i52n2hkbaja4e45bcs2pix5ajygne7zms52dk.py
# Source Nodes: [x_in], Original ATen: [aten.max_pool2d_with_indices]
# x_in => getitem_3, max_pool2d_with_indices
triton_poi_fused_max_pool2d_with_indices_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 56) % 56
    x0 = xindex % 56
    x3 = (xindex // 56)
    x4 = xindex
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-113) + (2*x0) + (224*x3)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-112) + (2*x0) + (224*x3)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-111) + (2*x0) + (224*x3)), tmp27, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + (2*x0) + (224*x3)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + ((2*x0) + (224*x3)), tmp41, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + (2*x0) + (224*x3)), tmp46, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 1 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (111 + (2*x0) + (224*x3)), tmp55, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (112 + (2*x0) + (224*x3)), tmp60, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (113 + (2*x0) + (224*x3)), tmp65, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tmp70 = tmp21 > tmp13
    tmp71 = (-112) + (2*x0) + (224*x1)
    tmp72 = (-113) + (2*x0) + (224*x1)
    tmp73 = tl.where(tmp70, tmp71, tmp72)
    tmp74 = tmp30 > tmp22
    tmp75 = (-111) + (2*x0) + (224*x1)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tmp77 = tmp39 > tmp31
    tmp78 = (-1) + (2*x0) + (224*x1)
    tmp79 = tl.where(tmp77, tmp78, tmp76)
    tmp80 = tmp44 > tmp40
    tmp81 = (2*x0) + (224*x1)
    tmp82 = tl.where(tmp80, tmp81, tmp79)
    tmp83 = tmp49 > tmp45
    tmp84 = 1 + (2*x0) + (224*x1)
    tmp85 = tl.where(tmp83, tmp84, tmp82)
    tmp86 = tmp58 > tmp50
    tmp87 = 111 + (2*x0) + (224*x1)
    tmp88 = tl.where(tmp86, tmp87, tmp85)
    tmp89 = tmp63 > tmp59
    tmp90 = 112 + (2*x0) + (224*x1)
    tmp91 = tl.where(tmp89, tmp90, tmp88)
    tmp92 = tmp68 > tmp64
    tmp93 = 113 + (2*x0) + (224*x1)
    tmp94 = tl.where(tmp92, tmp93, tmp91)
    tl.store(out_ptr0 + (x4), tmp69, None)
    tl.store(out_ptr1 + (x4), tmp94, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ig/cigspgxggf33pekwuuojqb3q7kkinyipdrisqmzqz3c3p3j5fpon.py
# Source Nodes: [x_5], Original ATen: [aten._native_batch_norm_legit_functional]
# x_5 => var_mean_1
triton_red_fused__native_batch_norm_legit_functional_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_4', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
    rnumel = 6272
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
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (401408*(r2 // 3136)) + (802816*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/eh/cehkgj3jymncevtqwow274rckaruf3ajnw444pvb7ccmj766sys2.py
# Source Nodes: [x_5, x_8], Original ATen: [aten._native_batch_norm_legit_functional]
# x_5 => add_6, add_7, add_8, mul_10, mul_11, mul_12, mul_8, mul_9, rsqrt_1, squeeze_4, var_mean_1
# x_8 => add_12, add_13, mul_16, mul_19
triton_per_fused__native_batch_norm_legit_functional_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_5', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'in_ptr5', 'in_ptr6', 'out_ptr10', 'out_ptr4', 'out_ptr6', 'out_ptr8']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, out_ptr8, out_ptr10, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp27 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp36 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
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
    tmp18 = 0.001
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = 0.1
    tmp22 = tmp13 * tmp21
    tmp24 = 0.9
    tmp25 = tmp23 * tmp24
    tmp26 = tmp22 + tmp25
    tmp28 = tmp27 * tmp24
    tmp29 = tmp22 + tmp28
    tmp30 = 1.0000398612827361
    tmp31 = tmp17 * tmp30
    tmp32 = tmp31 * tmp21
    tmp34 = tmp33 * tmp24
    tmp35 = tmp32 + tmp34
    tmp37 = tmp36 * tmp24
    tmp38 = tmp32 + tmp37
    tl.store(out_ptr2 + (x0), tmp20, xmask)
    tl.store(out_ptr4 + (x0), tmp26, xmask)
    tl.store(out_ptr6 + (x0), tmp29, xmask)
    tl.store(out_ptr8 + (x0), tmp35, xmask)
    tl.store(out_ptr10 + (x0), tmp38, xmask)
    tl.store(out_ptr0 + (x0), tmp13, xmask)
    tl.store(out_ptr1 + (x0), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ls/clsppfz6mx3yvsok3sn4jfku5sklprlgwzblkothkdlcuwmx254r.py
# Source Nodes: [x_10, x_5, x_7, x_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_10 => relu_2
# x_5 => add_6, add_9, mul_13, mul_7, rsqrt_1, sub_1, var_mean_1
# x_7 => relu_1
# x_8 => add_14, mul_20
triton_poi_fused__native_batch_norm_legit_functional_relu_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 25088.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp16 = tmp9 * tmp15
    tmp18 = tmp16 + tmp17
    tmp19 = triton_helpers.maximum(0, tmp18)
    tl.store(out_ptr0 + (x3), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gq/cgqoeybonhlbu7coaqvvgml5siepocpz2ptthku4un7pkmoklb6r.py
# Source Nodes: [x_11], Original ATen: [aten._native_batch_norm_legit_functional]
# x_11 => var_mean_3
triton_red_fused__native_batch_norm_legit_functional_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_7', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 800
    rnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 200
    x1 = (xindex // 200)
    tmp2_mean = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_m2 = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    tmp2_weight = tl.zeros([XBLOCK, RBLOCK], tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + ((3136*x0) + (627200*(r2 // 3136)) + (1254400*x1) + (r2 % 3136)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/52/c52fqa6tchiwf6ssc22mp44cp7xpflfnbvkyida2e7mgo5bgiaur.py
# Source Nodes: [x_11], Original ATen: [aten._native_batch_norm_legit_functional]
# x_11 => add_16, add_17, add_18, mul_22, mul_23, mul_24, mul_25, mul_26, rsqrt_3, squeeze_10, var_mean_3
triton_per_fused__native_batch_norm_legit_functional_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[256, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_8', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 200
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (200*r1)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0 + (200*r1)), rmask & xmask, other=0.0)
    tmp2 = tl.load(in_ptr2 + (x0 + (200*r1)), rmask & xmask, other=0.0)
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
    tmp18 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/go/cgo5477ldybdxlsh27umqheh5mjo5kqiu6yynvwi7la6cuarrozk.py
# Source Nodes: [x_11, x_13], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_11 => add_16, add_19, mul_21, mul_27, rsqrt_3, sub_3, var_mean_3
# x_13 => relu_3
triton_poi_fused__native_batch_norm_legit_functional_relu_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5017600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 200
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 25088.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wp/cwpte2deagztzwr2tmvv4sbrcczujuvkvzeej5qa6h5ke7v4mxcr.py
# Source Nodes: [cat_138], Original ATen: [aten.cat]
# cat_138 => cat_1
triton_poi_fused_cat_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7927808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3136) % 316
    x2 = (xindex // 990976)
    x3 = xindex % 990976
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (928256*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (865536*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 316, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = (-256) + x1
    tmp14 = tmp13 >= tmp1
    tmp15 = tl.full([1], 40, tl.int64)
    tmp16 = tmp13 < tmp15
    tmp17 = tmp16 & tmp10
    tmp18 = tl.load(in_ptr0 + (x3 + (928256*x2)), tmp17, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp13 >= tmp15
    tmp22 = tl.full([1], 60, tl.int64)
    tmp23 = tmp13 < tmp22
    tmp24 = tmp21 & tmp10
    tmp25 = tl.load(in_ptr1 + ((-125440) + x3 + (865536*x2)), tmp24, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp24, tmp25, tmp26)
    tmp28 = tl.where(tmp16, tmp20, tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp10, tmp28, tmp29)
    tmp31 = tl.where(tmp4, tmp9, tmp30)
    tl.store(out_ptr0 + (x4), tmp31, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fl/cfljwqb6n2omfz33e4beznfceicbjgvfkse4xtppavbeb5wzcrmu.py
# Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
# x_17 => add_27, add_28, add_29, mul_36, mul_37, mul_38, mul_39, mul_40, rsqrt_5, squeeze_16, var_mean_5
triton_red_fused__native_batch_norm_legit_functional_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_11', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 316
    rnumel = 25088
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
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (990976*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp5 = 25088.0
    tmp6 = tmp3 / tmp5
    tmp7 = 0.001
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = 0.1
    tmp11 = tmp2 * tmp10
    tmp13 = 0.9
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = 1.0000398612827361
    tmp17 = tmp6 * tmp16
    tmp18 = tmp17 * tmp10
    tmp20 = tmp19 * tmp13
    tmp21 = tmp18 + tmp20
    tl.store(out_ptr2 + (x0), tmp9, xmask)
    tl.store(out_ptr4 + (x0), tmp15, xmask)
    tl.store(out_ptr6 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nm/cnmn7jievqsjvf3hsxu6gpxzlxjyqrtw2uyvlhm5kpdyzbzn6sw3.py
# Source Nodes: [x_17, x_19], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_17 => add_27, add_30, mul_35, mul_41, rsqrt_5, sub_5, var_mean_5
# x_19 => relu_5
triton_poi_fused__native_batch_norm_legit_functional_relu_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7927808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 316
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 25088.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7w/c7wof4c2ys4k6gvg3xxjbferbck5ksg2c5epjpizfvbwfzv6jcd3.py
# Source Nodes: [cat_136], Original ATen: [aten.cat]
# cat_136 => cat_3
triton_poi_fused_cat_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8429568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3136) % 336
    x2 = (xindex // 1053696)
    x3 = xindex % 1053696
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (928256*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (865536*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x3 + (865536*x2)), tmp4, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 336, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = (-256) + x1
    tmp16 = tmp15 >= tmp1
    tmp17 = tl.full([1], 60, tl.int64)
    tmp18 = tmp15 < tmp17
    tmp19 = tmp18 & tmp12
    tmp20 = tl.full([1], 40, tl.int64)
    tmp21 = tmp15 < tmp20
    tmp22 = tmp21 & tmp19
    tmp23 = tl.load(in_ptr0 + (x3 + (928256*x2)), tmp22, other=0.0)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp22, tmp23, tmp24)
    tmp26 = tmp15 >= tmp20
    tmp27 = tmp26 & tmp19
    tmp28 = tl.load(in_ptr1 + ((-125440) + x3 + (865536*x2)), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tl.where(tmp21, tmp25, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp19, tmp31, tmp32)
    tmp34 = tmp15 >= tmp17
    tmp35 = tl.full([1], 80, tl.int64)
    tmp36 = tmp15 < tmp35
    tmp37 = tmp34 & tmp12
    tmp38 = tl.load(in_ptr2 + ((-188160) + x3 + (865536*x2)), tmp37, other=0.0)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = tl.where(tmp18, tmp33, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp12, tmp41, tmp42)
    tmp44 = tl.where(tmp4, tmp11, tmp43)
    tl.store(out_ptr0 + (x4), tmp44, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ad/cad762wzis2nku5pjh2xbgv4znwoqzyl4wc4efi4w5yua3v5r66b.py
# Source Nodes: [x_26], Original ATen: [aten._native_batch_norm_legit_functional]
# x_26 => add_43, add_44, add_45, mul_57, mul_58, mul_59, mul_60, mul_61, rsqrt_8, squeeze_25, var_mean_8
triton_red_fused__native_batch_norm_legit_functional_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_14', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 336
    rnumel = 25088
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
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (1053696*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp5 = 25088.0
    tmp6 = tmp3 / tmp5
    tmp7 = 0.001
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = 0.1
    tmp11 = tmp2 * tmp10
    tmp13 = 0.9
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = 1.0000398612827361
    tmp17 = tmp6 * tmp16
    tmp18 = tmp17 * tmp10
    tmp20 = tmp19 * tmp13
    tmp21 = tmp18 + tmp20
    tl.store(out_ptr2 + (x0), tmp9, xmask)
    tl.store(out_ptr4 + (x0), tmp15, xmask)
    tl.store(out_ptr6 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dy/cdywogsgpx7trt3n6gv7vsyrmkl2wa4pyi3rhrp7atcziuxtevl3.py
# Source Nodes: [x_26, x_28], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_26 => add_43, add_46, mul_56, mul_62, rsqrt_8, sub_8, var_mean_8
# x_28 => relu_8
triton_poi_fused__native_batch_norm_legit_functional_relu_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8429568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 336
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 25088.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zg/czgnrcidnkqzrcwecqp3dfyjbbpuziovo675du5hvxj34vzbyq2e.py
# Source Nodes: [cat_134], Original ATen: [aten.cat]
# cat_134 => cat_5
triton_poi_fused_cat_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8931328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3136) % 356
    x2 = (xindex // 1116416)
    x3 = xindex % 1116416
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (928256*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (865536*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x3 + (865536*x2)), tmp4, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + (x3 + (865536*x2)), tmp4, other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 356, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = (-256) + x1
    tmp18 = tmp17 >= tmp1
    tmp19 = tl.full([1], 80, tl.int64)
    tmp20 = tmp17 < tmp19
    tmp21 = tmp20 & tmp14
    tmp22 = tl.full([1], 60, tl.int64)
    tmp23 = tmp17 < tmp22
    tmp24 = tmp23 & tmp21
    tmp25 = tl.full([1], 40, tl.int64)
    tmp26 = tmp17 < tmp25
    tmp27 = tmp26 & tmp24
    tmp28 = tl.load(in_ptr0 + (x3 + (928256*x2)), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp17 >= tmp25
    tmp32 = tmp31 & tmp24
    tmp33 = tl.load(in_ptr1 + ((-125440) + x3 + (865536*x2)), tmp32, other=0.0)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp32, tmp33, tmp34)
    tmp36 = tl.where(tmp26, tmp30, tmp35)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp24, tmp36, tmp37)
    tmp39 = tmp17 >= tmp22
    tmp40 = tmp39 & tmp21
    tmp41 = tl.load(in_ptr2 + ((-188160) + x3 + (865536*x2)), tmp40, other=0.0)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp40, tmp41, tmp42)
    tmp44 = tl.where(tmp23, tmp38, tmp43)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp21, tmp44, tmp45)
    tmp47 = tmp17 >= tmp19
    tmp48 = tl.full([1], 100, tl.int64)
    tmp49 = tmp17 < tmp48
    tmp50 = tmp47 & tmp14
    tmp51 = tl.load(in_ptr3 + ((-250880) + x3 + (865536*x2)), tmp50, other=0.0)
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp50, tmp51, tmp52)
    tmp54 = tl.where(tmp20, tmp46, tmp53)
    tmp55 = tl.full(tmp54.shape, 0.0, tmp54.dtype)
    tmp56 = tl.where(tmp14, tmp54, tmp55)
    tmp57 = tl.where(tmp4, tmp13, tmp56)
    tl.store(out_ptr0 + (x4), tmp57, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pq/cpqc2fdb6cjoytud6ubnixiw7vpslfqfyupwqhsxr4vuapors4yp.py
# Source Nodes: [x_35], Original ATen: [aten._native_batch_norm_legit_functional]
# x_35 => add_59, add_60, add_61, mul_78, mul_79, mul_80, mul_81, mul_82, rsqrt_11, squeeze_34, var_mean_11
triton_red_fused__native_batch_norm_legit_functional_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_17', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 356
    rnumel = 25088
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
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (1116416*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp5 = 25088.0
    tmp6 = tmp3 / tmp5
    tmp7 = 0.001
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = 0.1
    tmp11 = tmp2 * tmp10
    tmp13 = 0.9
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = 1.0000398612827361
    tmp17 = tmp6 * tmp16
    tmp18 = tmp17 * tmp10
    tmp20 = tmp19 * tmp13
    tmp21 = tmp18 + tmp20
    tl.store(out_ptr2 + (x0), tmp9, xmask)
    tl.store(out_ptr4 + (x0), tmp15, xmask)
    tl.store(out_ptr6 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qf/cqf6xawbekoznugnouhy7cnrdigdso2kjyoaieljglz2bkywy2i3.py
# Source Nodes: [x_35, x_37], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_35 => add_59, add_62, mul_77, mul_83, rsqrt_11, sub_11, var_mean_11
# x_37 => relu_11
triton_poi_fused__native_batch_norm_legit_functional_relu_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8931328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 356
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 25088.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vf/cvfjn73wj7gtfszut3unbtabyqvagw2ptclydvyo4og3o2qh7maw.py
# Source Nodes: [cat_133], Original ATen: [aten.cat]
# cat_133 => cat_6
triton_poi_fused_cat_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3136) % 120
    x2 = (xindex // 376320)
    x3 = xindex % 376320
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 100, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 80, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.full([1], 60, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tmp9 & tmp7
    tmp11 = tl.full([1], 40, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp12 & tmp10
    tmp14 = tl.load(in_ptr0 + (802816 + x3 + (928256*x2)), tmp13, other=0.0)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = tmp0 >= tmp11
    tmp18 = tmp17 & tmp10
    tmp19 = tl.load(in_ptr1 + (677376 + x3 + (865536*x2)), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tl.where(tmp12, tmp16, tmp21)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp10, tmp22, tmp23)
    tmp25 = tmp0 >= tmp8
    tmp26 = tmp25 & tmp7
    tmp27 = tl.load(in_ptr2 + (614656 + x3 + (865536*x2)), tmp26, other=0.0)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp26, tmp27, tmp28)
    tmp30 = tl.where(tmp9, tmp24, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp7, tmp30, tmp31)
    tmp33 = tmp0 >= tmp5
    tmp34 = tmp33 & tmp4
    tmp35 = tl.load(in_ptr3 + (551936 + x3 + (865536*x2)), tmp34, other=0.0)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp34, tmp35, tmp36)
    tmp38 = tl.where(tmp6, tmp32, tmp37)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp4, tmp38, tmp39)
    tmp41 = tmp0 >= tmp3
    tmp42 = tl.full([1], 120, tl.int64)
    tmp43 = tmp0 < tmp42
    tmp44 = tl.load(in_ptr4 + (489216 + x3 + (865536*x2)), tmp41, other=0.0)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp41, tmp44, tmp45)
    tmp47 = tl.where(tmp4, tmp40, tmp46)
    tl.store(out_ptr0 + (x3 + (1179136*x2)), tmp47, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/s7/cs7d5bgrmcjfyicc5k4m4vf56ha46muzqk6imxg4d2qzgodysggd.py
# Source Nodes: [resid_3, x_s1_1, x_s1_2, x_s1_3], Original ATen: [aten.add]
# resid_3 => add_73
# x_s1_1 => add_25
# x_s1_2 => add_41
# x_s1_3 => add_57
triton_poi_fused_add_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 802816
    x1 = (xindex // 802816)
    tmp0 = tl.load(in_ptr0 + (x0 + (928256*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (865536*x1)), None)
    tmp3 = tl.load(in_ptr2 + (x0 + (865536*x1)), None)
    tmp5 = tl.load(in_ptr3 + (x0 + (865536*x1)), None)
    tmp7 = tl.load(in_ptr4 + (x0 + (865536*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x0 + (1179136*x1)), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ln/cln5ykgtobe6qpc7fszkcq5swqe42toshl5cc2bzby2nlaewhx7e.py
# Source Nodes: [x_44, x_47], Original ATen: [aten._native_batch_norm_legit_functional]
# x_44 => add_75, add_76, add_77, mul_100, mul_101, mul_102, mul_103, mul_99, rsqrt_14, squeeze_43, var_mean_14
# x_47 => add_81, add_82, mul_107, mul_110
triton_red_fused__native_batch_norm_legit_functional_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_21', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'in_ptr3', 'in_ptr4', 'out_ptr10', 'out_ptr4', 'out_ptr6', 'out_ptr8']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, out_ptr8, out_ptr10, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 376
    rnumel = 25088
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
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (1179136*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp16 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = 25088.0
    tmp6 = tmp3 / tmp5
    tmp7 = 0.001
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = 0.1
    tmp11 = tmp2 * tmp10
    tmp13 = 0.9
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp17 = tmp16 * tmp13
    tmp18 = tmp11 + tmp17
    tmp19 = 1.0000398612827361
    tmp20 = tmp6 * tmp19
    tmp21 = tmp20 * tmp10
    tmp23 = tmp22 * tmp13
    tmp24 = tmp21 + tmp23
    tmp26 = tmp25 * tmp13
    tmp27 = tmp21 + tmp26
    tl.store(out_ptr2 + (x0), tmp9, xmask)
    tl.store(out_ptr4 + (x0), tmp15, xmask)
    tl.store(out_ptr6 + (x0), tmp18, xmask)
    tl.store(out_ptr8 + (x0), tmp24, xmask)
    tl.store(out_ptr10 + (x0), tmp27, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r3/cr3e74q5bfzrcetcleecegytp7vlse5c2hfaqc6p5wxojwdpnn5a.py
# Source Nodes: [x_44, x_46, x_47, x_49], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_44 => add_75, add_78, mul_104, mul_98, rsqrt_14, sub_14, var_mean_14
# x_46 => relu_14
# x_47 => add_83, mul_111
# x_49 => relu_15
triton_poi_fused__native_batch_norm_legit_functional_relu_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9433088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 376
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 25088.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp16 = tmp9 * tmp15
    tmp18 = tmp16 + tmp17
    tmp19 = triton_helpers.maximum(0, tmp18)
    tl.store(out_ptr0 + (x3), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/u4/cu4yhpzcukp7icgng664e5ub3eodtlzurmrzomyfw7g53mhwv72q.py
# Source Nodes: [x_50], Original ATen: [aten._native_batch_norm_legit_functional]
# x_50 => add_85, add_86, add_87, mul_113, mul_114, mul_115, mul_116, mul_117, rsqrt_16, squeeze_49, var_mean_16
triton_red_fused__native_batch_norm_legit_functional_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_23', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 400
    rnumel = 25088
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
        r1 = rindex % 3136
        r2 = (rindex // 3136)
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (1254400*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp5 = 25088.0
    tmp6 = tmp3 / tmp5
    tmp7 = 0.001
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = 0.1
    tmp11 = tmp2 * tmp10
    tmp13 = 0.9
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp16 = 1.0000398612827361
    tmp17 = tmp6 * tmp16
    tmp18 = tmp17 * tmp10
    tmp20 = tmp19 * tmp13
    tmp21 = tmp18 + tmp20
    tl.store(out_ptr2 + (x0), tmp9, xmask)
    tl.store(out_ptr4 + (x0), tmp15, xmask)
    tl.store(out_ptr6 + (x0), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mz/cmzfdhluziz3jqb6cdc5j7lwrjlsdjmctww4s2mah5hxfx4zatgz.py
# Source Nodes: [x_50, x_52], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_50 => add_85, add_88, mul_112, mul_118, rsqrt_16, sub_16, var_mean_16
# x_52 => relu_16
triton_poi_fused__native_batch_norm_legit_functional_relu_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10035200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 400
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 25088.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/w6/cw644bpvijbzialc3d2qom7dljx5ajyiwb67bnv4zu5lcwc5ri4b.py
# Source Nodes: [x_53], Original ATen: [aten._native_batch_norm_legit_functional]
# x_53 => add_90, add_91, add_92, mul_120, mul_121, mul_122, mul_123, mul_124, rsqrt_17, squeeze_52, var_mean_17
triton_red_fused__native_batch_norm_legit_functional_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_25', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 400
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (313600*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/lu/clupyfv3rmqmea6qnulivbdneoawts3v662cinz4bih45na34s45.py
# Source Nodes: [x_53, x_55], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_53 => add_90, add_93, mul_119, mul_125, rsqrt_17, sub_17, var_mean_17
# x_55 => relu_17
triton_poi_fused__native_batch_norm_legit_functional_relu_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2508800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 400
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pk/cpkqss2ejkcczpbifygbc7ce5ifgph4rjtlyscrtkce2yftrj3ws.py
# Source Nodes: [cat_130], Original ATen: [aten.cat]
# cat_130 => cat_9
triton_poi_fused_cat_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4415488
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 704
    x2 = (xindex // 551936)
    x3 = xindex % 551936
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (501760*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (451584*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 704, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = (-512) + x1
    tmp14 = tmp13 >= tmp1
    tmp15 = tl.full([1], 128, tl.int64)
    tmp16 = tmp13 < tmp15
    tmp17 = tmp16 & tmp10
    tmp18 = tl.load(in_ptr0 + (x3 + (501760*x2)), tmp17, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp13 >= tmp15
    tmp22 = tl.full([1], 192, tl.int64)
    tmp23 = tmp13 < tmp22
    tmp24 = tmp21 & tmp10
    tmp25 = tl.load(in_ptr1 + ((-100352) + x3 + (451584*x2)), tmp24, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp24, tmp25, tmp26)
    tmp28 = tl.where(tmp16, tmp20, tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp10, tmp28, tmp29)
    tmp31 = tl.where(tmp4, tmp9, tmp30)
    tl.store(out_ptr0 + (x4), tmp31, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yw/cywcxporislfmzdqoaxlq433kqs3iq22geifkaa3i2inc3ym4zxt.py
# Source Nodes: [x_56], Original ATen: [aten._native_batch_norm_legit_functional]
# x_56 => add_96, add_97, add_98, mul_127, mul_128, mul_129, mul_130, mul_131, rsqrt_18, squeeze_55, var_mean_18
triton_red_fused__native_batch_norm_legit_functional_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_28', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 704
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (551936*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/6k/c6kdtzpni5yyjmewxulqlhd6rbk3t43b4kyoiwu7wl4xupbuwcfp.py
# Source Nodes: [x_56, x_58], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_56 => add_96, add_99, mul_126, mul_132, rsqrt_18, sub_18, var_mean_18
# x_58 => relu_18
triton_poi_fused__native_batch_norm_legit_functional_relu_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4415488
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 704
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qi/cqin7k4qwgiz22i7epgx7vy6djo77z7csggmwmvejg3a4lf3i3yl.py
# Source Nodes: [cat_128], Original ATen: [aten.cat]
# cat_128 => cat_11
triton_poi_fused_cat_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 768
    x2 = (xindex // 602112)
    x3 = xindex % 602112
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (501760*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (451584*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x3 + (451584*x2)), tmp4, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 768, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = (-512) + x1
    tmp16 = tmp15 >= tmp1
    tmp17 = tl.full([1], 192, tl.int64)
    tmp18 = tmp15 < tmp17
    tmp19 = tmp18 & tmp12
    tmp20 = tl.full([1], 128, tl.int64)
    tmp21 = tmp15 < tmp20
    tmp22 = tmp21 & tmp19
    tmp23 = tl.load(in_ptr0 + (x3 + (501760*x2)), tmp22, other=0.0)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp22, tmp23, tmp24)
    tmp26 = tmp15 >= tmp20
    tmp27 = tmp26 & tmp19
    tmp28 = tl.load(in_ptr1 + ((-100352) + x3 + (451584*x2)), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tl.where(tmp21, tmp25, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp19, tmp31, tmp32)
    tmp34 = tmp15 >= tmp17
    tmp35 = tl.full([1], 256, tl.int64)
    tmp36 = tmp15 < tmp35
    tmp37 = tmp34 & tmp12
    tmp38 = tl.load(in_ptr2 + ((-150528) + x3 + (451584*x2)), tmp37, other=0.0)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = tl.where(tmp18, tmp33, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp12, tmp41, tmp42)
    tmp44 = tl.where(tmp4, tmp11, tmp43)
    tl.store(out_ptr0 + (x4), tmp44, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lo/clooi37qvocmzacn5yoqt56yfh4sj5euh34w6fndippcc7wjg7lp.py
# Source Nodes: [x_65], Original ATen: [aten._native_batch_norm_legit_functional]
# x_65 => add_112, add_113, add_114, mul_148, mul_149, mul_150, mul_151, mul_152, rsqrt_21, squeeze_64, var_mean_21
triton_red_fused__native_batch_norm_legit_functional_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_31', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (602112*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/el/celjtp4ynkgnbeb24l3df44bq5vlayqdxxzler32jk3pds325nyl.py
# Source Nodes: [x_65, x_67], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_65 => add_112, add_115, mul_147, mul_153, rsqrt_21, sub_21, var_mean_21
# x_67 => relu_21
triton_poi_fused__native_batch_norm_legit_functional_relu_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 768
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ms/cmshs2tm74gduv3r4ywbgah266uty4vfqsesfl4end2nzbbqjwai.py
# Source Nodes: [cat_126], Original ATen: [aten.cat]
# cat_126 => cat_13
triton_poi_fused_cat_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5218304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 832
    x2 = (xindex // 652288)
    x3 = xindex % 652288
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (501760*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (451584*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x3 + (451584*x2)), tmp4, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + (x3 + (451584*x2)), tmp4, other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 832, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = (-512) + x1
    tmp18 = tmp17 >= tmp1
    tmp19 = tl.full([1], 256, tl.int64)
    tmp20 = tmp17 < tmp19
    tmp21 = tmp20 & tmp14
    tmp22 = tl.full([1], 192, tl.int64)
    tmp23 = tmp17 < tmp22
    tmp24 = tmp23 & tmp21
    tmp25 = tl.full([1], 128, tl.int64)
    tmp26 = tmp17 < tmp25
    tmp27 = tmp26 & tmp24
    tmp28 = tl.load(in_ptr0 + (x3 + (501760*x2)), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp17 >= tmp25
    tmp32 = tmp31 & tmp24
    tmp33 = tl.load(in_ptr1 + ((-100352) + x3 + (451584*x2)), tmp32, other=0.0)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp32, tmp33, tmp34)
    tmp36 = tl.where(tmp26, tmp30, tmp35)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp24, tmp36, tmp37)
    tmp39 = tmp17 >= tmp22
    tmp40 = tmp39 & tmp21
    tmp41 = tl.load(in_ptr2 + ((-150528) + x3 + (451584*x2)), tmp40, other=0.0)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp40, tmp41, tmp42)
    tmp44 = tl.where(tmp23, tmp38, tmp43)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp21, tmp44, tmp45)
    tmp47 = tmp17 >= tmp19
    tmp48 = tl.full([1], 320, tl.int64)
    tmp49 = tmp17 < tmp48
    tmp50 = tmp47 & tmp14
    tmp51 = tl.load(in_ptr3 + ((-200704) + x3 + (451584*x2)), tmp50, other=0.0)
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp50, tmp51, tmp52)
    tmp54 = tl.where(tmp20, tmp46, tmp53)
    tmp55 = tl.full(tmp54.shape, 0.0, tmp54.dtype)
    tmp56 = tl.where(tmp14, tmp54, tmp55)
    tmp57 = tl.where(tmp4, tmp13, tmp56)
    tl.store(out_ptr0 + (x4), tmp57, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kl/ckl67ymupswxpgyn5ksdv5ngtp4knmjpb4ckzozmbhj5ata2z74b.py
# Source Nodes: [x_74], Original ATen: [aten._native_batch_norm_legit_functional]
# x_74 => add_128, add_129, add_130, mul_169, mul_170, mul_171, mul_172, mul_173, rsqrt_24, squeeze_73, var_mean_24
triton_red_fused__native_batch_norm_legit_functional_34 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_34', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (652288*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/ss/css5s7z5jjocyx2ijygblx4t3busl4iaaspycsjjwd2gunvlva5e.py
# Source Nodes: [x_74, x_76], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_74 => add_128, add_131, mul_168, mul_174, rsqrt_24, sub_24, var_mean_24
# x_76 => relu_24
triton_poi_fused__native_batch_norm_legit_functional_relu_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5218304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 832
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/of/cof65m4muuhuvztlqfazt43nlbynyz676fntqnjkgq4p673a6hzs.py
# Source Nodes: [x_s1_5, x_s1_6, x_s1_7, x_s1_8], Original ATen: [aten.add]
# x_s1_5 => add_94
# x_s1_6 => add_110
# x_s1_7 => add_126
# x_s1_8 => add_142
triton_poi_fused_add_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 401408
    x1 = (xindex // 401408)
    tmp0 = tl.load(in_ptr0 + (x0 + (501760*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (451584*x1)), None)
    tmp3 = tl.load(in_ptr2 + (x0 + (451584*x1)), None)
    tmp5 = tl.load(in_ptr3 + (x0 + (451584*x1)), None)
    tmp7 = tl.load(in_ptr4 + (x0 + (451584*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x0 + (702464*x1)), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2l/c2lorn7j6nwurvisdnjl6hbawjflvgqyqh7bdv2touxmyxqw4bsf.py
# Source Nodes: [cat_125], Original ATen: [aten.cat]
# cat_125 => cat_14
triton_poi_fused_cat_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 384
    x2 = (xindex // 301056)
    x3 = xindex % 301056
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 320, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 256, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.full([1], 192, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tmp9 & tmp7
    tmp11 = tl.full([1], 128, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp12 & tmp10
    tmp14 = tl.load(in_ptr0 + (401408 + x3 + (501760*x2)), tmp13, other=0.0)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = tmp0 >= tmp11
    tmp18 = tmp17 & tmp10
    tmp19 = tl.load(in_ptr1 + (301056 + x3 + (451584*x2)), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tl.where(tmp12, tmp16, tmp21)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp10, tmp22, tmp23)
    tmp25 = tmp0 >= tmp8
    tmp26 = tmp25 & tmp7
    tmp27 = tl.load(in_ptr2 + (250880 + x3 + (451584*x2)), tmp26, other=0.0)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp26, tmp27, tmp28)
    tmp30 = tl.where(tmp9, tmp24, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp7, tmp30, tmp31)
    tmp33 = tmp0 >= tmp5
    tmp34 = tmp33 & tmp4
    tmp35 = tl.load(in_ptr3 + (200704 + x3 + (451584*x2)), tmp34, other=0.0)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp34, tmp35, tmp36)
    tmp38 = tl.where(tmp6, tmp32, tmp37)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp4, tmp38, tmp39)
    tmp41 = tmp0 >= tmp3
    tmp42 = tl.full([1], 384, tl.int64)
    tmp43 = tmp0 < tmp42
    tmp44 = tl.load(in_ptr4 + (150528 + x3 + (451584*x2)), tmp41, other=0.0)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp41, tmp44, tmp45)
    tmp47 = tl.where(tmp4, tmp40, tmp46)
    tl.store(out_ptr0 + (x3 + (702464*x2)), tmp47, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xp/cxpr7lhzwmzofur2imj7cwl6jejcdx3ksvazaigxf2xkyr67vf5t.py
# Source Nodes: [x_83], Original ATen: [aten._native_batch_norm_legit_functional]
# x_83 => add_144, add_145, add_146, mul_190, mul_191, mul_192, mul_193, mul_194, rsqrt_27, squeeze_82, var_mean_27
triton_red_fused__native_batch_norm_legit_functional_38 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_38', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 896
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (702464*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/7g/c7goeqzlqwm4e7u36hg34jy7ga22nysnae6ehinu33tjodyg7b7r.py
# Source Nodes: [x_83, x_85], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_83 => add_144, add_147, mul_189, mul_195, rsqrt_27, sub_27, var_mean_27
# x_85 => relu_27
triton_poi_fused__native_batch_norm_legit_functional_relu_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5619712
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 896
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pb/cpboclppph6voy3engpiyfd7i6yc6jbc2walmgsvtnnojseb655x.py
# Source Nodes: [cat_122], Original ATen: [aten.cat]
# cat_122 => cat_17
triton_poi_fused_cat_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6021120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 960
    x2 = (xindex // 752640)
    x3 = xindex % 752640
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (702464*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (451584*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 960, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = (-512) + x1
    tmp14 = tmp13 >= tmp1
    tmp15 = tl.full([1], 384, tl.int64)
    tmp16 = tmp13 < tmp15
    tmp17 = tmp16 & tmp10
    tmp18 = tl.load(in_ptr2 + ((-401408) + x3 + (702464*x2)), tmp17, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp13 >= tmp15
    tmp22 = tl.full([1], 448, tl.int64)
    tmp23 = tmp13 < tmp22
    tmp24 = tmp21 & tmp10
    tmp25 = tl.load(in_ptr1 + ((-301056) + x3 + (451584*x2)), tmp24, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp24, tmp25, tmp26)
    tmp28 = tl.where(tmp16, tmp20, tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp10, tmp28, tmp29)
    tmp31 = tl.where(tmp4, tmp9, tmp30)
    tl.store(out_ptr0 + (x4), tmp31, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ld/cldjptpxn6dzpbnpw2v4wxpzsyiwwsmafibrctjwjmyxj3w7lfqe.py
# Source Nodes: [x_92], Original ATen: [aten._native_batch_norm_legit_functional]
# x_92 => add_160, add_161, add_162, mul_211, mul_212, mul_213, mul_214, mul_215, rsqrt_30, squeeze_91, var_mean_30
triton_red_fused__native_batch_norm_legit_functional_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_41', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 960
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (752640*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/6d/c6dliqsr3xly2ao7tfvgs3h7nncbiny6jo7w2ibezsljtzii6alb.py
# Source Nodes: [x_92, x_94], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_92 => add_160, add_163, mul_210, mul_216, rsqrt_30, sub_30, var_mean_30
# x_94 => relu_30
triton_poi_fused__native_batch_norm_legit_functional_relu_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6021120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 960
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ky/ckylclyrp27mr7gb3x7vzqmjdkvy6kigibqajk6vy7legoja57p7.py
# Source Nodes: [cat_120], Original ATen: [aten.cat]
# cat_120 => cat_19
triton_poi_fused_cat_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 1024
    x2 = (xindex // 802816)
    x3 = xindex % 802816
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (702464*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (451584*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x3 + (451584*x2)), tmp4, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 1024, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = (-512) + x1
    tmp16 = tmp15 >= tmp1
    tmp17 = tl.full([1], 448, tl.int64)
    tmp18 = tmp15 < tmp17
    tmp19 = tmp18 & tmp12
    tmp20 = tl.full([1], 384, tl.int64)
    tmp21 = tmp15 < tmp20
    tmp22 = tmp21 & tmp19
    tmp23 = tl.load(in_ptr3 + ((-401408) + x3 + (702464*x2)), tmp22, other=0.0)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp22, tmp23, tmp24)
    tmp26 = tmp15 >= tmp20
    tmp27 = tmp26 & tmp19
    tmp28 = tl.load(in_ptr1 + ((-301056) + x3 + (451584*x2)), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tl.where(tmp21, tmp25, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp19, tmp31, tmp32)
    tmp34 = tmp15 >= tmp17
    tmp35 = tmp15 < tmp3
    tmp36 = tmp34 & tmp12
    tmp37 = tl.load(in_ptr2 + ((-351232) + x3 + (451584*x2)), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tl.where(tmp18, tmp33, tmp39)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp12, tmp40, tmp41)
    tmp43 = tl.where(tmp4, tmp11, tmp42)
    tl.store(out_ptr0 + (x4), tmp43, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/eg/cegjdb6ycysn5t5slzsdtasicj4vq4sy7tdsjsikazwbhap2uak2.py
# Source Nodes: [x_101], Original ATen: [aten._native_batch_norm_legit_functional]
# x_101 => add_176, add_177, add_178, mul_232, mul_233, mul_234, mul_235, mul_236, rsqrt_33, squeeze_100, var_mean_33
triton_red_fused__native_batch_norm_legit_functional_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_44', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/av/cavb5gmtyslrkdrlznaxfvrqy5mwuso2isunctrfv3lyx2hnkn6v.py
# Source Nodes: [x_101, x_103], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_101 => add_176, add_179, mul_231, mul_237, rsqrt_33, sub_33, var_mean_33
# x_103 => relu_33
triton_poi_fused__native_batch_norm_legit_functional_relu_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vq/cvqknldigrj4rcdudslhtx5kp62rg36fb73wyweppw34qmbjr53v.py
# Source Nodes: [cat_118], Original ATen: [aten.cat]
# cat_118 => cat_21
triton_poi_fused_cat_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6823936
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 1088
    x2 = (xindex // 852992)
    x3 = xindex % 852992
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (702464*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (451584*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x3 + (451584*x2)), tmp4, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + (x3 + (451584*x2)), tmp4, other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 1088, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = (-512) + x1
    tmp18 = tmp17 >= tmp1
    tmp19 = tmp17 < tmp3
    tmp20 = tmp19 & tmp14
    tmp21 = tl.full([1], 448, tl.int64)
    tmp22 = tmp17 < tmp21
    tmp23 = tmp22 & tmp20
    tmp24 = tl.full([1], 384, tl.int64)
    tmp25 = tmp17 < tmp24
    tmp26 = tmp25 & tmp23
    tmp27 = tl.load(in_ptr4 + ((-401408) + x3 + (702464*x2)), tmp26, other=0.0)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp26, tmp27, tmp28)
    tmp30 = tmp17 >= tmp24
    tmp31 = tmp30 & tmp23
    tmp32 = tl.load(in_ptr1 + ((-301056) + x3 + (451584*x2)), tmp31, other=0.0)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp31, tmp32, tmp33)
    tmp35 = tl.where(tmp25, tmp29, tmp34)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp23, tmp35, tmp36)
    tmp38 = tmp17 >= tmp21
    tmp39 = tmp38 & tmp20
    tmp40 = tl.load(in_ptr2 + ((-351232) + x3 + (451584*x2)), tmp39, other=0.0)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp39, tmp40, tmp41)
    tmp43 = tl.where(tmp22, tmp37, tmp42)
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp20, tmp43, tmp44)
    tmp46 = tmp17 >= tmp3
    tmp47 = tl.full([1], 576, tl.int64)
    tmp48 = tmp17 < tmp47
    tmp49 = tmp46 & tmp14
    tmp50 = tl.load(in_ptr3 + ((-401408) + x3 + (451584*x2)), tmp49, other=0.0)
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp49, tmp50, tmp51)
    tmp53 = tl.where(tmp19, tmp45, tmp52)
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp14, tmp53, tmp54)
    tmp56 = tl.where(tmp4, tmp13, tmp55)
    tl.store(out_ptr0 + (x4), tmp56, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fc/cfcqhchmor36rv5pvhdislqyeaisw4ulshaaov4asmr4f5hua53j.py
# Source Nodes: [x_110], Original ATen: [aten._native_batch_norm_legit_functional]
# x_110 => add_192, add_193, add_194, mul_253, mul_254, mul_255, mul_256, mul_257, rsqrt_36, squeeze_109, var_mean_36
triton_red_fused__native_batch_norm_legit_functional_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_47', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1088
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (852992*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/ki/ckiaapaasf5vhu4hmiwjrlhrrehsv2j4fv6ej7kufgycdp3uzych.py
# Source Nodes: [x_110, x_112], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_110 => add_192, add_195, mul_252, mul_258, rsqrt_36, sub_36, var_mean_36
# x_112 => relu_36
triton_poi_fused__native_batch_norm_legit_functional_relu_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6823936
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 1088
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dq/cdqo56qwlshfl72dlsi57t5ap2iofayd4bmbhwjfx3p5z5q2gscd.py
# Source Nodes: [cat_117], Original ATen: [aten.cat]
# cat_117 => cat_22
triton_poi_fused_cat_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4014080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 640
    x2 = (xindex // 501760)
    x3 = xindex % 501760
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 576, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 512, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.full([1], 448, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tmp9 & tmp7
    tmp11 = tl.full([1], 384, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp12 & tmp10
    tmp14 = tl.load(in_ptr0 + (x3 + (702464*x2)), tmp13, other=0.0)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = tmp0 >= tmp11
    tmp18 = tmp17 & tmp10
    tmp19 = tl.load(in_ptr1 + (100352 + x3 + (451584*x2)), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tl.where(tmp12, tmp16, tmp21)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp10, tmp22, tmp23)
    tmp25 = tmp0 >= tmp8
    tmp26 = tmp25 & tmp7
    tmp27 = tl.load(in_ptr2 + (50176 + x3 + (451584*x2)), tmp26, other=0.0)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp26, tmp27, tmp28)
    tmp30 = tl.where(tmp9, tmp24, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp7, tmp30, tmp31)
    tmp33 = tmp0 >= tmp5
    tmp34 = tmp33 & tmp4
    tmp35 = tl.load(in_ptr3 + (x3 + (451584*x2)), tmp34, other=0.0)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp34, tmp35, tmp36)
    tmp38 = tl.where(tmp6, tmp32, tmp37)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp4, tmp38, tmp39)
    tmp41 = tmp0 >= tmp3
    tmp42 = tl.full([1], 640, tl.int64)
    tmp43 = tmp0 < tmp42
    tmp44 = tl.load(in_ptr4 + ((-50176) + x3 + (451584*x2)), tmp41, other=0.0)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp41, tmp44, tmp45)
    tmp47 = tl.where(tmp4, tmp40, tmp46)
    tl.store(out_ptr0 + (x3 + (903168*x2)), tmp47, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/d6/cd6tclarhy44ed3ituqebj3wvygtd6qobouk2zbgw5khwzyxwubw.py
# Source Nodes: [resid_11, x_s1_10, x_s1_11, x_s1_9], Original ATen: [aten.add]
# resid_11 => add_206
# x_s1_10 => add_174
# x_s1_11 => add_190
# x_s1_9 => add_158
triton_poi_fused_add_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 401408
    x1 = (xindex // 401408)
    tmp0 = tl.load(in_ptr0 + (x0 + (702464*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (451584*x1)), None)
    tmp3 = tl.load(in_ptr2 + (x0 + (451584*x1)), None)
    tmp5 = tl.load(in_ptr3 + (x0 + (451584*x1)), None)
    tmp7 = tl.load(in_ptr4 + (x0 + (451584*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x0 + (903168*x1)), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/o6/co67rmkt656xorsjwyeu23l5bip6ao2zqifoq4rr4s6fzfwfevhp.py
# Source Nodes: [x_119, x_122], Original ATen: [aten._native_batch_norm_legit_functional]
# x_119 => add_208, add_209, add_210, mul_274, mul_275, mul_276, mul_277, mul_278, rsqrt_39, squeeze_118, var_mean_39
# x_122 => add_214, add_215, mul_282, mul_285
triton_red_fused__native_batch_norm_legit_functional_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_51', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'in_ptr3', 'in_ptr4', 'out_ptr10', 'out_ptr4', 'out_ptr6', 'out_ptr8']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, out_ptr8, out_ptr10, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1152
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (903168*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp16 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = 6272.0
    tmp6 = tmp3 / tmp5
    tmp7 = 0.001
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = 0.1
    tmp11 = tmp2 * tmp10
    tmp13 = 0.9
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp17 = tmp16 * tmp13
    tmp18 = tmp11 + tmp17
    tmp19 = 1.0001594642002871
    tmp20 = tmp6 * tmp19
    tmp21 = tmp20 * tmp10
    tmp23 = tmp22 * tmp13
    tmp24 = tmp21 + tmp23
    tmp26 = tmp25 * tmp13
    tmp27 = tmp21 + tmp26
    tl.store(out_ptr2 + (x0), tmp9, xmask)
    tl.store(out_ptr4 + (x0), tmp15, xmask)
    tl.store(out_ptr6 + (x0), tmp18, xmask)
    tl.store(out_ptr8 + (x0), tmp24, xmask)
    tl.store(out_ptr10 + (x0), tmp27, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3n/c3ngmwfbow2c7xez56t5zdeg4vdrafnsy432bvcda24ymt25dzrj.py
# Source Nodes: [x_119, x_121, x_122, x_124], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_119 => add_208, add_211, mul_273, mul_279, rsqrt_39, sub_39, var_mean_39
# x_121 => relu_39
# x_122 => add_216, mul_286
# x_124 => relu_40
triton_poi_fused__native_batch_norm_legit_functional_relu_52 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7225344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 1152
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp16 = tmp9 * tmp15
    tmp18 = tmp16 + tmp17
    tmp19 = triton_helpers.maximum(0, tmp18)
    tl.store(out_ptr0 + (x3), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mc/cmcmddzqw5nvzb475ibo3bbmqlsksvyiufqgngn75snkj7klkpnl.py
# Source Nodes: [x_125], Original ATen: [aten._native_batch_norm_legit_functional]
# x_125 => add_218, add_219, add_220, mul_288, mul_289, mul_290, mul_291, mul_292, rsqrt_41, squeeze_124, var_mean_41
triton_red_fused__native_batch_norm_legit_functional_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_53', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 800
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (627200*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/c2/cc25zlle344pxvcayjsb3p66yym5wbvulxo4a3kt4slpud7up7at.py
# Source Nodes: [x_125, x_127], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_125 => add_218, add_221, mul_287, mul_293, rsqrt_41, sub_41, var_mean_41
# x_127 => relu_41
triton_poi_fused__native_batch_norm_legit_functional_relu_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5017600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 800
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nw/cnwxnqywwtz5z6acugwoffwjj2o5dpxjz6sjvntvtdud5zc7tigr.py
# Source Nodes: [x_128], Original ATen: [aten._native_batch_norm_legit_functional]
# x_128 => add_223, add_224, add_225, mul_295, mul_296, mul_297, mul_298, mul_299, rsqrt_42, squeeze_127, var_mean_42
triton_red_fused__native_batch_norm_legit_functional_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_55', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 800
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (156800*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/sa/csatngfwavqwgjxahtdxryp4aluxvpp7dnvhdzrq3qvqjz2pgmof.py
# Source Nodes: [x_128, x_130], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_128 => add_223, add_226, mul_294, mul_300, rsqrt_42, sub_42, var_mean_42
# x_130 => relu_42
triton_poi_fused__native_batch_norm_legit_functional_relu_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_56', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1254400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 800
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6q/c6q2vzltidjnhck6i66fjxwduzzvke7pika2tdft7g5voucpxzhr.py
# Source Nodes: [cat_114], Original ATen: [aten.cat]
# cat_114 => cat_25
triton_poi_fused_cat_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1906688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1216
    x2 = (xindex // 238336)
    x3 = xindex % 238336
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (225792*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 1216, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = (-1024) + x1
    tmp14 = tmp13 >= tmp1
    tmp15 = tl.full([1], 128, tl.int64)
    tmp16 = tmp13 < tmp15
    tmp17 = tmp16 & tmp10
    tmp18 = tl.load(in_ptr0 + (x3 + (225792*x2)), tmp17, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp13 >= tmp15
    tmp22 = tl.full([1], 192, tl.int64)
    tmp23 = tmp13 < tmp22
    tmp24 = tmp21 & tmp10
    tmp25 = tl.load(in_ptr1 + ((-25088) + x3 + (213248*x2)), tmp24, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp24, tmp25, tmp26)
    tmp28 = tl.where(tmp16, tmp20, tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp10, tmp28, tmp29)
    tmp31 = tl.where(tmp4, tmp9, tmp30)
    tl.store(out_ptr0 + (x4), tmp31, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ei/ceillmmpenfke6xeuf4wrnrxgml7uwh5xsinxvrxrjgjrqxge5ko.py
# Source Nodes: [x_131], Original ATen: [aten._native_batch_norm_legit_functional]
# x_131 => add_229, add_230, add_231, mul_302, mul_303, mul_304, mul_305, mul_306, rsqrt_43, squeeze_130, var_mean_43
triton_red_fused__native_batch_norm_legit_functional_58 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_58', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1216
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (238336*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/i3/ci3i6knizmxd646xvaw2bmdh7nhxyzq4afgesmlu5m3ukjg2dd7j.py
# Source Nodes: [x_131, x_133], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_131 => add_229, add_232, mul_301, mul_307, rsqrt_43, sub_43, var_mean_43
# x_133 => relu_43
triton_poi_fused__native_batch_norm_legit_functional_relu_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1906688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1216
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2t/c2tyqp7ex2ic2b5icbvka75fdg7svrlxol7svide7n2dtmfs4znz.py
# Source Nodes: [cat_112], Original ATen: [aten.cat]
# cat_112 => cat_27
triton_poi_fused_cat_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2007040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1280
    x2 = (xindex // 250880)
    x3 = xindex % 250880
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (225792*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 1280, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = (-1024) + x1
    tmp16 = tmp15 >= tmp1
    tmp17 = tl.full([1], 192, tl.int64)
    tmp18 = tmp15 < tmp17
    tmp19 = tmp18 & tmp12
    tmp20 = tl.full([1], 128, tl.int64)
    tmp21 = tmp15 < tmp20
    tmp22 = tmp21 & tmp19
    tmp23 = tl.load(in_ptr0 + (x3 + (225792*x2)), tmp22, other=0.0)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp22, tmp23, tmp24)
    tmp26 = tmp15 >= tmp20
    tmp27 = tmp26 & tmp19
    tmp28 = tl.load(in_ptr1 + ((-25088) + x3 + (213248*x2)), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tl.where(tmp21, tmp25, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp19, tmp31, tmp32)
    tmp34 = tmp15 >= tmp17
    tmp35 = tl.full([1], 256, tl.int64)
    tmp36 = tmp15 < tmp35
    tmp37 = tmp34 & tmp12
    tmp38 = tl.load(in_ptr2 + ((-37632) + x3 + (213248*x2)), tmp37, other=0.0)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = tl.where(tmp18, tmp33, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp12, tmp41, tmp42)
    tmp44 = tl.where(tmp4, tmp11, tmp43)
    tl.store(out_ptr0 + (x4), tmp44, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/72/c72r5gkhgre6kkzs4hfcl3madnxl4ezqc32pf54s4dspid5ff4yh.py
# Source Nodes: [x_140], Original ATen: [aten._native_batch_norm_legit_functional]
# x_140 => add_245, add_246, add_247, mul_323, mul_324, mul_325, mul_326, mul_327, rsqrt_46, squeeze_139, var_mean_46
triton_red_fused__native_batch_norm_legit_functional_61 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_61', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1280
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (250880*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/5b/c5bsykyb57f7fbz4fxtybgff2f3to2c5qo4dzdvshmubiclvmovl.py
# Source Nodes: [x_140, x_142], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_140 => add_245, add_248, mul_322, mul_328, rsqrt_46, sub_46, var_mean_46
# x_142 => relu_46
triton_poi_fused__native_batch_norm_legit_functional_relu_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_62', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2007040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1280
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/id/cidnvu75bg7lsnzsrtnvl4n2z3kfitp5k47j2kmaiglofhktslr6.py
# Source Nodes: [cat_110], Original ATen: [aten.cat]
# cat_110 => cat_29
triton_poi_fused_cat_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_63', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2107392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1344
    x2 = (xindex // 263424)
    x3 = xindex % 263424
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (225792*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 1344, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = (-1024) + x1
    tmp18 = tmp17 >= tmp1
    tmp19 = tl.full([1], 256, tl.int64)
    tmp20 = tmp17 < tmp19
    tmp21 = tmp20 & tmp14
    tmp22 = tl.full([1], 192, tl.int64)
    tmp23 = tmp17 < tmp22
    tmp24 = tmp23 & tmp21
    tmp25 = tl.full([1], 128, tl.int64)
    tmp26 = tmp17 < tmp25
    tmp27 = tmp26 & tmp24
    tmp28 = tl.load(in_ptr0 + (x3 + (225792*x2)), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp17 >= tmp25
    tmp32 = tmp31 & tmp24
    tmp33 = tl.load(in_ptr1 + ((-25088) + x3 + (213248*x2)), tmp32, other=0.0)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp32, tmp33, tmp34)
    tmp36 = tl.where(tmp26, tmp30, tmp35)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp24, tmp36, tmp37)
    tmp39 = tmp17 >= tmp22
    tmp40 = tmp39 & tmp21
    tmp41 = tl.load(in_ptr2 + ((-37632) + x3 + (213248*x2)), tmp40, other=0.0)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp40, tmp41, tmp42)
    tmp44 = tl.where(tmp23, tmp38, tmp43)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp21, tmp44, tmp45)
    tmp47 = tmp17 >= tmp19
    tmp48 = tl.full([1], 320, tl.int64)
    tmp49 = tmp17 < tmp48
    tmp50 = tmp47 & tmp14
    tmp51 = tl.load(in_ptr3 + ((-50176) + x3 + (213248*x2)), tmp50, other=0.0)
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp50, tmp51, tmp52)
    tmp54 = tl.where(tmp20, tmp46, tmp53)
    tmp55 = tl.full(tmp54.shape, 0.0, tmp54.dtype)
    tmp56 = tl.where(tmp14, tmp54, tmp55)
    tmp57 = tl.where(tmp4, tmp13, tmp56)
    tl.store(out_ptr0 + (x4), tmp57, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/e7/ce7atkvdqrytgyjfxqyj2u5t4xmyr7c7cizlvs2fpkbpzk6vnxtv.py
# Source Nodes: [x_149], Original ATen: [aten._native_batch_norm_legit_functional]
# x_149 => add_261, add_262, add_263, mul_344, mul_345, mul_346, mul_347, mul_348, rsqrt_49, squeeze_148, var_mean_49
triton_red_fused__native_batch_norm_legit_functional_64 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_64', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1344
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (263424*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/ut/cutga7igoxckda5n5cd45vbqucti7sc7jsrxa5qnzeze2bp6ibjp.py
# Source Nodes: [x_149, x_151], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_149 => add_261, add_264, mul_343, mul_349, rsqrt_49, sub_49, var_mean_49
# x_151 => relu_49
triton_poi_fused__native_batch_norm_legit_functional_relu_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2107392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1344
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4i/c4ihfkcgmlnkemvk5jlkcc6njt2tfm67gaaasae6mrv4liesjfyk.py
# Source Nodes: [x_s1_13, x_s1_14, x_s1_15, x_s1_16], Original ATen: [aten.add]
# x_s1_13 => add_227
# x_s1_14 => add_243
# x_s1_15 => add_259
# x_s1_16 => add_275
triton_poi_fused_add_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_66', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 200704
    x1 = (xindex // 200704)
    tmp0 = tl.load(in_ptr0 + (x0 + (225792*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (213248*x1)), None)
    tmp3 = tl.load(in_ptr2 + (x0 + (213248*x1)), None)
    tmp5 = tl.load(in_ptr3 + (x0 + (213248*x1)), None)
    tmp7 = tl.load(in_ptr4 + (x0 + (213248*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x0 + (275968*x1)), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/by/cby2ivmvryftvrm44t7vcjgcqfl6rh34n5uwiywxdvffk4bb4b7c.py
# Source Nodes: [cat_109], Original ATen: [aten.cat]
# cat_109 => cat_30
triton_poi_fused_cat_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_67', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 384
    x2 = (xindex // 75264)
    x3 = xindex % 75264
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 320, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 256, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.full([1], 192, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tmp9 & tmp7
    tmp11 = tl.full([1], 128, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp12 & tmp10
    tmp14 = tl.load(in_ptr0 + (200704 + x3 + (225792*x2)), tmp13, other=0.0)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = tmp0 >= tmp11
    tmp18 = tmp17 & tmp10
    tmp19 = tl.load(in_ptr1 + (175616 + x3 + (213248*x2)), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tl.where(tmp12, tmp16, tmp21)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp10, tmp22, tmp23)
    tmp25 = tmp0 >= tmp8
    tmp26 = tmp25 & tmp7
    tmp27 = tl.load(in_ptr2 + (163072 + x3 + (213248*x2)), tmp26, other=0.0)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp26, tmp27, tmp28)
    tmp30 = tl.where(tmp9, tmp24, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp7, tmp30, tmp31)
    tmp33 = tmp0 >= tmp5
    tmp34 = tmp33 & tmp4
    tmp35 = tl.load(in_ptr3 + (150528 + x3 + (213248*x2)), tmp34, other=0.0)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp34, tmp35, tmp36)
    tmp38 = tl.where(tmp6, tmp32, tmp37)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp4, tmp38, tmp39)
    tmp41 = tmp0 >= tmp3
    tmp42 = tl.full([1], 384, tl.int64)
    tmp43 = tmp0 < tmp42
    tmp44 = tl.load(in_ptr4 + (137984 + x3 + (213248*x2)), tmp41, other=0.0)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp41, tmp44, tmp45)
    tmp47 = tl.where(tmp4, tmp40, tmp46)
    tl.store(out_ptr0 + (x3 + (275968*x2)), tmp47, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pr/cprjeumkepedykp5epfyigodbndjen3336lfldt7bvtrhrbrwo4k.py
# Source Nodes: [x_158], Original ATen: [aten._native_batch_norm_legit_functional]
# x_158 => add_277, add_278, add_279, mul_365, mul_366, mul_367, mul_368, mul_369, rsqrt_52, squeeze_157, var_mean_52
triton_red_fused__native_batch_norm_legit_functional_68 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_68', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1408
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (275968*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/sy/csyqlda7t7iojcubrtcnlrwb3lq6uf5ieb6bcg52jpgzdpjbiuam.py
# Source Nodes: [x_158, x_160], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_158 => add_277, add_280, mul_364, mul_370, rsqrt_52, sub_52, var_mean_52
# x_160 => relu_52
triton_poi_fused__native_batch_norm_legit_functional_relu_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_69', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2207744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1408
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6n/c6nedoi5hcmjwymc2i7i36cb6fjfpsaeizor7vp3j2ioilkajbic.py
# Source Nodes: [cat_106], Original ATen: [aten.cat]
# cat_106 => cat_33
triton_poi_fused_cat_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2308096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1472
    x2 = (xindex // 288512)
    x3 = xindex % 288512
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (275968*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 1472, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = (-1024) + x1
    tmp14 = tmp13 >= tmp1
    tmp15 = tl.full([1], 384, tl.int64)
    tmp16 = tmp13 < tmp15
    tmp17 = tmp16 & tmp10
    tmp18 = tl.load(in_ptr2 + ((-200704) + x3 + (275968*x2)), tmp17, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp13 >= tmp15
    tmp22 = tl.full([1], 448, tl.int64)
    tmp23 = tmp13 < tmp22
    tmp24 = tmp21 & tmp10
    tmp25 = tl.load(in_ptr1 + ((-75264) + x3 + (213248*x2)), tmp24, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp24, tmp25, tmp26)
    tmp28 = tl.where(tmp16, tmp20, tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp10, tmp28, tmp29)
    tmp31 = tl.where(tmp4, tmp9, tmp30)
    tl.store(out_ptr0 + (x4), tmp31, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/og/cogyp7fpbnrgwjdkmxzwcoys5xcgjle6ldt2nddf5teth3kne6rm.py
# Source Nodes: [x_167], Original ATen: [aten._native_batch_norm_legit_functional]
# x_167 => add_293, add_294, add_295, mul_386, mul_387, mul_388, mul_389, mul_390, rsqrt_55, squeeze_166, var_mean_55
triton_red_fused__native_batch_norm_legit_functional_71 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_71', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1472
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (288512*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/hm/chm32cxvmkgir45cz4r32qm355ho5j72uuwqxnrqkxveisaxbeav.py
# Source Nodes: [x_167, x_169], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_167 => add_293, add_296, mul_385, mul_391, rsqrt_55, sub_55, var_mean_55
# x_169 => relu_55
triton_poi_fused__native_batch_norm_legit_functional_relu_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_72', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2308096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1472
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/up/cupuwgjekt4xbtvwmunqlpdug63kqernhufunjl575b7ksry6nei.py
# Source Nodes: [cat_104], Original ATen: [aten.cat]
# cat_104 => cat_35
triton_poi_fused_cat_73 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1536
    x2 = (xindex // 301056)
    x3 = xindex % 301056
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (275968*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 1536, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = (-1024) + x1
    tmp16 = tmp15 >= tmp1
    tmp17 = tl.full([1], 448, tl.int64)
    tmp18 = tmp15 < tmp17
    tmp19 = tmp18 & tmp12
    tmp20 = tl.full([1], 384, tl.int64)
    tmp21 = tmp15 < tmp20
    tmp22 = tmp21 & tmp19
    tmp23 = tl.load(in_ptr3 + ((-200704) + x3 + (275968*x2)), tmp22, other=0.0)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp22, tmp23, tmp24)
    tmp26 = tmp15 >= tmp20
    tmp27 = tmp26 & tmp19
    tmp28 = tl.load(in_ptr1 + ((-75264) + x3 + (213248*x2)), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tl.where(tmp21, tmp25, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp19, tmp31, tmp32)
    tmp34 = tmp15 >= tmp17
    tmp35 = tl.full([1], 512, tl.int64)
    tmp36 = tmp15 < tmp35
    tmp37 = tmp34 & tmp12
    tmp38 = tl.load(in_ptr2 + ((-87808) + x3 + (213248*x2)), tmp37, other=0.0)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = tl.where(tmp18, tmp33, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp12, tmp41, tmp42)
    tmp44 = tl.where(tmp4, tmp11, tmp43)
    tl.store(out_ptr0 + (x4), tmp44, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/57/c57b5uj2afjvfks63mxvrdbd5e6xfxn2zfrn5rioe5tsycsrfd2y.py
# Source Nodes: [x_176], Original ATen: [aten._native_batch_norm_legit_functional]
# x_176 => add_309, add_310, add_311, mul_407, mul_408, mul_409, mul_410, mul_411, rsqrt_58, squeeze_175, var_mean_58
triton_red_fused__native_batch_norm_legit_functional_74 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_74', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1536
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (301056*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/nx/cnxofo5gza5vt3fdah2ycavipuqaxktjqz6ilhxftnqovi7p4mv2.py
# Source Nodes: [x_176, x_178], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_176 => add_309, add_312, mul_406, mul_412, rsqrt_58, sub_58, var_mean_58
# x_178 => relu_58
triton_poi_fused__native_batch_norm_legit_functional_relu_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_75', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1536
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5i/c5im2nog43xgpzzndci3nj7gzykvtf6ace5z5ytd7wezvtrexkhn.py
# Source Nodes: [cat_102], Original ATen: [aten.cat]
# cat_102 => cat_37
triton_poi_fused_cat_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_76', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2508800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1600
    x2 = (xindex // 313600)
    x3 = xindex % 313600
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (275968*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 1600, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = (-1024) + x1
    tmp18 = tmp17 >= tmp1
    tmp19 = tl.full([1], 512, tl.int64)
    tmp20 = tmp17 < tmp19
    tmp21 = tmp20 & tmp14
    tmp22 = tl.full([1], 448, tl.int64)
    tmp23 = tmp17 < tmp22
    tmp24 = tmp23 & tmp21
    tmp25 = tl.full([1], 384, tl.int64)
    tmp26 = tmp17 < tmp25
    tmp27 = tmp26 & tmp24
    tmp28 = tl.load(in_ptr4 + ((-200704) + x3 + (275968*x2)), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp17 >= tmp25
    tmp32 = tmp31 & tmp24
    tmp33 = tl.load(in_ptr1 + ((-75264) + x3 + (213248*x2)), tmp32, other=0.0)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp32, tmp33, tmp34)
    tmp36 = tl.where(tmp26, tmp30, tmp35)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp24, tmp36, tmp37)
    tmp39 = tmp17 >= tmp22
    tmp40 = tmp39 & tmp21
    tmp41 = tl.load(in_ptr2 + ((-87808) + x3 + (213248*x2)), tmp40, other=0.0)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp40, tmp41, tmp42)
    tmp44 = tl.where(tmp23, tmp38, tmp43)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp21, tmp44, tmp45)
    tmp47 = tmp17 >= tmp19
    tmp48 = tl.full([1], 576, tl.int64)
    tmp49 = tmp17 < tmp48
    tmp50 = tmp47 & tmp14
    tmp51 = tl.load(in_ptr3 + ((-100352) + x3 + (213248*x2)), tmp50, other=0.0)
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp50, tmp51, tmp52)
    tmp54 = tl.where(tmp20, tmp46, tmp53)
    tmp55 = tl.full(tmp54.shape, 0.0, tmp54.dtype)
    tmp56 = tl.where(tmp14, tmp54, tmp55)
    tmp57 = tl.where(tmp4, tmp13, tmp56)
    tl.store(out_ptr0 + (x4), tmp57, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3f/c3fn7zo2yu7lsqje5nls5i3ur6edn4eu53vhbjw5bim762vq4vma.py
# Source Nodes: [x_185], Original ATen: [aten._native_batch_norm_legit_functional]
# x_185 => add_325, add_326, add_327, mul_428, mul_429, mul_430, mul_431, mul_432, rsqrt_61, squeeze_184, var_mean_61
triton_red_fused__native_batch_norm_legit_functional_77 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_77', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1600
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (313600*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/yx/cyxa6zmijwlbhw3m5rerxsaljczx7fkbcdt2ktn5t4xsim64m3qm.py
# Source Nodes: [x_185, x_187], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_185 => add_325, add_328, mul_427, mul_433, rsqrt_61, sub_61, var_mean_61
# x_187 => relu_61
triton_poi_fused__native_batch_norm_legit_functional_relu_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_78', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2508800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1600
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vy/cvygxrfvy43gztnlrujhgjzwuc25bb37jarbzgd5ah2kyjk46vpv.py
# Source Nodes: [x_s1_17, x_s1_18, x_s1_19, x_s1_20], Original ATen: [aten.add]
# x_s1_17 => add_291
# x_s1_18 => add_307
# x_s1_19 => add_323
# x_s1_20 => add_339
triton_poi_fused_add_79 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 200704
    x1 = (xindex // 200704)
    tmp0 = tl.load(in_ptr0 + (x0 + (275968*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (213248*x1)), None)
    tmp3 = tl.load(in_ptr2 + (x0 + (213248*x1)), None)
    tmp5 = tl.load(in_ptr3 + (x0 + (213248*x1)), None)
    tmp7 = tl.load(in_ptr4 + (x0 + (213248*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x0 + (326144*x1)), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/47/c47pbyl2u7ocoqgtgbhz6l4etq2hswslog76n3l4al4xxrbormrp.py
# Source Nodes: [cat_101], Original ATen: [aten.cat]
# cat_101 => cat_38
triton_poi_fused_cat_80 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_80', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 640
    x2 = (xindex // 125440)
    x3 = xindex % 125440
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 576, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 512, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.full([1], 448, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tmp9 & tmp7
    tmp11 = tl.full([1], 384, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp12 & tmp10
    tmp14 = tl.load(in_ptr0 + (x3 + (275968*x2)), tmp13, other=0.0)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = tmp0 >= tmp11
    tmp18 = tmp17 & tmp10
    tmp19 = tl.load(in_ptr1 + (125440 + x3 + (213248*x2)), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tl.where(tmp12, tmp16, tmp21)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp10, tmp22, tmp23)
    tmp25 = tmp0 >= tmp8
    tmp26 = tmp25 & tmp7
    tmp27 = tl.load(in_ptr2 + (112896 + x3 + (213248*x2)), tmp26, other=0.0)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp26, tmp27, tmp28)
    tmp30 = tl.where(tmp9, tmp24, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp7, tmp30, tmp31)
    tmp33 = tmp0 >= tmp5
    tmp34 = tmp33 & tmp4
    tmp35 = tl.load(in_ptr3 + (100352 + x3 + (213248*x2)), tmp34, other=0.0)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp34, tmp35, tmp36)
    tmp38 = tl.where(tmp6, tmp32, tmp37)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp4, tmp38, tmp39)
    tmp41 = tmp0 >= tmp3
    tmp42 = tl.full([1], 640, tl.int64)
    tmp43 = tmp0 < tmp42
    tmp44 = tl.load(in_ptr4 + (87808 + x3 + (213248*x2)), tmp41, other=0.0)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp41, tmp44, tmp45)
    tmp47 = tl.where(tmp4, tmp40, tmp46)
    tl.store(out_ptr0 + (x3 + (326144*x2)), tmp47, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/py/cpywut5bcbpobx56ojxzixd27twnhnmr3ed4xwvogomlowbogrkt.py
# Source Nodes: [x_194], Original ATen: [aten._native_batch_norm_legit_functional]
# x_194 => add_341, add_342, add_343, mul_449, mul_450, mul_451, mul_452, mul_453, rsqrt_64, squeeze_193, var_mean_64
triton_red_fused__native_batch_norm_legit_functional_81 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_81', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1664
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (326144*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/vf/cvf2hbt46xsbsv6cgfhqxdk462vvdjlpoukaeqtxdfoweogkxhxt.py
# Source Nodes: [x_194, x_196], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_194 => add_341, add_344, mul_448, mul_454, rsqrt_64, sub_64, var_mean_64
# x_196 => relu_64
triton_poi_fused__native_batch_norm_legit_functional_relu_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_82', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2609152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1664
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ur/curlsl4ku4yn47wf7ybggyu5uu3hpuwik7o2hb7o7omybsex4gvf.py
# Source Nodes: [cat_98], Original ATen: [aten.cat]
# cat_98 => cat_41
triton_poi_fused_cat_83 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_83', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2709504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1728
    x2 = (xindex // 338688)
    x3 = xindex % 338688
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (326144*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 1728, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = (-1024) + x1
    tmp14 = tmp13 >= tmp1
    tmp15 = tl.full([1], 640, tl.int64)
    tmp16 = tmp13 < tmp15
    tmp17 = tmp16 & tmp10
    tmp18 = tl.load(in_ptr2 + ((-200704) + x3 + (326144*x2)), tmp17, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp13 >= tmp15
    tmp22 = tl.full([1], 704, tl.int64)
    tmp23 = tmp13 < tmp22
    tmp24 = tmp21 & tmp10
    tmp25 = tl.load(in_ptr1 + ((-125440) + x3 + (213248*x2)), tmp24, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp24, tmp25, tmp26)
    tmp28 = tl.where(tmp16, tmp20, tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp10, tmp28, tmp29)
    tmp31 = tl.where(tmp4, tmp9, tmp30)
    tl.store(out_ptr0 + (x4), tmp31, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/23/c23j5cwr3annruoumptlq5qs2y5irqb2st6rs3u43ptrjqdvkbcq.py
# Source Nodes: [x_203], Original ATen: [aten._native_batch_norm_legit_functional]
# x_203 => add_357, add_358, add_359, mul_470, mul_471, mul_472, mul_473, mul_474, rsqrt_67, squeeze_202, var_mean_67
triton_red_fused__native_batch_norm_legit_functional_84 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_84', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1728
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (338688*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/tj/ctj224hw735o3lmcqvfpvmiuttabpudxvqec7cy6ad5zf7zvbq2y.py
# Source Nodes: [x_203, x_205], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_203 => add_357, add_360, mul_469, mul_475, rsqrt_67, sub_67, var_mean_67
# x_205 => relu_67
triton_poi_fused__native_batch_norm_legit_functional_relu_85 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_85', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2709504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1728
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/m7/cm7vbycflsrak64rr5pjjm2vbinuwzf4ws5zffitgnigczgb45hf.py
# Source Nodes: [cat_96], Original ATen: [aten.cat]
# cat_96 => cat_43
triton_poi_fused_cat_86 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_86', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2809856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1792
    x2 = (xindex // 351232)
    x3 = xindex % 351232
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (326144*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 1792, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = (-1024) + x1
    tmp16 = tmp15 >= tmp1
    tmp17 = tl.full([1], 704, tl.int64)
    tmp18 = tmp15 < tmp17
    tmp19 = tmp18 & tmp12
    tmp20 = tl.full([1], 640, tl.int64)
    tmp21 = tmp15 < tmp20
    tmp22 = tmp21 & tmp19
    tmp23 = tl.load(in_ptr3 + ((-200704) + x3 + (326144*x2)), tmp22, other=0.0)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp22, tmp23, tmp24)
    tmp26 = tmp15 >= tmp20
    tmp27 = tmp26 & tmp19
    tmp28 = tl.load(in_ptr1 + ((-125440) + x3 + (213248*x2)), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tl.where(tmp21, tmp25, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp19, tmp31, tmp32)
    tmp34 = tmp15 >= tmp17
    tmp35 = tl.full([1], 768, tl.int64)
    tmp36 = tmp15 < tmp35
    tmp37 = tmp34 & tmp12
    tmp38 = tl.load(in_ptr2 + ((-137984) + x3 + (213248*x2)), tmp37, other=0.0)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = tl.where(tmp18, tmp33, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp12, tmp41, tmp42)
    tmp44 = tl.where(tmp4, tmp11, tmp43)
    tl.store(out_ptr0 + (x4), tmp44, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fk/cfkykrqncffrbapuekdojamwslmuekd22lh7d4wvdkit2gbrw4am.py
# Source Nodes: [x_212], Original ATen: [aten._native_batch_norm_legit_functional]
# x_212 => add_373, add_374, add_375, mul_491, mul_492, mul_493, mul_494, mul_495, rsqrt_70, squeeze_211, var_mean_70
triton_red_fused__native_batch_norm_legit_functional_87 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_87', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1792
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (351232*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/cl/ccllepi75eoeoziqjxhiuwrbfz6omopbwzbeanvafub5vligqpsc.py
# Source Nodes: [x_212, x_214], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_212 => add_373, add_376, mul_490, mul_496, rsqrt_70, sub_70, var_mean_70
# x_214 => relu_70
triton_poi_fused__native_batch_norm_legit_functional_relu_88 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_88', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2809856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1792
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7u/c7ucmmdmyqhudff2uuo6ipfbpjjgekjxfjjdllj3g5n7j6l6y4vq.py
# Source Nodes: [cat_94], Original ATen: [aten.cat]
# cat_94 => cat_45
triton_poi_fused_cat_89 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_89', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2910208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1856
    x2 = (xindex // 363776)
    x3 = xindex % 363776
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (326144*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 1856, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = (-1024) + x1
    tmp18 = tmp17 >= tmp1
    tmp19 = tl.full([1], 768, tl.int64)
    tmp20 = tmp17 < tmp19
    tmp21 = tmp20 & tmp14
    tmp22 = tl.full([1], 704, tl.int64)
    tmp23 = tmp17 < tmp22
    tmp24 = tmp23 & tmp21
    tmp25 = tl.full([1], 640, tl.int64)
    tmp26 = tmp17 < tmp25
    tmp27 = tmp26 & tmp24
    tmp28 = tl.load(in_ptr4 + ((-200704) + x3 + (326144*x2)), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp17 >= tmp25
    tmp32 = tmp31 & tmp24
    tmp33 = tl.load(in_ptr1 + ((-125440) + x3 + (213248*x2)), tmp32, other=0.0)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp32, tmp33, tmp34)
    tmp36 = tl.where(tmp26, tmp30, tmp35)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp24, tmp36, tmp37)
    tmp39 = tmp17 >= tmp22
    tmp40 = tmp39 & tmp21
    tmp41 = tl.load(in_ptr2 + ((-137984) + x3 + (213248*x2)), tmp40, other=0.0)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp40, tmp41, tmp42)
    tmp44 = tl.where(tmp23, tmp38, tmp43)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp21, tmp44, tmp45)
    tmp47 = tmp17 >= tmp19
    tmp48 = tl.full([1], 832, tl.int64)
    tmp49 = tmp17 < tmp48
    tmp50 = tmp47 & tmp14
    tmp51 = tl.load(in_ptr3 + ((-150528) + x3 + (213248*x2)), tmp50, other=0.0)
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp50, tmp51, tmp52)
    tmp54 = tl.where(tmp20, tmp46, tmp53)
    tmp55 = tl.full(tmp54.shape, 0.0, tmp54.dtype)
    tmp56 = tl.where(tmp14, tmp54, tmp55)
    tmp57 = tl.where(tmp4, tmp13, tmp56)
    tl.store(out_ptr0 + (x4), tmp57, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jt/cjt7dabped6zmeq6wdt7jkqeh5ds5zx5hxtfzblaewwop5msy3ol.py
# Source Nodes: [x_221], Original ATen: [aten._native_batch_norm_legit_functional]
# x_221 => add_389, add_390, add_391, mul_512, mul_513, mul_514, mul_515, mul_516, rsqrt_73, squeeze_220, var_mean_73
triton_red_fused__native_batch_norm_legit_functional_90 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_90', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1856
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (363776*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/qc/cqcvvf53dbotfku2mammpqtj4wixxggejckzkb3mr6fx66edizw3.py
# Source Nodes: [x_221, x_223], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_221 => add_389, add_392, mul_511, mul_517, rsqrt_73, sub_73, var_mean_73
# x_223 => relu_73
triton_poi_fused__native_batch_norm_legit_functional_relu_91 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_91', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2910208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1856
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vr/cvrzxoyo6kcs3vhlpqcnhyfrd62cbe2nftgc5techde4m5k6po3x.py
# Source Nodes: [x_s1_21, x_s1_22, x_s1_23, x_s1_24], Original ATen: [aten.add]
# x_s1_21 => add_355
# x_s1_22 => add_371
# x_s1_23 => add_387
# x_s1_24 => add_403
triton_poi_fused_add_92 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_92', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 200704
    x1 = (xindex // 200704)
    tmp0 = tl.load(in_ptr0 + (x0 + (326144*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (213248*x1)), None)
    tmp3 = tl.load(in_ptr2 + (x0 + (213248*x1)), None)
    tmp5 = tl.load(in_ptr3 + (x0 + (213248*x1)), None)
    tmp7 = tl.load(in_ptr4 + (x0 + (213248*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x0 + (376320*x1)), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zy/czyqkekj7rbvypvtdififie4fyyjkvekk6vuc444ldyjtb2bvaz6.py
# Source Nodes: [cat_93], Original ATen: [aten.cat]
# cat_93 => cat_46
triton_poi_fused_cat_93 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_93', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 896
    x2 = (xindex // 175616)
    x3 = xindex % 175616
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 832, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 768, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.full([1], 704, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tmp9 & tmp7
    tmp11 = tl.full([1], 640, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp12 & tmp10
    tmp14 = tl.load(in_ptr0 + (x3 + (326144*x2)), tmp13, other=0.0)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = tmp0 >= tmp11
    tmp18 = tmp17 & tmp10
    tmp19 = tl.load(in_ptr1 + (75264 + x3 + (213248*x2)), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tl.where(tmp12, tmp16, tmp21)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp10, tmp22, tmp23)
    tmp25 = tmp0 >= tmp8
    tmp26 = tmp25 & tmp7
    tmp27 = tl.load(in_ptr2 + (62720 + x3 + (213248*x2)), tmp26, other=0.0)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp26, tmp27, tmp28)
    tmp30 = tl.where(tmp9, tmp24, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp7, tmp30, tmp31)
    tmp33 = tmp0 >= tmp5
    tmp34 = tmp33 & tmp4
    tmp35 = tl.load(in_ptr3 + (50176 + x3 + (213248*x2)), tmp34, other=0.0)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp34, tmp35, tmp36)
    tmp38 = tl.where(tmp6, tmp32, tmp37)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp4, tmp38, tmp39)
    tmp41 = tmp0 >= tmp3
    tmp42 = tl.full([1], 896, tl.int64)
    tmp43 = tmp0 < tmp42
    tmp44 = tl.load(in_ptr4 + (37632 + x3 + (213248*x2)), tmp41, other=0.0)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp41, tmp44, tmp45)
    tmp47 = tl.where(tmp4, tmp40, tmp46)
    tl.store(out_ptr0 + (x3 + (376320*x2)), tmp47, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/u4/cu47rjqeqtpa2l36bwrrhajeis73n4kud3xamgixrrldo74sja57.py
# Source Nodes: [x_230], Original ATen: [aten._native_batch_norm_legit_functional]
# x_230 => add_405, add_406, add_407, mul_533, mul_534, mul_535, mul_536, mul_537, rsqrt_76, squeeze_229, var_mean_76
triton_red_fused__native_batch_norm_legit_functional_94 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_94', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1920
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (376320*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/y3/cy3f6yubyhbm5lq53nahe5gmrh2ifof476zcs4v65nkgiei7ry6m.py
# Source Nodes: [x_230, x_232], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_230 => add_405, add_408, mul_532, mul_538, rsqrt_76, sub_76, var_mean_76
# x_232 => relu_76
triton_poi_fused__native_batch_norm_legit_functional_relu_95 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_95', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3010560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1920
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/w4/cw4m7ps7fprhyieers6j2pmi7vqkcgcot5jqwjn3mersufkr2it2.py
# Source Nodes: [cat_90], Original ATen: [aten.cat]
# cat_90 => cat_49
triton_poi_fused_cat_96 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_96', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3110912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1984
    x2 = (xindex // 388864)
    x3 = xindex % 388864
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (376320*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 1984, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = (-1024) + x1
    tmp14 = tmp13 >= tmp1
    tmp15 = tl.full([1], 896, tl.int64)
    tmp16 = tmp13 < tmp15
    tmp17 = tmp16 & tmp10
    tmp18 = tl.load(in_ptr2 + ((-200704) + x3 + (376320*x2)), tmp17, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp13 >= tmp15
    tmp22 = tl.full([1], 960, tl.int64)
    tmp23 = tmp13 < tmp22
    tmp24 = tmp21 & tmp10
    tmp25 = tl.load(in_ptr1 + ((-175616) + x3 + (213248*x2)), tmp24, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp24, tmp25, tmp26)
    tmp28 = tl.where(tmp16, tmp20, tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp10, tmp28, tmp29)
    tmp31 = tl.where(tmp4, tmp9, tmp30)
    tl.store(out_ptr0 + (x4), tmp31, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hd/chde76q2d5sziuzvjrzu44l24pv76bkwthpvnogfyptgzzxuzspn.py
# Source Nodes: [x_239], Original ATen: [aten._native_batch_norm_legit_functional]
# x_239 => add_421, add_422, add_423, mul_554, mul_555, mul_556, mul_557, mul_558, rsqrt_79, squeeze_238, var_mean_79
triton_red_fused__native_batch_norm_legit_functional_97 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_97', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1984
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (388864*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/do/cdorh5orj75tvy4ix2bbjyt7fyqluux52fjiurdxvg56g42j3cjf.py
# Source Nodes: [x_239, x_241], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_239 => add_421, add_424, mul_553, mul_559, rsqrt_79, sub_79, var_mean_79
# x_241 => relu_79
triton_poi_fused__native_batch_norm_legit_functional_relu_98 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_98', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3110912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1984
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gm/cgmrbyk4a4g3r3tpci4yp3xrr62tfx4h5owkosqku624nafavpcg.py
# Source Nodes: [cat_88], Original ATen: [aten.cat]
# cat_88 => cat_51
triton_poi_fused_cat_99 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_99', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 2048
    x2 = (xindex // 401408)
    x3 = xindex % 401408
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (376320*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 2048, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = (-1024) + x1
    tmp16 = tmp15 >= tmp1
    tmp17 = tl.full([1], 960, tl.int64)
    tmp18 = tmp15 < tmp17
    tmp19 = tmp18 & tmp12
    tmp20 = tl.full([1], 896, tl.int64)
    tmp21 = tmp15 < tmp20
    tmp22 = tmp21 & tmp19
    tmp23 = tl.load(in_ptr3 + ((-200704) + x3 + (376320*x2)), tmp22, other=0.0)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp22, tmp23, tmp24)
    tmp26 = tmp15 >= tmp20
    tmp27 = tmp26 & tmp19
    tmp28 = tl.load(in_ptr1 + ((-175616) + x3 + (213248*x2)), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tl.where(tmp21, tmp25, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp19, tmp31, tmp32)
    tmp34 = tmp15 >= tmp17
    tmp35 = tmp15 < tmp3
    tmp36 = tmp34 & tmp12
    tmp37 = tl.load(in_ptr2 + ((-188160) + x3 + (213248*x2)), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tl.where(tmp18, tmp33, tmp39)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp12, tmp40, tmp41)
    tmp43 = tl.where(tmp4, tmp11, tmp42)
    tl.store(out_ptr0 + (x4), tmp43, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yx/cyxja5fwdjaebrljdgypydu666tvqzyddrl3ndy3x3bdnxhrzrzb.py
# Source Nodes: [x_248], Original ATen: [aten._native_batch_norm_legit_functional]
# x_248 => add_437, add_438, add_439, mul_575, mul_576, mul_577, mul_578, mul_579, rsqrt_82, squeeze_247, var_mean_82
triton_red_fused__native_batch_norm_legit_functional_100 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_100', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (401408*r2)), rmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr0 + (x0), tmp2, None)
    tl.store(out_ptr1 + (x0), tmp3, None)
    tmp12 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp5 = 1568.0
    tmp6 = tmp3 / tmp5
    tmp7 = 0.001
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
    tl.store(out_ptr2 + (x0), tmp9, None)
    tl.store(out_ptr4 + (x0), tmp15, None)
    tl.store(out_ptr6 + (x0), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hq/chq73uodps4reank2x62qgj3gfd7yoh5jb77me7jnz7ocltv5ear.py
# Source Nodes: [x_248, x_250], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_248 => add_437, add_440, mul_574, mul_580, rsqrt_82, sub_82, var_mean_82
# x_250 => relu_82
triton_poi_fused__native_batch_norm_legit_functional_relu_101 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_101', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 2048
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lw/clw4sebwh4xjqqju6ara4lhvofnjnnbzbx7xwpje2sgzsibrd4h5.py
# Source Nodes: [cat_86], Original ATen: [aten.cat]
# cat_86 => cat_53
triton_poi_fused_cat_102 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_102', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3311616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 2112
    x2 = (xindex // 413952)
    x3 = xindex % 413952
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (376320*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 2112, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = (-1024) + x1
    tmp18 = tmp17 >= tmp1
    tmp19 = tmp17 < tmp3
    tmp20 = tmp19 & tmp14
    tmp21 = tl.full([1], 960, tl.int64)
    tmp22 = tmp17 < tmp21
    tmp23 = tmp22 & tmp20
    tmp24 = tl.full([1], 896, tl.int64)
    tmp25 = tmp17 < tmp24
    tmp26 = tmp25 & tmp23
    tmp27 = tl.load(in_ptr4 + ((-200704) + x3 + (376320*x2)), tmp26, other=0.0)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp26, tmp27, tmp28)
    tmp30 = tmp17 >= tmp24
    tmp31 = tmp30 & tmp23
    tmp32 = tl.load(in_ptr1 + ((-175616) + x3 + (213248*x2)), tmp31, other=0.0)
    tmp33 = tl.full(tmp32.shape, 0.0, tmp32.dtype)
    tmp34 = tl.where(tmp31, tmp32, tmp33)
    tmp35 = tl.where(tmp25, tmp29, tmp34)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp23, tmp35, tmp36)
    tmp38 = tmp17 >= tmp21
    tmp39 = tmp38 & tmp20
    tmp40 = tl.load(in_ptr2 + ((-188160) + x3 + (213248*x2)), tmp39, other=0.0)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp39, tmp40, tmp41)
    tmp43 = tl.where(tmp22, tmp37, tmp42)
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp20, tmp43, tmp44)
    tmp46 = tmp17 >= tmp3
    tmp47 = tl.full([1], 1088, tl.int64)
    tmp48 = tmp17 < tmp47
    tmp49 = tmp46 & tmp14
    tmp50 = tl.load(in_ptr3 + ((-200704) + x3 + (213248*x2)), tmp49, other=0.0)
    tmp51 = tl.full(tmp50.shape, 0.0, tmp50.dtype)
    tmp52 = tl.where(tmp49, tmp50, tmp51)
    tmp53 = tl.where(tmp19, tmp45, tmp52)
    tmp54 = tl.full(tmp53.shape, 0.0, tmp53.dtype)
    tmp55 = tl.where(tmp14, tmp53, tmp54)
    tmp56 = tl.where(tmp4, tmp13, tmp55)
    tl.store(out_ptr0 + (x4), tmp56, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fs/cfsjmbba6mmzz3laghk7pdguihhv4drtvt6w2b6gcahr2vuzub5o.py
# Source Nodes: [x_257], Original ATen: [aten._native_batch_norm_legit_functional]
# x_257 => add_453, add_454, add_455, mul_596, mul_597, mul_598, mul_599, mul_600, rsqrt_85, squeeze_256, var_mean_85
triton_red_fused__native_batch_norm_legit_functional_103 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_103', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2112
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (413952*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/27/c273uqznk37z3vvyo56osx6x2undbpgwumc6v3bsi4gy4ydiapua.py
# Source Nodes: [x_257, x_259], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_257 => add_453, add_456, mul_595, mul_601, rsqrt_85, sub_85, var_mean_85
# x_259 => relu_85
triton_poi_fused__native_batch_norm_legit_functional_relu_104 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_104', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3311616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 2112
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3c/c3c7duz5ejkkcrr4mue4zpjrsetfi7bansn7ufn7jxkr5cdubjd3.py
# Source Nodes: [x_s1_25, x_s1_26, x_s1_27, x_s1_28], Original ATen: [aten.add]
# x_s1_25 => add_419
# x_s1_26 => add_435
# x_s1_27 => add_451
# x_s1_28 => add_467
triton_poi_fused_add_105 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_105', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 200704
    x1 = (xindex // 200704)
    tmp0 = tl.load(in_ptr0 + (x0 + (376320*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (213248*x1)), None)
    tmp3 = tl.load(in_ptr2 + (x0 + (213248*x1)), None)
    tmp5 = tl.load(in_ptr3 + (x0 + (213248*x1)), None)
    tmp7 = tl.load(in_ptr4 + (x0 + (213248*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x0 + (426496*x1)), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/je/cjeucu5mmihagla2m72m4u3mijuaoswzq667kbqhvwokvisetsxr.py
# Source Nodes: [cat_85], Original ATen: [aten.cat]
# cat_85 => cat_54
triton_poi_fused_cat_106 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_106', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1806336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1152
    x2 = (xindex // 225792)
    x3 = xindex % 225792
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1088, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 1024, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.full([1], 960, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tmp9 & tmp7
    tmp11 = tl.full([1], 896, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp12 & tmp10
    tmp14 = tl.load(in_ptr0 + (x3 + (376320*x2)), tmp13, other=0.0)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = tmp0 >= tmp11
    tmp18 = tmp17 & tmp10
    tmp19 = tl.load(in_ptr1 + (25088 + x3 + (213248*x2)), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tl.where(tmp12, tmp16, tmp21)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp10, tmp22, tmp23)
    tmp25 = tmp0 >= tmp8
    tmp26 = tmp25 & tmp7
    tmp27 = tl.load(in_ptr2 + (12544 + x3 + (213248*x2)), tmp26, other=0.0)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp26, tmp27, tmp28)
    tmp30 = tl.where(tmp9, tmp24, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp7, tmp30, tmp31)
    tmp33 = tmp0 >= tmp5
    tmp34 = tmp33 & tmp4
    tmp35 = tl.load(in_ptr3 + (x3 + (213248*x2)), tmp34, other=0.0)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp34, tmp35, tmp36)
    tmp38 = tl.where(tmp6, tmp32, tmp37)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp4, tmp38, tmp39)
    tmp41 = tmp0 >= tmp3
    tmp42 = tl.full([1], 1152, tl.int64)
    tmp43 = tmp0 < tmp42
    tmp44 = tl.load(in_ptr4 + ((-12544) + x3 + (213248*x2)), tmp41, other=0.0)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp41, tmp44, tmp45)
    tmp47 = tl.where(tmp4, tmp40, tmp46)
    tl.store(out_ptr0 + (x3 + (426496*x2)), tmp47, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6a/c6a2xdhw4gnyun4zsq4mju5rrecpppaurkffpa7suh7z3sczz734.py
# Source Nodes: [x_266], Original ATen: [aten._native_batch_norm_legit_functional]
# x_266 => add_469, add_470, add_471, mul_617, mul_618, mul_619, mul_620, mul_621, rsqrt_88, squeeze_265, var_mean_88
triton_red_fused__native_batch_norm_legit_functional_107 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_107', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2176
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (426496*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/xm/cxmnjwk74q5dhi6mpzru6cqae2aeu4gpurhn47khx6agbuew3ct5.py
# Source Nodes: [x_266, x_268], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_266 => add_469, add_472, mul_616, mul_622, rsqrt_88, sub_88, var_mean_88
# x_268 => relu_88
triton_poi_fused__native_batch_norm_legit_functional_relu_108 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_108', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3411968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 2176
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/44/c44zecf2laztuyhlxth4asztbugfzf4u2pcqac7wwf3p2lgluifv.py
# Source Nodes: [cat_82], Original ATen: [aten.cat]
# cat_82 => cat_57
triton_poi_fused_cat_109 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_109', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3512320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 2240
    x2 = (xindex // 439040)
    x3 = xindex % 439040
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (426496*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 2240, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = (-1024) + x1
    tmp14 = tmp13 >= tmp1
    tmp15 = tl.full([1], 1152, tl.int64)
    tmp16 = tmp13 < tmp15
    tmp17 = tmp16 & tmp10
    tmp18 = tl.load(in_ptr2 + ((-200704) + x3 + (426496*x2)), tmp17, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp13 >= tmp15
    tmp22 = tl.full([1], 1216, tl.int64)
    tmp23 = tmp13 < tmp22
    tmp24 = tmp21 & tmp10
    tmp25 = tl.load(in_ptr1 + ((-225792) + x3 + (213248*x2)), tmp24, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp24, tmp25, tmp26)
    tmp28 = tl.where(tmp16, tmp20, tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp10, tmp28, tmp29)
    tmp31 = tl.where(tmp4, tmp9, tmp30)
    tl.store(out_ptr0 + (x4), tmp31, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yp/cypkzapiprxyzyvvqcwajgzwmo3nr5np7vjjsw2ktpnllcrwglnb.py
# Source Nodes: [x_275], Original ATen: [aten._native_batch_norm_legit_functional]
# x_275 => add_485, add_486, add_487, mul_638, mul_639, mul_640, mul_641, mul_642, rsqrt_91, squeeze_274, var_mean_91
triton_red_fused__native_batch_norm_legit_functional_110 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_110', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2240
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (439040*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/ag/cagylhfdo7rfqmzbajingjrlfsw27n7wxng45fedphlpavfgnito.py
# Source Nodes: [x_275, x_277], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_275 => add_485, add_488, mul_637, mul_643, rsqrt_91, sub_91, var_mean_91
# x_277 => relu_91
triton_poi_fused__native_batch_norm_legit_functional_relu_111 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_111', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3512320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 2240
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2q/c2qmefrub2x7eactpzqljamkn54vddmncii23towsij56izcytms.py
# Source Nodes: [cat_80], Original ATen: [aten.cat]
# cat_80 => cat_59
triton_poi_fused_cat_112 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_112', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3612672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 2304
    x2 = (xindex // 451584)
    x3 = xindex % 451584
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (426496*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 2304, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = (-1024) + x1
    tmp16 = tmp15 >= tmp1
    tmp17 = tl.full([1], 1216, tl.int64)
    tmp18 = tmp15 < tmp17
    tmp19 = tmp18 & tmp12
    tmp20 = tl.full([1], 1152, tl.int64)
    tmp21 = tmp15 < tmp20
    tmp22 = tmp21 & tmp19
    tmp23 = tl.load(in_ptr3 + ((-200704) + x3 + (426496*x2)), tmp22, other=0.0)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp22, tmp23, tmp24)
    tmp26 = tmp15 >= tmp20
    tmp27 = tmp26 & tmp19
    tmp28 = tl.load(in_ptr1 + ((-225792) + x3 + (213248*x2)), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tl.where(tmp21, tmp25, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp19, tmp31, tmp32)
    tmp34 = tmp15 >= tmp17
    tmp35 = tl.full([1], 1280, tl.int64)
    tmp36 = tmp15 < tmp35
    tmp37 = tmp34 & tmp12
    tmp38 = tl.load(in_ptr2 + ((-238336) + x3 + (213248*x2)), tmp37, other=0.0)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = tl.where(tmp18, tmp33, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp12, tmp41, tmp42)
    tmp44 = tl.where(tmp4, tmp11, tmp43)
    tl.store(out_ptr0 + (x4), tmp44, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5e/c5e7cmdakit3hvzqsq7cyz7m5wmob2zcy5tejcwndomclydcep6k.py
# Source Nodes: [x_284], Original ATen: [aten._native_batch_norm_legit_functional]
# x_284 => add_501, add_502, add_503, mul_659, mul_660, mul_661, mul_662, mul_663, rsqrt_94, squeeze_283, var_mean_94
triton_red_fused__native_batch_norm_legit_functional_113 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_113', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2304
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (451584*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/hq/chqm7jsnmypokxpeczy5l3hbf4qo276etcs6p2s3se76taqdng4s.py
# Source Nodes: [x_284, x_286], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_284 => add_501, add_504, mul_658, mul_664, rsqrt_94, sub_94, var_mean_94
# x_286 => relu_94
triton_poi_fused__native_batch_norm_legit_functional_relu_114 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_114', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3612672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 2304
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5h/c5hgwu3srctxlqdxfboyle5slumxdrcfq7qtlcp2ebid254dzr2x.py
# Source Nodes: [cat_78], Original ATen: [aten.cat]
# cat_78 => cat_61
triton_poi_fused_cat_115 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_115', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3713024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 2368
    x2 = (xindex // 464128)
    x3 = xindex % 464128
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1024, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (426496*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 2368, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = (-1024) + x1
    tmp18 = tmp17 >= tmp1
    tmp19 = tl.full([1], 1280, tl.int64)
    tmp20 = tmp17 < tmp19
    tmp21 = tmp20 & tmp14
    tmp22 = tl.full([1], 1216, tl.int64)
    tmp23 = tmp17 < tmp22
    tmp24 = tmp23 & tmp21
    tmp25 = tl.full([1], 1152, tl.int64)
    tmp26 = tmp17 < tmp25
    tmp27 = tmp26 & tmp24
    tmp28 = tl.load(in_ptr4 + ((-200704) + x3 + (426496*x2)), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp17 >= tmp25
    tmp32 = tmp31 & tmp24
    tmp33 = tl.load(in_ptr1 + ((-225792) + x3 + (213248*x2)), tmp32, other=0.0)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp32, tmp33, tmp34)
    tmp36 = tl.where(tmp26, tmp30, tmp35)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp24, tmp36, tmp37)
    tmp39 = tmp17 >= tmp22
    tmp40 = tmp39 & tmp21
    tmp41 = tl.load(in_ptr2 + ((-238336) + x3 + (213248*x2)), tmp40, other=0.0)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp40, tmp41, tmp42)
    tmp44 = tl.where(tmp23, tmp38, tmp43)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp21, tmp44, tmp45)
    tmp47 = tmp17 >= tmp19
    tmp48 = tl.full([1], 1344, tl.int64)
    tmp49 = tmp17 < tmp48
    tmp50 = tmp47 & tmp14
    tmp51 = tl.load(in_ptr3 + ((-250880) + x3 + (213248*x2)), tmp50, other=0.0)
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp50, tmp51, tmp52)
    tmp54 = tl.where(tmp20, tmp46, tmp53)
    tmp55 = tl.full(tmp54.shape, 0.0, tmp54.dtype)
    tmp56 = tl.where(tmp14, tmp54, tmp55)
    tmp57 = tl.where(tmp4, tmp13, tmp56)
    tl.store(out_ptr0 + (x4), tmp57, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/33/c33aea4imotp7pvv36nxbbzrjzjduk4zea6gemgewbpap7qtoweh.py
# Source Nodes: [x_293], Original ATen: [aten._native_batch_norm_legit_functional]
# x_293 => add_517, add_518, add_519, mul_680, mul_681, mul_682, mul_683, mul_684, rsqrt_97, squeeze_292, var_mean_97
triton_red_fused__native_batch_norm_legit_functional_116 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_116', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2368
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (464128*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/wr/cwrrao2krs7oi4ygyjlmc2qcfm2kfcayts4mepng35l6gkoscg5s.py
# Source Nodes: [x_293, x_295], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_293 => add_517, add_520, mul_679, mul_685, rsqrt_97, sub_97, var_mean_97
# x_295 => relu_97
triton_poi_fused__native_batch_norm_legit_functional_relu_117 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_117', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3713024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 2368
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/uc/cucp67lh2wl2yszv7ld6nj3auxiyxbi6cv56za4v37llsmdi72ls.py
# Source Nodes: [cat_77], Original ATen: [aten.cat]
# cat_77 => cat_62
triton_poi_fused_cat_118 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_118', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2207744
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1408
    x2 = (xindex // 275968)
    x3 = xindex % 275968
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 1344, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.full([1], 1280, tl.int64)
    tmp6 = tmp0 < tmp5
    tmp7 = tmp6 & tmp4
    tmp8 = tl.full([1], 1216, tl.int64)
    tmp9 = tmp0 < tmp8
    tmp10 = tmp9 & tmp7
    tmp11 = tl.full([1], 1152, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = tmp12 & tmp10
    tmp14 = tl.load(in_ptr0 + (x3 + (426496*x2)), tmp13, other=0.0)
    tmp15 = tl.full(tmp14.shape, 0.0, tmp14.dtype)
    tmp16 = tl.where(tmp13, tmp14, tmp15)
    tmp17 = tmp0 >= tmp11
    tmp18 = tmp17 & tmp10
    tmp19 = tl.load(in_ptr1 + ((-25088) + x3 + (213248*x2)), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tl.where(tmp12, tmp16, tmp21)
    tmp23 = tl.full(tmp22.shape, 0.0, tmp22.dtype)
    tmp24 = tl.where(tmp10, tmp22, tmp23)
    tmp25 = tmp0 >= tmp8
    tmp26 = tmp25 & tmp7
    tmp27 = tl.load(in_ptr2 + ((-37632) + x3 + (213248*x2)), tmp26, other=0.0)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp26, tmp27, tmp28)
    tmp30 = tl.where(tmp9, tmp24, tmp29)
    tmp31 = tl.full(tmp30.shape, 0.0, tmp30.dtype)
    tmp32 = tl.where(tmp7, tmp30, tmp31)
    tmp33 = tmp0 >= tmp5
    tmp34 = tmp33 & tmp4
    tmp35 = tl.load(in_ptr3 + ((-50176) + x3 + (213248*x2)), tmp34, other=0.0)
    tmp36 = tl.full(tmp35.shape, 0.0, tmp35.dtype)
    tmp37 = tl.where(tmp34, tmp35, tmp36)
    tmp38 = tl.where(tmp6, tmp32, tmp37)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp4, tmp38, tmp39)
    tmp41 = tmp0 >= tmp3
    tmp42 = tl.full([1], 1408, tl.int64)
    tmp43 = tmp0 < tmp42
    tmp44 = tl.load(in_ptr4 + ((-62720) + x3 + (213248*x2)), tmp41, other=0.0)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp41, tmp44, tmp45)
    tmp47 = tl.where(tmp4, tmp40, tmp46)
    tl.store(out_ptr0 + (x3 + (476672*x2)), tmp47, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qb/cqbtaf4hd7zovp4shg5c6wmlptrvtvgwqg4r7xyg63jzbzyql3wn.py
# Source Nodes: [resid_31, x_s1_29, x_s1_30, x_s1_31], Original ATen: [aten.add]
# resid_31 => add_531
# x_s1_29 => add_483
# x_s1_30 => add_499
# x_s1_31 => add_515
triton_poi_fused_add_119 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_119', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 200704
    x1 = (xindex // 200704)
    tmp0 = tl.load(in_ptr0 + (x0 + (426496*x1)), None)
    tmp1 = tl.load(in_ptr1 + (x0 + (213248*x1)), None)
    tmp3 = tl.load(in_ptr2 + (x0 + (213248*x1)), None)
    tmp5 = tl.load(in_ptr3 + (x0 + (213248*x1)), None)
    tmp7 = tl.load(in_ptr4 + (x0 + (213248*x1)), None)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp8 = tmp6 + tmp7
    tl.store(out_ptr0 + (x0 + (476672*x1)), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sr/csrafqbkgesxdzhjezmvktv65xdqq3dy376o53ejv7och76jiihp.py
# Source Nodes: [x_302, x_305], Original ATen: [aten._native_batch_norm_legit_functional]
# x_302 => add_533, add_534, add_535, mul_701, mul_702, mul_703, mul_704, mul_705, rsqrt_100, squeeze_301, var_mean_100
# x_305 => add_539, add_540, mul_709, mul_712
triton_red_fused__native_batch_norm_legit_functional_120 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32', 13: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12, 13))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_120', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'in_ptr3', 'in_ptr4', 'out_ptr10', 'out_ptr4', 'out_ptr6', 'out_ptr8']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, out_ptr8, out_ptr10, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2432
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (476672*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp16 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp22 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp5 = 1568.0
    tmp6 = tmp3 / tmp5
    tmp7 = 0.001
    tmp8 = tmp6 + tmp7
    tmp9 = tl.math.rsqrt(tmp8)
    tmp10 = 0.1
    tmp11 = tmp2 * tmp10
    tmp13 = 0.9
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp17 = tmp16 * tmp13
    tmp18 = tmp11 + tmp17
    tmp19 = 1.0006381620931717
    tmp20 = tmp6 * tmp19
    tmp21 = tmp20 * tmp10
    tmp23 = tmp22 * tmp13
    tmp24 = tmp21 + tmp23
    tmp26 = tmp25 * tmp13
    tmp27 = tmp21 + tmp26
    tl.store(out_ptr2 + (x0), tmp9, xmask)
    tl.store(out_ptr4 + (x0), tmp15, xmask)
    tl.store(out_ptr6 + (x0), tmp18, xmask)
    tl.store(out_ptr8 + (x0), tmp24, xmask)
    tl.store(out_ptr10 + (x0), tmp27, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4f/c4ftm2pkvcenyvwlptuxx42jbwjczrlb4u324i2cmjn3nooe26qf.py
# Source Nodes: [x_302, x_304, x_305, x_307], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_302 => add_533, add_536, mul_700, mul_706, rsqrt_100, sub_100, var_mean_100
# x_304 => relu_100
# x_305 => add_541, mul_713
# x_307 => relu_101
triton_poi_fused__native_batch_norm_legit_functional_relu_121 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_121', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3813376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 2432
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp16 = tmp9 * tmp15
    tmp18 = tmp16 + tmp17
    tmp19 = triton_helpers.maximum(0, tmp18)
    tl.store(out_ptr0 + (x3), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp19, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bj/cbjfbfvvo43soozkgohvq5vighe2vplxzwtjcf57ozoe62vlrltd.py
# Source Nodes: [x_311], Original ATen: [aten._native_batch_norm_legit_functional]
# x_311 => add_548, add_549, add_550, mul_722, mul_723, mul_724, mul_725, mul_726, rsqrt_103, squeeze_310, var_mean_103
triton_per_fused__native_batch_norm_legit_functional_122 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_122', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel):
    xnumel = 1600
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (78400*r2)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 392, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = 392.0
    tmp18 = tmp16 / tmp17
    tmp19 = 0.001
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = 0.1
    tmp23 = tmp10 * tmp22
    tmp25 = 0.9
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 + tmp26
    tmp28 = 1.0025575447570332
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


# kernel path: /tmp/torchinductor_youkaichao/tc/ctctyg3zovtbkah4l3aevc2wdgceifhd4gghxbrbits5gv3ijlqo.py
# Source Nodes: [x_311, x_313], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_311 => add_548, add_551, mul_721, mul_727, rsqrt_103, sub_103, var_mean_103
# x_313 => relu_103
triton_poi_fused__native_batch_norm_legit_functional_relu_123 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_123', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 627200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 1600
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fn/cfniptb4ubn44idpohkek2yxzld5qg6glbsvzpbnoqefegguaksq.py
# Source Nodes: [cat_74], Original ATen: [aten.cat]
# cat_74 => cat_65
triton_poi_fused_cat_124 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_124', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 953344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 49) % 2432
    x2 = (xindex // 119168)
    x3 = xindex % 119168
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2048, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (112896*x2)), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (106624*x2)), tmp4 & xmask, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.full(tmp7.shape, 0.0, tmp7.dtype)
    tmp9 = tl.where(tmp4, tmp7, tmp8)
    tmp10 = tmp0 >= tmp3
    tmp11 = tl.full([1], 2432, tl.int64)
    tmp12 = tmp0 < tmp11
    tmp13 = (-2048) + x1
    tmp14 = tmp13 >= tmp1
    tmp15 = tl.full([1], 256, tl.int64)
    tmp16 = tmp13 < tmp15
    tmp17 = tmp16 & tmp10
    tmp18 = tl.load(in_ptr0 + (x3 + (112896*x2)), tmp17 & xmask, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp13 >= tmp15
    tmp22 = tl.full([1], 384, tl.int64)
    tmp23 = tmp13 < tmp22
    tmp24 = tmp21 & tmp10
    tmp25 = tl.load(in_ptr1 + ((-12544) + x3 + (106624*x2)), tmp24 & xmask, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp24, tmp25, tmp26)
    tmp28 = tl.where(tmp16, tmp20, tmp27)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp10, tmp28, tmp29)
    tmp31 = tl.where(tmp4, tmp9, tmp30)
    tl.store(out_ptr0 + (x4), tmp31, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dz/cdz62x4qugthkkk46tenro6jeah5bwbnnfd23tjuvmmz4te3qh3g.py
# Source Nodes: [x_314], Original ATen: [aten._native_batch_norm_legit_functional]
# x_314 => add_554, add_555, add_556, mul_729, mul_730, mul_731, mul_732, mul_733, rsqrt_104, squeeze_313, var_mean_104
triton_per_fused__native_batch_norm_legit_functional_125 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_125', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel):
    xnumel = 2432
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (119168*r2)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 392, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = 392.0
    tmp18 = tmp16 / tmp17
    tmp19 = 0.001
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = 0.1
    tmp23 = tmp10 * tmp22
    tmp25 = 0.9
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 + tmp26
    tmp28 = 1.0025575447570332
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


# kernel path: /tmp/torchinductor_youkaichao/ab/cabqwawi2fonlrdtq33tcao6bopuz6dnitzo5g4nki3oxcshsctg.py
# Source Nodes: [x_314, x_316], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_314 => add_554, add_557, mul_728, mul_734, rsqrt_104, sub_104, var_mean_104
# x_316 => relu_104
triton_poi_fused__native_batch_norm_legit_functional_relu_126 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_126', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 953344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 2432
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/di/cdidcpfval2fwoaoorq3su3say5st456mby6oh722jvmulivo667.py
# Source Nodes: [cat_72], Original ATen: [aten.cat]
# cat_72 => cat_67
triton_poi_fused_cat_127 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_127', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 49) % 2560
    x2 = (xindex // 125440)
    x3 = xindex % 125440
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2048, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (112896*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (106624*x2)), tmp4, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x3 + (106624*x2)), tmp4, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.full(tmp9.shape, 0.0, tmp9.dtype)
    tmp11 = tl.where(tmp4, tmp9, tmp10)
    tmp12 = tmp0 >= tmp3
    tmp13 = tl.full([1], 2560, tl.int64)
    tmp14 = tmp0 < tmp13
    tmp15 = (-2048) + x1
    tmp16 = tmp15 >= tmp1
    tmp17 = tl.full([1], 384, tl.int64)
    tmp18 = tmp15 < tmp17
    tmp19 = tmp18 & tmp12
    tmp20 = tl.full([1], 256, tl.int64)
    tmp21 = tmp15 < tmp20
    tmp22 = tmp21 & tmp19
    tmp23 = tl.load(in_ptr0 + (x3 + (112896*x2)), tmp22, other=0.0)
    tmp24 = tl.full(tmp23.shape, 0.0, tmp23.dtype)
    tmp25 = tl.where(tmp22, tmp23, tmp24)
    tmp26 = tmp15 >= tmp20
    tmp27 = tmp26 & tmp19
    tmp28 = tl.load(in_ptr1 + ((-12544) + x3 + (106624*x2)), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tl.where(tmp21, tmp25, tmp30)
    tmp32 = tl.full(tmp31.shape, 0.0, tmp31.dtype)
    tmp33 = tl.where(tmp19, tmp31, tmp32)
    tmp34 = tmp15 >= tmp17
    tmp35 = tl.full([1], 512, tl.int64)
    tmp36 = tmp15 < tmp35
    tmp37 = tmp34 & tmp12
    tmp38 = tl.load(in_ptr2 + ((-18816) + x3 + (106624*x2)), tmp37, other=0.0)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = tl.where(tmp18, tmp33, tmp40)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp12, tmp41, tmp42)
    tmp44 = tl.where(tmp4, tmp11, tmp43)
    tl.store(out_ptr0 + (x4), tmp44, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pv/cpv2zjtcw3o2mvanyzgdbsdzx25kqq5fl2deoizpxtyodbqjw4d3.py
# Source Nodes: [x_323], Original ATen: [aten._native_batch_norm_legit_functional]
# x_323 => add_570, add_571, add_572, mul_750, mul_751, mul_752, mul_753, mul_754, rsqrt_107, squeeze_322, var_mean_107
triton_per_fused__native_batch_norm_legit_functional_128 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_128', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel):
    xnumel = 2560
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (125440*r2)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 392, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = 392.0
    tmp18 = tmp16 / tmp17
    tmp19 = 0.001
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = 0.1
    tmp23 = tmp10 * tmp22
    tmp25 = 0.9
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 + tmp26
    tmp28 = 1.0025575447570332
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


# kernel path: /tmp/torchinductor_youkaichao/ma/cmanuqxg2r5scbt4fyn2vxljktek2ukkh3lm6u65nqntmlcpyf5b.py
# Source Nodes: [x_323, x_325], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_323 => add_570, add_573, mul_749, mul_755, rsqrt_107, sub_107, var_mean_107
# x_325 => relu_107
triton_poi_fused__native_batch_norm_legit_functional_relu_129 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_129', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 2560
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bz/cbza2nvegjpj6hqimw2vt4p3o7lcxdaw2sc55mdinmrz2anx2ndb.py
# Source Nodes: [cat_70], Original ATen: [aten.cat]
# cat_70 => cat_69
triton_poi_fused_cat_130 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_130', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1053696
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 49) % 2688
    x2 = (xindex // 131712)
    x3 = xindex % 131712
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 2048, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (112896*x2)), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x3 + (106624*x2)), tmp4 & xmask, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (x3 + (106624*x2)), tmp4 & xmask, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + (x3 + (106624*x2)), tmp4 & xmask, other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1], 2688, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = (-2048) + x1
    tmp18 = tmp17 >= tmp1
    tmp19 = tl.full([1], 512, tl.int64)
    tmp20 = tmp17 < tmp19
    tmp21 = tmp20 & tmp14
    tmp22 = tl.full([1], 384, tl.int64)
    tmp23 = tmp17 < tmp22
    tmp24 = tmp23 & tmp21
    tmp25 = tl.full([1], 256, tl.int64)
    tmp26 = tmp17 < tmp25
    tmp27 = tmp26 & tmp24
    tmp28 = tl.load(in_ptr0 + (x3 + (112896*x2)), tmp27 & xmask, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp17 >= tmp25
    tmp32 = tmp31 & tmp24
    tmp33 = tl.load(in_ptr1 + ((-12544) + x3 + (106624*x2)), tmp32 & xmask, other=0.0)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp32, tmp33, tmp34)
    tmp36 = tl.where(tmp26, tmp30, tmp35)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp24, tmp36, tmp37)
    tmp39 = tmp17 >= tmp22
    tmp40 = tmp39 & tmp21
    tmp41 = tl.load(in_ptr2 + ((-18816) + x3 + (106624*x2)), tmp40 & xmask, other=0.0)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp40, tmp41, tmp42)
    tmp44 = tl.where(tmp23, tmp38, tmp43)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp21, tmp44, tmp45)
    tmp47 = tmp17 >= tmp19
    tmp48 = tl.full([1], 640, tl.int64)
    tmp49 = tmp17 < tmp48
    tmp50 = tmp47 & tmp14
    tmp51 = tl.load(in_ptr3 + ((-25088) + x3 + (106624*x2)), tmp50 & xmask, other=0.0)
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp50, tmp51, tmp52)
    tmp54 = tl.where(tmp20, tmp46, tmp53)
    tmp55 = tl.full(tmp54.shape, 0.0, tmp54.dtype)
    tmp56 = tl.where(tmp14, tmp54, tmp55)
    tmp57 = tl.where(tmp4, tmp13, tmp56)
    tl.store(out_ptr0 + (x4), tmp57, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/re/creowf5eezdre4ibrgp3455yoifxw2lrwq7bmgkvclygfmytqoxx.py
# Source Nodes: [x_333], Original ATen: [aten._native_batch_norm_legit_functional]
# x_333 => add_586, add_587, add_588, mul_771, mul_772, mul_773, mul_774, mul_775, rsqrt_110, squeeze_331, var_mean_110
triton_per_fused__native_batch_norm_legit_functional_131 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 512],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_131', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel):
    xnumel = 2688
    XBLOCK: tl.constexpr = 1
    rnumel = 392
    RBLOCK: tl.constexpr = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex % 49
    r2 = (rindex // 49)
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (131712*r2)), rmask & xmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 392, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask & xmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = 392.0
    tmp18 = tmp16 / tmp17
    tmp19 = 0.001
    tmp20 = tmp18 + tmp19
    tmp21 = tl.math.rsqrt(tmp20)
    tmp22 = 0.1
    tmp23 = tmp10 * tmp22
    tmp25 = 0.9
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 + tmp26
    tmp28 = 1.0025575447570332
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


# kernel path: /tmp/torchinductor_youkaichao/45/c45gh72rfbomcdn5wti2hwqtg3i4vyu7gafg54it6y2mj3337n7r.py
# Source Nodes: [x_333, x_336, x_337], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu, aten.threshold_backward]
# x_333 => add_586, add_589, mul_770, mul_776, rsqrt_110, sub_110, var_mean_110
# x_336 => relu_110
# x_337 => mean
triton_per_fused__native_batch_norm_legit_functional_mean_relu_threshold_backward_132 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_mean_relu_threshold_backward_132', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 21504
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 2688
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
    tmp5 = tmp3 / tmp4
    tmp6 = 0.001
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tmp17 = tl.broadcast_to(tmp14, [XBLOCK, RBLOCK])
    tmp19 = tl.where(rmask & xmask, tmp17, 0)
    tmp20 = tl.sum(tmp19, 1)[:, None]
    tmp21 = 49.0
    tmp22 = tmp20 / tmp21
    tl.store(out_ptr1 + (r2 + (49*x3)), tmp16, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5m/c5mj27sa3kkf3mu62nr3fufpgz3qapm54wcy3xlzndeqhdbsc72x.py
# Source Nodes: [pred, x_340], Original ATen: [aten.convolution, aten.view]
# pred => view
# x_340 => convolution_110
triton_poi_fused_convolution_view_133 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_view_133', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1000
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x2), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zw/czwvifz6nxkdrhkynnvfeki6np7shxvotew4jlr5h3c7wirfy2lu.py
# Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]

triton_poi_fused_native_batch_norm_backward_134 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_native_batch_norm_backward_134', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tu/ctuz2zh65z4z4k3sp23r7azyn737s6basmcaanrtz5vmyigaxyvo.py
# Source Nodes: [add_], Original ATen: [aten.add]
# add_ => add
triton_poi_fused_add_135 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_135', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668 = args
    args.clear()
    assert_size_stride(primals_1, (128, ), (1, ))
    assert_size_stride(primals_2, (128, ), (1, ))
    assert_size_stride(primals_3, (128, ), (1, ))
    assert_size_stride(primals_4, (128, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_6, (128, ), (1, ))
    assert_size_stride(primals_7, (200, ), (1, ))
    assert_size_stride(primals_8, (200, ), (1, ))
    assert_size_stride(primals_9, (200, ), (1, ))
    assert_size_stride(primals_10, (200, ), (1, ))
    assert_size_stride(primals_11, (316, ), (1, ))
    assert_size_stride(primals_12, (316, ), (1, ))
    assert_size_stride(primals_13, (200, ), (1, ))
    assert_size_stride(primals_14, (200, ), (1, ))
    assert_size_stride(primals_15, (200, ), (1, ))
    assert_size_stride(primals_16, (200, ), (1, ))
    assert_size_stride(primals_17, (336, ), (1, ))
    assert_size_stride(primals_18, (336, ), (1, ))
    assert_size_stride(primals_19, (200, ), (1, ))
    assert_size_stride(primals_20, (200, ), (1, ))
    assert_size_stride(primals_21, (200, ), (1, ))
    assert_size_stride(primals_22, (200, ), (1, ))
    assert_size_stride(primals_23, (356, ), (1, ))
    assert_size_stride(primals_24, (356, ), (1, ))
    assert_size_stride(primals_25, (200, ), (1, ))
    assert_size_stride(primals_26, (200, ), (1, ))
    assert_size_stride(primals_27, (200, ), (1, ))
    assert_size_stride(primals_28, (200, ), (1, ))
    assert_size_stride(primals_29, (376, ), (1, ))
    assert_size_stride(primals_30, (376, ), (1, ))
    assert_size_stride(primals_31, (376, ), (1, ))
    assert_size_stride(primals_32, (376, ), (1, ))
    assert_size_stride(primals_33, (400, ), (1, ))
    assert_size_stride(primals_34, (400, ), (1, ))
    assert_size_stride(primals_35, (400, ), (1, ))
    assert_size_stride(primals_36, (400, ), (1, ))
    assert_size_stride(primals_37, (704, ), (1, ))
    assert_size_stride(primals_38, (704, ), (1, ))
    assert_size_stride(primals_39, (400, ), (1, ))
    assert_size_stride(primals_40, (400, ), (1, ))
    assert_size_stride(primals_41, (400, ), (1, ))
    assert_size_stride(primals_42, (400, ), (1, ))
    assert_size_stride(primals_43, (768, ), (1, ))
    assert_size_stride(primals_44, (768, ), (1, ))
    assert_size_stride(primals_45, (400, ), (1, ))
    assert_size_stride(primals_46, (400, ), (1, ))
    assert_size_stride(primals_47, (400, ), (1, ))
    assert_size_stride(primals_48, (400, ), (1, ))
    assert_size_stride(primals_49, (832, ), (1, ))
    assert_size_stride(primals_50, (832, ), (1, ))
    assert_size_stride(primals_51, (400, ), (1, ))
    assert_size_stride(primals_52, (400, ), (1, ))
    assert_size_stride(primals_53, (400, ), (1, ))
    assert_size_stride(primals_54, (400, ), (1, ))
    assert_size_stride(primals_55, (896, ), (1, ))
    assert_size_stride(primals_56, (896, ), (1, ))
    assert_size_stride(primals_57, (400, ), (1, ))
    assert_size_stride(primals_58, (400, ), (1, ))
    assert_size_stride(primals_59, (400, ), (1, ))
    assert_size_stride(primals_60, (400, ), (1, ))
    assert_size_stride(primals_61, (960, ), (1, ))
    assert_size_stride(primals_62, (960, ), (1, ))
    assert_size_stride(primals_63, (400, ), (1, ))
    assert_size_stride(primals_64, (400, ), (1, ))
    assert_size_stride(primals_65, (400, ), (1, ))
    assert_size_stride(primals_66, (400, ), (1, ))
    assert_size_stride(primals_67, (1024, ), (1, ))
    assert_size_stride(primals_68, (1024, ), (1, ))
    assert_size_stride(primals_69, (400, ), (1, ))
    assert_size_stride(primals_70, (400, ), (1, ))
    assert_size_stride(primals_71, (400, ), (1, ))
    assert_size_stride(primals_72, (400, ), (1, ))
    assert_size_stride(primals_73, (1088, ), (1, ))
    assert_size_stride(primals_74, (1088, ), (1, ))
    assert_size_stride(primals_75, (400, ), (1, ))
    assert_size_stride(primals_76, (400, ), (1, ))
    assert_size_stride(primals_77, (400, ), (1, ))
    assert_size_stride(primals_78, (400, ), (1, ))
    assert_size_stride(primals_79, (1152, ), (1, ))
    assert_size_stride(primals_80, (1152, ), (1, ))
    assert_size_stride(primals_81, (1152, ), (1, ))
    assert_size_stride(primals_82, (1152, ), (1, ))
    assert_size_stride(primals_83, (800, ), (1, ))
    assert_size_stride(primals_84, (800, ), (1, ))
    assert_size_stride(primals_85, (800, ), (1, ))
    assert_size_stride(primals_86, (800, ), (1, ))
    assert_size_stride(primals_87, (1216, ), (1, ))
    assert_size_stride(primals_88, (1216, ), (1, ))
    assert_size_stride(primals_89, (800, ), (1, ))
    assert_size_stride(primals_90, (800, ), (1, ))
    assert_size_stride(primals_91, (800, ), (1, ))
    assert_size_stride(primals_92, (800, ), (1, ))
    assert_size_stride(primals_93, (1280, ), (1, ))
    assert_size_stride(primals_94, (1280, ), (1, ))
    assert_size_stride(primals_95, (800, ), (1, ))
    assert_size_stride(primals_96, (800, ), (1, ))
    assert_size_stride(primals_97, (800, ), (1, ))
    assert_size_stride(primals_98, (800, ), (1, ))
    assert_size_stride(primals_99, (1344, ), (1, ))
    assert_size_stride(primals_100, (1344, ), (1, ))
    assert_size_stride(primals_101, (800, ), (1, ))
    assert_size_stride(primals_102, (800, ), (1, ))
    assert_size_stride(primals_103, (800, ), (1, ))
    assert_size_stride(primals_104, (800, ), (1, ))
    assert_size_stride(primals_105, (1408, ), (1, ))
    assert_size_stride(primals_106, (1408, ), (1, ))
    assert_size_stride(primals_107, (800, ), (1, ))
    assert_size_stride(primals_108, (800, ), (1, ))
    assert_size_stride(primals_109, (800, ), (1, ))
    assert_size_stride(primals_110, (800, ), (1, ))
    assert_size_stride(primals_111, (1472, ), (1, ))
    assert_size_stride(primals_112, (1472, ), (1, ))
    assert_size_stride(primals_113, (800, ), (1, ))
    assert_size_stride(primals_114, (800, ), (1, ))
    assert_size_stride(primals_115, (800, ), (1, ))
    assert_size_stride(primals_116, (800, ), (1, ))
    assert_size_stride(primals_117, (1536, ), (1, ))
    assert_size_stride(primals_118, (1536, ), (1, ))
    assert_size_stride(primals_119, (800, ), (1, ))
    assert_size_stride(primals_120, (800, ), (1, ))
    assert_size_stride(primals_121, (800, ), (1, ))
    assert_size_stride(primals_122, (800, ), (1, ))
    assert_size_stride(primals_123, (1600, ), (1, ))
    assert_size_stride(primals_124, (1600, ), (1, ))
    assert_size_stride(primals_125, (800, ), (1, ))
    assert_size_stride(primals_126, (800, ), (1, ))
    assert_size_stride(primals_127, (800, ), (1, ))
    assert_size_stride(primals_128, (800, ), (1, ))
    assert_size_stride(primals_129, (1664, ), (1, ))
    assert_size_stride(primals_130, (1664, ), (1, ))
    assert_size_stride(primals_131, (800, ), (1, ))
    assert_size_stride(primals_132, (800, ), (1, ))
    assert_size_stride(primals_133, (800, ), (1, ))
    assert_size_stride(primals_134, (800, ), (1, ))
    assert_size_stride(primals_135, (1728, ), (1, ))
    assert_size_stride(primals_136, (1728, ), (1, ))
    assert_size_stride(primals_137, (800, ), (1, ))
    assert_size_stride(primals_138, (800, ), (1, ))
    assert_size_stride(primals_139, (800, ), (1, ))
    assert_size_stride(primals_140, (800, ), (1, ))
    assert_size_stride(primals_141, (1792, ), (1, ))
    assert_size_stride(primals_142, (1792, ), (1, ))
    assert_size_stride(primals_143, (800, ), (1, ))
    assert_size_stride(primals_144, (800, ), (1, ))
    assert_size_stride(primals_145, (800, ), (1, ))
    assert_size_stride(primals_146, (800, ), (1, ))
    assert_size_stride(primals_147, (1856, ), (1, ))
    assert_size_stride(primals_148, (1856, ), (1, ))
    assert_size_stride(primals_149, (800, ), (1, ))
    assert_size_stride(primals_150, (800, ), (1, ))
    assert_size_stride(primals_151, (800, ), (1, ))
    assert_size_stride(primals_152, (800, ), (1, ))
    assert_size_stride(primals_153, (1920, ), (1, ))
    assert_size_stride(primals_154, (1920, ), (1, ))
    assert_size_stride(primals_155, (800, ), (1, ))
    assert_size_stride(primals_156, (800, ), (1, ))
    assert_size_stride(primals_157, (800, ), (1, ))
    assert_size_stride(primals_158, (800, ), (1, ))
    assert_size_stride(primals_159, (1984, ), (1, ))
    assert_size_stride(primals_160, (1984, ), (1, ))
    assert_size_stride(primals_161, (800, ), (1, ))
    assert_size_stride(primals_162, (800, ), (1, ))
    assert_size_stride(primals_163, (800, ), (1, ))
    assert_size_stride(primals_164, (800, ), (1, ))
    assert_size_stride(primals_165, (2048, ), (1, ))
    assert_size_stride(primals_166, (2048, ), (1, ))
    assert_size_stride(primals_167, (800, ), (1, ))
    assert_size_stride(primals_168, (800, ), (1, ))
    assert_size_stride(primals_169, (800, ), (1, ))
    assert_size_stride(primals_170, (800, ), (1, ))
    assert_size_stride(primals_171, (2112, ), (1, ))
    assert_size_stride(primals_172, (2112, ), (1, ))
    assert_size_stride(primals_173, (800, ), (1, ))
    assert_size_stride(primals_174, (800, ), (1, ))
    assert_size_stride(primals_175, (800, ), (1, ))
    assert_size_stride(primals_176, (800, ), (1, ))
    assert_size_stride(primals_177, (2176, ), (1, ))
    assert_size_stride(primals_178, (2176, ), (1, ))
    assert_size_stride(primals_179, (800, ), (1, ))
    assert_size_stride(primals_180, (800, ), (1, ))
    assert_size_stride(primals_181, (800, ), (1, ))
    assert_size_stride(primals_182, (800, ), (1, ))
    assert_size_stride(primals_183, (2240, ), (1, ))
    assert_size_stride(primals_184, (2240, ), (1, ))
    assert_size_stride(primals_185, (800, ), (1, ))
    assert_size_stride(primals_186, (800, ), (1, ))
    assert_size_stride(primals_187, (800, ), (1, ))
    assert_size_stride(primals_188, (800, ), (1, ))
    assert_size_stride(primals_189, (2304, ), (1, ))
    assert_size_stride(primals_190, (2304, ), (1, ))
    assert_size_stride(primals_191, (800, ), (1, ))
    assert_size_stride(primals_192, (800, ), (1, ))
    assert_size_stride(primals_193, (800, ), (1, ))
    assert_size_stride(primals_194, (800, ), (1, ))
    assert_size_stride(primals_195, (2368, ), (1, ))
    assert_size_stride(primals_196, (2368, ), (1, ))
    assert_size_stride(primals_197, (800, ), (1, ))
    assert_size_stride(primals_198, (800, ), (1, ))
    assert_size_stride(primals_199, (800, ), (1, ))
    assert_size_stride(primals_200, (800, ), (1, ))
    assert_size_stride(primals_201, (2432, ), (1, ))
    assert_size_stride(primals_202, (2432, ), (1, ))
    assert_size_stride(primals_203, (2432, ), (1, ))
    assert_size_stride(primals_204, (2432, ), (1, ))
    assert_size_stride(primals_205, (1600, ), (1, ))
    assert_size_stride(primals_206, (1600, ), (1, ))
    assert_size_stride(primals_207, (1600, ), (1, ))
    assert_size_stride(primals_208, (1600, ), (1, ))
    assert_size_stride(primals_209, (2432, ), (1, ))
    assert_size_stride(primals_210, (2432, ), (1, ))
    assert_size_stride(primals_211, (1600, ), (1, ))
    assert_size_stride(primals_212, (1600, ), (1, ))
    assert_size_stride(primals_213, (1600, ), (1, ))
    assert_size_stride(primals_214, (1600, ), (1, ))
    assert_size_stride(primals_215, (2560, ), (1, ))
    assert_size_stride(primals_216, (2560, ), (1, ))
    assert_size_stride(primals_217, (1600, ), (1, ))
    assert_size_stride(primals_218, (1600, ), (1, ))
    assert_size_stride(primals_219, (1600, ), (1, ))
    assert_size_stride(primals_220, (1600, ), (1, ))
    assert_size_stride(primals_221, (2688, ), (1, ))
    assert_size_stride(primals_222, (2688, ), (1, ))
    assert_size_stride(primals_223, (128, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_224, (296, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_225, (200, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_226, (200, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_227, (276, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(primals_228, (200, 316, 1, 1), (316, 1, 1, 1))
    assert_size_stride(primals_229, (200, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_230, (276, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(primals_231, (200, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(primals_232, (200, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_233, (276, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(primals_234, (200, 356, 1, 1), (356, 1, 1, 1))
    assert_size_stride(primals_235, (200, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_236, (276, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(primals_237, (640, 376, 1, 1), (376, 1, 1, 1))
    assert_size_stride(primals_238, (400, 376, 1, 1), (376, 1, 1, 1))
    assert_size_stride(primals_239, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_240, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(primals_241, (400, 704, 1, 1), (704, 1, 1, 1))
    assert_size_stride(primals_242, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_243, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(primals_244, (400, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_245, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_246, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(primals_247, (400, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(primals_248, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_249, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(primals_250, (400, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_251, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_252, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(primals_253, (400, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_254, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_255, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(primals_256, (400, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_257, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_258, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(primals_259, (400, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(primals_260, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_261, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(primals_262, (1152, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_263, (800, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(primals_264, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_265, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_266, (800, 1216, 1, 1), (1216, 1, 1, 1))
    assert_size_stride(primals_267, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_268, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_269, (800, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(primals_270, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_271, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_272, (800, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(primals_273, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_274, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_275, (800, 1408, 1, 1), (1408, 1, 1, 1))
    assert_size_stride(primals_276, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_277, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_278, (800, 1472, 1, 1), (1472, 1, 1, 1))
    assert_size_stride(primals_279, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_280, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_281, (800, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(primals_282, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_283, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_284, (800, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(primals_285, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_286, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_287, (800, 1664, 1, 1), (1664, 1, 1, 1))
    assert_size_stride(primals_288, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_289, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_290, (800, 1728, 1, 1), (1728, 1, 1, 1))
    assert_size_stride(primals_291, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_292, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_293, (800, 1792, 1, 1), (1792, 1, 1, 1))
    assert_size_stride(primals_294, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_295, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_296, (800, 1856, 1, 1), (1856, 1, 1, 1))
    assert_size_stride(primals_297, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_298, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_299, (800, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(primals_300, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_301, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_302, (800, 1984, 1, 1), (1984, 1, 1, 1))
    assert_size_stride(primals_303, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_304, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_305, (800, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_306, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_307, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_308, (800, 2112, 1, 1), (2112, 1, 1, 1))
    assert_size_stride(primals_309, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_310, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_311, (800, 2176, 1, 1), (2176, 1, 1, 1))
    assert_size_stride(primals_312, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_313, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_314, (800, 2240, 1, 1), (2240, 1, 1, 1))
    assert_size_stride(primals_315, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_316, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_317, (800, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(primals_318, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_319, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_320, (800, 2368, 1, 1), (2368, 1, 1, 1))
    assert_size_stride(primals_321, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_322, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_323, (2304, 2432, 1, 1), (2432, 1, 1, 1))
    assert_size_stride(primals_324, (1600, 2432, 1, 1), (2432, 1, 1, 1))
    assert_size_stride(primals_325, (1600, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_326, (2176, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(primals_327, (1600, 2432, 1, 1), (2432, 1, 1, 1))
    assert_size_stride(primals_328, (1600, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_329, (2176, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(primals_330, (1600, 2560, 1, 1), (2560, 1, 1, 1))
    assert_size_stride(primals_331, (1600, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_332, (2176, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(primals_333, (1000, 2688, 1, 1), (2688, 1, 1, 1))
    assert_size_stride(primals_334, (1000, ), (1, ))
    assert_size_stride(primals_335, (), ())
    assert_size_stride(primals_336, (128, ), (1, ))
    assert_size_stride(primals_337, (128, ), (1, ))
    assert_size_stride(primals_338, (), ())
    assert_size_stride(primals_339, (128, ), (1, ))
    assert_size_stride(primals_340, (128, ), (1, ))
    assert_size_stride(primals_341, (), ())
    assert_size_stride(primals_342, (128, ), (1, ))
    assert_size_stride(primals_343, (128, ), (1, ))
    assert_size_stride(primals_344, (), ())
    assert_size_stride(primals_345, (200, ), (1, ))
    assert_size_stride(primals_346, (200, ), (1, ))
    assert_size_stride(primals_347, (), ())
    assert_size_stride(primals_348, (200, ), (1, ))
    assert_size_stride(primals_349, (200, ), (1, ))
    assert_size_stride(primals_350, (), ())
    assert_size_stride(primals_351, (316, ), (1, ))
    assert_size_stride(primals_352, (316, ), (1, ))
    assert_size_stride(primals_353, (), ())
    assert_size_stride(primals_354, (200, ), (1, ))
    assert_size_stride(primals_355, (200, ), (1, ))
    assert_size_stride(primals_356, (), ())
    assert_size_stride(primals_357, (200, ), (1, ))
    assert_size_stride(primals_358, (200, ), (1, ))
    assert_size_stride(primals_359, (), ())
    assert_size_stride(primals_360, (336, ), (1, ))
    assert_size_stride(primals_361, (336, ), (1, ))
    assert_size_stride(primals_362, (), ())
    assert_size_stride(primals_363, (200, ), (1, ))
    assert_size_stride(primals_364, (200, ), (1, ))
    assert_size_stride(primals_365, (), ())
    assert_size_stride(primals_366, (200, ), (1, ))
    assert_size_stride(primals_367, (200, ), (1, ))
    assert_size_stride(primals_368, (), ())
    assert_size_stride(primals_369, (356, ), (1, ))
    assert_size_stride(primals_370, (356, ), (1, ))
    assert_size_stride(primals_371, (), ())
    assert_size_stride(primals_372, (200, ), (1, ))
    assert_size_stride(primals_373, (200, ), (1, ))
    assert_size_stride(primals_374, (), ())
    assert_size_stride(primals_375, (200, ), (1, ))
    assert_size_stride(primals_376, (200, ), (1, ))
    assert_size_stride(primals_377, (), ())
    assert_size_stride(primals_378, (376, ), (1, ))
    assert_size_stride(primals_379, (376, ), (1, ))
    assert_size_stride(primals_380, (), ())
    assert_size_stride(primals_381, (376, ), (1, ))
    assert_size_stride(primals_382, (376, ), (1, ))
    assert_size_stride(primals_383, (), ())
    assert_size_stride(primals_384, (400, ), (1, ))
    assert_size_stride(primals_385, (400, ), (1, ))
    assert_size_stride(primals_386, (), ())
    assert_size_stride(primals_387, (400, ), (1, ))
    assert_size_stride(primals_388, (400, ), (1, ))
    assert_size_stride(primals_389, (), ())
    assert_size_stride(primals_390, (704, ), (1, ))
    assert_size_stride(primals_391, (704, ), (1, ))
    assert_size_stride(primals_392, (), ())
    assert_size_stride(primals_393, (400, ), (1, ))
    assert_size_stride(primals_394, (400, ), (1, ))
    assert_size_stride(primals_395, (), ())
    assert_size_stride(primals_396, (400, ), (1, ))
    assert_size_stride(primals_397, (400, ), (1, ))
    assert_size_stride(primals_398, (), ())
    assert_size_stride(primals_399, (768, ), (1, ))
    assert_size_stride(primals_400, (768, ), (1, ))
    assert_size_stride(primals_401, (), ())
    assert_size_stride(primals_402, (400, ), (1, ))
    assert_size_stride(primals_403, (400, ), (1, ))
    assert_size_stride(primals_404, (), ())
    assert_size_stride(primals_405, (400, ), (1, ))
    assert_size_stride(primals_406, (400, ), (1, ))
    assert_size_stride(primals_407, (), ())
    assert_size_stride(primals_408, (832, ), (1, ))
    assert_size_stride(primals_409, (832, ), (1, ))
    assert_size_stride(primals_410, (), ())
    assert_size_stride(primals_411, (400, ), (1, ))
    assert_size_stride(primals_412, (400, ), (1, ))
    assert_size_stride(primals_413, (), ())
    assert_size_stride(primals_414, (400, ), (1, ))
    assert_size_stride(primals_415, (400, ), (1, ))
    assert_size_stride(primals_416, (), ())
    assert_size_stride(primals_417, (896, ), (1, ))
    assert_size_stride(primals_418, (896, ), (1, ))
    assert_size_stride(primals_419, (), ())
    assert_size_stride(primals_420, (400, ), (1, ))
    assert_size_stride(primals_421, (400, ), (1, ))
    assert_size_stride(primals_422, (), ())
    assert_size_stride(primals_423, (400, ), (1, ))
    assert_size_stride(primals_424, (400, ), (1, ))
    assert_size_stride(primals_425, (), ())
    assert_size_stride(primals_426, (960, ), (1, ))
    assert_size_stride(primals_427, (960, ), (1, ))
    assert_size_stride(primals_428, (), ())
    assert_size_stride(primals_429, (400, ), (1, ))
    assert_size_stride(primals_430, (400, ), (1, ))
    assert_size_stride(primals_431, (), ())
    assert_size_stride(primals_432, (400, ), (1, ))
    assert_size_stride(primals_433, (400, ), (1, ))
    assert_size_stride(primals_434, (), ())
    assert_size_stride(primals_435, (1024, ), (1, ))
    assert_size_stride(primals_436, (1024, ), (1, ))
    assert_size_stride(primals_437, (), ())
    assert_size_stride(primals_438, (400, ), (1, ))
    assert_size_stride(primals_439, (400, ), (1, ))
    assert_size_stride(primals_440, (), ())
    assert_size_stride(primals_441, (400, ), (1, ))
    assert_size_stride(primals_442, (400, ), (1, ))
    assert_size_stride(primals_443, (), ())
    assert_size_stride(primals_444, (1088, ), (1, ))
    assert_size_stride(primals_445, (1088, ), (1, ))
    assert_size_stride(primals_446, (), ())
    assert_size_stride(primals_447, (400, ), (1, ))
    assert_size_stride(primals_448, (400, ), (1, ))
    assert_size_stride(primals_449, (), ())
    assert_size_stride(primals_450, (400, ), (1, ))
    assert_size_stride(primals_451, (400, ), (1, ))
    assert_size_stride(primals_452, (), ())
    assert_size_stride(primals_453, (1152, ), (1, ))
    assert_size_stride(primals_454, (1152, ), (1, ))
    assert_size_stride(primals_455, (), ())
    assert_size_stride(primals_456, (1152, ), (1, ))
    assert_size_stride(primals_457, (1152, ), (1, ))
    assert_size_stride(primals_458, (), ())
    assert_size_stride(primals_459, (800, ), (1, ))
    assert_size_stride(primals_460, (800, ), (1, ))
    assert_size_stride(primals_461, (), ())
    assert_size_stride(primals_462, (800, ), (1, ))
    assert_size_stride(primals_463, (800, ), (1, ))
    assert_size_stride(primals_464, (), ())
    assert_size_stride(primals_465, (1216, ), (1, ))
    assert_size_stride(primals_466, (1216, ), (1, ))
    assert_size_stride(primals_467, (), ())
    assert_size_stride(primals_468, (800, ), (1, ))
    assert_size_stride(primals_469, (800, ), (1, ))
    assert_size_stride(primals_470, (), ())
    assert_size_stride(primals_471, (800, ), (1, ))
    assert_size_stride(primals_472, (800, ), (1, ))
    assert_size_stride(primals_473, (), ())
    assert_size_stride(primals_474, (1280, ), (1, ))
    assert_size_stride(primals_475, (1280, ), (1, ))
    assert_size_stride(primals_476, (), ())
    assert_size_stride(primals_477, (800, ), (1, ))
    assert_size_stride(primals_478, (800, ), (1, ))
    assert_size_stride(primals_479, (), ())
    assert_size_stride(primals_480, (800, ), (1, ))
    assert_size_stride(primals_481, (800, ), (1, ))
    assert_size_stride(primals_482, (), ())
    assert_size_stride(primals_483, (1344, ), (1, ))
    assert_size_stride(primals_484, (1344, ), (1, ))
    assert_size_stride(primals_485, (), ())
    assert_size_stride(primals_486, (800, ), (1, ))
    assert_size_stride(primals_487, (800, ), (1, ))
    assert_size_stride(primals_488, (), ())
    assert_size_stride(primals_489, (800, ), (1, ))
    assert_size_stride(primals_490, (800, ), (1, ))
    assert_size_stride(primals_491, (), ())
    assert_size_stride(primals_492, (1408, ), (1, ))
    assert_size_stride(primals_493, (1408, ), (1, ))
    assert_size_stride(primals_494, (), ())
    assert_size_stride(primals_495, (800, ), (1, ))
    assert_size_stride(primals_496, (800, ), (1, ))
    assert_size_stride(primals_497, (), ())
    assert_size_stride(primals_498, (800, ), (1, ))
    assert_size_stride(primals_499, (800, ), (1, ))
    assert_size_stride(primals_500, (), ())
    assert_size_stride(primals_501, (1472, ), (1, ))
    assert_size_stride(primals_502, (1472, ), (1, ))
    assert_size_stride(primals_503, (), ())
    assert_size_stride(primals_504, (800, ), (1, ))
    assert_size_stride(primals_505, (800, ), (1, ))
    assert_size_stride(primals_506, (), ())
    assert_size_stride(primals_507, (800, ), (1, ))
    assert_size_stride(primals_508, (800, ), (1, ))
    assert_size_stride(primals_509, (), ())
    assert_size_stride(primals_510, (1536, ), (1, ))
    assert_size_stride(primals_511, (1536, ), (1, ))
    assert_size_stride(primals_512, (), ())
    assert_size_stride(primals_513, (800, ), (1, ))
    assert_size_stride(primals_514, (800, ), (1, ))
    assert_size_stride(primals_515, (), ())
    assert_size_stride(primals_516, (800, ), (1, ))
    assert_size_stride(primals_517, (800, ), (1, ))
    assert_size_stride(primals_518, (), ())
    assert_size_stride(primals_519, (1600, ), (1, ))
    assert_size_stride(primals_520, (1600, ), (1, ))
    assert_size_stride(primals_521, (), ())
    assert_size_stride(primals_522, (800, ), (1, ))
    assert_size_stride(primals_523, (800, ), (1, ))
    assert_size_stride(primals_524, (), ())
    assert_size_stride(primals_525, (800, ), (1, ))
    assert_size_stride(primals_526, (800, ), (1, ))
    assert_size_stride(primals_527, (), ())
    assert_size_stride(primals_528, (1664, ), (1, ))
    assert_size_stride(primals_529, (1664, ), (1, ))
    assert_size_stride(primals_530, (), ())
    assert_size_stride(primals_531, (800, ), (1, ))
    assert_size_stride(primals_532, (800, ), (1, ))
    assert_size_stride(primals_533, (), ())
    assert_size_stride(primals_534, (800, ), (1, ))
    assert_size_stride(primals_535, (800, ), (1, ))
    assert_size_stride(primals_536, (), ())
    assert_size_stride(primals_537, (1728, ), (1, ))
    assert_size_stride(primals_538, (1728, ), (1, ))
    assert_size_stride(primals_539, (), ())
    assert_size_stride(primals_540, (800, ), (1, ))
    assert_size_stride(primals_541, (800, ), (1, ))
    assert_size_stride(primals_542, (), ())
    assert_size_stride(primals_543, (800, ), (1, ))
    assert_size_stride(primals_544, (800, ), (1, ))
    assert_size_stride(primals_545, (), ())
    assert_size_stride(primals_546, (1792, ), (1, ))
    assert_size_stride(primals_547, (1792, ), (1, ))
    assert_size_stride(primals_548, (), ())
    assert_size_stride(primals_549, (800, ), (1, ))
    assert_size_stride(primals_550, (800, ), (1, ))
    assert_size_stride(primals_551, (), ())
    assert_size_stride(primals_552, (800, ), (1, ))
    assert_size_stride(primals_553, (800, ), (1, ))
    assert_size_stride(primals_554, (), ())
    assert_size_stride(primals_555, (1856, ), (1, ))
    assert_size_stride(primals_556, (1856, ), (1, ))
    assert_size_stride(primals_557, (), ())
    assert_size_stride(primals_558, (800, ), (1, ))
    assert_size_stride(primals_559, (800, ), (1, ))
    assert_size_stride(primals_560, (), ())
    assert_size_stride(primals_561, (800, ), (1, ))
    assert_size_stride(primals_562, (800, ), (1, ))
    assert_size_stride(primals_563, (), ())
    assert_size_stride(primals_564, (1920, ), (1, ))
    assert_size_stride(primals_565, (1920, ), (1, ))
    assert_size_stride(primals_566, (), ())
    assert_size_stride(primals_567, (800, ), (1, ))
    assert_size_stride(primals_568, (800, ), (1, ))
    assert_size_stride(primals_569, (), ())
    assert_size_stride(primals_570, (800, ), (1, ))
    assert_size_stride(primals_571, (800, ), (1, ))
    assert_size_stride(primals_572, (), ())
    assert_size_stride(primals_573, (1984, ), (1, ))
    assert_size_stride(primals_574, (1984, ), (1, ))
    assert_size_stride(primals_575, (), ())
    assert_size_stride(primals_576, (800, ), (1, ))
    assert_size_stride(primals_577, (800, ), (1, ))
    assert_size_stride(primals_578, (), ())
    assert_size_stride(primals_579, (800, ), (1, ))
    assert_size_stride(primals_580, (800, ), (1, ))
    assert_size_stride(primals_581, (), ())
    assert_size_stride(primals_582, (2048, ), (1, ))
    assert_size_stride(primals_583, (2048, ), (1, ))
    assert_size_stride(primals_584, (), ())
    assert_size_stride(primals_585, (800, ), (1, ))
    assert_size_stride(primals_586, (800, ), (1, ))
    assert_size_stride(primals_587, (), ())
    assert_size_stride(primals_588, (800, ), (1, ))
    assert_size_stride(primals_589, (800, ), (1, ))
    assert_size_stride(primals_590, (), ())
    assert_size_stride(primals_591, (2112, ), (1, ))
    assert_size_stride(primals_592, (2112, ), (1, ))
    assert_size_stride(primals_593, (), ())
    assert_size_stride(primals_594, (800, ), (1, ))
    assert_size_stride(primals_595, (800, ), (1, ))
    assert_size_stride(primals_596, (), ())
    assert_size_stride(primals_597, (800, ), (1, ))
    assert_size_stride(primals_598, (800, ), (1, ))
    assert_size_stride(primals_599, (), ())
    assert_size_stride(primals_600, (2176, ), (1, ))
    assert_size_stride(primals_601, (2176, ), (1, ))
    assert_size_stride(primals_602, (), ())
    assert_size_stride(primals_603, (800, ), (1, ))
    assert_size_stride(primals_604, (800, ), (1, ))
    assert_size_stride(primals_605, (), ())
    assert_size_stride(primals_606, (800, ), (1, ))
    assert_size_stride(primals_607, (800, ), (1, ))
    assert_size_stride(primals_608, (), ())
    assert_size_stride(primals_609, (2240, ), (1, ))
    assert_size_stride(primals_610, (2240, ), (1, ))
    assert_size_stride(primals_611, (), ())
    assert_size_stride(primals_612, (800, ), (1, ))
    assert_size_stride(primals_613, (800, ), (1, ))
    assert_size_stride(primals_614, (), ())
    assert_size_stride(primals_615, (800, ), (1, ))
    assert_size_stride(primals_616, (800, ), (1, ))
    assert_size_stride(primals_617, (), ())
    assert_size_stride(primals_618, (2304, ), (1, ))
    assert_size_stride(primals_619, (2304, ), (1, ))
    assert_size_stride(primals_620, (), ())
    assert_size_stride(primals_621, (800, ), (1, ))
    assert_size_stride(primals_622, (800, ), (1, ))
    assert_size_stride(primals_623, (), ())
    assert_size_stride(primals_624, (800, ), (1, ))
    assert_size_stride(primals_625, (800, ), (1, ))
    assert_size_stride(primals_626, (), ())
    assert_size_stride(primals_627, (2368, ), (1, ))
    assert_size_stride(primals_628, (2368, ), (1, ))
    assert_size_stride(primals_629, (), ())
    assert_size_stride(primals_630, (800, ), (1, ))
    assert_size_stride(primals_631, (800, ), (1, ))
    assert_size_stride(primals_632, (), ())
    assert_size_stride(primals_633, (800, ), (1, ))
    assert_size_stride(primals_634, (800, ), (1, ))
    assert_size_stride(primals_635, (), ())
    assert_size_stride(primals_636, (2432, ), (1, ))
    assert_size_stride(primals_637, (2432, ), (1, ))
    assert_size_stride(primals_638, (), ())
    assert_size_stride(primals_639, (2432, ), (1, ))
    assert_size_stride(primals_640, (2432, ), (1, ))
    assert_size_stride(primals_641, (), ())
    assert_size_stride(primals_642, (1600, ), (1, ))
    assert_size_stride(primals_643, (1600, ), (1, ))
    assert_size_stride(primals_644, (), ())
    assert_size_stride(primals_645, (1600, ), (1, ))
    assert_size_stride(primals_646, (1600, ), (1, ))
    assert_size_stride(primals_647, (), ())
    assert_size_stride(primals_648, (2432, ), (1, ))
    assert_size_stride(primals_649, (2432, ), (1, ))
    assert_size_stride(primals_650, (), ())
    assert_size_stride(primals_651, (1600, ), (1, ))
    assert_size_stride(primals_652, (1600, ), (1, ))
    assert_size_stride(primals_653, (), ())
    assert_size_stride(primals_654, (1600, ), (1, ))
    assert_size_stride(primals_655, (1600, ), (1, ))
    assert_size_stride(primals_656, (), ())
    assert_size_stride(primals_657, (2560, ), (1, ))
    assert_size_stride(primals_658, (2560, ), (1, ))
    assert_size_stride(primals_659, (), ())
    assert_size_stride(primals_660, (1600, ), (1, ))
    assert_size_stride(primals_661, (1600, ), (1, ))
    assert_size_stride(primals_662, (), ())
    assert_size_stride(primals_663, (1600, ), (1, ))
    assert_size_stride(primals_664, (1600, ), (1, ))
    assert_size_stride(primals_665, (), ())
    assert_size_stride(primals_666, (2688, ), (1, ))
    assert_size_stride(primals_667, (2688, ), (1, ))
    assert_size_stride(primals_668, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_668, primals_223, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 128, 112, 112), (1605632, 12544, 112, 1))
        buf1 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_cuda_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_0.run(buf0, buf1, buf2, buf3, 512, 25088, grid=grid(512), stream=stream0)
        buf4 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf7 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_1.run(buf1, buf2, buf3, primals_336, primals_337, buf4, buf5, buf7, primals_336, primals_337, 128, 4, grid=grid(128), stream=stream0)
        del primals_336
        del primals_337
        buf8 = empty((8, 128, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1, x_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_2.run(buf0, buf4, buf5, primals_1, primals_2, buf8, 12845056, grid=grid(12845056), stream=stream0)
        del primals_2
        buf9 = empty((8, 128, 56, 56), device='cuda', dtype=torch.float32)
        buf10 = empty((8, 128, 56, 56), device='cuda', dtype=torch.int64)
        # Source Nodes: [x_in], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_3.run(buf8, buf9, buf10, 3211264, grid=grid(3211264), stream=stream0)
        buf11 = buf3; del buf3  # reuse
        buf12 = buf2; del buf2  # reuse
        buf13 = buf1; del buf1  # reuse
        # Source Nodes: [x_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf9, buf11, buf12, buf13, 512, 6272, grid=grid(512), stream=stream0)
        buf14 = buf5; del buf5  # reuse
        buf15 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf17 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_5, x_8], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf11, buf12, buf13, primals_339, primals_342, primals_340, primals_343, buf14, buf15, buf17, primals_339, primals_342, primals_340, primals_343, 128, 4, grid=grid(128), stream=stream0)
        del buf11
        del buf12
        del buf13
        del primals_339
        del primals_340
        del primals_342
        del primals_343
        buf18 = empty((8, 128, 56, 56), device='cuda', dtype=torch.float32)
        buf20 = empty((8, 128, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_10, x_5, x_7, x_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_6.run(buf9, buf14, buf15, primals_3, primals_4, primals_5, primals_6, buf18, buf20, 3211264, grid=grid(3211264), stream=stream0)
        del buf15
        del primals_4
        del primals_6
        # Source Nodes: [x_s], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, primals_224, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (8, 296, 56, 56), (928256, 3136, 56, 1))
        # Source Nodes: [x_in_1], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_225, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (8, 200, 56, 56), (627200, 3136, 56, 1))
        buf22 = empty_strided((1, 200, 1, 1, 4), (800, 1, 800, 800, 200), device='cuda', dtype=torch.float32)
        buf23 = empty_strided((1, 200, 1, 1, 4), (800, 1, 800, 800, 200), device='cuda', dtype=torch.float32)
        buf24 = empty_strided((1, 200, 1, 1, 4), (800, 1, 800, 800, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf21, buf22, buf23, buf24, 800, 6272, grid=grid(800), stream=stream0)
        buf25 = empty_strided((1, 200, 1, 1), (200, 1, 200, 200), device='cuda', dtype=torch.float32)
        buf26 = empty_strided((1, 200, 1, 1), (200, 1, 200, 200), device='cuda', dtype=torch.float32)
        buf28 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_8.run(buf22, buf23, buf24, primals_345, primals_346, buf25, buf26, buf28, primals_345, primals_346, 200, 4, grid=grid(200), stream=stream0)
        del primals_345
        del primals_346
        buf29 = empty((8, 200, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11, x_13], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_9.run(buf21, buf25, buf26, primals_7, primals_8, buf29, 5017600, grid=grid(5017600), stream=stream0)
        del primals_8
        # Source Nodes: [x_in_2], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_226, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf30, (8, 200, 56, 56), (627200, 3136, 56, 1))
        buf31 = buf24; del buf24  # reuse
        buf32 = buf23; del buf23  # reuse
        buf33 = buf22; del buf22  # reuse
        # Source Nodes: [x_14], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf30, buf31, buf32, buf33, 800, 6272, grid=grid(800), stream=stream0)
        buf34 = buf26; del buf26  # reuse
        buf35 = empty_strided((1, 200, 1, 1), (200, 1, 200, 200), device='cuda', dtype=torch.float32)
        buf37 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_14], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_8.run(buf31, buf32, buf33, primals_348, primals_349, buf34, buf35, buf37, primals_348, primals_349, 200, 4, grid=grid(200), stream=stream0)
        del primals_348
        del primals_349
        buf38 = empty((8, 200, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_14, x_16], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_9.run(buf30, buf34, buf35, primals_9, primals_10, buf38, 5017600, grid=grid(5017600), stream=stream0)
        del primals_10
        # Source Nodes: [x_in_3], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, primals_227, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (8, 276, 56, 56), (865536, 3136, 56, 1))
        buf40 = empty((8, 316, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_138], Original ATen: [aten.cat]
        triton_poi_fused_cat_10.run(buf19, buf39, buf40, 7927808, grid=grid(7927808), stream=stream0)
        buf41 = empty_strided((1, 316, 1, 1), (316, 1, 316, 316), device='cuda', dtype=torch.float32)
        buf42 = empty_strided((1, 316, 1, 1), (316, 1, 316, 316), device='cuda', dtype=torch.float32)
        buf44 = empty((316, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf40, primals_351, primals_352, buf41, buf42, buf44, primals_351, primals_352, 316, 25088, grid=grid(316), stream=stream0)
        del primals_351
        del primals_352
        buf45 = empty((8, 316, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17, x_19], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_12.run(buf40, buf41, buf42, primals_11, primals_12, buf45, 7927808, grid=grid(7927808), stream=stream0)
        del buf42
        del primals_12
        # Source Nodes: [x_in_5], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_228, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (8, 200, 56, 56), (627200, 3136, 56, 1))
        buf47 = buf33; del buf33  # reuse
        buf48 = buf32; del buf32  # reuse
        buf49 = buf31; del buf31  # reuse
        # Source Nodes: [x_20], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf46, buf47, buf48, buf49, 800, 6272, grid=grid(800), stream=stream0)
        buf50 = buf35; del buf35  # reuse
        buf51 = empty_strided((1, 200, 1, 1), (200, 1, 200, 200), device='cuda', dtype=torch.float32)
        buf53 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_20], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_8.run(buf47, buf48, buf49, primals_354, primals_355, buf50, buf51, buf53, primals_354, primals_355, 200, 4, grid=grid(200), stream=stream0)
        del primals_354
        del primals_355
        buf54 = empty((8, 200, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_20, x_22], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_9.run(buf46, buf50, buf51, primals_13, primals_14, buf54, 5017600, grid=grid(5017600), stream=stream0)
        del primals_14
        # Source Nodes: [x_in_6], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, primals_229, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf55, (8, 200, 56, 56), (627200, 3136, 56, 1))
        buf56 = buf49; del buf49  # reuse
        buf57 = buf48; del buf48  # reuse
        buf58 = buf47; del buf47  # reuse
        # Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf55, buf56, buf57, buf58, 800, 6272, grid=grid(800), stream=stream0)
        buf59 = buf51; del buf51  # reuse
        buf60 = empty_strided((1, 200, 1, 1), (200, 1, 200, 200), device='cuda', dtype=torch.float32)
        buf62 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_23], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_8.run(buf56, buf57, buf58, primals_357, primals_358, buf59, buf60, buf62, primals_357, primals_358, 200, 4, grid=grid(200), stream=stream0)
        del primals_357
        del primals_358
        buf63 = empty((8, 200, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_23, x_25], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_9.run(buf55, buf59, buf60, primals_15, primals_16, buf63, 5017600, grid=grid(5017600), stream=stream0)
        del primals_16
        # Source Nodes: [x_in_7], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_230, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (8, 276, 56, 56), (865536, 3136, 56, 1))
        buf65 = empty((8, 336, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_136], Original ATen: [aten.cat]
        triton_poi_fused_cat_13.run(buf19, buf39, buf64, buf65, 8429568, grid=grid(8429568), stream=stream0)
        buf66 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf67 = empty_strided((1, 336, 1, 1), (336, 1, 336, 336), device='cuda', dtype=torch.float32)
        buf69 = empty((336, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_26], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_14.run(buf65, primals_360, primals_361, buf66, buf67, buf69, primals_360, primals_361, 336, 25088, grid=grid(336), stream=stream0)
        del primals_360
        del primals_361
        buf70 = empty((8, 336, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_26, x_28], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_15.run(buf65, buf66, buf67, primals_17, primals_18, buf70, 8429568, grid=grid(8429568), stream=stream0)
        del buf67
        del primals_18
        # Source Nodes: [x_in_9], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_231, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (8, 200, 56, 56), (627200, 3136, 56, 1))
        buf72 = buf58; del buf58  # reuse
        buf73 = buf57; del buf57  # reuse
        buf74 = buf56; del buf56  # reuse
        # Source Nodes: [x_29], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf71, buf72, buf73, buf74, 800, 6272, grid=grid(800), stream=stream0)
        buf75 = buf60; del buf60  # reuse
        buf76 = empty_strided((1, 200, 1, 1), (200, 1, 200, 200), device='cuda', dtype=torch.float32)
        buf78 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_29], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_8.run(buf72, buf73, buf74, primals_363, primals_364, buf75, buf76, buf78, primals_363, primals_364, 200, 4, grid=grid(200), stream=stream0)
        del primals_363
        del primals_364
        buf79 = empty((8, 200, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_29, x_31], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_9.run(buf71, buf75, buf76, primals_19, primals_20, buf79, 5017600, grid=grid(5017600), stream=stream0)
        del primals_20
        # Source Nodes: [x_in_10], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_232, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf80, (8, 200, 56, 56), (627200, 3136, 56, 1))
        buf81 = buf74; del buf74  # reuse
        buf82 = buf73; del buf73  # reuse
        buf83 = buf72; del buf72  # reuse
        # Source Nodes: [x_32], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf80, buf81, buf82, buf83, 800, 6272, grid=grid(800), stream=stream0)
        buf84 = buf76; del buf76  # reuse
        buf85 = empty_strided((1, 200, 1, 1), (200, 1, 200, 200), device='cuda', dtype=torch.float32)
        buf87 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_8.run(buf81, buf82, buf83, primals_366, primals_367, buf84, buf85, buf87, primals_366, primals_367, 200, 4, grid=grid(200), stream=stream0)
        del primals_366
        del primals_367
        buf88 = empty((8, 200, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32, x_34], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_9.run(buf80, buf84, buf85, primals_21, primals_22, buf88, 5017600, grid=grid(5017600), stream=stream0)
        del primals_22
        # Source Nodes: [x_in_11], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, primals_233, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (8, 276, 56, 56), (865536, 3136, 56, 1))
        buf90 = empty((8, 356, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_134], Original ATen: [aten.cat]
        triton_poi_fused_cat_16.run(buf19, buf39, buf64, buf89, buf90, 8931328, grid=grid(8931328), stream=stream0)
        buf91 = empty_strided((1, 356, 1, 1), (356, 1, 356, 356), device='cuda', dtype=torch.float32)
        buf92 = empty_strided((1, 356, 1, 1), (356, 1, 356, 356), device='cuda', dtype=torch.float32)
        buf94 = empty((356, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_35], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_17.run(buf90, primals_369, primals_370, buf91, buf92, buf94, primals_369, primals_370, 356, 25088, grid=grid(356), stream=stream0)
        del primals_369
        del primals_370
        buf95 = empty((8, 356, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_35, x_37], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_18.run(buf90, buf91, buf92, primals_23, primals_24, buf95, 8931328, grid=grid(8931328), stream=stream0)
        del buf92
        del primals_24
        # Source Nodes: [x_in_13], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, primals_234, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (8, 200, 56, 56), (627200, 3136, 56, 1))
        buf97 = buf83; del buf83  # reuse
        buf98 = buf82; del buf82  # reuse
        buf99 = buf81; del buf81  # reuse
        # Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf96, buf97, buf98, buf99, 800, 6272, grid=grid(800), stream=stream0)
        buf100 = buf85; del buf85  # reuse
        buf101 = empty_strided((1, 200, 1, 1), (200, 1, 200, 200), device='cuda', dtype=torch.float32)
        buf103 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_38], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_8.run(buf97, buf98, buf99, primals_372, primals_373, buf100, buf101, buf103, primals_372, primals_373, 200, 4, grid=grid(200), stream=stream0)
        del primals_372
        del primals_373
        buf104 = empty((8, 200, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_38, x_40], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_9.run(buf96, buf100, buf101, primals_25, primals_26, buf104, 5017600, grid=grid(5017600), stream=stream0)
        del primals_26
        # Source Nodes: [x_in_14], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, primals_235, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf105, (8, 200, 56, 56), (627200, 3136, 56, 1))
        buf106 = buf99; del buf99  # reuse
        buf107 = buf98; del buf98  # reuse
        buf108 = buf97; del buf97  # reuse
        # Source Nodes: [x_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf105, buf106, buf107, buf108, 800, 6272, grid=grid(800), stream=stream0)
        buf109 = buf101; del buf101  # reuse
        buf110 = empty_strided((1, 200, 1, 1), (200, 1, 200, 200), device='cuda', dtype=torch.float32)
        buf112 = empty((200, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_8.run(buf106, buf107, buf108, primals_375, primals_376, buf109, buf110, buf112, primals_375, primals_376, 200, 4, grid=grid(200), stream=stream0)
        del primals_375
        del primals_376
        buf113 = empty((8, 200, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_41, x_43], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_9.run(buf105, buf109, buf110, primals_27, primals_28, buf113, 5017600, grid=grid(5017600), stream=stream0)
        del buf110
        del primals_28
        # Source Nodes: [x_in_15], Original ATen: [aten.convolution]
        buf114 = extern_kernels.convolution(buf113, primals_236, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (8, 276, 56, 56), (865536, 3136, 56, 1))
        buf117 = empty((8, 376, 56, 56), device='cuda', dtype=torch.float32)
        buf115 = reinterpret_tensor(buf117, (8, 120, 56, 56), (1179136, 3136, 56, 1), 802816)  # alias
        # Source Nodes: [cat_133], Original ATen: [aten.cat]
        triton_poi_fused_cat_19.run(buf19, buf39, buf64, buf89, buf114, buf115, 3010560, grid=grid(3010560), stream=stream0)
        buf116 = reinterpret_tensor(buf117, (8, 256, 56, 56), (1179136, 3136, 56, 1), 0)  # alias
        # Source Nodes: [resid_3, x_s1_1, x_s1_2, x_s1_3], Original ATen: [aten.add]
        triton_poi_fused_add_20.run(buf19, buf39, buf64, buf89, buf114, buf116, 6422528, grid=grid(6422528), stream=stream0)
        del buf114
        del buf19
        del buf39
        del buf64
        del buf89
        buf118 = empty_strided((1, 376, 1, 1), (376, 1, 376, 376), device='cuda', dtype=torch.float32)
        buf119 = empty_strided((1, 376, 1, 1), (376, 1, 376, 376), device='cuda', dtype=torch.float32)
        buf121 = empty((376, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44, x_47], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf117, primals_378, primals_381, primals_379, primals_382, buf118, buf119, buf121, primals_378, primals_381, primals_379, primals_382, 376, 25088, grid=grid(376), stream=stream0)
        del primals_378
        del primals_379
        del primals_381
        del primals_382
        buf122 = empty((8, 376, 56, 56), device='cuda', dtype=torch.float32)
        buf124 = empty((8, 376, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44, x_46, x_47, x_49], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_22.run(buf117, buf118, buf119, primals_29, primals_30, primals_31, primals_32, buf122, buf124, 9433088, grid=grid(9433088), stream=stream0)
        del buf119
        del primals_30
        del primals_32
        # Source Nodes: [x_s_1], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, primals_237, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (8, 640, 28, 28), (501760, 784, 28, 1))
        # Source Nodes: [x_in_17], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, primals_238, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (8, 400, 56, 56), (1254400, 3136, 56, 1))
        buf126 = empty_strided((1, 400, 1, 1), (400, 1, 400, 400), device='cuda', dtype=torch.float32)
        buf127 = empty_strided((1, 400, 1, 1), (400, 1, 400, 400), device='cuda', dtype=torch.float32)
        buf129 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_50], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_23.run(buf125, primals_384, primals_385, buf126, buf127, buf129, primals_384, primals_385, 400, 25088, grid=grid(400), stream=stream0)
        del primals_384
        del primals_385
        buf130 = empty((8, 400, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_50, x_52], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_24.run(buf125, buf126, buf127, primals_33, primals_34, buf130, 10035200, grid=grid(10035200), stream=stream0)
        del primals_34
        # Source Nodes: [x_in_18], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, primals_239, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf131, (8, 400, 28, 28), (313600, 784, 28, 1))
        buf132 = buf127; del buf127  # reuse
        buf133 = empty_strided((1, 400, 1, 1), (400, 1, 400, 400), device='cuda', dtype=torch.float32)
        buf135 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_53], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf131, primals_387, primals_388, buf132, buf133, buf135, primals_387, primals_388, 400, 6272, grid=grid(400), stream=stream0)
        del primals_387
        del primals_388
        buf136 = empty((8, 400, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_53, x_55], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf131, buf132, buf133, primals_35, primals_36, buf136, 2508800, grid=grid(2508800), stream=stream0)
        del primals_36
        # Source Nodes: [x_in_19], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf136, primals_240, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (8, 576, 28, 28), (451584, 784, 28, 1))
        buf138 = empty((8, 704, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_130], Original ATen: [aten.cat]
        triton_poi_fused_cat_27.run(buf123, buf137, buf138, 4415488, grid=grid(4415488), stream=stream0)
        buf139 = empty_strided((1, 704, 1, 1), (704, 1, 704, 704), device='cuda', dtype=torch.float32)
        buf140 = empty_strided((1, 704, 1, 1), (704, 1, 704, 704), device='cuda', dtype=torch.float32)
        buf142 = empty((704, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_56], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_28.run(buf138, primals_390, primals_391, buf139, buf140, buf142, primals_390, primals_391, 704, 6272, grid=grid(704), stream=stream0)
        del primals_390
        del primals_391
        buf143 = empty((8, 704, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_56, x_58], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_29.run(buf138, buf139, buf140, primals_37, primals_38, buf143, 4415488, grid=grid(4415488), stream=stream0)
        del buf140
        del primals_38
        # Source Nodes: [x_in_21], Original ATen: [aten.convolution]
        buf144 = extern_kernels.convolution(buf143, primals_241, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (8, 400, 28, 28), (313600, 784, 28, 1))
        buf145 = buf133; del buf133  # reuse
        buf146 = empty_strided((1, 400, 1, 1), (400, 1, 400, 400), device='cuda', dtype=torch.float32)
        buf148 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_59], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf144, primals_393, primals_394, buf145, buf146, buf148, primals_393, primals_394, 400, 6272, grid=grid(400), stream=stream0)
        del primals_393
        del primals_394
        buf149 = empty((8, 400, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_59, x_61], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf144, buf145, buf146, primals_39, primals_40, buf149, 2508800, grid=grid(2508800), stream=stream0)
        del primals_40
        # Source Nodes: [x_in_22], Original ATen: [aten.convolution]
        buf150 = extern_kernels.convolution(buf149, primals_242, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf150, (8, 400, 28, 28), (313600, 784, 28, 1))
        buf151 = buf146; del buf146  # reuse
        buf152 = empty_strided((1, 400, 1, 1), (400, 1, 400, 400), device='cuda', dtype=torch.float32)
        buf154 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_62], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf150, primals_396, primals_397, buf151, buf152, buf154, primals_396, primals_397, 400, 6272, grid=grid(400), stream=stream0)
        del primals_396
        del primals_397
        buf155 = empty((8, 400, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_62, x_64], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf150, buf151, buf152, primals_41, primals_42, buf155, 2508800, grid=grid(2508800), stream=stream0)
        del primals_42
        # Source Nodes: [x_in_23], Original ATen: [aten.convolution]
        buf156 = extern_kernels.convolution(buf155, primals_243, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (8, 576, 28, 28), (451584, 784, 28, 1))
        buf157 = empty((8, 768, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_128], Original ATen: [aten.cat]
        triton_poi_fused_cat_30.run(buf123, buf137, buf156, buf157, 4816896, grid=grid(4816896), stream=stream0)
        buf158 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf159 = empty_strided((1, 768, 1, 1), (768, 1, 768, 768), device='cuda', dtype=torch.float32)
        buf161 = empty((768, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_65], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_31.run(buf157, primals_399, primals_400, buf158, buf159, buf161, primals_399, primals_400, 768, 6272, grid=grid(768), stream=stream0)
        del primals_399
        del primals_400
        buf162 = empty((8, 768, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_65, x_67], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_32.run(buf157, buf158, buf159, primals_43, primals_44, buf162, 4816896, grid=grid(4816896), stream=stream0)
        del buf159
        del primals_44
        # Source Nodes: [x_in_25], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, primals_244, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (8, 400, 28, 28), (313600, 784, 28, 1))
        buf164 = buf152; del buf152  # reuse
        buf165 = empty_strided((1, 400, 1, 1), (400, 1, 400, 400), device='cuda', dtype=torch.float32)
        buf167 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf163, primals_402, primals_403, buf164, buf165, buf167, primals_402, primals_403, 400, 6272, grid=grid(400), stream=stream0)
        del primals_402
        del primals_403
        buf168 = empty((8, 400, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_68, x_70], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf163, buf164, buf165, primals_45, primals_46, buf168, 2508800, grid=grid(2508800), stream=stream0)
        del primals_46
        # Source Nodes: [x_in_26], Original ATen: [aten.convolution]
        buf169 = extern_kernels.convolution(buf168, primals_245, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf169, (8, 400, 28, 28), (313600, 784, 28, 1))
        buf170 = buf165; del buf165  # reuse
        buf171 = empty_strided((1, 400, 1, 1), (400, 1, 400, 400), device='cuda', dtype=torch.float32)
        buf173 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_71], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf169, primals_405, primals_406, buf170, buf171, buf173, primals_405, primals_406, 400, 6272, grid=grid(400), stream=stream0)
        del primals_405
        del primals_406
        buf174 = empty((8, 400, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_71, x_73], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf169, buf170, buf171, primals_47, primals_48, buf174, 2508800, grid=grid(2508800), stream=stream0)
        del primals_48
        # Source Nodes: [x_in_27], Original ATen: [aten.convolution]
        buf175 = extern_kernels.convolution(buf174, primals_246, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (8, 576, 28, 28), (451584, 784, 28, 1))
        buf176 = empty((8, 832, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_126], Original ATen: [aten.cat]
        triton_poi_fused_cat_33.run(buf123, buf137, buf156, buf175, buf176, 5218304, grid=grid(5218304), stream=stream0)
        buf177 = empty_strided((1, 832, 1, 1), (832, 1, 832, 832), device='cuda', dtype=torch.float32)
        buf178 = empty_strided((1, 832, 1, 1), (832, 1, 832, 832), device='cuda', dtype=torch.float32)
        buf180 = empty((832, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_34.run(buf176, primals_408, primals_409, buf177, buf178, buf180, primals_408, primals_409, 832, 6272, grid=grid(832), stream=stream0)
        del primals_408
        del primals_409
        buf181 = empty((8, 832, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74, x_76], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_35.run(buf176, buf177, buf178, primals_49, primals_50, buf181, 5218304, grid=grid(5218304), stream=stream0)
        del buf178
        del primals_50
        # Source Nodes: [x_in_29], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, primals_247, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (8, 400, 28, 28), (313600, 784, 28, 1))
        buf183 = buf171; del buf171  # reuse
        buf184 = empty_strided((1, 400, 1, 1), (400, 1, 400, 400), device='cuda', dtype=torch.float32)
        buf186 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_77], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf182, primals_411, primals_412, buf183, buf184, buf186, primals_411, primals_412, 400, 6272, grid=grid(400), stream=stream0)
        del primals_411
        del primals_412
        buf187 = empty((8, 400, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_77, x_79], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf182, buf183, buf184, primals_51, primals_52, buf187, 2508800, grid=grid(2508800), stream=stream0)
        del primals_52
        # Source Nodes: [x_in_30], Original ATen: [aten.convolution]
        buf188 = extern_kernels.convolution(buf187, primals_248, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf188, (8, 400, 28, 28), (313600, 784, 28, 1))
        buf189 = buf184; del buf184  # reuse
        buf190 = empty_strided((1, 400, 1, 1), (400, 1, 400, 400), device='cuda', dtype=torch.float32)
        buf192 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_80], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf188, primals_414, primals_415, buf189, buf190, buf192, primals_414, primals_415, 400, 6272, grid=grid(400), stream=stream0)
        del primals_414
        del primals_415
        buf193 = empty((8, 400, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_80, x_82], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf188, buf189, buf190, primals_53, primals_54, buf193, 2508800, grid=grid(2508800), stream=stream0)
        del primals_54
        # Source Nodes: [x_in_31], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf193, primals_249, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (8, 576, 28, 28), (451584, 784, 28, 1))
        buf197 = empty((8, 896, 28, 28), device='cuda', dtype=torch.float32)
        buf195 = reinterpret_tensor(buf197, (8, 512, 28, 28), (702464, 784, 28, 1), 0)  # alias
        # Source Nodes: [x_s1_5, x_s1_6, x_s1_7, x_s1_8], Original ATen: [aten.add]
        triton_poi_fused_add_36.run(buf123, buf137, buf156, buf175, buf194, buf195, 3211264, grid=grid(3211264), stream=stream0)
        buf196 = reinterpret_tensor(buf197, (8, 384, 28, 28), (702464, 784, 28, 1), 401408)  # alias
        # Source Nodes: [cat_125], Original ATen: [aten.cat]
        triton_poi_fused_cat_37.run(buf123, buf137, buf156, buf175, buf194, buf196, 2408448, grid=grid(2408448), stream=stream0)
        del buf123
        del buf137
        del buf156
        del buf175
        del buf194
        buf198 = empty_strided((1, 896, 1, 1), (896, 1, 896, 896), device='cuda', dtype=torch.float32)
        buf199 = empty_strided((1, 896, 1, 1), (896, 1, 896, 896), device='cuda', dtype=torch.float32)
        buf201 = empty((896, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_83], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_38.run(buf197, primals_417, primals_418, buf198, buf199, buf201, primals_417, primals_418, 896, 6272, grid=grid(896), stream=stream0)
        del primals_417
        del primals_418
        buf202 = empty((8, 896, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_83, x_85], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_39.run(buf197, buf198, buf199, primals_55, primals_56, buf202, 5619712, grid=grid(5619712), stream=stream0)
        del buf199
        del primals_56
        # Source Nodes: [x_in_33], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, primals_250, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (8, 400, 28, 28), (313600, 784, 28, 1))
        buf204 = buf190; del buf190  # reuse
        buf205 = empty_strided((1, 400, 1, 1), (400, 1, 400, 400), device='cuda', dtype=torch.float32)
        buf207 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_86], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf203, primals_420, primals_421, buf204, buf205, buf207, primals_420, primals_421, 400, 6272, grid=grid(400), stream=stream0)
        del primals_420
        del primals_421
        buf208 = empty((8, 400, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_86, x_88], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf203, buf204, buf205, primals_57, primals_58, buf208, 2508800, grid=grid(2508800), stream=stream0)
        del primals_58
        # Source Nodes: [x_in_34], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf208, primals_251, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf209, (8, 400, 28, 28), (313600, 784, 28, 1))
        buf210 = buf205; del buf205  # reuse
        buf211 = empty_strided((1, 400, 1, 1), (400, 1, 400, 400), device='cuda', dtype=torch.float32)
        buf213 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf209, primals_423, primals_424, buf210, buf211, buf213, primals_423, primals_424, 400, 6272, grid=grid(400), stream=stream0)
        del primals_423
        del primals_424
        buf214 = empty((8, 400, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_89, x_91], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf209, buf210, buf211, primals_59, primals_60, buf214, 2508800, grid=grid(2508800), stream=stream0)
        del primals_60
        # Source Nodes: [x_in_35], Original ATen: [aten.convolution]
        buf215 = extern_kernels.convolution(buf214, primals_252, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf215, (8, 576, 28, 28), (451584, 784, 28, 1))
        buf216 = empty((8, 960, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_122], Original ATen: [aten.cat]
        triton_poi_fused_cat_40.run(buf195, buf215, buf196, buf216, 6021120, grid=grid(6021120), stream=stream0)
        buf217 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf218 = empty_strided((1, 960, 1, 1), (960, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf220 = empty((960, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_92], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_41.run(buf216, primals_426, primals_427, buf217, buf218, buf220, primals_426, primals_427, 960, 6272, grid=grid(960), stream=stream0)
        del primals_426
        del primals_427
        buf221 = empty((8, 960, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_92, x_94], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_42.run(buf216, buf217, buf218, primals_61, primals_62, buf221, 6021120, grid=grid(6021120), stream=stream0)
        del buf218
        del primals_62
        # Source Nodes: [x_in_37], Original ATen: [aten.convolution]
        buf222 = extern_kernels.convolution(buf221, primals_253, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf222, (8, 400, 28, 28), (313600, 784, 28, 1))
        buf223 = buf211; del buf211  # reuse
        buf224 = empty_strided((1, 400, 1, 1), (400, 1, 400, 400), device='cuda', dtype=torch.float32)
        buf226 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_95], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf222, primals_429, primals_430, buf223, buf224, buf226, primals_429, primals_430, 400, 6272, grid=grid(400), stream=stream0)
        del primals_429
        del primals_430
        buf227 = empty((8, 400, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_95, x_97], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf222, buf223, buf224, primals_63, primals_64, buf227, 2508800, grid=grid(2508800), stream=stream0)
        del primals_64
        # Source Nodes: [x_in_38], Original ATen: [aten.convolution]
        buf228 = extern_kernels.convolution(buf227, primals_254, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf228, (8, 400, 28, 28), (313600, 784, 28, 1))
        buf229 = buf224; del buf224  # reuse
        buf230 = empty_strided((1, 400, 1, 1), (400, 1, 400, 400), device='cuda', dtype=torch.float32)
        buf232 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_98], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf228, primals_432, primals_433, buf229, buf230, buf232, primals_432, primals_433, 400, 6272, grid=grid(400), stream=stream0)
        del primals_432
        del primals_433
        buf233 = empty((8, 400, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_100, x_98], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf228, buf229, buf230, primals_65, primals_66, buf233, 2508800, grid=grid(2508800), stream=stream0)
        del primals_66
        # Source Nodes: [x_in_39], Original ATen: [aten.convolution]
        buf234 = extern_kernels.convolution(buf233, primals_255, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf234, (8, 576, 28, 28), (451584, 784, 28, 1))
        buf235 = empty((8, 1024, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_120], Original ATen: [aten.cat]
        triton_poi_fused_cat_43.run(buf195, buf215, buf234, buf196, buf235, 6422528, grid=grid(6422528), stream=stream0)
        buf236 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf237 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf239 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_101], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_44.run(buf235, primals_435, primals_436, buf236, buf237, buf239, primals_435, primals_436, 1024, 6272, grid=grid(1024), stream=stream0)
        del primals_435
        del primals_436
        buf240 = empty((8, 1024, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_101, x_103], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_45.run(buf235, buf236, buf237, primals_67, primals_68, buf240, 6422528, grid=grid(6422528), stream=stream0)
        del buf237
        del primals_68
        # Source Nodes: [x_in_41], Original ATen: [aten.convolution]
        buf241 = extern_kernels.convolution(buf240, primals_256, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (8, 400, 28, 28), (313600, 784, 28, 1))
        buf242 = buf230; del buf230  # reuse
        buf243 = empty_strided((1, 400, 1, 1), (400, 1, 400, 400), device='cuda', dtype=torch.float32)
        buf245 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_104], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf241, primals_438, primals_439, buf242, buf243, buf245, primals_438, primals_439, 400, 6272, grid=grid(400), stream=stream0)
        del primals_438
        del primals_439
        buf246 = empty((8, 400, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_104, x_106], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf241, buf242, buf243, primals_69, primals_70, buf246, 2508800, grid=grid(2508800), stream=stream0)
        del primals_70
        # Source Nodes: [x_in_42], Original ATen: [aten.convolution]
        buf247 = extern_kernels.convolution(buf246, primals_257, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf247, (8, 400, 28, 28), (313600, 784, 28, 1))
        buf248 = buf243; del buf243  # reuse
        buf249 = empty_strided((1, 400, 1, 1), (400, 1, 400, 400), device='cuda', dtype=torch.float32)
        buf251 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_107], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf247, primals_441, primals_442, buf248, buf249, buf251, primals_441, primals_442, 400, 6272, grid=grid(400), stream=stream0)
        del primals_441
        del primals_442
        buf252 = empty((8, 400, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_107, x_109], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf247, buf248, buf249, primals_71, primals_72, buf252, 2508800, grid=grid(2508800), stream=stream0)
        del primals_72
        # Source Nodes: [x_in_43], Original ATen: [aten.convolution]
        buf253 = extern_kernels.convolution(buf252, primals_258, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (8, 576, 28, 28), (451584, 784, 28, 1))
        buf254 = empty((8, 1088, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_118], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf195, buf215, buf234, buf253, buf196, buf254, 6823936, grid=grid(6823936), stream=stream0)
        buf255 = empty_strided((1, 1088, 1, 1), (1088, 1, 1088, 1088), device='cuda', dtype=torch.float32)
        buf256 = empty_strided((1, 1088, 1, 1), (1088, 1, 1088, 1088), device='cuda', dtype=torch.float32)
        buf258 = empty((1088, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_110], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_47.run(buf254, primals_444, primals_445, buf255, buf256, buf258, primals_444, primals_445, 1088, 6272, grid=grid(1088), stream=stream0)
        del primals_444
        del primals_445
        buf259 = empty((8, 1088, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_110, x_112], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_48.run(buf254, buf255, buf256, primals_73, primals_74, buf259, 6823936, grid=grid(6823936), stream=stream0)
        del buf256
        del primals_74
        # Source Nodes: [x_in_45], Original ATen: [aten.convolution]
        buf260 = extern_kernels.convolution(buf259, primals_259, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf260, (8, 400, 28, 28), (313600, 784, 28, 1))
        buf261 = buf249; del buf249  # reuse
        buf262 = empty_strided((1, 400, 1, 1), (400, 1, 400, 400), device='cuda', dtype=torch.float32)
        buf264 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_113], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf260, primals_447, primals_448, buf261, buf262, buf264, primals_447, primals_448, 400, 6272, grid=grid(400), stream=stream0)
        del primals_447
        del primals_448
        buf265 = empty((8, 400, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_113, x_115], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf260, buf261, buf262, primals_75, primals_76, buf265, 2508800, grid=grid(2508800), stream=stream0)
        del primals_76
        # Source Nodes: [x_in_46], Original ATen: [aten.convolution]
        buf266 = extern_kernels.convolution(buf265, primals_260, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf266, (8, 400, 28, 28), (313600, 784, 28, 1))
        buf267 = buf262; del buf262  # reuse
        buf268 = empty_strided((1, 400, 1, 1), (400, 1, 400, 400), device='cuda', dtype=torch.float32)
        buf270 = empty((400, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_116], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_25.run(buf266, primals_450, primals_451, buf267, buf268, buf270, primals_450, primals_451, 400, 6272, grid=grid(400), stream=stream0)
        del primals_450
        del primals_451
        buf271 = empty((8, 400, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_116, x_118], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf266, buf267, buf268, primals_77, primals_78, buf271, 2508800, grid=grid(2508800), stream=stream0)
        del buf268
        del primals_78
        # Source Nodes: [x_in_47], Original ATen: [aten.convolution]
        buf272 = extern_kernels.convolution(buf271, primals_261, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf272, (8, 576, 28, 28), (451584, 784, 28, 1))
        buf275 = empty((8, 1152, 28, 28), device='cuda', dtype=torch.float32)
        buf273 = reinterpret_tensor(buf275, (8, 640, 28, 28), (903168, 784, 28, 1), 401408)  # alias
        # Source Nodes: [cat_117], Original ATen: [aten.cat]
        triton_poi_fused_cat_49.run(buf196, buf215, buf234, buf253, buf272, buf273, 4014080, grid=grid(4014080), stream=stream0)
        buf274 = reinterpret_tensor(buf275, (8, 512, 28, 28), (903168, 784, 28, 1), 0)  # alias
        # Source Nodes: [resid_11, x_s1_10, x_s1_11, x_s1_9], Original ATen: [aten.add]
        triton_poi_fused_add_50.run(buf195, buf215, buf234, buf253, buf272, buf274, 3211264, grid=grid(3211264), stream=stream0)
        del buf215
        del buf234
        buf276 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf277 = empty_strided((1, 1152, 1, 1), (1152, 1, 1152, 1152), device='cuda', dtype=torch.float32)
        buf279 = empty((1152, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_119, x_122], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_51.run(buf275, primals_453, primals_456, primals_454, primals_457, buf276, buf277, buf279, primals_453, primals_456, primals_454, primals_457, 1152, 6272, grid=grid(1152), stream=stream0)
        del primals_453
        del primals_454
        del primals_456
        del primals_457
        buf280 = empty((8, 1152, 28, 28), device='cuda', dtype=torch.float32)
        buf282 = empty((8, 1152, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_119, x_121, x_122, x_124], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_52.run(buf275, buf276, buf277, primals_79, primals_80, primals_81, primals_82, buf280, buf282, 7225344, grid=grid(7225344), stream=stream0)
        del buf277
        del primals_80
        del primals_82
        # Source Nodes: [x_s_2], Original ATen: [aten.convolution]
        buf281 = extern_kernels.convolution(buf280, primals_262, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf281, (8, 1152, 14, 14), (225792, 196, 14, 1))
        # Source Nodes: [x_in_49], Original ATen: [aten.convolution]
        buf283 = extern_kernels.convolution(buf282, primals_263, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (8, 800, 28, 28), (627200, 784, 28, 1))
        buf284 = reinterpret_tensor(buf108, (1, 800, 1, 1), (800, 1, 800, 800), 0); del buf108  # reuse
        buf285 = reinterpret_tensor(buf107, (1, 800, 1, 1), (800, 1, 800, 800), 0); del buf107  # reuse
        buf287 = reinterpret_tensor(buf106, (800, ), (1, ), 0); del buf106  # reuse
        # Source Nodes: [x_125], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_53.run(buf283, primals_459, primals_460, buf284, buf285, buf287, primals_459, primals_460, 800, 6272, grid=grid(800), stream=stream0)
        del primals_459
        del primals_460
        buf288 = empty((8, 800, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_125, x_127], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_54.run(buf283, buf284, buf285, primals_83, primals_84, buf288, 5017600, grid=grid(5017600), stream=stream0)
        del primals_84
        # Source Nodes: [x_in_50], Original ATen: [aten.convolution]
        buf289 = extern_kernels.convolution(buf288, primals_264, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf289, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf290 = buf285; del buf285  # reuse
        buf291 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf293 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_128], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf289, primals_462, primals_463, buf290, buf291, buf293, primals_462, primals_463, 800, 1568, grid=grid(800), stream=stream0)
        del primals_462
        del primals_463
        buf294 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_128, x_130], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf289, buf290, buf291, primals_85, primals_86, buf294, 1254400, grid=grid(1254400), stream=stream0)
        del primals_86
        # Source Nodes: [x_in_51], Original ATen: [aten.convolution]
        buf295 = extern_kernels.convolution(buf294, primals_265, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf295, (8, 1088, 14, 14), (213248, 196, 14, 1))
        buf296 = empty((8, 1216, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_114], Original ATen: [aten.cat]
        triton_poi_fused_cat_57.run(buf281, buf295, buf296, 1906688, grid=grid(1906688), stream=stream0)
        buf297 = empty_strided((1, 1216, 1, 1), (1216, 1, 1216, 1216), device='cuda', dtype=torch.float32)
        buf298 = empty_strided((1, 1216, 1, 1), (1216, 1, 1216, 1216), device='cuda', dtype=torch.float32)
        buf300 = empty((1216, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_131], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_58.run(buf296, primals_465, primals_466, buf297, buf298, buf300, primals_465, primals_466, 1216, 1568, grid=grid(1216), stream=stream0)
        del primals_465
        del primals_466
        buf301 = empty((8, 1216, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_131, x_133], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_59.run(buf296, buf297, buf298, primals_87, primals_88, buf301, 1906688, grid=grid(1906688), stream=stream0)
        del buf298
        del primals_88
        # Source Nodes: [x_in_53], Original ATen: [aten.convolution]
        buf302 = extern_kernels.convolution(buf301, primals_266, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf302, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf303 = buf291; del buf291  # reuse
        buf304 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf306 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_134], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf302, primals_468, primals_469, buf303, buf304, buf306, primals_468, primals_469, 800, 1568, grid=grid(800), stream=stream0)
        del primals_468
        del primals_469
        buf307 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_134, x_136], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf302, buf303, buf304, primals_89, primals_90, buf307, 1254400, grid=grid(1254400), stream=stream0)
        del primals_90
        # Source Nodes: [x_in_54], Original ATen: [aten.convolution]
        buf308 = extern_kernels.convolution(buf307, primals_267, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf308, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf309 = buf304; del buf304  # reuse
        buf310 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf312 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_137], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf308, primals_471, primals_472, buf309, buf310, buf312, primals_471, primals_472, 800, 1568, grid=grid(800), stream=stream0)
        del primals_471
        del primals_472
        buf313 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_137, x_139], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf308, buf309, buf310, primals_91, primals_92, buf313, 1254400, grid=grid(1254400), stream=stream0)
        del primals_92
        # Source Nodes: [x_in_55], Original ATen: [aten.convolution]
        buf314 = extern_kernels.convolution(buf313, primals_268, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf314, (8, 1088, 14, 14), (213248, 196, 14, 1))
        buf315 = empty((8, 1280, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_112], Original ATen: [aten.cat]
        triton_poi_fused_cat_60.run(buf281, buf295, buf314, buf315, 2007040, grid=grid(2007040), stream=stream0)
        buf316 = empty_strided((1, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda', dtype=torch.float32)
        buf317 = empty_strided((1, 1280, 1, 1), (1280, 1, 1280, 1280), device='cuda', dtype=torch.float32)
        buf319 = empty((1280, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_140], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_61.run(buf315, primals_474, primals_475, buf316, buf317, buf319, primals_474, primals_475, 1280, 1568, grid=grid(1280), stream=stream0)
        del primals_474
        del primals_475
        buf320 = empty((8, 1280, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_140, x_142], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_62.run(buf315, buf316, buf317, primals_93, primals_94, buf320, 2007040, grid=grid(2007040), stream=stream0)
        del buf317
        del primals_94
        # Source Nodes: [x_in_57], Original ATen: [aten.convolution]
        buf321 = extern_kernels.convolution(buf320, primals_269, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf321, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf322 = buf310; del buf310  # reuse
        buf323 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf325 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_143], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf321, primals_477, primals_478, buf322, buf323, buf325, primals_477, primals_478, 800, 1568, grid=grid(800), stream=stream0)
        del primals_477
        del primals_478
        buf326 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_143, x_145], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf321, buf322, buf323, primals_95, primals_96, buf326, 1254400, grid=grid(1254400), stream=stream0)
        del primals_96
        # Source Nodes: [x_in_58], Original ATen: [aten.convolution]
        buf327 = extern_kernels.convolution(buf326, primals_270, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf327, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf328 = buf323; del buf323  # reuse
        buf329 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf331 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_146], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf327, primals_480, primals_481, buf328, buf329, buf331, primals_480, primals_481, 800, 1568, grid=grid(800), stream=stream0)
        del primals_480
        del primals_481
        buf332 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_146, x_148], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf327, buf328, buf329, primals_97, primals_98, buf332, 1254400, grid=grid(1254400), stream=stream0)
        del primals_98
        # Source Nodes: [x_in_59], Original ATen: [aten.convolution]
        buf333 = extern_kernels.convolution(buf332, primals_271, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf333, (8, 1088, 14, 14), (213248, 196, 14, 1))
        buf334 = empty((8, 1344, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_110], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf281, buf295, buf314, buf333, buf334, 2107392, grid=grid(2107392), stream=stream0)
        buf335 = empty_strided((1, 1344, 1, 1), (1344, 1, 1344, 1344), device='cuda', dtype=torch.float32)
        buf336 = empty_strided((1, 1344, 1, 1), (1344, 1, 1344, 1344), device='cuda', dtype=torch.float32)
        buf338 = empty((1344, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_149], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_64.run(buf334, primals_483, primals_484, buf335, buf336, buf338, primals_483, primals_484, 1344, 1568, grid=grid(1344), stream=stream0)
        del primals_483
        del primals_484
        buf339 = empty((8, 1344, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_149, x_151], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_65.run(buf334, buf335, buf336, primals_99, primals_100, buf339, 2107392, grid=grid(2107392), stream=stream0)
        del buf336
        del primals_100
        # Source Nodes: [x_in_61], Original ATen: [aten.convolution]
        buf340 = extern_kernels.convolution(buf339, primals_272, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf340, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf341 = buf329; del buf329  # reuse
        buf342 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf344 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_152], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf340, primals_486, primals_487, buf341, buf342, buf344, primals_486, primals_487, 800, 1568, grid=grid(800), stream=stream0)
        del primals_486
        del primals_487
        buf345 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_152, x_154], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf340, buf341, buf342, primals_101, primals_102, buf345, 1254400, grid=grid(1254400), stream=stream0)
        del primals_102
        # Source Nodes: [x_in_62], Original ATen: [aten.convolution]
        buf346 = extern_kernels.convolution(buf345, primals_273, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf346, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf347 = buf342; del buf342  # reuse
        buf348 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf350 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_155], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf346, primals_489, primals_490, buf347, buf348, buf350, primals_489, primals_490, 800, 1568, grid=grid(800), stream=stream0)
        del primals_489
        del primals_490
        buf351 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_155, x_157], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf346, buf347, buf348, primals_103, primals_104, buf351, 1254400, grid=grid(1254400), stream=stream0)
        del primals_104
        # Source Nodes: [x_in_63], Original ATen: [aten.convolution]
        buf352 = extern_kernels.convolution(buf351, primals_274, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf352, (8, 1088, 14, 14), (213248, 196, 14, 1))
        buf355 = empty((8, 1408, 14, 14), device='cuda', dtype=torch.float32)
        buf353 = reinterpret_tensor(buf355, (8, 1024, 14, 14), (275968, 196, 14, 1), 0)  # alias
        # Source Nodes: [x_s1_13, x_s1_14, x_s1_15, x_s1_16], Original ATen: [aten.add]
        triton_poi_fused_add_66.run(buf281, buf295, buf314, buf333, buf352, buf353, 1605632, grid=grid(1605632), stream=stream0)
        buf354 = reinterpret_tensor(buf355, (8, 384, 14, 14), (275968, 196, 14, 1), 200704)  # alias
        # Source Nodes: [cat_109], Original ATen: [aten.cat]
        triton_poi_fused_cat_67.run(buf281, buf295, buf314, buf333, buf352, buf354, 602112, grid=grid(602112), stream=stream0)
        del buf281
        del buf295
        del buf314
        del buf333
        del buf352
        buf356 = empty_strided((1, 1408, 1, 1), (1408, 1, 1408, 1408), device='cuda', dtype=torch.float32)
        buf357 = empty_strided((1, 1408, 1, 1), (1408, 1, 1408, 1408), device='cuda', dtype=torch.float32)
        buf359 = empty((1408, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_158], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_68.run(buf355, primals_492, primals_493, buf356, buf357, buf359, primals_492, primals_493, 1408, 1568, grid=grid(1408), stream=stream0)
        del primals_492
        del primals_493
        buf360 = empty((8, 1408, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_158, x_160], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_69.run(buf355, buf356, buf357, primals_105, primals_106, buf360, 2207744, grid=grid(2207744), stream=stream0)
        del buf357
        del primals_106
        # Source Nodes: [x_in_65], Original ATen: [aten.convolution]
        buf361 = extern_kernels.convolution(buf360, primals_275, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf361, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf362 = buf348; del buf348  # reuse
        buf363 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf365 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_161], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf361, primals_495, primals_496, buf362, buf363, buf365, primals_495, primals_496, 800, 1568, grid=grid(800), stream=stream0)
        del primals_495
        del primals_496
        buf366 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_161, x_163], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf361, buf362, buf363, primals_107, primals_108, buf366, 1254400, grid=grid(1254400), stream=stream0)
        del primals_108
        # Source Nodes: [x_in_66], Original ATen: [aten.convolution]
        buf367 = extern_kernels.convolution(buf366, primals_276, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf367, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf368 = buf363; del buf363  # reuse
        buf369 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf371 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_164], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf367, primals_498, primals_499, buf368, buf369, buf371, primals_498, primals_499, 800, 1568, grid=grid(800), stream=stream0)
        del primals_498
        del primals_499
        buf372 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_164, x_166], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf367, buf368, buf369, primals_109, primals_110, buf372, 1254400, grid=grid(1254400), stream=stream0)
        del primals_110
        # Source Nodes: [x_in_67], Original ATen: [aten.convolution]
        buf373 = extern_kernels.convolution(buf372, primals_277, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf373, (8, 1088, 14, 14), (213248, 196, 14, 1))
        buf374 = empty((8, 1472, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_106], Original ATen: [aten.cat]
        triton_poi_fused_cat_70.run(buf353, buf373, buf354, buf374, 2308096, grid=grid(2308096), stream=stream0)
        buf375 = empty_strided((1, 1472, 1, 1), (1472, 1, 1472, 1472), device='cuda', dtype=torch.float32)
        buf376 = empty_strided((1, 1472, 1, 1), (1472, 1, 1472, 1472), device='cuda', dtype=torch.float32)
        buf378 = empty((1472, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_167], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_71.run(buf374, primals_501, primals_502, buf375, buf376, buf378, primals_501, primals_502, 1472, 1568, grid=grid(1472), stream=stream0)
        del primals_501
        del primals_502
        buf379 = empty((8, 1472, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_167, x_169], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_72.run(buf374, buf375, buf376, primals_111, primals_112, buf379, 2308096, grid=grid(2308096), stream=stream0)
        del buf376
        del primals_112
        # Source Nodes: [x_in_69], Original ATen: [aten.convolution]
        buf380 = extern_kernels.convolution(buf379, primals_278, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf380, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf381 = buf369; del buf369  # reuse
        buf382 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf384 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_170], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf380, primals_504, primals_505, buf381, buf382, buf384, primals_504, primals_505, 800, 1568, grid=grid(800), stream=stream0)
        del primals_504
        del primals_505
        buf385 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_170, x_172], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf380, buf381, buf382, primals_113, primals_114, buf385, 1254400, grid=grid(1254400), stream=stream0)
        del primals_114
        # Source Nodes: [x_in_70], Original ATen: [aten.convolution]
        buf386 = extern_kernels.convolution(buf385, primals_279, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf386, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf387 = buf382; del buf382  # reuse
        buf388 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf390 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_173], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf386, primals_507, primals_508, buf387, buf388, buf390, primals_507, primals_508, 800, 1568, grid=grid(800), stream=stream0)
        del primals_507
        del primals_508
        buf391 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_173, x_175], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf386, buf387, buf388, primals_115, primals_116, buf391, 1254400, grid=grid(1254400), stream=stream0)
        del primals_116
        # Source Nodes: [x_in_71], Original ATen: [aten.convolution]
        buf392 = extern_kernels.convolution(buf391, primals_280, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf392, (8, 1088, 14, 14), (213248, 196, 14, 1))
        buf393 = empty((8, 1536, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_104], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf353, buf373, buf392, buf354, buf393, 2408448, grid=grid(2408448), stream=stream0)
        buf394 = empty_strided((1, 1536, 1, 1), (1536, 1, 1536, 1536), device='cuda', dtype=torch.float32)
        buf395 = empty_strided((1, 1536, 1, 1), (1536, 1, 1536, 1536), device='cuda', dtype=torch.float32)
        buf397 = empty((1536, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_176], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_74.run(buf393, primals_510, primals_511, buf394, buf395, buf397, primals_510, primals_511, 1536, 1568, grid=grid(1536), stream=stream0)
        del primals_510
        del primals_511
        buf398 = empty((8, 1536, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_176, x_178], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_75.run(buf393, buf394, buf395, primals_117, primals_118, buf398, 2408448, grid=grid(2408448), stream=stream0)
        del buf395
        del primals_118
        # Source Nodes: [x_in_73], Original ATen: [aten.convolution]
        buf399 = extern_kernels.convolution(buf398, primals_281, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf399, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf400 = buf388; del buf388  # reuse
        buf401 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf403 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_179], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf399, primals_513, primals_514, buf400, buf401, buf403, primals_513, primals_514, 800, 1568, grid=grid(800), stream=stream0)
        del primals_513
        del primals_514
        buf404 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_179, x_181], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf399, buf400, buf401, primals_119, primals_120, buf404, 1254400, grid=grid(1254400), stream=stream0)
        del primals_120
        # Source Nodes: [x_in_74], Original ATen: [aten.convolution]
        buf405 = extern_kernels.convolution(buf404, primals_282, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf405, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf406 = buf401; del buf401  # reuse
        buf407 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf409 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_182], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf405, primals_516, primals_517, buf406, buf407, buf409, primals_516, primals_517, 800, 1568, grid=grid(800), stream=stream0)
        del primals_516
        del primals_517
        buf410 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_182, x_184], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf405, buf406, buf407, primals_121, primals_122, buf410, 1254400, grid=grid(1254400), stream=stream0)
        del primals_122
        # Source Nodes: [x_in_75], Original ATen: [aten.convolution]
        buf411 = extern_kernels.convolution(buf410, primals_283, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf411, (8, 1088, 14, 14), (213248, 196, 14, 1))
        buf412 = empty((8, 1600, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_102], Original ATen: [aten.cat]
        triton_poi_fused_cat_76.run(buf353, buf373, buf392, buf411, buf354, buf412, 2508800, grid=grid(2508800), stream=stream0)
        buf413 = empty_strided((1, 1600, 1, 1), (1600, 1, 1600, 1600), device='cuda', dtype=torch.float32)
        buf414 = empty_strided((1, 1600, 1, 1), (1600, 1, 1600, 1600), device='cuda', dtype=torch.float32)
        buf416 = empty((1600, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_185], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_77.run(buf412, primals_519, primals_520, buf413, buf414, buf416, primals_519, primals_520, 1600, 1568, grid=grid(1600), stream=stream0)
        del primals_519
        del primals_520
        buf417 = empty((8, 1600, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_185, x_187], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_78.run(buf412, buf413, buf414, primals_123, primals_124, buf417, 2508800, grid=grid(2508800), stream=stream0)
        del primals_124
        # Source Nodes: [x_in_77], Original ATen: [aten.convolution]
        buf418 = extern_kernels.convolution(buf417, primals_284, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf418, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf419 = buf407; del buf407  # reuse
        buf420 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf422 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_188], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf418, primals_522, primals_523, buf419, buf420, buf422, primals_522, primals_523, 800, 1568, grid=grid(800), stream=stream0)
        del primals_522
        del primals_523
        buf423 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_188, x_190], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf418, buf419, buf420, primals_125, primals_126, buf423, 1254400, grid=grid(1254400), stream=stream0)
        del primals_126
        # Source Nodes: [x_in_78], Original ATen: [aten.convolution]
        buf424 = extern_kernels.convolution(buf423, primals_285, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf424, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf425 = buf420; del buf420  # reuse
        buf426 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf428 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_191], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf424, primals_525, primals_526, buf425, buf426, buf428, primals_525, primals_526, 800, 1568, grid=grid(800), stream=stream0)
        del primals_525
        del primals_526
        buf429 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_191, x_193], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf424, buf425, buf426, primals_127, primals_128, buf429, 1254400, grid=grid(1254400), stream=stream0)
        del primals_128
        # Source Nodes: [x_in_79], Original ATen: [aten.convolution]
        buf430 = extern_kernels.convolution(buf429, primals_286, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf430, (8, 1088, 14, 14), (213248, 196, 14, 1))
        buf433 = empty((8, 1664, 14, 14), device='cuda', dtype=torch.float32)
        buf431 = reinterpret_tensor(buf433, (8, 1024, 14, 14), (326144, 196, 14, 1), 0)  # alias
        # Source Nodes: [x_s1_17, x_s1_18, x_s1_19, x_s1_20], Original ATen: [aten.add]
        triton_poi_fused_add_79.run(buf353, buf373, buf392, buf411, buf430, buf431, 1605632, grid=grid(1605632), stream=stream0)
        buf432 = reinterpret_tensor(buf433, (8, 640, 14, 14), (326144, 196, 14, 1), 200704)  # alias
        # Source Nodes: [cat_101], Original ATen: [aten.cat]
        triton_poi_fused_cat_80.run(buf354, buf373, buf392, buf411, buf430, buf432, 1003520, grid=grid(1003520), stream=stream0)
        del buf373
        del buf392
        del buf411
        del buf430
        buf434 = empty_strided((1, 1664, 1, 1), (1664, 1, 1664, 1664), device='cuda', dtype=torch.float32)
        buf435 = empty_strided((1, 1664, 1, 1), (1664, 1, 1664, 1664), device='cuda', dtype=torch.float32)
        buf437 = empty((1664, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_194], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_81.run(buf433, primals_528, primals_529, buf434, buf435, buf437, primals_528, primals_529, 1664, 1568, grid=grid(1664), stream=stream0)
        del primals_528
        del primals_529
        buf438 = empty((8, 1664, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_194, x_196], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_82.run(buf433, buf434, buf435, primals_129, primals_130, buf438, 2609152, grid=grid(2609152), stream=stream0)
        del buf435
        del primals_130
        # Source Nodes: [x_in_81], Original ATen: [aten.convolution]
        buf439 = extern_kernels.convolution(buf438, primals_287, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf439, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf440 = buf426; del buf426  # reuse
        buf441 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf443 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_197], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf439, primals_531, primals_532, buf440, buf441, buf443, primals_531, primals_532, 800, 1568, grid=grid(800), stream=stream0)
        del primals_531
        del primals_532
        buf444 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_197, x_199], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf439, buf440, buf441, primals_131, primals_132, buf444, 1254400, grid=grid(1254400), stream=stream0)
        del primals_132
        # Source Nodes: [x_in_82], Original ATen: [aten.convolution]
        buf445 = extern_kernels.convolution(buf444, primals_288, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf445, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf446 = buf441; del buf441  # reuse
        buf447 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf449 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_200], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf445, primals_534, primals_535, buf446, buf447, buf449, primals_534, primals_535, 800, 1568, grid=grid(800), stream=stream0)
        del primals_534
        del primals_535
        buf450 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_200, x_202], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf445, buf446, buf447, primals_133, primals_134, buf450, 1254400, grid=grid(1254400), stream=stream0)
        del primals_134
        # Source Nodes: [x_in_83], Original ATen: [aten.convolution]
        buf451 = extern_kernels.convolution(buf450, primals_289, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf451, (8, 1088, 14, 14), (213248, 196, 14, 1))
        buf452 = empty((8, 1728, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_98], Original ATen: [aten.cat]
        triton_poi_fused_cat_83.run(buf431, buf451, buf432, buf452, 2709504, grid=grid(2709504), stream=stream0)
        buf453 = empty_strided((1, 1728, 1, 1), (1728, 1, 1728, 1728), device='cuda', dtype=torch.float32)
        buf454 = empty_strided((1, 1728, 1, 1), (1728, 1, 1728, 1728), device='cuda', dtype=torch.float32)
        buf456 = empty((1728, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_203], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_84.run(buf452, primals_537, primals_538, buf453, buf454, buf456, primals_537, primals_538, 1728, 1568, grid=grid(1728), stream=stream0)
        del primals_537
        del primals_538
        buf457 = empty((8, 1728, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_203, x_205], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_85.run(buf452, buf453, buf454, primals_135, primals_136, buf457, 2709504, grid=grid(2709504), stream=stream0)
        del buf454
        del primals_136
        # Source Nodes: [x_in_85], Original ATen: [aten.convolution]
        buf458 = extern_kernels.convolution(buf457, primals_290, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf458, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf459 = buf447; del buf447  # reuse
        buf460 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf462 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_206], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf458, primals_540, primals_541, buf459, buf460, buf462, primals_540, primals_541, 800, 1568, grid=grid(800), stream=stream0)
        del primals_540
        del primals_541
        buf463 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_206, x_208], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf458, buf459, buf460, primals_137, primals_138, buf463, 1254400, grid=grid(1254400), stream=stream0)
        del primals_138
        # Source Nodes: [x_in_86], Original ATen: [aten.convolution]
        buf464 = extern_kernels.convolution(buf463, primals_291, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf464, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf465 = buf460; del buf460  # reuse
        buf466 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf468 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_209], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf464, primals_543, primals_544, buf465, buf466, buf468, primals_543, primals_544, 800, 1568, grid=grid(800), stream=stream0)
        del primals_543
        del primals_544
        buf469 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_209, x_211], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf464, buf465, buf466, primals_139, primals_140, buf469, 1254400, grid=grid(1254400), stream=stream0)
        del primals_140
        # Source Nodes: [x_in_87], Original ATen: [aten.convolution]
        buf470 = extern_kernels.convolution(buf469, primals_292, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf470, (8, 1088, 14, 14), (213248, 196, 14, 1))
        buf471 = empty((8, 1792, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_96], Original ATen: [aten.cat]
        triton_poi_fused_cat_86.run(buf431, buf451, buf470, buf432, buf471, 2809856, grid=grid(2809856), stream=stream0)
        buf472 = empty_strided((1, 1792, 1, 1), (1792, 1, 1792, 1792), device='cuda', dtype=torch.float32)
        buf473 = empty_strided((1, 1792, 1, 1), (1792, 1, 1792, 1792), device='cuda', dtype=torch.float32)
        buf475 = empty((1792, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_212], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_87.run(buf471, primals_546, primals_547, buf472, buf473, buf475, primals_546, primals_547, 1792, 1568, grid=grid(1792), stream=stream0)
        del primals_546
        del primals_547
        buf476 = empty((8, 1792, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_212, x_214], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_88.run(buf471, buf472, buf473, primals_141, primals_142, buf476, 2809856, grid=grid(2809856), stream=stream0)
        del buf473
        del primals_142
        # Source Nodes: [x_in_89], Original ATen: [aten.convolution]
        buf477 = extern_kernels.convolution(buf476, primals_293, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf477, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf478 = buf466; del buf466  # reuse
        buf479 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf481 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_215], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf477, primals_549, primals_550, buf478, buf479, buf481, primals_549, primals_550, 800, 1568, grid=grid(800), stream=stream0)
        del primals_549
        del primals_550
        buf482 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_215, x_217], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf477, buf478, buf479, primals_143, primals_144, buf482, 1254400, grid=grid(1254400), stream=stream0)
        del primals_144
        # Source Nodes: [x_in_90], Original ATen: [aten.convolution]
        buf483 = extern_kernels.convolution(buf482, primals_294, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf483, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf484 = buf479; del buf479  # reuse
        buf485 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf487 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_218], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf483, primals_552, primals_553, buf484, buf485, buf487, primals_552, primals_553, 800, 1568, grid=grid(800), stream=stream0)
        del primals_552
        del primals_553
        buf488 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_218, x_220], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf483, buf484, buf485, primals_145, primals_146, buf488, 1254400, grid=grid(1254400), stream=stream0)
        del primals_146
        # Source Nodes: [x_in_91], Original ATen: [aten.convolution]
        buf489 = extern_kernels.convolution(buf488, primals_295, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf489, (8, 1088, 14, 14), (213248, 196, 14, 1))
        buf490 = empty((8, 1856, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_94], Original ATen: [aten.cat]
        triton_poi_fused_cat_89.run(buf431, buf451, buf470, buf489, buf432, buf490, 2910208, grid=grid(2910208), stream=stream0)
        buf491 = empty_strided((1, 1856, 1, 1), (1856, 1, 1856, 1856), device='cuda', dtype=torch.float32)
        buf492 = empty_strided((1, 1856, 1, 1), (1856, 1, 1856, 1856), device='cuda', dtype=torch.float32)
        buf494 = empty((1856, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_221], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_90.run(buf490, primals_555, primals_556, buf491, buf492, buf494, primals_555, primals_556, 1856, 1568, grid=grid(1856), stream=stream0)
        del primals_555
        del primals_556
        buf495 = empty((8, 1856, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_221, x_223], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_91.run(buf490, buf491, buf492, primals_147, primals_148, buf495, 2910208, grid=grid(2910208), stream=stream0)
        del buf492
        del primals_148
        # Source Nodes: [x_in_93], Original ATen: [aten.convolution]
        buf496 = extern_kernels.convolution(buf495, primals_296, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf496, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf497 = buf485; del buf485  # reuse
        buf498 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf500 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_224], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf496, primals_558, primals_559, buf497, buf498, buf500, primals_558, primals_559, 800, 1568, grid=grid(800), stream=stream0)
        del primals_558
        del primals_559
        buf501 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_224, x_226], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf496, buf497, buf498, primals_149, primals_150, buf501, 1254400, grid=grid(1254400), stream=stream0)
        del primals_150
        # Source Nodes: [x_in_94], Original ATen: [aten.convolution]
        buf502 = extern_kernels.convolution(buf501, primals_297, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf502, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf503 = buf498; del buf498  # reuse
        buf504 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf506 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_227], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf502, primals_561, primals_562, buf503, buf504, buf506, primals_561, primals_562, 800, 1568, grid=grid(800), stream=stream0)
        del primals_561
        del primals_562
        buf507 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_227, x_229], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf502, buf503, buf504, primals_151, primals_152, buf507, 1254400, grid=grid(1254400), stream=stream0)
        del primals_152
        # Source Nodes: [x_in_95], Original ATen: [aten.convolution]
        buf508 = extern_kernels.convolution(buf507, primals_298, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf508, (8, 1088, 14, 14), (213248, 196, 14, 1))
        buf511 = empty((8, 1920, 14, 14), device='cuda', dtype=torch.float32)
        buf509 = reinterpret_tensor(buf511, (8, 1024, 14, 14), (376320, 196, 14, 1), 0)  # alias
        # Source Nodes: [x_s1_21, x_s1_22, x_s1_23, x_s1_24], Original ATen: [aten.add]
        triton_poi_fused_add_92.run(buf431, buf451, buf470, buf489, buf508, buf509, 1605632, grid=grid(1605632), stream=stream0)
        buf510 = reinterpret_tensor(buf511, (8, 896, 14, 14), (376320, 196, 14, 1), 200704)  # alias
        # Source Nodes: [cat_93], Original ATen: [aten.cat]
        triton_poi_fused_cat_93.run(buf432, buf451, buf470, buf489, buf508, buf510, 1404928, grid=grid(1404928), stream=stream0)
        del buf451
        del buf470
        del buf489
        del buf508
        buf512 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf513 = empty_strided((1, 1920, 1, 1), (1920, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf515 = empty((1920, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_230], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_94.run(buf511, primals_564, primals_565, buf512, buf513, buf515, primals_564, primals_565, 1920, 1568, grid=grid(1920), stream=stream0)
        del primals_564
        del primals_565
        buf516 = empty((8, 1920, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_230, x_232], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_95.run(buf511, buf512, buf513, primals_153, primals_154, buf516, 3010560, grid=grid(3010560), stream=stream0)
        del buf513
        del primals_154
        # Source Nodes: [x_in_97], Original ATen: [aten.convolution]
        buf517 = extern_kernels.convolution(buf516, primals_299, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf517, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf518 = buf504; del buf504  # reuse
        buf519 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf521 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_233], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf517, primals_567, primals_568, buf518, buf519, buf521, primals_567, primals_568, 800, 1568, grid=grid(800), stream=stream0)
        del primals_567
        del primals_568
        buf522 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_233, x_235], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf517, buf518, buf519, primals_155, primals_156, buf522, 1254400, grid=grid(1254400), stream=stream0)
        del primals_156
        # Source Nodes: [x_in_98], Original ATen: [aten.convolution]
        buf523 = extern_kernels.convolution(buf522, primals_300, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf523, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf524 = buf519; del buf519  # reuse
        buf525 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf527 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_236], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf523, primals_570, primals_571, buf524, buf525, buf527, primals_570, primals_571, 800, 1568, grid=grid(800), stream=stream0)
        del primals_570
        del primals_571
        buf528 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_236, x_238], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf523, buf524, buf525, primals_157, primals_158, buf528, 1254400, grid=grid(1254400), stream=stream0)
        del primals_158
        # Source Nodes: [x_in_99], Original ATen: [aten.convolution]
        buf529 = extern_kernels.convolution(buf528, primals_301, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf529, (8, 1088, 14, 14), (213248, 196, 14, 1))
        buf530 = empty((8, 1984, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_90], Original ATen: [aten.cat]
        triton_poi_fused_cat_96.run(buf509, buf529, buf510, buf530, 3110912, grid=grid(3110912), stream=stream0)
        buf531 = empty_strided((1, 1984, 1, 1), (1984, 1, 1984, 1984), device='cuda', dtype=torch.float32)
        buf532 = empty_strided((1, 1984, 1, 1), (1984, 1, 1984, 1984), device='cuda', dtype=torch.float32)
        buf534 = empty((1984, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_239], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_97.run(buf530, primals_573, primals_574, buf531, buf532, buf534, primals_573, primals_574, 1984, 1568, grid=grid(1984), stream=stream0)
        del primals_573
        del primals_574
        buf535 = empty((8, 1984, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_239, x_241], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_98.run(buf530, buf531, buf532, primals_159, primals_160, buf535, 3110912, grid=grid(3110912), stream=stream0)
        del buf532
        del primals_160
        # Source Nodes: [x_in_101], Original ATen: [aten.convolution]
        buf536 = extern_kernels.convolution(buf535, primals_302, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf536, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf537 = buf525; del buf525  # reuse
        buf538 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf540 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_242], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf536, primals_576, primals_577, buf537, buf538, buf540, primals_576, primals_577, 800, 1568, grid=grid(800), stream=stream0)
        del primals_576
        del primals_577
        buf541 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_242, x_244], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf536, buf537, buf538, primals_161, primals_162, buf541, 1254400, grid=grid(1254400), stream=stream0)
        del primals_162
        # Source Nodes: [x_in_102], Original ATen: [aten.convolution]
        buf542 = extern_kernels.convolution(buf541, primals_303, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf542, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf543 = buf538; del buf538  # reuse
        buf544 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf546 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_245], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf542, primals_579, primals_580, buf543, buf544, buf546, primals_579, primals_580, 800, 1568, grid=grid(800), stream=stream0)
        del primals_579
        del primals_580
        buf547 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_245, x_247], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf542, buf543, buf544, primals_163, primals_164, buf547, 1254400, grid=grid(1254400), stream=stream0)
        del primals_164
        # Source Nodes: [x_in_103], Original ATen: [aten.convolution]
        buf548 = extern_kernels.convolution(buf547, primals_304, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf548, (8, 1088, 14, 14), (213248, 196, 14, 1))
        buf549 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_88], Original ATen: [aten.cat]
        triton_poi_fused_cat_99.run(buf509, buf529, buf548, buf510, buf549, 3211264, grid=grid(3211264), stream=stream0)
        buf550 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf551 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf553 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_248], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_100.run(buf549, primals_582, primals_583, buf550, buf551, buf553, primals_582, primals_583, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_582
        del primals_583
        buf554 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_248, x_250], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_101.run(buf549, buf550, buf551, primals_165, primals_166, buf554, 3211264, grid=grid(3211264), stream=stream0)
        del buf551
        del primals_166
        # Source Nodes: [x_in_105], Original ATen: [aten.convolution]
        buf555 = extern_kernels.convolution(buf554, primals_305, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf555, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf556 = buf544; del buf544  # reuse
        buf557 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf559 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_251], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf555, primals_585, primals_586, buf556, buf557, buf559, primals_585, primals_586, 800, 1568, grid=grid(800), stream=stream0)
        del primals_585
        del primals_586
        buf560 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_251, x_253], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf555, buf556, buf557, primals_167, primals_168, buf560, 1254400, grid=grid(1254400), stream=stream0)
        del primals_168
        # Source Nodes: [x_in_106], Original ATen: [aten.convolution]
        buf561 = extern_kernels.convolution(buf560, primals_306, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf561, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf562 = buf557; del buf557  # reuse
        buf563 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf565 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_254], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf561, primals_588, primals_589, buf562, buf563, buf565, primals_588, primals_589, 800, 1568, grid=grid(800), stream=stream0)
        del primals_588
        del primals_589
        buf566 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_254, x_256], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf561, buf562, buf563, primals_169, primals_170, buf566, 1254400, grid=grid(1254400), stream=stream0)
        del primals_170
        # Source Nodes: [x_in_107], Original ATen: [aten.convolution]
        buf567 = extern_kernels.convolution(buf566, primals_307, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf567, (8, 1088, 14, 14), (213248, 196, 14, 1))
        buf568 = empty((8, 2112, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_86], Original ATen: [aten.cat]
        triton_poi_fused_cat_102.run(buf509, buf529, buf548, buf567, buf510, buf568, 3311616, grid=grid(3311616), stream=stream0)
        buf569 = empty_strided((1, 2112, 1, 1), (2112, 1, 2112, 2112), device='cuda', dtype=torch.float32)
        buf570 = empty_strided((1, 2112, 1, 1), (2112, 1, 2112, 2112), device='cuda', dtype=torch.float32)
        buf572 = empty((2112, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_257], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_103.run(buf568, primals_591, primals_592, buf569, buf570, buf572, primals_591, primals_592, 2112, 1568, grid=grid(2112), stream=stream0)
        del primals_591
        del primals_592
        buf573 = empty((8, 2112, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_257, x_259], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_104.run(buf568, buf569, buf570, primals_171, primals_172, buf573, 3311616, grid=grid(3311616), stream=stream0)
        del buf570
        del primals_172
        # Source Nodes: [x_in_109], Original ATen: [aten.convolution]
        buf574 = extern_kernels.convolution(buf573, primals_308, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf574, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf575 = buf563; del buf563  # reuse
        buf576 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf578 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_260], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf574, primals_594, primals_595, buf575, buf576, buf578, primals_594, primals_595, 800, 1568, grid=grid(800), stream=stream0)
        del primals_594
        del primals_595
        buf579 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_260, x_262], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf574, buf575, buf576, primals_173, primals_174, buf579, 1254400, grid=grid(1254400), stream=stream0)
        del primals_174
        # Source Nodes: [x_in_110], Original ATen: [aten.convolution]
        buf580 = extern_kernels.convolution(buf579, primals_309, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf580, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf581 = buf576; del buf576  # reuse
        buf582 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf584 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_263], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf580, primals_597, primals_598, buf581, buf582, buf584, primals_597, primals_598, 800, 1568, grid=grid(800), stream=stream0)
        del primals_597
        del primals_598
        buf585 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_263, x_265], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf580, buf581, buf582, primals_175, primals_176, buf585, 1254400, grid=grid(1254400), stream=stream0)
        del primals_176
        # Source Nodes: [x_in_111], Original ATen: [aten.convolution]
        buf586 = extern_kernels.convolution(buf585, primals_310, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf586, (8, 1088, 14, 14), (213248, 196, 14, 1))
        buf589 = empty((8, 2176, 14, 14), device='cuda', dtype=torch.float32)
        buf587 = reinterpret_tensor(buf589, (8, 1024, 14, 14), (426496, 196, 14, 1), 0)  # alias
        # Source Nodes: [x_s1_25, x_s1_26, x_s1_27, x_s1_28], Original ATen: [aten.add]
        triton_poi_fused_add_105.run(buf509, buf529, buf548, buf567, buf586, buf587, 1605632, grid=grid(1605632), stream=stream0)
        buf588 = reinterpret_tensor(buf589, (8, 1152, 14, 14), (426496, 196, 14, 1), 200704)  # alias
        # Source Nodes: [cat_85], Original ATen: [aten.cat]
        triton_poi_fused_cat_106.run(buf510, buf529, buf548, buf567, buf586, buf588, 1806336, grid=grid(1806336), stream=stream0)
        del buf529
        del buf548
        del buf567
        del buf586
        buf590 = empty_strided((1, 2176, 1, 1), (2176, 1, 2176, 2176), device='cuda', dtype=torch.float32)
        buf591 = empty_strided((1, 2176, 1, 1), (2176, 1, 2176, 2176), device='cuda', dtype=torch.float32)
        buf593 = empty((2176, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_266], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_107.run(buf589, primals_600, primals_601, buf590, buf591, buf593, primals_600, primals_601, 2176, 1568, grid=grid(2176), stream=stream0)
        del primals_600
        del primals_601
        buf594 = empty((8, 2176, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_266, x_268], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_108.run(buf589, buf590, buf591, primals_177, primals_178, buf594, 3411968, grid=grid(3411968), stream=stream0)
        del buf591
        del primals_178
        # Source Nodes: [x_in_113], Original ATen: [aten.convolution]
        buf595 = extern_kernels.convolution(buf594, primals_311, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf595, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf596 = buf582; del buf582  # reuse
        buf597 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf599 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_269], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf595, primals_603, primals_604, buf596, buf597, buf599, primals_603, primals_604, 800, 1568, grid=grid(800), stream=stream0)
        del primals_603
        del primals_604
        buf600 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_269, x_271], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf595, buf596, buf597, primals_179, primals_180, buf600, 1254400, grid=grid(1254400), stream=stream0)
        del primals_180
        # Source Nodes: [x_in_114], Original ATen: [aten.convolution]
        buf601 = extern_kernels.convolution(buf600, primals_312, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf601, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf602 = buf597; del buf597  # reuse
        buf603 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf605 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_272], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf601, primals_606, primals_607, buf602, buf603, buf605, primals_606, primals_607, 800, 1568, grid=grid(800), stream=stream0)
        del primals_606
        del primals_607
        buf606 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_272, x_274], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf601, buf602, buf603, primals_181, primals_182, buf606, 1254400, grid=grid(1254400), stream=stream0)
        del primals_182
        # Source Nodes: [x_in_115], Original ATen: [aten.convolution]
        buf607 = extern_kernels.convolution(buf606, primals_313, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf607, (8, 1088, 14, 14), (213248, 196, 14, 1))
        buf608 = empty((8, 2240, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_82], Original ATen: [aten.cat]
        triton_poi_fused_cat_109.run(buf587, buf607, buf588, buf608, 3512320, grid=grid(3512320), stream=stream0)
        buf609 = empty_strided((1, 2240, 1, 1), (2240, 1, 2240, 2240), device='cuda', dtype=torch.float32)
        buf610 = empty_strided((1, 2240, 1, 1), (2240, 1, 2240, 2240), device='cuda', dtype=torch.float32)
        buf612 = empty((2240, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_275], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_110.run(buf608, primals_609, primals_610, buf609, buf610, buf612, primals_609, primals_610, 2240, 1568, grid=grid(2240), stream=stream0)
        del primals_609
        del primals_610
        buf613 = empty((8, 2240, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_275, x_277], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_111.run(buf608, buf609, buf610, primals_183, primals_184, buf613, 3512320, grid=grid(3512320), stream=stream0)
        del buf610
        del primals_184
        # Source Nodes: [x_in_117], Original ATen: [aten.convolution]
        buf614 = extern_kernels.convolution(buf613, primals_314, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf614, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf615 = buf603; del buf603  # reuse
        buf616 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf618 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf614, primals_612, primals_613, buf615, buf616, buf618, primals_612, primals_613, 800, 1568, grid=grid(800), stream=stream0)
        del primals_612
        del primals_613
        buf619 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_278, x_280], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf614, buf615, buf616, primals_185, primals_186, buf619, 1254400, grid=grid(1254400), stream=stream0)
        del primals_186
        # Source Nodes: [x_in_118], Original ATen: [aten.convolution]
        buf620 = extern_kernels.convolution(buf619, primals_315, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf620, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf621 = buf616; del buf616  # reuse
        buf622 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf624 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_281], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf620, primals_615, primals_616, buf621, buf622, buf624, primals_615, primals_616, 800, 1568, grid=grid(800), stream=stream0)
        del primals_615
        del primals_616
        buf625 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_281, x_283], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf620, buf621, buf622, primals_187, primals_188, buf625, 1254400, grid=grid(1254400), stream=stream0)
        del primals_188
        # Source Nodes: [x_in_119], Original ATen: [aten.convolution]
        buf626 = extern_kernels.convolution(buf625, primals_316, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf626, (8, 1088, 14, 14), (213248, 196, 14, 1))
        buf627 = reinterpret_tensor(buf272, (8, 2304, 14, 14), (451584, 196, 14, 1), 0); del buf272  # reuse
        # Source Nodes: [cat_80], Original ATen: [aten.cat]
        triton_poi_fused_cat_112.run(buf587, buf607, buf626, buf588, buf627, 3612672, grid=grid(3612672), stream=stream0)
        buf628 = empty_strided((1, 2304, 1, 1), (2304, 1, 2304, 2304), device='cuda', dtype=torch.float32)
        buf629 = empty_strided((1, 2304, 1, 1), (2304, 1, 2304, 2304), device='cuda', dtype=torch.float32)
        buf631 = empty((2304, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_284], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_113.run(buf627, primals_618, primals_619, buf628, buf629, buf631, primals_618, primals_619, 2304, 1568, grid=grid(2304), stream=stream0)
        del primals_618
        del primals_619
        buf632 = reinterpret_tensor(buf253, (8, 2304, 14, 14), (451584, 196, 14, 1), 0); del buf253  # reuse
        # Source Nodes: [x_284, x_286], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_114.run(buf627, buf628, buf629, primals_189, primals_190, buf632, 3612672, grid=grid(3612672), stream=stream0)
        del buf629
        del primals_190
        # Source Nodes: [x_in_121], Original ATen: [aten.convolution]
        buf633 = extern_kernels.convolution(buf632, primals_317, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf633, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf634 = buf622; del buf622  # reuse
        buf635 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf637 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_287], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf633, primals_621, primals_622, buf634, buf635, buf637, primals_621, primals_622, 800, 1568, grid=grid(800), stream=stream0)
        del primals_621
        del primals_622
        buf638 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_287, x_289], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf633, buf634, buf635, primals_191, primals_192, buf638, 1254400, grid=grid(1254400), stream=stream0)
        del primals_192
        # Source Nodes: [x_in_122], Original ATen: [aten.convolution]
        buf639 = extern_kernels.convolution(buf638, primals_318, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf639, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf640 = buf635; del buf635  # reuse
        buf641 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf643 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_290], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf639, primals_624, primals_625, buf640, buf641, buf643, primals_624, primals_625, 800, 1568, grid=grid(800), stream=stream0)
        del primals_624
        del primals_625
        buf644 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_290, x_292], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf639, buf640, buf641, primals_193, primals_194, buf644, 1254400, grid=grid(1254400), stream=stream0)
        del primals_194
        # Source Nodes: [x_in_123], Original ATen: [aten.convolution]
        buf645 = extern_kernels.convolution(buf644, primals_319, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf645, (8, 1088, 14, 14), (213248, 196, 14, 1))
        buf646 = empty((8, 2368, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_78], Original ATen: [aten.cat]
        triton_poi_fused_cat_115.run(buf587, buf607, buf626, buf645, buf588, buf646, 3713024, grid=grid(3713024), stream=stream0)
        buf647 = empty_strided((1, 2368, 1, 1), (2368, 1, 2368, 2368), device='cuda', dtype=torch.float32)
        buf648 = empty_strided((1, 2368, 1, 1), (2368, 1, 2368, 2368), device='cuda', dtype=torch.float32)
        buf650 = empty((2368, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_293], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_116.run(buf646, primals_627, primals_628, buf647, buf648, buf650, primals_627, primals_628, 2368, 1568, grid=grid(2368), stream=stream0)
        del primals_627
        del primals_628
        buf651 = empty((8, 2368, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_293, x_295], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_117.run(buf646, buf647, buf648, primals_195, primals_196, buf651, 3713024, grid=grid(3713024), stream=stream0)
        del buf648
        del primals_196
        # Source Nodes: [x_in_125], Original ATen: [aten.convolution]
        buf652 = extern_kernels.convolution(buf651, primals_320, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf652, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf653 = buf641; del buf641  # reuse
        buf654 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf656 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_296], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf652, primals_630, primals_631, buf653, buf654, buf656, primals_630, primals_631, 800, 1568, grid=grid(800), stream=stream0)
        del primals_630
        del primals_631
        buf657 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_296, x_298], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf652, buf653, buf654, primals_197, primals_198, buf657, 1254400, grid=grid(1254400), stream=stream0)
        del primals_198
        # Source Nodes: [x_in_126], Original ATen: [aten.convolution]
        buf658 = extern_kernels.convolution(buf657, primals_321, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf658, (8, 800, 14, 14), (156800, 196, 14, 1))
        buf659 = buf654; del buf654  # reuse
        buf660 = empty_strided((1, 800, 1, 1), (800, 1, 800, 800), device='cuda', dtype=torch.float32)
        buf662 = empty((800, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_299], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_55.run(buf658, primals_633, primals_634, buf659, buf660, buf662, primals_633, primals_634, 800, 1568, grid=grid(800), stream=stream0)
        del primals_633
        del primals_634
        buf663 = empty((8, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_299, x_301], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_56.run(buf658, buf659, buf660, primals_199, primals_200, buf663, 1254400, grid=grid(1254400), stream=stream0)
        del buf660
        del primals_200
        # Source Nodes: [x_in_127], Original ATen: [aten.convolution]
        buf664 = extern_kernels.convolution(buf663, primals_322, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf664, (8, 1088, 14, 14), (213248, 196, 14, 1))
        buf667 = empty((8, 2432, 14, 14), device='cuda', dtype=torch.float32)
        buf665 = reinterpret_tensor(buf667, (8, 1408, 14, 14), (476672, 196, 14, 1), 200704)  # alias
        # Source Nodes: [cat_77], Original ATen: [aten.cat]
        triton_poi_fused_cat_118.run(buf588, buf607, buf626, buf645, buf664, buf665, 2207744, grid=grid(2207744), stream=stream0)
        buf666 = reinterpret_tensor(buf667, (8, 1024, 14, 14), (476672, 196, 14, 1), 0)  # alias
        # Source Nodes: [resid_31, x_s1_29, x_s1_30, x_s1_31], Original ATen: [aten.add]
        triton_poi_fused_add_119.run(buf587, buf607, buf626, buf645, buf664, buf666, 1605632, grid=grid(1605632), stream=stream0)
        del buf607
        del buf626
        del buf645
        del buf664
        buf668 = empty_strided((1, 2432, 1, 1), (2432, 1, 2432, 2432), device='cuda', dtype=torch.float32)
        buf669 = empty_strided((1, 2432, 1, 1), (2432, 1, 2432, 2432), device='cuda', dtype=torch.float32)
        buf671 = empty((2432, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_302, x_305], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_120.run(buf667, primals_636, primals_639, primals_637, primals_640, buf668, buf669, buf671, primals_636, primals_639, primals_637, primals_640, 2432, 1568, grid=grid(2432), stream=stream0)
        del primals_636
        del primals_637
        del primals_639
        del primals_640
        buf672 = empty((8, 2432, 14, 14), device='cuda', dtype=torch.float32)
        buf674 = empty((8, 2432, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_302, x_304, x_305, x_307], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_121.run(buf667, buf668, buf669, primals_201, primals_202, primals_203, primals_204, buf672, buf674, 3813376, grid=grid(3813376), stream=stream0)
        del primals_202
        del primals_204
        # Source Nodes: [x_s_3], Original ATen: [aten.convolution]
        buf673 = extern_kernels.convolution(buf672, primals_323, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf673, (8, 2304, 7, 7), (112896, 49, 7, 1))
        # Source Nodes: [x_in_129], Original ATen: [aten.convolution]
        buf675 = extern_kernels.convolution(buf674, primals_324, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf675, (8, 1600, 14, 14), (313600, 196, 14, 1))
        buf676 = buf414; del buf414  # reuse
        buf677 = empty_strided((1, 1600, 1, 1), (1600, 1, 1600, 1600), device='cuda', dtype=torch.float32)
        buf679 = empty((1600, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_308], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_77.run(buf675, primals_642, primals_643, buf676, buf677, buf679, primals_642, primals_643, 1600, 1568, grid=grid(1600), stream=stream0)
        del primals_642
        del primals_643
        buf680 = empty((8, 1600, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_308, x_310], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_78.run(buf675, buf676, buf677, primals_205, primals_206, buf680, 2508800, grid=grid(2508800), stream=stream0)
        del primals_206
        # Source Nodes: [x_in_130], Original ATen: [aten.convolution]
        buf681 = extern_kernels.convolution(buf680, primals_325, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf681, (8, 1600, 7, 7), (78400, 49, 7, 1))
        buf682 = buf677; del buf677  # reuse
        buf683 = empty_strided((1, 1600, 1, 1), (1600, 1, 1600, 1600), device='cuda', dtype=torch.float32)
        buf685 = empty((1600, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_311], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_122.run(buf681, primals_645, primals_646, buf682, buf683, buf685, primals_645, primals_646, 1600, 392, grid=grid(1600), stream=stream0)
        del primals_645
        del primals_646
        buf686 = empty((8, 1600, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_311, x_313], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_123.run(buf681, buf682, buf683, primals_207, primals_208, buf686, 627200, grid=grid(627200), stream=stream0)
        del primals_208
        # Source Nodes: [x_in_131], Original ATen: [aten.convolution]
        buf687 = extern_kernels.convolution(buf686, primals_326, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf687, (8, 2176, 7, 7), (106624, 49, 7, 1))
        buf688 = empty((8, 2432, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_74], Original ATen: [aten.cat]
        triton_poi_fused_cat_124.run(buf673, buf687, buf688, 953344, grid=grid(953344), stream=stream0)
        buf689 = buf669; del buf669  # reuse
        buf690 = empty_strided((1, 2432, 1, 1), (2432, 1, 2432, 2432), device='cuda', dtype=torch.float32)
        buf692 = empty((2432, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_314], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_125.run(buf688, primals_648, primals_649, buf689, buf690, buf692, primals_648, primals_649, 2432, 392, grid=grid(2432), stream=stream0)
        del primals_648
        del primals_649
        buf693 = empty((8, 2432, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_314, x_316], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_126.run(buf688, buf689, buf690, primals_209, primals_210, buf693, 953344, grid=grid(953344), stream=stream0)
        del buf690
        del primals_210
        # Source Nodes: [x_in_133], Original ATen: [aten.convolution]
        buf694 = extern_kernels.convolution(buf693, primals_327, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf694, (8, 1600, 7, 7), (78400, 49, 7, 1))
        buf695 = buf683; del buf683  # reuse
        buf696 = empty_strided((1, 1600, 1, 1), (1600, 1, 1600, 1600), device='cuda', dtype=torch.float32)
        buf698 = empty((1600, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_317], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_122.run(buf694, primals_651, primals_652, buf695, buf696, buf698, primals_651, primals_652, 1600, 392, grid=grid(1600), stream=stream0)
        del primals_651
        del primals_652
        buf699 = empty((8, 1600, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_317, x_319], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_123.run(buf694, buf695, buf696, primals_211, primals_212, buf699, 627200, grid=grid(627200), stream=stream0)
        del primals_212
        # Source Nodes: [x_in_134], Original ATen: [aten.convolution]
        buf700 = extern_kernels.convolution(buf699, primals_328, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf700, (8, 1600, 7, 7), (78400, 49, 7, 1))
        buf701 = buf696; del buf696  # reuse
        buf702 = empty_strided((1, 1600, 1, 1), (1600, 1, 1600, 1600), device='cuda', dtype=torch.float32)
        buf704 = empty((1600, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_320], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_122.run(buf700, primals_654, primals_655, buf701, buf702, buf704, primals_654, primals_655, 1600, 392, grid=grid(1600), stream=stream0)
        del primals_654
        del primals_655
        buf705 = empty((8, 1600, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_320, x_322], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_123.run(buf700, buf701, buf702, primals_213, primals_214, buf705, 627200, grid=grid(627200), stream=stream0)
        del primals_214
        # Source Nodes: [x_in_135], Original ATen: [aten.convolution]
        buf706 = extern_kernels.convolution(buf705, primals_329, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf706, (8, 2176, 7, 7), (106624, 49, 7, 1))
        buf707 = empty((8, 2560, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_72], Original ATen: [aten.cat]
        triton_poi_fused_cat_127.run(buf673, buf687, buf706, buf707, 1003520, grid=grid(1003520), stream=stream0)
        buf708 = empty_strided((1, 2560, 1, 1), (2560, 1, 2560, 2560), device='cuda', dtype=torch.float32)
        buf709 = empty_strided((1, 2560, 1, 1), (2560, 1, 2560, 2560), device='cuda', dtype=torch.float32)
        buf711 = empty((2560, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_323], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_128.run(buf707, primals_657, primals_658, buf708, buf709, buf711, primals_657, primals_658, 2560, 392, grid=grid(2560), stream=stream0)
        del primals_657
        del primals_658
        buf712 = empty((8, 2560, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_323, x_325], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_129.run(buf707, buf708, buf709, primals_215, primals_216, buf712, 1003520, grid=grid(1003520), stream=stream0)
        del buf709
        del primals_216
        # Source Nodes: [x_in_137], Original ATen: [aten.convolution]
        buf713 = extern_kernels.convolution(buf712, primals_330, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf713, (8, 1600, 7, 7), (78400, 49, 7, 1))
        buf714 = buf702; del buf702  # reuse
        buf715 = empty_strided((1, 1600, 1, 1), (1600, 1, 1600, 1600), device='cuda', dtype=torch.float32)
        buf717 = empty((1600, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_326], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_122.run(buf713, primals_660, primals_661, buf714, buf715, buf717, primals_660, primals_661, 1600, 392, grid=grid(1600), stream=stream0)
        del primals_660
        del primals_661
        buf718 = empty((8, 1600, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_326, x_328], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_123.run(buf713, buf714, buf715, primals_217, primals_218, buf718, 627200, grid=grid(627200), stream=stream0)
        del primals_218
        # Source Nodes: [x_in_138], Original ATen: [aten.convolution]
        buf719 = extern_kernels.convolution(buf718, primals_331, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf719, (8, 1600, 7, 7), (78400, 49, 7, 1))
        buf720 = buf715; del buf715  # reuse
        buf721 = empty_strided((1, 1600, 1, 1), (1600, 1, 1600, 1600), device='cuda', dtype=torch.float32)
        buf723 = empty((1600, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_329], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_122.run(buf719, primals_663, primals_664, buf720, buf721, buf723, primals_663, primals_664, 1600, 392, grid=grid(1600), stream=stream0)
        del primals_663
        del primals_664
        buf724 = empty((8, 1600, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_329, x_331], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_123.run(buf719, buf720, buf721, primals_219, primals_220, buf724, 627200, grid=grid(627200), stream=stream0)
        del buf721
        del primals_220
        # Source Nodes: [x_in_139], Original ATen: [aten.convolution]
        buf725 = extern_kernels.convolution(buf724, primals_332, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf725, (8, 2176, 7, 7), (106624, 49, 7, 1))
        buf726 = empty((8, 2688, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_70], Original ATen: [aten.cat]
        triton_poi_fused_cat_130.run(buf673, buf687, buf706, buf725, buf726, 1053696, grid=grid(1053696), stream=stream0)
        del buf673
        del buf687
        del buf706
        del buf725
        buf727 = empty_strided((1, 2688, 1, 1), (2688, 1, 2688, 2688), device='cuda', dtype=torch.float32)
        buf728 = empty_strided((1, 2688, 1, 1), (2688, 1, 2688, 2688), device='cuda', dtype=torch.float32)
        buf730 = empty((2688, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_333], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_131.run(buf726, primals_666, primals_667, buf727, buf728, buf730, primals_666, primals_667, 2688, 392, grid=grid(2688), stream=stream0)
        del primals_666
        del primals_667
        buf736 = empty((8, 2688, 7, 7), device='cuda', dtype=torch.bool)
        buf732 = empty_strided((8, 2688, 1, 1), (2688, 1, 21504, 21504), device='cuda', dtype=torch.float32)
        buf733 = reinterpret_tensor(buf732, (8, 2688, 1, 1), (2688, 1, 1, 1), 0); del buf732  # reuse
        # Source Nodes: [x_333, x_336, x_337], Original ATen: [aten._native_batch_norm_legit_functional, aten.mean, aten.relu, aten.threshold_backward]
        triton_per_fused__native_batch_norm_legit_functional_mean_relu_threshold_backward_132.run(buf733, buf726, buf727, buf728, primals_221, primals_222, buf736, 21504, 49, grid=grid(21504), stream=stream0)
        del buf728
        del primals_222
        # Source Nodes: [x_340], Original ATen: [aten.convolution]
        buf734 = extern_kernels.convolution(buf733, primals_333, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf734, (8, 1000, 1, 1), (1000, 1, 1, 1))
        buf735 = reinterpret_tensor(buf734, (8, 1000), (1000, 1), 0); del buf734  # reuse
        # Source Nodes: [pred, x_340], Original ATen: [aten.convolution, aten.view]
        triton_poi_fused_convolution_view_133.run(buf735, primals_334, 8000, grid=grid(8000), stream=stream0)
        del primals_334
        buf737 = buf9; del buf9  # reuse
        # Source Nodes: [], Original ATen: [aten.native_batch_norm_backward]
        triton_poi_fused_native_batch_norm_backward_134.run(buf737, buf14, 3211264, grid=grid(3211264), stream=stream0)
        del buf14
        # Source Nodes: [add_], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_335, primals_335, 1, grid=grid(1), stream=stream0)
        del primals_335
        # Source Nodes: [add__1], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_338, primals_338, 1, grid=grid(1), stream=stream0)
        del primals_338
        # Source Nodes: [add__2], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_341, primals_341, 1, grid=grid(1), stream=stream0)
        del primals_341
        # Source Nodes: [add__3], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_344, primals_344, 1, grid=grid(1), stream=stream0)
        del primals_344
        # Source Nodes: [add__4], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_347, primals_347, 1, grid=grid(1), stream=stream0)
        del primals_347
        # Source Nodes: [add__5], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_350, primals_350, 1, grid=grid(1), stream=stream0)
        del primals_350
        # Source Nodes: [add__6], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_353, primals_353, 1, grid=grid(1), stream=stream0)
        del primals_353
        # Source Nodes: [add__7], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_356, primals_356, 1, grid=grid(1), stream=stream0)
        del primals_356
        # Source Nodes: [add__8], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_359, primals_359, 1, grid=grid(1), stream=stream0)
        del primals_359
        # Source Nodes: [add__9], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_362, primals_362, 1, grid=grid(1), stream=stream0)
        del primals_362
        # Source Nodes: [add__10], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_365, primals_365, 1, grid=grid(1), stream=stream0)
        del primals_365
        # Source Nodes: [add__11], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_368, primals_368, 1, grid=grid(1), stream=stream0)
        del primals_368
        # Source Nodes: [add__12], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_371, primals_371, 1, grid=grid(1), stream=stream0)
        del primals_371
        # Source Nodes: [add__13], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_374, primals_374, 1, grid=grid(1), stream=stream0)
        del primals_374
        # Source Nodes: [add__14], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_377, primals_377, 1, grid=grid(1), stream=stream0)
        del primals_377
        # Source Nodes: [add__15], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_380, primals_380, 1, grid=grid(1), stream=stream0)
        del primals_380
        # Source Nodes: [add__16], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_383, primals_383, 1, grid=grid(1), stream=stream0)
        del primals_383
        # Source Nodes: [add__17], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_386, primals_386, 1, grid=grid(1), stream=stream0)
        del primals_386
        # Source Nodes: [add__18], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_389, primals_389, 1, grid=grid(1), stream=stream0)
        del primals_389
        # Source Nodes: [add__19], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_392, primals_392, 1, grid=grid(1), stream=stream0)
        del primals_392
        # Source Nodes: [add__20], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_395, primals_395, 1, grid=grid(1), stream=stream0)
        del primals_395
        # Source Nodes: [add__21], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_398, primals_398, 1, grid=grid(1), stream=stream0)
        del primals_398
        # Source Nodes: [add__22], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_401, primals_401, 1, grid=grid(1), stream=stream0)
        del primals_401
        # Source Nodes: [add__23], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_404, primals_404, 1, grid=grid(1), stream=stream0)
        del primals_404
        # Source Nodes: [add__24], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_407, primals_407, 1, grid=grid(1), stream=stream0)
        del primals_407
        # Source Nodes: [add__25], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_410, primals_410, 1, grid=grid(1), stream=stream0)
        del primals_410
        # Source Nodes: [add__26], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_413, primals_413, 1, grid=grid(1), stream=stream0)
        del primals_413
        # Source Nodes: [add__27], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_416, primals_416, 1, grid=grid(1), stream=stream0)
        del primals_416
        # Source Nodes: [add__28], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_419, primals_419, 1, grid=grid(1), stream=stream0)
        del primals_419
        # Source Nodes: [add__29], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_422, primals_422, 1, grid=grid(1), stream=stream0)
        del primals_422
        # Source Nodes: [add__30], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_425, primals_425, 1, grid=grid(1), stream=stream0)
        del primals_425
        # Source Nodes: [add__31], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_428, primals_428, 1, grid=grid(1), stream=stream0)
        del primals_428
        # Source Nodes: [add__32], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_431, primals_431, 1, grid=grid(1), stream=stream0)
        del primals_431
        # Source Nodes: [add__33], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_434, primals_434, 1, grid=grid(1), stream=stream0)
        del primals_434
        # Source Nodes: [add__34], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_437, primals_437, 1, grid=grid(1), stream=stream0)
        del primals_437
        # Source Nodes: [add__35], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_440, primals_440, 1, grid=grid(1), stream=stream0)
        del primals_440
        # Source Nodes: [add__36], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_443, primals_443, 1, grid=grid(1), stream=stream0)
        del primals_443
        # Source Nodes: [add__37], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_446, primals_446, 1, grid=grid(1), stream=stream0)
        del primals_446
        # Source Nodes: [add__38], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_449, primals_449, 1, grid=grid(1), stream=stream0)
        del primals_449
        # Source Nodes: [add__39], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_452, primals_452, 1, grid=grid(1), stream=stream0)
        del primals_452
        # Source Nodes: [add__40], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_455, primals_455, 1, grid=grid(1), stream=stream0)
        del primals_455
        # Source Nodes: [add__41], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_458, primals_458, 1, grid=grid(1), stream=stream0)
        del primals_458
        # Source Nodes: [add__42], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_461, primals_461, 1, grid=grid(1), stream=stream0)
        del primals_461
        # Source Nodes: [add__43], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_464, primals_464, 1, grid=grid(1), stream=stream0)
        del primals_464
        # Source Nodes: [add__44], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_467, primals_467, 1, grid=grid(1), stream=stream0)
        del primals_467
        # Source Nodes: [add__45], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_470, primals_470, 1, grid=grid(1), stream=stream0)
        del primals_470
        # Source Nodes: [add__46], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_473, primals_473, 1, grid=grid(1), stream=stream0)
        del primals_473
        # Source Nodes: [add__47], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_476, primals_476, 1, grid=grid(1), stream=stream0)
        del primals_476
        # Source Nodes: [add__48], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_479, primals_479, 1, grid=grid(1), stream=stream0)
        del primals_479
        # Source Nodes: [add__49], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_482, primals_482, 1, grid=grid(1), stream=stream0)
        del primals_482
        # Source Nodes: [add__50], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_485, primals_485, 1, grid=grid(1), stream=stream0)
        del primals_485
        # Source Nodes: [add__51], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_488, primals_488, 1, grid=grid(1), stream=stream0)
        del primals_488
        # Source Nodes: [add__52], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_491, primals_491, 1, grid=grid(1), stream=stream0)
        del primals_491
        # Source Nodes: [add__53], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_494, primals_494, 1, grid=grid(1), stream=stream0)
        del primals_494
        # Source Nodes: [add__54], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_497, primals_497, 1, grid=grid(1), stream=stream0)
        del primals_497
        # Source Nodes: [add__55], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_500, primals_500, 1, grid=grid(1), stream=stream0)
        del primals_500
        # Source Nodes: [add__56], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_503, primals_503, 1, grid=grid(1), stream=stream0)
        del primals_503
        # Source Nodes: [add__57], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_506, primals_506, 1, grid=grid(1), stream=stream0)
        del primals_506
        # Source Nodes: [add__58], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_509, primals_509, 1, grid=grid(1), stream=stream0)
        del primals_509
        # Source Nodes: [add__59], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_512, primals_512, 1, grid=grid(1), stream=stream0)
        del primals_512
        # Source Nodes: [add__60], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_515, primals_515, 1, grid=grid(1), stream=stream0)
        del primals_515
        # Source Nodes: [add__61], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_518, primals_518, 1, grid=grid(1), stream=stream0)
        del primals_518
        # Source Nodes: [add__62], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_521, primals_521, 1, grid=grid(1), stream=stream0)
        del primals_521
        # Source Nodes: [add__63], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_524, primals_524, 1, grid=grid(1), stream=stream0)
        del primals_524
        # Source Nodes: [add__64], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_527, primals_527, 1, grid=grid(1), stream=stream0)
        del primals_527
        # Source Nodes: [add__65], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_530, primals_530, 1, grid=grid(1), stream=stream0)
        del primals_530
        # Source Nodes: [add__66], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_533, primals_533, 1, grid=grid(1), stream=stream0)
        del primals_533
        # Source Nodes: [add__67], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_536, primals_536, 1, grid=grid(1), stream=stream0)
        del primals_536
        # Source Nodes: [add__68], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_539, primals_539, 1, grid=grid(1), stream=stream0)
        del primals_539
        # Source Nodes: [add__69], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_542, primals_542, 1, grid=grid(1), stream=stream0)
        del primals_542
        # Source Nodes: [add__70], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_545, primals_545, 1, grid=grid(1), stream=stream0)
        del primals_545
        # Source Nodes: [add__71], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_548, primals_548, 1, grid=grid(1), stream=stream0)
        del primals_548
        # Source Nodes: [add__72], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_551, primals_551, 1, grid=grid(1), stream=stream0)
        del primals_551
        # Source Nodes: [add__73], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_554, primals_554, 1, grid=grid(1), stream=stream0)
        del primals_554
        # Source Nodes: [add__74], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_557, primals_557, 1, grid=grid(1), stream=stream0)
        del primals_557
        # Source Nodes: [add__75], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_560, primals_560, 1, grid=grid(1), stream=stream0)
        del primals_560
        # Source Nodes: [add__76], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_563, primals_563, 1, grid=grid(1), stream=stream0)
        del primals_563
        # Source Nodes: [add__77], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_566, primals_566, 1, grid=grid(1), stream=stream0)
        del primals_566
        # Source Nodes: [add__78], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_569, primals_569, 1, grid=grid(1), stream=stream0)
        del primals_569
        # Source Nodes: [add__79], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_572, primals_572, 1, grid=grid(1), stream=stream0)
        del primals_572
        # Source Nodes: [add__80], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_575, primals_575, 1, grid=grid(1), stream=stream0)
        del primals_575
        # Source Nodes: [add__81], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_578, primals_578, 1, grid=grid(1), stream=stream0)
        del primals_578
        # Source Nodes: [add__82], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_581, primals_581, 1, grid=grid(1), stream=stream0)
        del primals_581
        # Source Nodes: [add__83], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_584, primals_584, 1, grid=grid(1), stream=stream0)
        del primals_584
        # Source Nodes: [add__84], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_587, primals_587, 1, grid=grid(1), stream=stream0)
        del primals_587
        # Source Nodes: [add__85], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_590, primals_590, 1, grid=grid(1), stream=stream0)
        del primals_590
        # Source Nodes: [add__86], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_593, primals_593, 1, grid=grid(1), stream=stream0)
        del primals_593
        # Source Nodes: [add__87], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_596, primals_596, 1, grid=grid(1), stream=stream0)
        del primals_596
        # Source Nodes: [add__88], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_599, primals_599, 1, grid=grid(1), stream=stream0)
        del primals_599
        # Source Nodes: [add__89], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_602, primals_602, 1, grid=grid(1), stream=stream0)
        del primals_602
        # Source Nodes: [add__90], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_605, primals_605, 1, grid=grid(1), stream=stream0)
        del primals_605
        # Source Nodes: [add__91], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_608, primals_608, 1, grid=grid(1), stream=stream0)
        del primals_608
        # Source Nodes: [add__92], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_611, primals_611, 1, grid=grid(1), stream=stream0)
        del primals_611
        # Source Nodes: [add__93], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_614, primals_614, 1, grid=grid(1), stream=stream0)
        del primals_614
        # Source Nodes: [add__94], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_617, primals_617, 1, grid=grid(1), stream=stream0)
        del primals_617
        # Source Nodes: [add__95], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_620, primals_620, 1, grid=grid(1), stream=stream0)
        del primals_620
        # Source Nodes: [add__96], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_623, primals_623, 1, grid=grid(1), stream=stream0)
        del primals_623
        # Source Nodes: [add__97], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_626, primals_626, 1, grid=grid(1), stream=stream0)
        del primals_626
        # Source Nodes: [add__98], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_629, primals_629, 1, grid=grid(1), stream=stream0)
        del primals_629
        # Source Nodes: [add__99], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_632, primals_632, 1, grid=grid(1), stream=stream0)
        del primals_632
        # Source Nodes: [add__100], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_635, primals_635, 1, grid=grid(1), stream=stream0)
        del primals_635
        # Source Nodes: [add__101], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_638, primals_638, 1, grid=grid(1), stream=stream0)
        del primals_638
        # Source Nodes: [add__102], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_641, primals_641, 1, grid=grid(1), stream=stream0)
        del primals_641
        # Source Nodes: [add__103], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_644, primals_644, 1, grid=grid(1), stream=stream0)
        del primals_644
        # Source Nodes: [add__104], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_647, primals_647, 1, grid=grid(1), stream=stream0)
        del primals_647
        # Source Nodes: [add__105], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_650, primals_650, 1, grid=grid(1), stream=stream0)
        del primals_650
        # Source Nodes: [add__106], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_653, primals_653, 1, grid=grid(1), stream=stream0)
        del primals_653
        # Source Nodes: [add__107], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_656, primals_656, 1, grid=grid(1), stream=stream0)
        del primals_656
        # Source Nodes: [add__108], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_659, primals_659, 1, grid=grid(1), stream=stream0)
        del primals_659
        # Source Nodes: [add__109], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_662, primals_662, 1, grid=grid(1), stream=stream0)
        del primals_662
        # Source Nodes: [add__110], Original ATen: [aten.add]
        triton_poi_fused_add_135.run(primals_665, primals_665, 1, grid=grid(1), stream=stream0)
        del primals_665
        return (buf735, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_81, primals_83, primals_85, primals_87, primals_89, primals_91, primals_93, primals_95, primals_97, primals_99, primals_101, primals_103, primals_105, primals_107, primals_109, primals_111, primals_113, primals_115, primals_117, primals_119, primals_121, primals_123, primals_125, primals_127, primals_129, primals_131, primals_133, primals_135, primals_137, primals_139, primals_141, primals_143, primals_145, primals_147, primals_149, primals_151, primals_153, primals_155, primals_157, primals_159, primals_161, primals_163, primals_165, primals_167, primals_169, primals_171, primals_173, primals_175, primals_177, primals_179, primals_181, primals_183, primals_185, primals_187, primals_189, primals_191, primals_193, primals_195, primals_197, primals_199, primals_201, primals_203, primals_205, primals_207, primals_209, primals_211, primals_213, primals_215, primals_217, primals_219, primals_221, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_668, buf0, buf7, buf8, buf10, buf17, buf18, buf20, buf21, buf28, buf29, buf30, buf37, buf38, buf40, buf44, buf45, buf46, buf53, buf54, buf55, buf62, buf63, buf65, buf69, buf70, buf71, buf78, buf79, buf80, buf87, buf88, buf90, buf94, buf95, buf96, buf103, buf104, buf105, buf112, buf113, buf117, buf121, buf122, buf124, buf125, buf129, buf130, buf131, buf135, buf136, buf138, buf142, buf143, buf144, buf148, buf149, buf150, buf154, buf155, buf157, buf161, buf162, buf163, buf167, buf168, buf169, buf173, buf174, buf176, buf180, buf181, buf182, buf186, buf187, buf188, buf192, buf193, buf197, buf201, buf202, buf203, buf207, buf208, buf209, buf213, buf214, buf216, buf220, buf221, buf222, buf226, buf227, buf228, buf232, buf233, buf235, buf239, buf240, buf241, buf245, buf246, buf247, buf251, buf252, buf254, buf258, buf259, buf260, buf264, buf265, buf266, buf270, buf271, buf275, buf279, buf280, buf282, buf283, buf287, buf288, buf289, buf293, buf294, buf296, buf300, buf301, buf302, buf306, buf307, buf308, buf312, buf313, buf315, buf319, buf320, buf321, buf325, buf326, buf327, buf331, buf332, buf334, buf338, buf339, buf340, buf344, buf345, buf346, buf350, buf351, buf355, buf359, buf360, buf361, buf365, buf366, buf367, buf371, buf372, buf374, buf378, buf379, buf380, buf384, buf385, buf386, buf390, buf391, buf393, buf397, buf398, buf399, buf403, buf404, buf405, buf409, buf410, buf412, buf416, buf417, buf418, buf422, buf423, buf424, buf428, buf429, buf433, buf437, buf438, buf439, buf443, buf444, buf445, buf449, buf450, buf452, buf456, buf457, buf458, buf462, buf463, buf464, buf468, buf469, buf471, buf475, buf476, buf477, buf481, buf482, buf483, buf487, buf488, buf490, buf494, buf495, buf496, buf500, buf501, buf502, buf506, buf507, buf511, buf515, buf516, buf517, buf521, buf522, buf523, buf527, buf528, buf530, buf534, buf535, buf536, buf540, buf541, buf542, buf546, buf547, buf549, buf553, buf554, buf555, buf559, buf560, buf561, buf565, buf566, buf568, buf572, buf573, buf574, buf578, buf579, buf580, buf584, buf585, buf589, buf593, buf594, buf595, buf599, buf600, buf601, buf605, buf606, buf608, buf612, buf613, buf614, buf618, buf619, buf620, buf624, buf625, buf627, buf631, buf632, buf633, buf637, buf638, buf639, buf643, buf644, buf646, buf650, buf651, buf652, buf656, buf657, buf658, buf662, buf663, buf667, buf671, buf672, buf674, buf675, buf679, buf680, buf681, buf685, buf686, buf688, buf692, buf693, buf694, buf698, buf699, buf700, buf704, buf705, buf707, buf711, buf712, buf713, buf717, buf718, buf719, buf723, buf724, buf726, buf730, buf733, buf736, reinterpret_tensor(buf727, (1, 2688, 1, 1), (2688, 1, 1, 1), 0), reinterpret_tensor(buf720, (1, 1600, 1, 1), (1600, 1, 1, 1), 0), reinterpret_tensor(buf714, (1, 1600, 1, 1), (1600, 1, 1, 1), 0), reinterpret_tensor(buf708, (1, 2560, 1, 1), (2560, 1, 1, 1), 0), reinterpret_tensor(buf701, (1, 1600, 1, 1), (1600, 1, 1, 1), 0), reinterpret_tensor(buf695, (1, 1600, 1, 1), (1600, 1, 1, 1), 0), reinterpret_tensor(buf689, (1, 2432, 1, 1), (2432, 1, 1, 1), 0), reinterpret_tensor(buf682, (1, 1600, 1, 1), (1600, 1, 1, 1), 0), reinterpret_tensor(buf676, (1, 1600, 1, 1), (1600, 1, 1, 1), 0), reinterpret_tensor(buf668, (1, 2432, 1, 1), (2432, 1, 1, 1), 0), reinterpret_tensor(buf659, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf653, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf647, (1, 2368, 1, 1), (2368, 1, 1, 1), 0), reinterpret_tensor(buf640, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf634, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf628, (1, 2304, 1, 1), (2304, 1, 1, 1), 0), reinterpret_tensor(buf621, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf615, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf609, (1, 2240, 1, 1), (2240, 1, 1, 1), 0), reinterpret_tensor(buf602, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf596, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf590, (1, 2176, 1, 1), (2176, 1, 1, 1), 0), reinterpret_tensor(buf581, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf575, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf569, (1, 2112, 1, 1), (2112, 1, 1, 1), 0), reinterpret_tensor(buf562, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf556, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf550, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf543, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf537, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf531, (1, 1984, 1, 1), (1984, 1, 1, 1), 0), reinterpret_tensor(buf524, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf518, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf512, (1, 1920, 1, 1), (1920, 1, 1, 1), 0), reinterpret_tensor(buf503, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf497, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf491, (1, 1856, 1, 1), (1856, 1, 1, 1), 0), reinterpret_tensor(buf484, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf478, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf472, (1, 1792, 1, 1), (1792, 1, 1, 1), 0), reinterpret_tensor(buf465, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf459, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf453, (1, 1728, 1, 1), (1728, 1, 1, 1), 0), reinterpret_tensor(buf446, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf440, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf434, (1, 1664, 1, 1), (1664, 1, 1, 1), 0), reinterpret_tensor(buf425, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf419, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf413, (1, 1600, 1, 1), (1600, 1, 1, 1), 0), reinterpret_tensor(buf406, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf400, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf394, (1, 1536, 1, 1), (1536, 1, 1, 1), 0), reinterpret_tensor(buf387, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf381, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf375, (1, 1472, 1, 1), (1472, 1, 1, 1), 0), reinterpret_tensor(buf368, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf362, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf356, (1, 1408, 1, 1), (1408, 1, 1, 1), 0), reinterpret_tensor(buf347, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf341, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf335, (1, 1344, 1, 1), (1344, 1, 1, 1), 0), reinterpret_tensor(buf328, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf322, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf316, (1, 1280, 1, 1), (1280, 1, 1, 1), 0), reinterpret_tensor(buf309, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf303, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf297, (1, 1216, 1, 1), (1216, 1, 1, 1), 0), reinterpret_tensor(buf290, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf284, (1, 800, 1, 1), (800, 1, 1, 1), 0), reinterpret_tensor(buf276, (1, 1152, 1, 1), (1152, 1, 1, 1), 0), reinterpret_tensor(buf267, (1, 400, 1, 1), (400, 1, 1, 1), 0), reinterpret_tensor(buf261, (1, 400, 1, 1), (400, 1, 1, 1), 0), reinterpret_tensor(buf255, (1, 1088, 1, 1), (1088, 1, 1, 1), 0), reinterpret_tensor(buf248, (1, 400, 1, 1), (400, 1, 1, 1), 0), reinterpret_tensor(buf242, (1, 400, 1, 1), (400, 1, 1, 1), 0), reinterpret_tensor(buf236, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf229, (1, 400, 1, 1), (400, 1, 1, 1), 0), reinterpret_tensor(buf223, (1, 400, 1, 1), (400, 1, 1, 1), 0), reinterpret_tensor(buf217, (1, 960, 1, 1), (960, 1, 1, 1), 0), reinterpret_tensor(buf210, (1, 400, 1, 1), (400, 1, 1, 1), 0), reinterpret_tensor(buf204, (1, 400, 1, 1), (400, 1, 1, 1), 0), reinterpret_tensor(buf198, (1, 896, 1, 1), (896, 1, 1, 1), 0), reinterpret_tensor(buf189, (1, 400, 1, 1), (400, 1, 1, 1), 0), reinterpret_tensor(buf183, (1, 400, 1, 1), (400, 1, 1, 1), 0), reinterpret_tensor(buf177, (1, 832, 1, 1), (832, 1, 1, 1), 0), reinterpret_tensor(buf170, (1, 400, 1, 1), (400, 1, 1, 1), 0), reinterpret_tensor(buf164, (1, 400, 1, 1), (400, 1, 1, 1), 0), reinterpret_tensor(buf158, (1, 768, 1, 1), (768, 1, 1, 1), 0), reinterpret_tensor(buf151, (1, 400, 1, 1), (400, 1, 1, 1), 0), reinterpret_tensor(buf145, (1, 400, 1, 1), (400, 1, 1, 1), 0), reinterpret_tensor(buf139, (1, 704, 1, 1), (704, 1, 1, 1), 0), reinterpret_tensor(buf132, (1, 400, 1, 1), (400, 1, 1, 1), 0), reinterpret_tensor(buf126, (1, 400, 1, 1), (400, 1, 1, 1), 0), reinterpret_tensor(buf118, (1, 376, 1, 1), (376, 1, 1, 1), 0), reinterpret_tensor(buf109, (1, 200, 1, 1), (200, 1, 1, 1), 0), reinterpret_tensor(buf100, (1, 200, 1, 1), (200, 1, 1, 1), 0), reinterpret_tensor(buf91, (1, 356, 1, 1), (356, 1, 1, 1), 0), reinterpret_tensor(buf84, (1, 200, 1, 1), (200, 1, 1, 1), 0), reinterpret_tensor(buf75, (1, 200, 1, 1), (200, 1, 1, 1), 0), reinterpret_tensor(buf66, (1, 336, 1, 1), (336, 1, 1, 1), 0), reinterpret_tensor(buf59, (1, 200, 1, 1), (200, 1, 1, 1), 0), reinterpret_tensor(buf50, (1, 200, 1, 1), (200, 1, 1, 1), 0), reinterpret_tensor(buf41, (1, 316, 1, 1), (316, 1, 1, 1), 0), reinterpret_tensor(buf34, (1, 200, 1, 1), (200, 1, 1, 1), 0), reinterpret_tensor(buf25, (1, 200, 1, 1), (200, 1, 1, 1), 0), buf737, reinterpret_tensor(buf4, (1, 128, 1, 1), (128, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((316, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((316, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((356, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((356, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((1216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((1216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((1472, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((1472, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((1664, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((1664, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((1728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((1728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((1856, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((1856, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((1984, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((1984, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((2112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((2112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((2176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((2176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((2368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((2368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((2688, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((2688, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((128, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((296, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((200, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((200, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((276, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((200, 316, 1, 1), (316, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((200, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((276, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((200, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((200, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((276, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((200, 356, 1, 1), (356, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((200, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((276, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((640, 376, 1, 1), (376, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((400, 376, 1, 1), (376, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((400, 704, 1, 1), (704, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((400, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((400, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((400, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((400, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((400, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((400, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((1152, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((800, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((800, 1216, 1, 1), (1216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((800, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((800, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((800, 1408, 1, 1), (1408, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((800, 1472, 1, 1), (1472, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((800, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((800, 1600, 1, 1), (1600, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((800, 1664, 1, 1), (1664, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((800, 1728, 1, 1), (1728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((800, 1792, 1, 1), (1792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((800, 1856, 1, 1), (1856, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((800, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((800, 1984, 1, 1), (1984, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((800, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((800, 2112, 1, 1), (2112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((800, 2176, 1, 1), (2176, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((800, 2240, 1, 1), (2240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((800, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((800, 2368, 1, 1), (2368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((2304, 2432, 1, 1), (2432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((1600, 2432, 1, 1), (2432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((1600, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((2176, 1600, 1, 1), (1600, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((1600, 2432, 1, 1), (2432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((1600, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((2176, 1600, 1, 1), (1600, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((1600, 2560, 1, 1), (2560, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((1600, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((2176, 1600, 1, 1), (1600, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((1000, 2688, 1, 1), (2688, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_336 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_339 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_342 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_345 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_348 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_351 = rand_strided((316, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((316, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_354 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_357 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_360 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_363 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_366 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_369 = rand_strided((356, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((356, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_372 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_375 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_378 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_381 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_384 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_387 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_390 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_393 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_396 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_399 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_402 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_405 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_408 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_411 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_414 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_417 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_420 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_423 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_426 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_429 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_432 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_435 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_438 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_441 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_444 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_447 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_450 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_453 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_456 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_459 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_462 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_465 = rand_strided((1216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((1216, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_468 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_471 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_474 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_477 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_480 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_483 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_486 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_489 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_492 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_495 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_498 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_501 = rand_strided((1472, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((1472, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_504 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_507 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_510 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_513 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_516 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_519 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_522 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_525 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_528 = rand_strided((1664, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((1664, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_531 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_534 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_537 = rand_strided((1728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((1728, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_540 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_543 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_546 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_549 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_552 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_555 = rand_strided((1856, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((1856, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_558 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_561 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_564 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_567 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_570 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_573 = rand_strided((1984, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((1984, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_576 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_579 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_582 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_585 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_588 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_591 = rand_strided((2112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((2112, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_594 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_597 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_600 = rand_strided((2176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((2176, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_603 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_606 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_609 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_612 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_615 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_618 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_621 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_624 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_627 = rand_strided((2368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((2368, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_629 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_630 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_632 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_633 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_635 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_636 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_638 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_639 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_641 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_642 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_644 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_645 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_647 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_648 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_650 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_651 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_653 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_654 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_656 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_657 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_659 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_660 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_661 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_662 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_663 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_664 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_665 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_666 = rand_strided((2688, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_667 = rand_strided((2688, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_668 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('dpn107', benchmark_compiled_module)
