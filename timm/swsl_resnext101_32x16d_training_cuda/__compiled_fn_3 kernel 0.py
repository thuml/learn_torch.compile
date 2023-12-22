
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


# kernel path: /tmp/torchinductor_youkaichao/cx/ccx7xbhxmbqep7kdwzfoge3fu7arpi53l4bjepquecwvapm2ibzw.py
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
    size_hints=[1024, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_0', 'mutated_arg_names': []}
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 832
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
        tmp3 = tl.load(in_ptr0 + ((12544*x1) + (802816*(((r2 + (7720*x0)) // 12544) % 8)) + ((r2 + (7720*x0)) % 12544)), rmask & tmp2 & xmask, eviction_policy='evict_last', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/j7/cj7577cqg4n66b3xqkcmg6a3vhwdqbcq52nfscszzitfddxwo2am.py
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
    size_hints=[64, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_1', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_youkaichao/7y/c7ylhig4hyzsq4bww3ib2lx7xey7cwthx3cxpj4dxnottbujjgx5.py
# Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_1 => add_1, add_4, mul, mul_6, rsqrt, sub, var_mean
# x_2 => relu
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
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 64
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
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pk/cpkszetm6aeu3vqgbdymgqmt2tyvsuvvturvaidzvpud35vyricn.py
# Source Nodes: [shortcut], Original ATen: [aten.max_pool2d_with_indices]
# shortcut => getitem_2, getitem_3
triton_poi_fused_max_pool2d_with_indices_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
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


# kernel path: /tmp/torchinductor_youkaichao/6r/c6rj7hbj5f5anryy2mgh7rhwahwozvvwieuygzjeldcnjdunhjcr.py
# Source Nodes: [x_5], Original ATen: [aten._native_batch_norm_legit_functional]
# x_5 => add_6, add_7, add_8, mul_10, mul_11, mul_12, mul_8, mul_9, rsqrt_1, squeeze_4, var_mean_1
triton_red_fused__native_batch_norm_legit_functional_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_4', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
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
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (1605632*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/zi/cziwbqez5lwumxo7gxguud3ur4tizgyjd4ziar73xtfwq437qnqj.py
# Source Nodes: [x_5, x_6], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_5 => add_6, add_9, mul_13, mul_7, rsqrt_1, sub_1, var_mean_1
# x_6 => relu_1
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
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 512
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
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/so/cso5lohapvutzsgdtlpyc3jyvihuvoxef52xck554wfdxetfjh7e.py
# Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
# x_13 => add_16, add_17, add_18, mul_22, mul_23, mul_24, mul_25, mul_26, rsqrt_3, squeeze_10, var_mean_3
triton_red_fused__native_batch_norm_legit_functional_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_6', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
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
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (802816*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/j2/cj2afz65z742uhlpp4xec3alddzvbdg2dkiswzzyai7dgnparbwl.py
# Source Nodes: [shortcut_1, shortcut_2, x_13, x_14], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_1 => add_21, add_24, mul_28, mul_34, rsqrt_4, sub_4, var_mean_4
# shortcut_2 => relu_3
# x_13 => add_16, add_19, mul_21, mul_27, rsqrt_3, sub_3, var_mean_3
# x_14 => add_25
triton_poi_fused__native_batch_norm_legit_functional_add_relu_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 256
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
    tmp4 = 25088.0
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


# kernel path: /tmp/torchinductor_youkaichao/hb/chbrvbtsatibbzmc5dfhill633m3lzlzbqqffqtthh7itp4ikezn.py
# Source Nodes: [shortcut_3, x_25, x_26], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_3 => relu_6
# x_25 => add_37, add_40, mul_49, mul_55, rsqrt_7, sub_7, var_mean_7
# x_26 => add_41
triton_poi_fused__native_batch_norm_legit_functional_add_relu_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x3), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 25088.0
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


# kernel path: /tmp/torchinductor_youkaichao/fy/cfywnwol3574fpbnokpadmrkuxkwuilqbr73uklqhgsafl22lkea.py
# Source Nodes: [x_42], Original ATen: [aten._native_batch_norm_legit_functional]
# x_42 => add_59, add_60, add_61, mul_78, mul_79, mul_80, mul_81, mul_82, rsqrt_11, squeeze_34, var_mean_11
triton_red_fused__native_batch_norm_legit_functional_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 32768],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_9', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
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
        tmp0 = tl.load(in_ptr0 + (r1 + (3136*x0) + (3211264*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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
    tmp7 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/al/calhqlb4jwzhe27a6hbxdxrjqbmvi7cy6saosw5aeqfhlkgicesf.py
# Source Nodes: [x_42, x_43], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_42 => add_59, add_62, mul_77, mul_83, rsqrt_11, sub_11, var_mean_11
# x_43 => relu_10
triton_poi_fused__native_batch_norm_legit_functional_relu_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 1024
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
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bt/cbtnel5o6d5g7g2qorefx4gozxl3glceum7nvnzsywvabb43wj37.py
# Source Nodes: [x_45], Original ATen: [aten._native_batch_norm_legit_functional]
# x_45 => add_64, add_65, add_66, mul_85, mul_86, mul_87, mul_88, mul_89, rsqrt_12, squeeze_37, var_mean_12
triton_red_fused__native_batch_norm_legit_functional_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_11', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/5v/c5vloglft25lwrkk6msimmpyn5zrjlvuyazvgxuprm2gsha5lfgd.py
# Source Nodes: [x_45, x_47], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_45 => add_64, add_67, mul_84, mul_90, rsqrt_12, sub_12, var_mean_12
# x_47 => relu_11
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/do/cdoq7bpoqcbyadzi6mbpyxyqqchi72b53jayq3l43qom42pjiksr.py
# Source Nodes: [x_50], Original ATen: [aten._native_batch_norm_legit_functional]
# x_50 => add_69, add_70, add_71, mul_92, mul_93, mul_94, mul_95, mul_96, rsqrt_13, squeeze_40, var_mean_13
triton_red_fused__native_batch_norm_legit_functional_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_13', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (401408*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/t4/ct4gsoe5zajylwpzllpvzoefviohp4j4khcjwz3bgi4gcjniutjy.py
# Source Nodes: [shortcut_5, shortcut_6, x_50, x_51], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_5 => add_74, add_77, mul_104, mul_98, rsqrt_14, sub_14, var_mean_14
# shortcut_6 => relu_12
# x_50 => add_69, add_72, mul_91, mul_97, rsqrt_13, sub_13, var_mean_13
# x_51 => add_78
triton_poi_fused__native_batch_norm_legit_functional_add_relu_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 512
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
    tmp4 = 6272.0
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


# kernel path: /tmp/torchinductor_youkaichao/yf/cyfdhthjsf4ba6xnstscdsrfgapzxfunt2v6lwbdllxdb6sxkzjn.py
# Source Nodes: [shortcut_7, x_62, x_63], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_7 => relu_15
# x_62 => add_90, add_93, mul_119, mul_125, rsqrt_17, sub_17, var_mean_17
# x_63 => add_94
triton_poi_fused__native_batch_norm_legit_functional_add_relu_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x3), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 6272.0
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


# kernel path: /tmp/torchinductor_youkaichao/u6/cu634cfhetfgrkybld7lkg6a7loii6inog5ewzjjo4fkdbzzfgfq.py
# Source Nodes: [x_91], Original ATen: [aten._native_batch_norm_legit_functional]
# x_91 => add_128, add_129, add_130, mul_169, mul_170, mul_171, mul_172, mul_173, rsqrt_24, squeeze_73, var_mean_24
triton_red_fused__native_batch_norm_legit_functional_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_16', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 2048
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (1605632*r2)), rmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr2 + (x0), tmp9, None)
    tl.store(out_ptr4 + (x0), tmp15, None)
    tl.store(out_ptr6 + (x0), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/km/ckmfgeyaecpnhc6wgwgb37ibym3eop6bep32nsugpy44yvwtkuou.py
# Source Nodes: [x_91, x_92], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_91 => add_128, add_131, mul_168, mul_174, rsqrt_24, sub_24, var_mean_24
# x_92 => relu_22
triton_poi_fused__native_batch_norm_legit_functional_relu_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 2048
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
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3t/c3te5p36nrvtgzpmyiwpfgnseo53w3cdsfifvvmmyki7satigqz2.py
# Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_functional]
# x_94 => add_133, add_134, add_135, mul_176, mul_177, mul_178, mul_179, mul_180, rsqrt_25, squeeze_76, var_mean_25
triton_red_fused__native_batch_norm_legit_functional_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_18', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
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
    tl.store(out_ptr2 + (x0), tmp9, None)
    tl.store(out_ptr4 + (x0), tmp15, None)
    tl.store(out_ptr6 + (x0), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/76/c76tso36ogc7m5ykezbvj76rlccpaqxpmgzpf4qaqdq66ctjb3td.py
# Source Nodes: [x_94, x_96], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_94 => add_133, add_136, mul_175, mul_181, rsqrt_25, sub_25, var_mean_25
# x_96 => relu_23
triton_poi_fused__native_batch_norm_legit_functional_relu_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_19', 'mutated_arg_names': []},
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
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.math.rsqrt(tmp7)
    tmp9 = tmp2 * tmp8
    tmp11 = tmp9 * tmp10
    tmp13 = tmp11 + tmp12
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/az/cazwjf4qtbzeeyxuubvymwwwpsrb3loalg76eehwqngqpuqhalwg.py
# Source Nodes: [x_99], Original ATen: [aten._native_batch_norm_legit_functional]
# x_99 => add_138, add_139, add_140, mul_183, mul_184, mul_185, mul_186, mul_187, rsqrt_26, squeeze_79, var_mean_26
triton_red_fused__native_batch_norm_legit_functional_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_20', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1024
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/hq/chqoq72p4pg56omrvciv7mbw4o2fh2kqpdsrqo4ax52ut7h2zzha.py
# Source Nodes: [shortcut_10, shortcut_11, x_100, x_99], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_10 => add_143, add_146, mul_189, mul_195, rsqrt_27, sub_27, var_mean_27
# shortcut_11 => relu_24
# x_100 => add_147
# x_99 => add_138, add_141, mul_182, mul_188, rsqrt_26, sub_26, var_mean_26
triton_poi_fused__native_batch_norm_legit_functional_add_relu_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1024
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
    tmp4 = 1568.0
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


# kernel path: /tmp/torchinductor_youkaichao/dd/cdd2o6mjbh7xcag6lyhu5hxrwhkdgw3fibqgr3t5dc4xtqgqr5v3.py
# Source Nodes: [shortcut_12, x_111, x_112], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_12 => relu_27
# x_111 => add_159, add_162, mul_210, mul_216, rsqrt_30, sub_30, var_mean_30
# x_112 => add_163
triton_poi_fused__native_batch_norm_legit_functional_add_relu_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x3), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 1568.0
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


# kernel path: /tmp/torchinductor_youkaichao/er/cers5rkujvysq3fksmtv3t47cfr7acawicsatbizhi2dbf6qznla.py
# Source Nodes: [x_368], Original ATen: [aten._native_batch_norm_legit_functional]
# x_368 => add_501, add_502, add_503, mul_659, mul_660, mul_661, mul_662, mul_663, rsqrt_94, squeeze_283, var_mean_94
triton_red_fused__native_batch_norm_legit_functional_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_23', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4096
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (802816*r2)), rmask, eviction_policy='evict_first', other=0.0)
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
    tl.store(out_ptr2 + (x0), tmp9, None)
    tl.store(out_ptr4 + (x0), tmp15, None)
    tl.store(out_ptr6 + (x0), tmp21, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/um/cummjcdrbaklhbfkx5t3sz63tc3g3drlsfvfnmr7sibjunmqlf37.py
# Source Nodes: [x_368, x_369], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_368 => add_501, add_504, mul_658, mul_664, rsqrt_94, sub_94, var_mean_94
# x_369 => relu_91
triton_poi_fused__native_batch_norm_legit_functional_relu_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 4096
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
    tmp14 = triton_helpers.maximum(0, tmp13)
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/d7/cd7q6rtsumv22w3ch2dpwvewd4in5mzhqifrnkusdwvjqypv3r27.py
# Source Nodes: [x_371], Original ATen: [aten._native_batch_norm_legit_functional]
# x_371 => add_506, add_507, add_508, mul_666, mul_667, mul_668, mul_669, mul_670, rsqrt_95, squeeze_286, var_mean_95
triton_per_fused__native_batch_norm_legit_functional_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_25', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel):
    xnumel = 4096
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (200704*r2)), rmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 392, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = 392.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
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
    tl.store(out_ptr2 + (x0), tmp21, None)
    tl.store(out_ptr4 + (x0), tmp27, None)
    tl.store(out_ptr6 + (x0), tmp33, None)
    tl.store(out_ptr0 + (x0), tmp10, None)
    tl.store(out_ptr1 + (x0), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rp/crpr6ovhdfwyf5n7npldhhjioechzzvbspwb5ckd6srzp3oblrgy.py
# Source Nodes: [x_371, x_373], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
# x_371 => add_506, add_509, mul_665, mul_671, rsqrt_95, sub_95, var_mean_95
# x_373 => relu_92
triton_poi_fused__native_batch_norm_legit_functional_relu_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 4096
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
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


# kernel path: /tmp/torchinductor_youkaichao/wp/cwpm427h56mqarc7thqwsmgtsvssckqr4qg3ntkwyd7hdxyqz7ba.py
# Source Nodes: [x_376], Original ATen: [aten._native_batch_norm_legit_functional]
# x_376 => add_511, add_512, add_513, mul_673, mul_674, mul_675, mul_676, mul_677, rsqrt_96, squeeze_289, var_mean_96
triton_per_fused__native_batch_norm_legit_functional_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_27', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel):
    xnumel = 2048
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (100352*r2)), rmask, other=0.0)
    tmp24 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp1 = tl.broadcast_to(tmp0, [RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp6 = tl.where(rmask, tmp4, 0)
    tmp7 = triton_helpers.promote_to_tensor(tl.sum(tmp6, 0))
    tmp8 = tl.full([1], 392, tl.int32)
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp7 / tmp9
    tmp11 = tmp1 - tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tl.broadcast_to(tmp12, [RBLOCK])
    tmp15 = tl.where(rmask, tmp13, 0)
    tmp16 = triton_helpers.promote_to_tensor(tl.sum(tmp15, 0))
    tmp17 = 392.0
    tmp18 = tmp16 / tmp17
    tmp19 = 1e-05
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
    tl.store(out_ptr2 + (x0), tmp21, None)
    tl.store(out_ptr4 + (x0), tmp27, None)
    tl.store(out_ptr6 + (x0), tmp33, None)
    tl.store(out_ptr0 + (x0), tmp10, None)
    tl.store(out_ptr1 + (x0), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wf/cwfh4rcyk5aqneczblh5keys6n5kcnrhef4phbj4qbcxdkd5d5ho.py
# Source Nodes: [shortcut_34, shortcut_35, x_376, x_377], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_34 => add_516, add_519, mul_679, mul_685, rsqrt_97, sub_97, var_mean_97
# shortcut_35 => relu_93
# x_376 => add_511, add_514, mul_672, mul_678, rsqrt_96, sub_96, var_mean_96
# x_377 => add_520
triton_poi_fused__native_batch_norm_legit_functional_add_relu_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_28', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 2048
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
    tmp4 = 392.0
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


# kernel path: /tmp/torchinductor_youkaichao/cb/ccbituhv3zppbpn24eutcbjobfteu6gkrqf642gxkwu6gbed6qhj.py
# Source Nodes: [shortcut_36, x_388, x_389], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# shortcut_36 => relu_96
# x_388 => add_532, add_535, mul_700, mul_706, rsqrt_100, sub_100, var_mean_100
# x_389 => add_536
triton_poi_fused__native_batch_norm_legit_functional_add_relu_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 2048
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (x3), None)
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
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


# kernel path: /tmp/torchinductor_youkaichao/bw/cbw6fgw5ma4tyumwrmccow25asbphkymf42wi2lcjkb4udjihdj7.py
# Source Nodes: [x_400, x_401, x_404, x_405, x_407], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.mean, aten.relu, aten.threshold_backward, aten.view]
# x_400 => add_548, add_551, mul_721, mul_727, rsqrt_103, sub_103, var_mean_103
# x_401 => add_552
# x_404 => relu_99
# x_405 => mean
# x_407 => view
triton_per_fused__native_batch_norm_legit_functional_add_mean_relu_threshold_backward_view_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_add_mean_relu_threshold_backward_view_30', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 2048
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp14 = tl.load(in_ptr5 + (r2 + (49*x3)), rmask, other=0.0)
    tmp2 = tmp0 - tmp1
    tmp4 = 392.0
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
    tmp23 = 49.0
    tmp24 = tmp22 / tmp23
    tl.store(out_ptr1 + (r2 + (49*x3)), tmp18, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp24, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gk/cgkkxpozguxolu24fykrmdlsft2xxi3cwfz4ozfrtcaylngg6u5p.py
# Source Nodes: [x_1], Original ATen: [aten.add]
# x_1 => add
triton_poi_fused_add_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_31', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (512, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_5, (512, ), (1, ))
    assert_size_stride(primals_6, (512, ), (1, ))
    assert_size_stride(primals_7, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_8, (512, ), (1, ))
    assert_size_stride(primals_9, (512, ), (1, ))
    assert_size_stride(primals_10, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_11, (256, ), (1, ))
    assert_size_stride(primals_12, (256, ), (1, ))
    assert_size_stride(primals_13, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_16, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_17, (512, ), (1, ))
    assert_size_stride(primals_18, (512, ), (1, ))
    assert_size_stride(primals_19, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_20, (512, ), (1, ))
    assert_size_stride(primals_21, (512, ), (1, ))
    assert_size_stride(primals_22, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_24, (256, ), (1, ))
    assert_size_stride(primals_25, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_26, (512, ), (1, ))
    assert_size_stride(primals_27, (512, ), (1, ))
    assert_size_stride(primals_28, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_29, (512, ), (1, ))
    assert_size_stride(primals_30, (512, ), (1, ))
    assert_size_stride(primals_31, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_32, (256, ), (1, ))
    assert_size_stride(primals_33, (256, ), (1, ))
    assert_size_stride(primals_34, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_35, (1024, ), (1, ))
    assert_size_stride(primals_36, (1024, ), (1, ))
    assert_size_stride(primals_37, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_38, (1024, ), (1, ))
    assert_size_stride(primals_39, (1024, ), (1, ))
    assert_size_stride(primals_40, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_42, (512, ), (1, ))
    assert_size_stride(primals_43, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_44, (512, ), (1, ))
    assert_size_stride(primals_45, (512, ), (1, ))
    assert_size_stride(primals_46, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_47, (1024, ), (1, ))
    assert_size_stride(primals_48, (1024, ), (1, ))
    assert_size_stride(primals_49, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_50, (1024, ), (1, ))
    assert_size_stride(primals_51, (1024, ), (1, ))
    assert_size_stride(primals_52, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_53, (512, ), (1, ))
    assert_size_stride(primals_54, (512, ), (1, ))
    assert_size_stride(primals_55, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_56, (1024, ), (1, ))
    assert_size_stride(primals_57, (1024, ), (1, ))
    assert_size_stride(primals_58, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_59, (1024, ), (1, ))
    assert_size_stride(primals_60, (1024, ), (1, ))
    assert_size_stride(primals_61, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_62, (512, ), (1, ))
    assert_size_stride(primals_63, (512, ), (1, ))
    assert_size_stride(primals_64, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_65, (1024, ), (1, ))
    assert_size_stride(primals_66, (1024, ), (1, ))
    assert_size_stride(primals_67, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_68, (1024, ), (1, ))
    assert_size_stride(primals_69, (1024, ), (1, ))
    assert_size_stride(primals_70, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_71, (512, ), (1, ))
    assert_size_stride(primals_72, (512, ), (1, ))
    assert_size_stride(primals_73, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_74, (2048, ), (1, ))
    assert_size_stride(primals_75, (2048, ), (1, ))
    assert_size_stride(primals_76, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_77, (2048, ), (1, ))
    assert_size_stride(primals_78, (2048, ), (1, ))
    assert_size_stride(primals_79, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_80, (1024, ), (1, ))
    assert_size_stride(primals_81, (1024, ), (1, ))
    assert_size_stride(primals_82, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_83, (1024, ), (1, ))
    assert_size_stride(primals_84, (1024, ), (1, ))
    assert_size_stride(primals_85, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_86, (2048, ), (1, ))
    assert_size_stride(primals_87, (2048, ), (1, ))
    assert_size_stride(primals_88, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_89, (2048, ), (1, ))
    assert_size_stride(primals_90, (2048, ), (1, ))
    assert_size_stride(primals_91, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_92, (1024, ), (1, ))
    assert_size_stride(primals_93, (1024, ), (1, ))
    assert_size_stride(primals_94, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_95, (2048, ), (1, ))
    assert_size_stride(primals_96, (2048, ), (1, ))
    assert_size_stride(primals_97, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_98, (2048, ), (1, ))
    assert_size_stride(primals_99, (2048, ), (1, ))
    assert_size_stride(primals_100, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_101, (1024, ), (1, ))
    assert_size_stride(primals_102, (1024, ), (1, ))
    assert_size_stride(primals_103, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_104, (2048, ), (1, ))
    assert_size_stride(primals_105, (2048, ), (1, ))
    assert_size_stride(primals_106, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_107, (2048, ), (1, ))
    assert_size_stride(primals_108, (2048, ), (1, ))
    assert_size_stride(primals_109, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_110, (1024, ), (1, ))
    assert_size_stride(primals_111, (1024, ), (1, ))
    assert_size_stride(primals_112, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_113, (2048, ), (1, ))
    assert_size_stride(primals_114, (2048, ), (1, ))
    assert_size_stride(primals_115, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_116, (2048, ), (1, ))
    assert_size_stride(primals_117, (2048, ), (1, ))
    assert_size_stride(primals_118, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_119, (1024, ), (1, ))
    assert_size_stride(primals_120, (1024, ), (1, ))
    assert_size_stride(primals_121, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_122, (2048, ), (1, ))
    assert_size_stride(primals_123, (2048, ), (1, ))
    assert_size_stride(primals_124, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_125, (2048, ), (1, ))
    assert_size_stride(primals_126, (2048, ), (1, ))
    assert_size_stride(primals_127, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_128, (1024, ), (1, ))
    assert_size_stride(primals_129, (1024, ), (1, ))
    assert_size_stride(primals_130, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_131, (2048, ), (1, ))
    assert_size_stride(primals_132, (2048, ), (1, ))
    assert_size_stride(primals_133, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_134, (2048, ), (1, ))
    assert_size_stride(primals_135, (2048, ), (1, ))
    assert_size_stride(primals_136, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_137, (1024, ), (1, ))
    assert_size_stride(primals_138, (1024, ), (1, ))
    assert_size_stride(primals_139, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_140, (2048, ), (1, ))
    assert_size_stride(primals_141, (2048, ), (1, ))
    assert_size_stride(primals_142, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_143, (2048, ), (1, ))
    assert_size_stride(primals_144, (2048, ), (1, ))
    assert_size_stride(primals_145, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_146, (1024, ), (1, ))
    assert_size_stride(primals_147, (1024, ), (1, ))
    assert_size_stride(primals_148, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_149, (2048, ), (1, ))
    assert_size_stride(primals_150, (2048, ), (1, ))
    assert_size_stride(primals_151, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_152, (2048, ), (1, ))
    assert_size_stride(primals_153, (2048, ), (1, ))
    assert_size_stride(primals_154, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_155, (1024, ), (1, ))
    assert_size_stride(primals_156, (1024, ), (1, ))
    assert_size_stride(primals_157, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_158, (2048, ), (1, ))
    assert_size_stride(primals_159, (2048, ), (1, ))
    assert_size_stride(primals_160, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_161, (2048, ), (1, ))
    assert_size_stride(primals_162, (2048, ), (1, ))
    assert_size_stride(primals_163, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_164, (1024, ), (1, ))
    assert_size_stride(primals_165, (1024, ), (1, ))
    assert_size_stride(primals_166, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_167, (2048, ), (1, ))
    assert_size_stride(primals_168, (2048, ), (1, ))
    assert_size_stride(primals_169, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_170, (2048, ), (1, ))
    assert_size_stride(primals_171, (2048, ), (1, ))
    assert_size_stride(primals_172, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_173, (1024, ), (1, ))
    assert_size_stride(primals_174, (1024, ), (1, ))
    assert_size_stride(primals_175, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_176, (2048, ), (1, ))
    assert_size_stride(primals_177, (2048, ), (1, ))
    assert_size_stride(primals_178, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_179, (2048, ), (1, ))
    assert_size_stride(primals_180, (2048, ), (1, ))
    assert_size_stride(primals_181, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_182, (1024, ), (1, ))
    assert_size_stride(primals_183, (1024, ), (1, ))
    assert_size_stride(primals_184, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_185, (2048, ), (1, ))
    assert_size_stride(primals_186, (2048, ), (1, ))
    assert_size_stride(primals_187, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_188, (2048, ), (1, ))
    assert_size_stride(primals_189, (2048, ), (1, ))
    assert_size_stride(primals_190, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_191, (1024, ), (1, ))
    assert_size_stride(primals_192, (1024, ), (1, ))
    assert_size_stride(primals_193, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_194, (2048, ), (1, ))
    assert_size_stride(primals_195, (2048, ), (1, ))
    assert_size_stride(primals_196, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_197, (2048, ), (1, ))
    assert_size_stride(primals_198, (2048, ), (1, ))
    assert_size_stride(primals_199, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_200, (1024, ), (1, ))
    assert_size_stride(primals_201, (1024, ), (1, ))
    assert_size_stride(primals_202, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_203, (2048, ), (1, ))
    assert_size_stride(primals_204, (2048, ), (1, ))
    assert_size_stride(primals_205, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_206, (2048, ), (1, ))
    assert_size_stride(primals_207, (2048, ), (1, ))
    assert_size_stride(primals_208, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_209, (1024, ), (1, ))
    assert_size_stride(primals_210, (1024, ), (1, ))
    assert_size_stride(primals_211, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_212, (2048, ), (1, ))
    assert_size_stride(primals_213, (2048, ), (1, ))
    assert_size_stride(primals_214, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_215, (2048, ), (1, ))
    assert_size_stride(primals_216, (2048, ), (1, ))
    assert_size_stride(primals_217, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_218, (1024, ), (1, ))
    assert_size_stride(primals_219, (1024, ), (1, ))
    assert_size_stride(primals_220, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_221, (2048, ), (1, ))
    assert_size_stride(primals_222, (2048, ), (1, ))
    assert_size_stride(primals_223, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_224, (2048, ), (1, ))
    assert_size_stride(primals_225, (2048, ), (1, ))
    assert_size_stride(primals_226, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_227, (1024, ), (1, ))
    assert_size_stride(primals_228, (1024, ), (1, ))
    assert_size_stride(primals_229, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_230, (2048, ), (1, ))
    assert_size_stride(primals_231, (2048, ), (1, ))
    assert_size_stride(primals_232, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_233, (2048, ), (1, ))
    assert_size_stride(primals_234, (2048, ), (1, ))
    assert_size_stride(primals_235, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_236, (1024, ), (1, ))
    assert_size_stride(primals_237, (1024, ), (1, ))
    assert_size_stride(primals_238, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_239, (2048, ), (1, ))
    assert_size_stride(primals_240, (2048, ), (1, ))
    assert_size_stride(primals_241, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_242, (2048, ), (1, ))
    assert_size_stride(primals_243, (2048, ), (1, ))
    assert_size_stride(primals_244, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_245, (1024, ), (1, ))
    assert_size_stride(primals_246, (1024, ), (1, ))
    assert_size_stride(primals_247, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_248, (2048, ), (1, ))
    assert_size_stride(primals_249, (2048, ), (1, ))
    assert_size_stride(primals_250, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_251, (2048, ), (1, ))
    assert_size_stride(primals_252, (2048, ), (1, ))
    assert_size_stride(primals_253, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_254, (1024, ), (1, ))
    assert_size_stride(primals_255, (1024, ), (1, ))
    assert_size_stride(primals_256, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_257, (2048, ), (1, ))
    assert_size_stride(primals_258, (2048, ), (1, ))
    assert_size_stride(primals_259, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_260, (2048, ), (1, ))
    assert_size_stride(primals_261, (2048, ), (1, ))
    assert_size_stride(primals_262, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_263, (1024, ), (1, ))
    assert_size_stride(primals_264, (1024, ), (1, ))
    assert_size_stride(primals_265, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_266, (2048, ), (1, ))
    assert_size_stride(primals_267, (2048, ), (1, ))
    assert_size_stride(primals_268, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_269, (2048, ), (1, ))
    assert_size_stride(primals_270, (2048, ), (1, ))
    assert_size_stride(primals_271, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_272, (1024, ), (1, ))
    assert_size_stride(primals_273, (1024, ), (1, ))
    assert_size_stride(primals_274, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_275, (2048, ), (1, ))
    assert_size_stride(primals_276, (2048, ), (1, ))
    assert_size_stride(primals_277, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_278, (2048, ), (1, ))
    assert_size_stride(primals_279, (2048, ), (1, ))
    assert_size_stride(primals_280, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_281, (1024, ), (1, ))
    assert_size_stride(primals_282, (1024, ), (1, ))
    assert_size_stride(primals_283, (4096, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_284, (4096, ), (1, ))
    assert_size_stride(primals_285, (4096, ), (1, ))
    assert_size_stride(primals_286, (4096, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_287, (4096, ), (1, ))
    assert_size_stride(primals_288, (4096, ), (1, ))
    assert_size_stride(primals_289, (2048, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(primals_290, (2048, ), (1, ))
    assert_size_stride(primals_291, (2048, ), (1, ))
    assert_size_stride(primals_292, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_293, (2048, ), (1, ))
    assert_size_stride(primals_294, (2048, ), (1, ))
    assert_size_stride(primals_295, (4096, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_296, (4096, ), (1, ))
    assert_size_stride(primals_297, (4096, ), (1, ))
    assert_size_stride(primals_298, (4096, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_299, (4096, ), (1, ))
    assert_size_stride(primals_300, (4096, ), (1, ))
    assert_size_stride(primals_301, (2048, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(primals_302, (2048, ), (1, ))
    assert_size_stride(primals_303, (2048, ), (1, ))
    assert_size_stride(primals_304, (4096, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_305, (4096, ), (1, ))
    assert_size_stride(primals_306, (4096, ), (1, ))
    assert_size_stride(primals_307, (4096, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_308, (4096, ), (1, ))
    assert_size_stride(primals_309, (4096, ), (1, ))
    assert_size_stride(primals_310, (2048, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(primals_311, (2048, ), (1, ))
    assert_size_stride(primals_312, (2048, ), (1, ))
    assert_size_stride(primals_313, (1000, 2048), (2048, 1))
    assert_size_stride(primals_314, (1000, ), (1, ))
    assert_size_stride(primals_315, (64, ), (1, ))
    assert_size_stride(primals_316, (64, ), (1, ))
    assert_size_stride(primals_317, (), ())
    assert_size_stride(primals_318, (512, ), (1, ))
    assert_size_stride(primals_319, (512, ), (1, ))
    assert_size_stride(primals_320, (), ())
    assert_size_stride(primals_321, (512, ), (1, ))
    assert_size_stride(primals_322, (512, ), (1, ))
    assert_size_stride(primals_323, (), ())
    assert_size_stride(primals_324, (256, ), (1, ))
    assert_size_stride(primals_325, (256, ), (1, ))
    assert_size_stride(primals_326, (), ())
    assert_size_stride(primals_327, (256, ), (1, ))
    assert_size_stride(primals_328, (256, ), (1, ))
    assert_size_stride(primals_329, (), ())
    assert_size_stride(primals_330, (512, ), (1, ))
    assert_size_stride(primals_331, (512, ), (1, ))
    assert_size_stride(primals_332, (), ())
    assert_size_stride(primals_333, (512, ), (1, ))
    assert_size_stride(primals_334, (512, ), (1, ))
    assert_size_stride(primals_335, (), ())
    assert_size_stride(primals_336, (256, ), (1, ))
    assert_size_stride(primals_337, (256, ), (1, ))
    assert_size_stride(primals_338, (), ())
    assert_size_stride(primals_339, (512, ), (1, ))
    assert_size_stride(primals_340, (512, ), (1, ))
    assert_size_stride(primals_341, (), ())
    assert_size_stride(primals_342, (512, ), (1, ))
    assert_size_stride(primals_343, (512, ), (1, ))
    assert_size_stride(primals_344, (), ())
    assert_size_stride(primals_345, (256, ), (1, ))
    assert_size_stride(primals_346, (256, ), (1, ))
    assert_size_stride(primals_347, (), ())
    assert_size_stride(primals_348, (1024, ), (1, ))
    assert_size_stride(primals_349, (1024, ), (1, ))
    assert_size_stride(primals_350, (), ())
    assert_size_stride(primals_351, (1024, ), (1, ))
    assert_size_stride(primals_352, (1024, ), (1, ))
    assert_size_stride(primals_353, (), ())
    assert_size_stride(primals_354, (512, ), (1, ))
    assert_size_stride(primals_355, (512, ), (1, ))
    assert_size_stride(primals_356, (), ())
    assert_size_stride(primals_357, (512, ), (1, ))
    assert_size_stride(primals_358, (512, ), (1, ))
    assert_size_stride(primals_359, (), ())
    assert_size_stride(primals_360, (1024, ), (1, ))
    assert_size_stride(primals_361, (1024, ), (1, ))
    assert_size_stride(primals_362, (), ())
    assert_size_stride(primals_363, (1024, ), (1, ))
    assert_size_stride(primals_364, (1024, ), (1, ))
    assert_size_stride(primals_365, (), ())
    assert_size_stride(primals_366, (512, ), (1, ))
    assert_size_stride(primals_367, (512, ), (1, ))
    assert_size_stride(primals_368, (), ())
    assert_size_stride(primals_369, (1024, ), (1, ))
    assert_size_stride(primals_370, (1024, ), (1, ))
    assert_size_stride(primals_371, (), ())
    assert_size_stride(primals_372, (1024, ), (1, ))
    assert_size_stride(primals_373, (1024, ), (1, ))
    assert_size_stride(primals_374, (), ())
    assert_size_stride(primals_375, (512, ), (1, ))
    assert_size_stride(primals_376, (512, ), (1, ))
    assert_size_stride(primals_377, (), ())
    assert_size_stride(primals_378, (1024, ), (1, ))
    assert_size_stride(primals_379, (1024, ), (1, ))
    assert_size_stride(primals_380, (), ())
    assert_size_stride(primals_381, (1024, ), (1, ))
    assert_size_stride(primals_382, (1024, ), (1, ))
    assert_size_stride(primals_383, (), ())
    assert_size_stride(primals_384, (512, ), (1, ))
    assert_size_stride(primals_385, (512, ), (1, ))
    assert_size_stride(primals_386, (), ())
    assert_size_stride(primals_387, (2048, ), (1, ))
    assert_size_stride(primals_388, (2048, ), (1, ))
    assert_size_stride(primals_389, (), ())
    assert_size_stride(primals_390, (2048, ), (1, ))
    assert_size_stride(primals_391, (2048, ), (1, ))
    assert_size_stride(primals_392, (), ())
    assert_size_stride(primals_393, (1024, ), (1, ))
    assert_size_stride(primals_394, (1024, ), (1, ))
    assert_size_stride(primals_395, (), ())
    assert_size_stride(primals_396, (1024, ), (1, ))
    assert_size_stride(primals_397, (1024, ), (1, ))
    assert_size_stride(primals_398, (), ())
    assert_size_stride(primals_399, (2048, ), (1, ))
    assert_size_stride(primals_400, (2048, ), (1, ))
    assert_size_stride(primals_401, (), ())
    assert_size_stride(primals_402, (2048, ), (1, ))
    assert_size_stride(primals_403, (2048, ), (1, ))
    assert_size_stride(primals_404, (), ())
    assert_size_stride(primals_405, (1024, ), (1, ))
    assert_size_stride(primals_406, (1024, ), (1, ))
    assert_size_stride(primals_407, (), ())
    assert_size_stride(primals_408, (2048, ), (1, ))
    assert_size_stride(primals_409, (2048, ), (1, ))
    assert_size_stride(primals_410, (), ())
    assert_size_stride(primals_411, (2048, ), (1, ))
    assert_size_stride(primals_412, (2048, ), (1, ))
    assert_size_stride(primals_413, (), ())
    assert_size_stride(primals_414, (1024, ), (1, ))
    assert_size_stride(primals_415, (1024, ), (1, ))
    assert_size_stride(primals_416, (), ())
    assert_size_stride(primals_417, (2048, ), (1, ))
    assert_size_stride(primals_418, (2048, ), (1, ))
    assert_size_stride(primals_419, (), ())
    assert_size_stride(primals_420, (2048, ), (1, ))
    assert_size_stride(primals_421, (2048, ), (1, ))
    assert_size_stride(primals_422, (), ())
    assert_size_stride(primals_423, (1024, ), (1, ))
    assert_size_stride(primals_424, (1024, ), (1, ))
    assert_size_stride(primals_425, (), ())
    assert_size_stride(primals_426, (2048, ), (1, ))
    assert_size_stride(primals_427, (2048, ), (1, ))
    assert_size_stride(primals_428, (), ())
    assert_size_stride(primals_429, (2048, ), (1, ))
    assert_size_stride(primals_430, (2048, ), (1, ))
    assert_size_stride(primals_431, (), ())
    assert_size_stride(primals_432, (1024, ), (1, ))
    assert_size_stride(primals_433, (1024, ), (1, ))
    assert_size_stride(primals_434, (), ())
    assert_size_stride(primals_435, (2048, ), (1, ))
    assert_size_stride(primals_436, (2048, ), (1, ))
    assert_size_stride(primals_437, (), ())
    assert_size_stride(primals_438, (2048, ), (1, ))
    assert_size_stride(primals_439, (2048, ), (1, ))
    assert_size_stride(primals_440, (), ())
    assert_size_stride(primals_441, (1024, ), (1, ))
    assert_size_stride(primals_442, (1024, ), (1, ))
    assert_size_stride(primals_443, (), ())
    assert_size_stride(primals_444, (2048, ), (1, ))
    assert_size_stride(primals_445, (2048, ), (1, ))
    assert_size_stride(primals_446, (), ())
    assert_size_stride(primals_447, (2048, ), (1, ))
    assert_size_stride(primals_448, (2048, ), (1, ))
    assert_size_stride(primals_449, (), ())
    assert_size_stride(primals_450, (1024, ), (1, ))
    assert_size_stride(primals_451, (1024, ), (1, ))
    assert_size_stride(primals_452, (), ())
    assert_size_stride(primals_453, (2048, ), (1, ))
    assert_size_stride(primals_454, (2048, ), (1, ))
    assert_size_stride(primals_455, (), ())
    assert_size_stride(primals_456, (2048, ), (1, ))
    assert_size_stride(primals_457, (2048, ), (1, ))
    assert_size_stride(primals_458, (), ())
    assert_size_stride(primals_459, (1024, ), (1, ))
    assert_size_stride(primals_460, (1024, ), (1, ))
    assert_size_stride(primals_461, (), ())
    assert_size_stride(primals_462, (2048, ), (1, ))
    assert_size_stride(primals_463, (2048, ), (1, ))
    assert_size_stride(primals_464, (), ())
    assert_size_stride(primals_465, (2048, ), (1, ))
    assert_size_stride(primals_466, (2048, ), (1, ))
    assert_size_stride(primals_467, (), ())
    assert_size_stride(primals_468, (1024, ), (1, ))
    assert_size_stride(primals_469, (1024, ), (1, ))
    assert_size_stride(primals_470, (), ())
    assert_size_stride(primals_471, (2048, ), (1, ))
    assert_size_stride(primals_472, (2048, ), (1, ))
    assert_size_stride(primals_473, (), ())
    assert_size_stride(primals_474, (2048, ), (1, ))
    assert_size_stride(primals_475, (2048, ), (1, ))
    assert_size_stride(primals_476, (), ())
    assert_size_stride(primals_477, (1024, ), (1, ))
    assert_size_stride(primals_478, (1024, ), (1, ))
    assert_size_stride(primals_479, (), ())
    assert_size_stride(primals_480, (2048, ), (1, ))
    assert_size_stride(primals_481, (2048, ), (1, ))
    assert_size_stride(primals_482, (), ())
    assert_size_stride(primals_483, (2048, ), (1, ))
    assert_size_stride(primals_484, (2048, ), (1, ))
    assert_size_stride(primals_485, (), ())
    assert_size_stride(primals_486, (1024, ), (1, ))
    assert_size_stride(primals_487, (1024, ), (1, ))
    assert_size_stride(primals_488, (), ())
    assert_size_stride(primals_489, (2048, ), (1, ))
    assert_size_stride(primals_490, (2048, ), (1, ))
    assert_size_stride(primals_491, (), ())
    assert_size_stride(primals_492, (2048, ), (1, ))
    assert_size_stride(primals_493, (2048, ), (1, ))
    assert_size_stride(primals_494, (), ())
    assert_size_stride(primals_495, (1024, ), (1, ))
    assert_size_stride(primals_496, (1024, ), (1, ))
    assert_size_stride(primals_497, (), ())
    assert_size_stride(primals_498, (2048, ), (1, ))
    assert_size_stride(primals_499, (2048, ), (1, ))
    assert_size_stride(primals_500, (), ())
    assert_size_stride(primals_501, (2048, ), (1, ))
    assert_size_stride(primals_502, (2048, ), (1, ))
    assert_size_stride(primals_503, (), ())
    assert_size_stride(primals_504, (1024, ), (1, ))
    assert_size_stride(primals_505, (1024, ), (1, ))
    assert_size_stride(primals_506, (), ())
    assert_size_stride(primals_507, (2048, ), (1, ))
    assert_size_stride(primals_508, (2048, ), (1, ))
    assert_size_stride(primals_509, (), ())
    assert_size_stride(primals_510, (2048, ), (1, ))
    assert_size_stride(primals_511, (2048, ), (1, ))
    assert_size_stride(primals_512, (), ())
    assert_size_stride(primals_513, (1024, ), (1, ))
    assert_size_stride(primals_514, (1024, ), (1, ))
    assert_size_stride(primals_515, (), ())
    assert_size_stride(primals_516, (2048, ), (1, ))
    assert_size_stride(primals_517, (2048, ), (1, ))
    assert_size_stride(primals_518, (), ())
    assert_size_stride(primals_519, (2048, ), (1, ))
    assert_size_stride(primals_520, (2048, ), (1, ))
    assert_size_stride(primals_521, (), ())
    assert_size_stride(primals_522, (1024, ), (1, ))
    assert_size_stride(primals_523, (1024, ), (1, ))
    assert_size_stride(primals_524, (), ())
    assert_size_stride(primals_525, (2048, ), (1, ))
    assert_size_stride(primals_526, (2048, ), (1, ))
    assert_size_stride(primals_527, (), ())
    assert_size_stride(primals_528, (2048, ), (1, ))
    assert_size_stride(primals_529, (2048, ), (1, ))
    assert_size_stride(primals_530, (), ())
    assert_size_stride(primals_531, (1024, ), (1, ))
    assert_size_stride(primals_532, (1024, ), (1, ))
    assert_size_stride(primals_533, (), ())
    assert_size_stride(primals_534, (2048, ), (1, ))
    assert_size_stride(primals_535, (2048, ), (1, ))
    assert_size_stride(primals_536, (), ())
    assert_size_stride(primals_537, (2048, ), (1, ))
    assert_size_stride(primals_538, (2048, ), (1, ))
    assert_size_stride(primals_539, (), ())
    assert_size_stride(primals_540, (1024, ), (1, ))
    assert_size_stride(primals_541, (1024, ), (1, ))
    assert_size_stride(primals_542, (), ())
    assert_size_stride(primals_543, (2048, ), (1, ))
    assert_size_stride(primals_544, (2048, ), (1, ))
    assert_size_stride(primals_545, (), ())
    assert_size_stride(primals_546, (2048, ), (1, ))
    assert_size_stride(primals_547, (2048, ), (1, ))
    assert_size_stride(primals_548, (), ())
    assert_size_stride(primals_549, (1024, ), (1, ))
    assert_size_stride(primals_550, (1024, ), (1, ))
    assert_size_stride(primals_551, (), ())
    assert_size_stride(primals_552, (2048, ), (1, ))
    assert_size_stride(primals_553, (2048, ), (1, ))
    assert_size_stride(primals_554, (), ())
    assert_size_stride(primals_555, (2048, ), (1, ))
    assert_size_stride(primals_556, (2048, ), (1, ))
    assert_size_stride(primals_557, (), ())
    assert_size_stride(primals_558, (1024, ), (1, ))
    assert_size_stride(primals_559, (1024, ), (1, ))
    assert_size_stride(primals_560, (), ())
    assert_size_stride(primals_561, (2048, ), (1, ))
    assert_size_stride(primals_562, (2048, ), (1, ))
    assert_size_stride(primals_563, (), ())
    assert_size_stride(primals_564, (2048, ), (1, ))
    assert_size_stride(primals_565, (2048, ), (1, ))
    assert_size_stride(primals_566, (), ())
    assert_size_stride(primals_567, (1024, ), (1, ))
    assert_size_stride(primals_568, (1024, ), (1, ))
    assert_size_stride(primals_569, (), ())
    assert_size_stride(primals_570, (2048, ), (1, ))
    assert_size_stride(primals_571, (2048, ), (1, ))
    assert_size_stride(primals_572, (), ())
    assert_size_stride(primals_573, (2048, ), (1, ))
    assert_size_stride(primals_574, (2048, ), (1, ))
    assert_size_stride(primals_575, (), ())
    assert_size_stride(primals_576, (1024, ), (1, ))
    assert_size_stride(primals_577, (1024, ), (1, ))
    assert_size_stride(primals_578, (), ())
    assert_size_stride(primals_579, (2048, ), (1, ))
    assert_size_stride(primals_580, (2048, ), (1, ))
    assert_size_stride(primals_581, (), ())
    assert_size_stride(primals_582, (2048, ), (1, ))
    assert_size_stride(primals_583, (2048, ), (1, ))
    assert_size_stride(primals_584, (), ())
    assert_size_stride(primals_585, (1024, ), (1, ))
    assert_size_stride(primals_586, (1024, ), (1, ))
    assert_size_stride(primals_587, (), ())
    assert_size_stride(primals_588, (2048, ), (1, ))
    assert_size_stride(primals_589, (2048, ), (1, ))
    assert_size_stride(primals_590, (), ())
    assert_size_stride(primals_591, (2048, ), (1, ))
    assert_size_stride(primals_592, (2048, ), (1, ))
    assert_size_stride(primals_593, (), ())
    assert_size_stride(primals_594, (1024, ), (1, ))
    assert_size_stride(primals_595, (1024, ), (1, ))
    assert_size_stride(primals_596, (), ())
    assert_size_stride(primals_597, (4096, ), (1, ))
    assert_size_stride(primals_598, (4096, ), (1, ))
    assert_size_stride(primals_599, (), ())
    assert_size_stride(primals_600, (4096, ), (1, ))
    assert_size_stride(primals_601, (4096, ), (1, ))
    assert_size_stride(primals_602, (), ())
    assert_size_stride(primals_603, (2048, ), (1, ))
    assert_size_stride(primals_604, (2048, ), (1, ))
    assert_size_stride(primals_605, (), ())
    assert_size_stride(primals_606, (2048, ), (1, ))
    assert_size_stride(primals_607, (2048, ), (1, ))
    assert_size_stride(primals_608, (), ())
    assert_size_stride(primals_609, (4096, ), (1, ))
    assert_size_stride(primals_610, (4096, ), (1, ))
    assert_size_stride(primals_611, (), ())
    assert_size_stride(primals_612, (4096, ), (1, ))
    assert_size_stride(primals_613, (4096, ), (1, ))
    assert_size_stride(primals_614, (), ())
    assert_size_stride(primals_615, (2048, ), (1, ))
    assert_size_stride(primals_616, (2048, ), (1, ))
    assert_size_stride(primals_617, (), ())
    assert_size_stride(primals_618, (4096, ), (1, ))
    assert_size_stride(primals_619, (4096, ), (1, ))
    assert_size_stride(primals_620, (), ())
    assert_size_stride(primals_621, (4096, ), (1, ))
    assert_size_stride(primals_622, (4096, ), (1, ))
    assert_size_stride(primals_623, (), ())
    assert_size_stride(primals_624, (2048, ), (1, ))
    assert_size_stride(primals_625, (2048, ), (1, ))
    assert_size_stride(primals_626, (), ())
    assert_size_stride(primals_627, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_627, primals_1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 64, 112, 112), (802816, 12544, 112, 1))
        buf1 = empty_strided((1, 64, 1, 1, 13), (832, 13, 832, 832, 1), device='cuda', dtype=torch.float32)
        buf2 = empty_strided((1, 64, 1, 1, 13), (832, 13, 832, 832, 1), device='cuda', dtype=torch.float32)
        buf3 = empty_strided((1, 64, 1, 1, 13), (832, 13, 832, 832, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        stream0 = get_cuda_stream(0)
        triton_red_fused__native_batch_norm_legit_functional_0.run(buf0, buf1, buf2, buf3, 832, 7720, grid=grid(832), stream=stream0)
        buf4 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf5 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf7 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_1.run(buf1, buf2, buf3, primals_315, primals_316, buf4, buf5, buf7, primals_315, primals_316, 64, 13, grid=grid(64), stream=stream0)
        del buf1
        del buf2
        del buf3
        del primals_315
        del primals_316
        buf8 = empty((8, 64, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_2.run(buf0, buf4, buf5, primals_2, primals_3, buf8, 6422528, grid=grid(6422528), stream=stream0)
        del buf5
        del primals_3
        buf9 = empty((8, 64, 56, 56), device='cuda', dtype=torch.float32)
        buf10 = empty((8, 64, 56, 56), device='cuda', dtype=torch.int64)
        # Source Nodes: [shortcut], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_3.run(buf8, buf9, buf10, 1605632, grid=grid(1605632), stream=stream0)
        # Source Nodes: [x_4], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf9, primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (8, 512, 56, 56), (1605632, 3136, 56, 1))
        buf12 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf13 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf15 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf11, primals_318, primals_319, buf12, buf13, buf15, primals_318, primals_319, 512, 25088, grid=grid(512), stream=stream0)
        del primals_318
        del primals_319
        buf16 = empty((8, 512, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_5, x_6], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf11, buf12, buf13, primals_5, primals_6, buf16, 12845056, grid=grid(12845056), stream=stream0)
        del primals_6
        # Source Nodes: [x_7], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, primals_7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf17, (8, 512, 56, 56), (1605632, 3136, 56, 1))
        buf18 = buf13; del buf13  # reuse
        buf19 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf21 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_8], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf17, primals_321, primals_322, buf18, buf19, buf21, primals_321, primals_322, 512, 25088, grid=grid(512), stream=stream0)
        del primals_321
        del primals_322
        buf22 = empty((8, 512, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_10, x_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf17, buf18, buf19, primals_8, primals_9, buf22, 12845056, grid=grid(12845056), stream=stream0)
        del primals_9
        # Source Nodes: [x_12], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_10, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (8, 256, 56, 56), (802816, 3136, 56, 1))
        buf24 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf25 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf27 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_6.run(buf23, primals_324, primals_325, buf24, buf25, buf27, primals_324, primals_325, 256, 25088, grid=grid(256), stream=stream0)
        del primals_324
        del primals_325
        # Source Nodes: [getattr_l__mod___layer1___0___downsample_0], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf9, primals_13, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (8, 256, 56, 56), (802816, 3136, 56, 1))
        buf29 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf30 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf32 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_6.run(buf28, primals_327, primals_328, buf29, buf30, buf32, primals_327, primals_328, 256, 25088, grid=grid(256), stream=stream0)
        del primals_327
        del primals_328
        buf33 = empty((8, 256, 56, 56), device='cuda', dtype=torch.float32)
        buf34 = buf33; del buf33  # reuse
        # Source Nodes: [shortcut_1, shortcut_2, x_13, x_14], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_7.run(buf34, buf23, buf24, buf25, primals_11, primals_12, buf28, buf29, buf30, primals_14, primals_15, 6422528, grid=grid(6422528), stream=stream0)
        del primals_12
        del primals_15
        # Source Nodes: [x_16], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf34, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (8, 512, 56, 56), (1605632, 3136, 56, 1))
        buf36 = buf19; del buf19  # reuse
        buf37 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf39 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf35, primals_330, primals_331, buf36, buf37, buf39, primals_330, primals_331, 512, 25088, grid=grid(512), stream=stream0)
        del primals_330
        del primals_331
        buf40 = empty((8, 512, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17, x_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf35, buf36, buf37, primals_17, primals_18, buf40, 12845056, grid=grid(12845056), stream=stream0)
        del primals_18
        # Source Nodes: [x_19], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, primals_19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf41, (8, 512, 56, 56), (1605632, 3136, 56, 1))
        buf42 = buf37; del buf37  # reuse
        buf43 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf45 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_20], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf41, primals_333, primals_334, buf42, buf43, buf45, primals_333, primals_334, 512, 25088, grid=grid(512), stream=stream0)
        del primals_333
        del primals_334
        buf46 = empty((8, 512, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_20, x_22], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf41, buf42, buf43, primals_20, primals_21, buf46, 12845056, grid=grid(12845056), stream=stream0)
        del primals_21
        # Source Nodes: [x_24], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (8, 256, 56, 56), (802816, 3136, 56, 1))
        buf48 = buf30; del buf30  # reuse
        buf49 = buf25; del buf25  # reuse
        buf51 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_25], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_6.run(buf47, primals_336, primals_337, buf48, buf49, buf51, primals_336, primals_337, 256, 25088, grid=grid(256), stream=stream0)
        del primals_336
        del primals_337
        buf52 = empty((8, 256, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_3, x_25, x_26], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_8.run(buf47, buf48, buf49, primals_23, primals_24, buf34, buf52, 6422528, grid=grid(6422528), stream=stream0)
        del primals_24
        # Source Nodes: [x_28], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_25, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (8, 512, 56, 56), (1605632, 3136, 56, 1))
        buf54 = buf43; del buf43  # reuse
        buf55 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf57 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_29], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf53, primals_339, primals_340, buf54, buf55, buf57, primals_339, primals_340, 512, 25088, grid=grid(512), stream=stream0)
        del primals_339
        del primals_340
        buf58 = empty((8, 512, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_29, x_30], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf53, buf54, buf55, primals_26, primals_27, buf58, 12845056, grid=grid(12845056), stream=stream0)
        del primals_27
        # Source Nodes: [x_31], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, primals_28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf59, (8, 512, 56, 56), (1605632, 3136, 56, 1))
        buf60 = buf55; del buf55  # reuse
        buf61 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf63 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf59, primals_342, primals_343, buf60, buf61, buf63, primals_342, primals_343, 512, 25088, grid=grid(512), stream=stream0)
        del primals_342
        del primals_343
        buf64 = empty((8, 512, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_32, x_34], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_5.run(buf59, buf60, buf61, primals_29, primals_30, buf64, 12845056, grid=grid(12845056), stream=stream0)
        del primals_30
        # Source Nodes: [x_36], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, primals_31, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (8, 256, 56, 56), (802816, 3136, 56, 1))
        buf66 = buf49; del buf49  # reuse
        buf67 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf69 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_37], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_6.run(buf65, primals_345, primals_346, buf66, buf67, buf69, primals_345, primals_346, 256, 25088, grid=grid(256), stream=stream0)
        del primals_345
        del primals_346
        buf70 = empty((8, 256, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_4, x_37, x_38], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_8.run(buf65, buf66, buf67, primals_32, primals_33, buf52, buf70, 6422528, grid=grid(6422528), stream=stream0)
        del buf67
        del primals_33
        # Source Nodes: [x_41], Original ATen: [aten.convolution]
        buf71 = extern_kernels.convolution(buf70, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (8, 1024, 56, 56), (3211264, 3136, 56, 1))
        buf72 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf73 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf75 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_42], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_9.run(buf71, primals_348, primals_349, buf72, buf73, buf75, primals_348, primals_349, 1024, 25088, grid=grid(1024), stream=stream0)
        del primals_348
        del primals_349
        buf76 = empty((8, 1024, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_42, x_43], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_10.run(buf71, buf72, buf73, primals_35, primals_36, buf76, 25690112, grid=grid(25690112), stream=stream0)
        del primals_36
        # Source Nodes: [x_44], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf76, primals_37, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf77, (8, 1024, 28, 28), (802816, 784, 28, 1))
        buf78 = buf73; del buf73  # reuse
        buf79 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf81 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_45], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf77, primals_351, primals_352, buf78, buf79, buf81, primals_351, primals_352, 1024, 6272, grid=grid(1024), stream=stream0)
        del primals_351
        del primals_352
        buf82 = empty((8, 1024, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_45, x_47], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_12.run(buf77, buf78, buf79, primals_38, primals_39, buf82, 6422528, grid=grid(6422528), stream=stream0)
        del primals_39
        # Source Nodes: [x_49], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, primals_40, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (8, 512, 28, 28), (401408, 784, 28, 1))
        buf84 = buf61; del buf61  # reuse
        buf85 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf87 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_50], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf83, primals_354, primals_355, buf84, buf85, buf87, primals_354, primals_355, 512, 6272, grid=grid(512), stream=stream0)
        del primals_354
        del primals_355
        # Source Nodes: [getattr_l__mod___layer2___0___downsample_0], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf70, primals_43, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (8, 512, 28, 28), (401408, 784, 28, 1))
        buf89 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf90 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf92 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf88, primals_357, primals_358, buf89, buf90, buf92, primals_357, primals_358, 512, 6272, grid=grid(512), stream=stream0)
        del primals_357
        del primals_358
        buf93 = empty((8, 512, 28, 28), device='cuda', dtype=torch.float32)
        buf94 = buf93; del buf93  # reuse
        # Source Nodes: [shortcut_5, shortcut_6, x_50, x_51], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_14.run(buf94, buf83, buf84, buf85, primals_41, primals_42, buf88, buf89, buf90, primals_44, primals_45, 3211264, grid=grid(3211264), stream=stream0)
        del primals_42
        del primals_45
        # Source Nodes: [x_53], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, primals_46, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (8, 1024, 28, 28), (802816, 784, 28, 1))
        buf96 = buf79; del buf79  # reuse
        buf97 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf99 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf95, primals_360, primals_361, buf96, buf97, buf99, primals_360, primals_361, 1024, 6272, grid=grid(1024), stream=stream0)
        del primals_360
        del primals_361
        buf100 = empty((8, 1024, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_54, x_55], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_12.run(buf95, buf96, buf97, primals_47, primals_48, buf100, 6422528, grid=grid(6422528), stream=stream0)
        del primals_48
        # Source Nodes: [x_56], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_49, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf101, (8, 1024, 28, 28), (802816, 784, 28, 1))
        buf102 = buf97; del buf97  # reuse
        buf103 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf105 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf101, primals_363, primals_364, buf102, buf103, buf105, primals_363, primals_364, 1024, 6272, grid=grid(1024), stream=stream0)
        del primals_363
        del primals_364
        buf106 = empty((8, 1024, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_57, x_59], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_12.run(buf101, buf102, buf103, primals_50, primals_51, buf106, 6422528, grid=grid(6422528), stream=stream0)
        del primals_51
        # Source Nodes: [x_61], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (8, 512, 28, 28), (401408, 784, 28, 1))
        buf108 = buf90; del buf90  # reuse
        buf109 = buf85; del buf85  # reuse
        buf111 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_62], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf107, primals_366, primals_367, buf108, buf109, buf111, primals_366, primals_367, 512, 6272, grid=grid(512), stream=stream0)
        del primals_366
        del primals_367
        buf112 = empty((8, 512, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_7, x_62, x_63], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_15.run(buf107, buf108, buf109, primals_53, primals_54, buf94, buf112, 3211264, grid=grid(3211264), stream=stream0)
        del primals_54
        # Source Nodes: [x_65], Original ATen: [aten.convolution]
        buf113 = extern_kernels.convolution(buf112, primals_55, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (8, 1024, 28, 28), (802816, 784, 28, 1))
        buf114 = buf103; del buf103  # reuse
        buf115 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf117 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_66], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf113, primals_369, primals_370, buf114, buf115, buf117, primals_369, primals_370, 1024, 6272, grid=grid(1024), stream=stream0)
        del primals_369
        del primals_370
        buf118 = empty((8, 1024, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_66, x_67], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_12.run(buf113, buf114, buf115, primals_56, primals_57, buf118, 6422528, grid=grid(6422528), stream=stream0)
        del primals_57
        # Source Nodes: [x_68], Original ATen: [aten.convolution]
        buf119 = extern_kernels.convolution(buf118, primals_58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf119, (8, 1024, 28, 28), (802816, 784, 28, 1))
        buf120 = buf115; del buf115  # reuse
        buf121 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf123 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_69], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf119, primals_372, primals_373, buf120, buf121, buf123, primals_372, primals_373, 1024, 6272, grid=grid(1024), stream=stream0)
        del primals_372
        del primals_373
        buf124 = empty((8, 1024, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_69, x_71], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_12.run(buf119, buf120, buf121, primals_59, primals_60, buf124, 6422528, grid=grid(6422528), stream=stream0)
        del primals_60
        # Source Nodes: [x_73], Original ATen: [aten.convolution]
        buf125 = extern_kernels.convolution(buf124, primals_61, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (8, 512, 28, 28), (401408, 784, 28, 1))
        buf126 = buf109; del buf109  # reuse
        buf127 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf129 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_74], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf125, primals_375, primals_376, buf126, buf127, buf129, primals_375, primals_376, 512, 6272, grid=grid(512), stream=stream0)
        del primals_375
        del primals_376
        buf130 = empty((8, 512, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_8, x_74, x_75], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_15.run(buf125, buf126, buf127, primals_62, primals_63, buf112, buf130, 3211264, grid=grid(3211264), stream=stream0)
        del primals_63
        # Source Nodes: [x_77], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, primals_64, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (8, 1024, 28, 28), (802816, 784, 28, 1))
        buf132 = buf121; del buf121  # reuse
        buf133 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf135 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_78], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf131, primals_378, primals_379, buf132, buf133, buf135, primals_378, primals_379, 1024, 6272, grid=grid(1024), stream=stream0)
        del primals_378
        del primals_379
        buf136 = empty((8, 1024, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_78, x_79], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_12.run(buf131, buf132, buf133, primals_65, primals_66, buf136, 6422528, grid=grid(6422528), stream=stream0)
        del primals_66
        # Source Nodes: [x_80], Original ATen: [aten.convolution]
        buf137 = extern_kernels.convolution(buf136, primals_67, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf137, (8, 1024, 28, 28), (802816, 784, 28, 1))
        buf138 = buf133; del buf133  # reuse
        buf139 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf141 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_81], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf137, primals_381, primals_382, buf138, buf139, buf141, primals_381, primals_382, 1024, 6272, grid=grid(1024), stream=stream0)
        del primals_381
        del primals_382
        buf142 = empty((8, 1024, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_81, x_83], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_12.run(buf137, buf138, buf139, primals_68, primals_69, buf142, 6422528, grid=grid(6422528), stream=stream0)
        del primals_69
        # Source Nodes: [x_85], Original ATen: [aten.convolution]
        buf143 = extern_kernels.convolution(buf142, primals_70, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (8, 512, 28, 28), (401408, 784, 28, 1))
        buf144 = buf127; del buf127  # reuse
        buf145 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf147 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_86], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_13.run(buf143, primals_384, primals_385, buf144, buf145, buf147, primals_384, primals_385, 512, 6272, grid=grid(512), stream=stream0)
        del primals_384
        del primals_385
        buf148 = empty((8, 512, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_9, x_86, x_87], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_15.run(buf143, buf144, buf145, primals_71, primals_72, buf130, buf148, 3211264, grid=grid(3211264), stream=stream0)
        del buf145
        del primals_72
        # Source Nodes: [x_90], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, primals_73, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (8, 2048, 28, 28), (1605632, 784, 28, 1))
        buf150 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf151 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf153 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_91], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_16.run(buf149, primals_387, primals_388, buf150, buf151, buf153, primals_387, primals_388, 2048, 6272, grid=grid(2048), stream=stream0)
        del primals_387
        del primals_388
        buf154 = empty((8, 2048, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_91, x_92], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_17.run(buf149, buf150, buf151, primals_74, primals_75, buf154, 12845056, grid=grid(12845056), stream=stream0)
        del primals_75
        # Source Nodes: [x_93], Original ATen: [aten.convolution]
        buf155 = extern_kernels.convolution(buf154, primals_76, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf155, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf156 = buf151; del buf151  # reuse
        buf157 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf159 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_94], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf155, primals_390, primals_391, buf156, buf157, buf159, primals_390, primals_391, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_390
        del primals_391
        buf160 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_94, x_96], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf155, buf156, buf157, primals_77, primals_78, buf160, 3211264, grid=grid(3211264), stream=stream0)
        del primals_78
        # Source Nodes: [x_98], Original ATen: [aten.convolution]
        buf161 = extern_kernels.convolution(buf160, primals_79, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf162 = buf139; del buf139  # reuse
        buf163 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf165 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_99], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf161, primals_393, primals_394, buf162, buf163, buf165, primals_393, primals_394, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_393
        del primals_394
        # Source Nodes: [getattr_l__mod___layer3___0___downsample_0], Original ATen: [aten.convolution]
        buf166 = extern_kernels.convolution(buf148, primals_82, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf167 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf168 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf170 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_10], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf166, primals_396, primals_397, buf167, buf168, buf170, primals_396, primals_397, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_396
        del primals_397
        buf171 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        buf172 = buf171; del buf171  # reuse
        # Source Nodes: [shortcut_10, shortcut_11, x_100, x_99], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_21.run(buf172, buf161, buf162, buf163, primals_80, primals_81, buf166, buf167, buf168, primals_83, primals_84, 1605632, grid=grid(1605632), stream=stream0)
        del primals_81
        del primals_84
        # Source Nodes: [x_102], Original ATen: [aten.convolution]
        buf173 = extern_kernels.convolution(buf172, primals_85, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf174 = buf157; del buf157  # reuse
        buf175 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf177 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_103], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf173, primals_399, primals_400, buf174, buf175, buf177, primals_399, primals_400, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_399
        del primals_400
        buf178 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_103, x_104], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf173, buf174, buf175, primals_86, primals_87, buf178, 3211264, grid=grid(3211264), stream=stream0)
        del primals_87
        # Source Nodes: [x_105], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, primals_88, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf179, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf180 = buf175; del buf175  # reuse
        buf181 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf183 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_106], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf179, primals_402, primals_403, buf180, buf181, buf183, primals_402, primals_403, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_402
        del primals_403
        buf184 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_106, x_108], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf179, buf180, buf181, primals_89, primals_90, buf184, 3211264, grid=grid(3211264), stream=stream0)
        del primals_90
        # Source Nodes: [x_110], Original ATen: [aten.convolution]
        buf185 = extern_kernels.convolution(buf184, primals_91, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf186 = buf168; del buf168  # reuse
        buf187 = buf163; del buf163  # reuse
        buf189 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_111], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf185, primals_405, primals_406, buf186, buf187, buf189, primals_405, primals_406, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_405
        del primals_406
        buf190 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_12, x_111, x_112], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf185, buf186, buf187, primals_92, primals_93, buf172, buf190, 1605632, grid=grid(1605632), stream=stream0)
        del primals_93
        # Source Nodes: [x_114], Original ATen: [aten.convolution]
        buf191 = extern_kernels.convolution(buf190, primals_94, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf192 = buf181; del buf181  # reuse
        buf193 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf195 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_115], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf191, primals_408, primals_409, buf192, buf193, buf195, primals_408, primals_409, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_408
        del primals_409
        buf196 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_115, x_116], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf191, buf192, buf193, primals_95, primals_96, buf196, 3211264, grid=grid(3211264), stream=stream0)
        del primals_96
        # Source Nodes: [x_117], Original ATen: [aten.convolution]
        buf197 = extern_kernels.convolution(buf196, primals_97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf197, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf198 = buf193; del buf193  # reuse
        buf199 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf201 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_118], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf197, primals_411, primals_412, buf198, buf199, buf201, primals_411, primals_412, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_411
        del primals_412
        buf202 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_118, x_120], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf197, buf198, buf199, primals_98, primals_99, buf202, 3211264, grid=grid(3211264), stream=stream0)
        del primals_99
        # Source Nodes: [x_122], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, primals_100, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf204 = buf187; del buf187  # reuse
        buf205 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf207 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_123], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf203, primals_414, primals_415, buf204, buf205, buf207, primals_414, primals_415, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_414
        del primals_415
        buf208 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_13, x_123, x_124], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf203, buf204, buf205, primals_101, primals_102, buf190, buf208, 1605632, grid=grid(1605632), stream=stream0)
        del primals_102
        # Source Nodes: [x_126], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf208, primals_103, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf210 = buf199; del buf199  # reuse
        buf211 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf213 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_127], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf209, primals_417, primals_418, buf210, buf211, buf213, primals_417, primals_418, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_417
        del primals_418
        buf214 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_127, x_128], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf209, buf210, buf211, primals_104, primals_105, buf214, 3211264, grid=grid(3211264), stream=stream0)
        del primals_105
        # Source Nodes: [x_129], Original ATen: [aten.convolution]
        buf215 = extern_kernels.convolution(buf214, primals_106, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf215, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf216 = buf211; del buf211  # reuse
        buf217 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf219 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_130], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf215, primals_420, primals_421, buf216, buf217, buf219, primals_420, primals_421, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_420
        del primals_421
        buf220 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_130, x_132], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf215, buf216, buf217, primals_107, primals_108, buf220, 3211264, grid=grid(3211264), stream=stream0)
        del primals_108
        # Source Nodes: [x_134], Original ATen: [aten.convolution]
        buf221 = extern_kernels.convolution(buf220, primals_109, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf222 = buf205; del buf205  # reuse
        buf223 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf225 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_135], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf221, primals_423, primals_424, buf222, buf223, buf225, primals_423, primals_424, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_423
        del primals_424
        buf226 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_14, x_135, x_136], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf221, buf222, buf223, primals_110, primals_111, buf208, buf226, 1605632, grid=grid(1605632), stream=stream0)
        del primals_111
        # Source Nodes: [x_138], Original ATen: [aten.convolution]
        buf227 = extern_kernels.convolution(buf226, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf227, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf228 = buf217; del buf217  # reuse
        buf229 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf231 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_139], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf227, primals_426, primals_427, buf228, buf229, buf231, primals_426, primals_427, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_426
        del primals_427
        buf232 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_139, x_140], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf227, buf228, buf229, primals_113, primals_114, buf232, 3211264, grid=grid(3211264), stream=stream0)
        del primals_114
        # Source Nodes: [x_141], Original ATen: [aten.convolution]
        buf233 = extern_kernels.convolution(buf232, primals_115, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf233, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf234 = buf229; del buf229  # reuse
        buf235 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf237 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_142], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf233, primals_429, primals_430, buf234, buf235, buf237, primals_429, primals_430, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_429
        del primals_430
        buf238 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_142, x_144], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf233, buf234, buf235, primals_116, primals_117, buf238, 3211264, grid=grid(3211264), stream=stream0)
        del primals_117
        # Source Nodes: [x_146], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf238, primals_118, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf240 = buf223; del buf223  # reuse
        buf241 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf243 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_147], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf239, primals_432, primals_433, buf240, buf241, buf243, primals_432, primals_433, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_432
        del primals_433
        buf244 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_15, x_147, x_148], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf239, buf240, buf241, primals_119, primals_120, buf226, buf244, 1605632, grid=grid(1605632), stream=stream0)
        del primals_120
        # Source Nodes: [x_150], Original ATen: [aten.convolution]
        buf245 = extern_kernels.convolution(buf244, primals_121, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf245, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf246 = buf235; del buf235  # reuse
        buf247 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf249 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_151], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf245, primals_435, primals_436, buf246, buf247, buf249, primals_435, primals_436, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_435
        del primals_436
        buf250 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_151, x_152], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf245, buf246, buf247, primals_122, primals_123, buf250, 3211264, grid=grid(3211264), stream=stream0)
        del primals_123
        # Source Nodes: [x_153], Original ATen: [aten.convolution]
        buf251 = extern_kernels.convolution(buf250, primals_124, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf251, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf252 = buf247; del buf247  # reuse
        buf253 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf255 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_154], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf251, primals_438, primals_439, buf252, buf253, buf255, primals_438, primals_439, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_438
        del primals_439
        buf256 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_154, x_156], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf251, buf252, buf253, primals_125, primals_126, buf256, 3211264, grid=grid(3211264), stream=stream0)
        del primals_126
        # Source Nodes: [x_158], Original ATen: [aten.convolution]
        buf257 = extern_kernels.convolution(buf256, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf257, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf258 = buf241; del buf241  # reuse
        buf259 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf261 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_159], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf257, primals_441, primals_442, buf258, buf259, buf261, primals_441, primals_442, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_441
        del primals_442
        buf262 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_16, x_159, x_160], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf257, buf258, buf259, primals_128, primals_129, buf244, buf262, 1605632, grid=grid(1605632), stream=stream0)
        del primals_129
        # Source Nodes: [x_162], Original ATen: [aten.convolution]
        buf263 = extern_kernels.convolution(buf262, primals_130, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf263, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf264 = buf253; del buf253  # reuse
        buf265 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf267 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_163], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf263, primals_444, primals_445, buf264, buf265, buf267, primals_444, primals_445, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_444
        del primals_445
        buf268 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_163, x_164], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf263, buf264, buf265, primals_131, primals_132, buf268, 3211264, grid=grid(3211264), stream=stream0)
        del primals_132
        # Source Nodes: [x_165], Original ATen: [aten.convolution]
        buf269 = extern_kernels.convolution(buf268, primals_133, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf269, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf270 = buf265; del buf265  # reuse
        buf271 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf273 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_166], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf269, primals_447, primals_448, buf270, buf271, buf273, primals_447, primals_448, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_447
        del primals_448
        buf274 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_166, x_168], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf269, buf270, buf271, primals_134, primals_135, buf274, 3211264, grid=grid(3211264), stream=stream0)
        del primals_135
        # Source Nodes: [x_170], Original ATen: [aten.convolution]
        buf275 = extern_kernels.convolution(buf274, primals_136, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf275, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf276 = buf259; del buf259  # reuse
        buf277 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf279 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_171], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf275, primals_450, primals_451, buf276, buf277, buf279, primals_450, primals_451, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_450
        del primals_451
        buf280 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_17, x_171, x_172], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf275, buf276, buf277, primals_137, primals_138, buf262, buf280, 1605632, grid=grid(1605632), stream=stream0)
        del primals_138
        # Source Nodes: [x_174], Original ATen: [aten.convolution]
        buf281 = extern_kernels.convolution(buf280, primals_139, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf281, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf282 = buf271; del buf271  # reuse
        buf283 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf285 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_175], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf281, primals_453, primals_454, buf282, buf283, buf285, primals_453, primals_454, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_453
        del primals_454
        buf286 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_175, x_176], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf281, buf282, buf283, primals_140, primals_141, buf286, 3211264, grid=grid(3211264), stream=stream0)
        del primals_141
        # Source Nodes: [x_177], Original ATen: [aten.convolution]
        buf287 = extern_kernels.convolution(buf286, primals_142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf287, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf288 = buf283; del buf283  # reuse
        buf289 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf291 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf287, primals_456, primals_457, buf288, buf289, buf291, primals_456, primals_457, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_456
        del primals_457
        buf292 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_178, x_180], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf287, buf288, buf289, primals_143, primals_144, buf292, 3211264, grid=grid(3211264), stream=stream0)
        del primals_144
        # Source Nodes: [x_182], Original ATen: [aten.convolution]
        buf293 = extern_kernels.convolution(buf292, primals_145, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf293, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf294 = buf277; del buf277  # reuse
        buf295 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf297 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_183], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf293, primals_459, primals_460, buf294, buf295, buf297, primals_459, primals_460, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_459
        del primals_460
        buf298 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_18, x_183, x_184], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf293, buf294, buf295, primals_146, primals_147, buf280, buf298, 1605632, grid=grid(1605632), stream=stream0)
        del primals_147
        # Source Nodes: [x_186], Original ATen: [aten.convolution]
        buf299 = extern_kernels.convolution(buf298, primals_148, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf299, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf300 = buf289; del buf289  # reuse
        buf301 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf303 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_187], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf299, primals_462, primals_463, buf300, buf301, buf303, primals_462, primals_463, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_462
        del primals_463
        buf304 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_187, x_188], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf299, buf300, buf301, primals_149, primals_150, buf304, 3211264, grid=grid(3211264), stream=stream0)
        del primals_150
        # Source Nodes: [x_189], Original ATen: [aten.convolution]
        buf305 = extern_kernels.convolution(buf304, primals_151, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf305, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf306 = buf301; del buf301  # reuse
        buf307 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf309 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_190], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf305, primals_465, primals_466, buf306, buf307, buf309, primals_465, primals_466, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_465
        del primals_466
        buf310 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_190, x_192], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf305, buf306, buf307, primals_152, primals_153, buf310, 3211264, grid=grid(3211264), stream=stream0)
        del primals_153
        # Source Nodes: [x_194], Original ATen: [aten.convolution]
        buf311 = extern_kernels.convolution(buf310, primals_154, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf311, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf312 = buf295; del buf295  # reuse
        buf313 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf315 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_195], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf311, primals_468, primals_469, buf312, buf313, buf315, primals_468, primals_469, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_468
        del primals_469
        buf316 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_19, x_195, x_196], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf311, buf312, buf313, primals_155, primals_156, buf298, buf316, 1605632, grid=grid(1605632), stream=stream0)
        del primals_156
        # Source Nodes: [x_198], Original ATen: [aten.convolution]
        buf317 = extern_kernels.convolution(buf316, primals_157, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf317, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf318 = buf307; del buf307  # reuse
        buf319 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf321 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_199], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf317, primals_471, primals_472, buf318, buf319, buf321, primals_471, primals_472, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_471
        del primals_472
        buf322 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_199, x_200], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf317, buf318, buf319, primals_158, primals_159, buf322, 3211264, grid=grid(3211264), stream=stream0)
        del primals_159
        # Source Nodes: [x_201], Original ATen: [aten.convolution]
        buf323 = extern_kernels.convolution(buf322, primals_160, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf323, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf324 = buf319; del buf319  # reuse
        buf325 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf327 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_202], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf323, primals_474, primals_475, buf324, buf325, buf327, primals_474, primals_475, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_474
        del primals_475
        buf328 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_202, x_204], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf323, buf324, buf325, primals_161, primals_162, buf328, 3211264, grid=grid(3211264), stream=stream0)
        del primals_162
        # Source Nodes: [x_206], Original ATen: [aten.convolution]
        buf329 = extern_kernels.convolution(buf328, primals_163, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf329, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf330 = buf313; del buf313  # reuse
        buf331 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf333 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_207], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf329, primals_477, primals_478, buf330, buf331, buf333, primals_477, primals_478, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_477
        del primals_478
        buf334 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_20, x_207, x_208], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf329, buf330, buf331, primals_164, primals_165, buf316, buf334, 1605632, grid=grid(1605632), stream=stream0)
        del primals_165
        # Source Nodes: [x_210], Original ATen: [aten.convolution]
        buf335 = extern_kernels.convolution(buf334, primals_166, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf335, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf336 = buf325; del buf325  # reuse
        buf337 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf339 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf335, primals_480, primals_481, buf336, buf337, buf339, primals_480, primals_481, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_480
        del primals_481
        buf340 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_211, x_212], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf335, buf336, buf337, primals_167, primals_168, buf340, 3211264, grid=grid(3211264), stream=stream0)
        del primals_168
        # Source Nodes: [x_213], Original ATen: [aten.convolution]
        buf341 = extern_kernels.convolution(buf340, primals_169, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf341, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf342 = buf337; del buf337  # reuse
        buf343 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf345 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_214], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf341, primals_483, primals_484, buf342, buf343, buf345, primals_483, primals_484, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_483
        del primals_484
        buf346 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_214, x_216], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf341, buf342, buf343, primals_170, primals_171, buf346, 3211264, grid=grid(3211264), stream=stream0)
        del primals_171
        # Source Nodes: [x_218], Original ATen: [aten.convolution]
        buf347 = extern_kernels.convolution(buf346, primals_172, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf347, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf348 = buf331; del buf331  # reuse
        buf349 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf351 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_219], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf347, primals_486, primals_487, buf348, buf349, buf351, primals_486, primals_487, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_486
        del primals_487
        buf352 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_21, x_219, x_220], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf347, buf348, buf349, primals_173, primals_174, buf334, buf352, 1605632, grid=grid(1605632), stream=stream0)
        del primals_174
        # Source Nodes: [x_222], Original ATen: [aten.convolution]
        buf353 = extern_kernels.convolution(buf352, primals_175, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf353, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf354 = buf343; del buf343  # reuse
        buf355 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf357 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_223], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf353, primals_489, primals_490, buf354, buf355, buf357, primals_489, primals_490, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_489
        del primals_490
        buf358 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_223, x_224], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf353, buf354, buf355, primals_176, primals_177, buf358, 3211264, grid=grid(3211264), stream=stream0)
        del primals_177
        # Source Nodes: [x_225], Original ATen: [aten.convolution]
        buf359 = extern_kernels.convolution(buf358, primals_178, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf359, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf360 = buf355; del buf355  # reuse
        buf361 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf363 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_226], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf359, primals_492, primals_493, buf360, buf361, buf363, primals_492, primals_493, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_492
        del primals_493
        buf364 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_226, x_228], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf359, buf360, buf361, primals_179, primals_180, buf364, 3211264, grid=grid(3211264), stream=stream0)
        del primals_180
        # Source Nodes: [x_230], Original ATen: [aten.convolution]
        buf365 = extern_kernels.convolution(buf364, primals_181, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf365, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf366 = buf349; del buf349  # reuse
        buf367 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf369 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_231], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf365, primals_495, primals_496, buf366, buf367, buf369, primals_495, primals_496, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_495
        del primals_496
        buf370 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_22, x_231, x_232], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf365, buf366, buf367, primals_182, primals_183, buf352, buf370, 1605632, grid=grid(1605632), stream=stream0)
        del primals_183
        # Source Nodes: [x_234], Original ATen: [aten.convolution]
        buf371 = extern_kernels.convolution(buf370, primals_184, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf371, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf372 = buf361; del buf361  # reuse
        buf373 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf375 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_235], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf371, primals_498, primals_499, buf372, buf373, buf375, primals_498, primals_499, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_498
        del primals_499
        buf376 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_235, x_236], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf371, buf372, buf373, primals_185, primals_186, buf376, 3211264, grid=grid(3211264), stream=stream0)
        del primals_186
        # Source Nodes: [x_237], Original ATen: [aten.convolution]
        buf377 = extern_kernels.convolution(buf376, primals_187, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf377, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf378 = buf373; del buf373  # reuse
        buf379 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf381 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_238], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf377, primals_501, primals_502, buf378, buf379, buf381, primals_501, primals_502, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_501
        del primals_502
        buf382 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_238, x_240], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf377, buf378, buf379, primals_188, primals_189, buf382, 3211264, grid=grid(3211264), stream=stream0)
        del primals_189
        # Source Nodes: [x_242], Original ATen: [aten.convolution]
        buf383 = extern_kernels.convolution(buf382, primals_190, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf383, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf384 = buf367; del buf367  # reuse
        buf385 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf387 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_243], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf383, primals_504, primals_505, buf384, buf385, buf387, primals_504, primals_505, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_504
        del primals_505
        buf388 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_23, x_243, x_244], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf383, buf384, buf385, primals_191, primals_192, buf370, buf388, 1605632, grid=grid(1605632), stream=stream0)
        del primals_192
        # Source Nodes: [x_246], Original ATen: [aten.convolution]
        buf389 = extern_kernels.convolution(buf388, primals_193, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf389, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf390 = buf379; del buf379  # reuse
        buf391 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf393 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_247], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf389, primals_507, primals_508, buf390, buf391, buf393, primals_507, primals_508, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_507
        del primals_508
        buf394 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_247, x_248], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf389, buf390, buf391, primals_194, primals_195, buf394, 3211264, grid=grid(3211264), stream=stream0)
        del primals_195
        # Source Nodes: [x_249], Original ATen: [aten.convolution]
        buf395 = extern_kernels.convolution(buf394, primals_196, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf395, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf396 = buf391; del buf391  # reuse
        buf397 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf399 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_250], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf395, primals_510, primals_511, buf396, buf397, buf399, primals_510, primals_511, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_510
        del primals_511
        buf400 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_250, x_252], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf395, buf396, buf397, primals_197, primals_198, buf400, 3211264, grid=grid(3211264), stream=stream0)
        del primals_198
        # Source Nodes: [x_254], Original ATen: [aten.convolution]
        buf401 = extern_kernels.convolution(buf400, primals_199, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf401, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf402 = buf385; del buf385  # reuse
        buf403 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf405 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_255], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf401, primals_513, primals_514, buf402, buf403, buf405, primals_513, primals_514, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_513
        del primals_514
        buf406 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_24, x_255, x_256], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf401, buf402, buf403, primals_200, primals_201, buf388, buf406, 1605632, grid=grid(1605632), stream=stream0)
        del primals_201
        # Source Nodes: [x_258], Original ATen: [aten.convolution]
        buf407 = extern_kernels.convolution(buf406, primals_202, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf407, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf408 = buf397; del buf397  # reuse
        buf409 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf411 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_259], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf407, primals_516, primals_517, buf408, buf409, buf411, primals_516, primals_517, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_516
        del primals_517
        buf412 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_259, x_260], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf407, buf408, buf409, primals_203, primals_204, buf412, 3211264, grid=grid(3211264), stream=stream0)
        del primals_204
        # Source Nodes: [x_261], Original ATen: [aten.convolution]
        buf413 = extern_kernels.convolution(buf412, primals_205, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf413, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf414 = buf409; del buf409  # reuse
        buf415 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf417 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_262], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf413, primals_519, primals_520, buf414, buf415, buf417, primals_519, primals_520, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_519
        del primals_520
        buf418 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_262, x_264], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf413, buf414, buf415, primals_206, primals_207, buf418, 3211264, grid=grid(3211264), stream=stream0)
        del primals_207
        # Source Nodes: [x_266], Original ATen: [aten.convolution]
        buf419 = extern_kernels.convolution(buf418, primals_208, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf419, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf420 = buf403; del buf403  # reuse
        buf421 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf423 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_267], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf419, primals_522, primals_523, buf420, buf421, buf423, primals_522, primals_523, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_522
        del primals_523
        buf424 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_25, x_267, x_268], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf419, buf420, buf421, primals_209, primals_210, buf406, buf424, 1605632, grid=grid(1605632), stream=stream0)
        del primals_210
        # Source Nodes: [x_270], Original ATen: [aten.convolution]
        buf425 = extern_kernels.convolution(buf424, primals_211, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf425, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf426 = buf415; del buf415  # reuse
        buf427 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf429 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_271], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf425, primals_525, primals_526, buf426, buf427, buf429, primals_525, primals_526, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_525
        del primals_526
        buf430 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_271, x_272], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf425, buf426, buf427, primals_212, primals_213, buf430, 3211264, grid=grid(3211264), stream=stream0)
        del primals_213
        # Source Nodes: [x_273], Original ATen: [aten.convolution]
        buf431 = extern_kernels.convolution(buf430, primals_214, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf431, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf432 = buf427; del buf427  # reuse
        buf433 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf435 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_274], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf431, primals_528, primals_529, buf432, buf433, buf435, primals_528, primals_529, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_528
        del primals_529
        buf436 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_274, x_276], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf431, buf432, buf433, primals_215, primals_216, buf436, 3211264, grid=grid(3211264), stream=stream0)
        del primals_216
        # Source Nodes: [x_278], Original ATen: [aten.convolution]
        buf437 = extern_kernels.convolution(buf436, primals_217, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf437, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf438 = buf421; del buf421  # reuse
        buf439 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf441 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_279], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf437, primals_531, primals_532, buf438, buf439, buf441, primals_531, primals_532, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_531
        del primals_532
        buf442 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_26, x_279, x_280], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf437, buf438, buf439, primals_218, primals_219, buf424, buf442, 1605632, grid=grid(1605632), stream=stream0)
        del primals_219
        # Source Nodes: [x_282], Original ATen: [aten.convolution]
        buf443 = extern_kernels.convolution(buf442, primals_220, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf443, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf444 = buf433; del buf433  # reuse
        buf445 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf447 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_283], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf443, primals_534, primals_535, buf444, buf445, buf447, primals_534, primals_535, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_534
        del primals_535
        buf448 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_283, x_284], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf443, buf444, buf445, primals_221, primals_222, buf448, 3211264, grid=grid(3211264), stream=stream0)
        del primals_222
        # Source Nodes: [x_285], Original ATen: [aten.convolution]
        buf449 = extern_kernels.convolution(buf448, primals_223, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf449, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf450 = buf445; del buf445  # reuse
        buf451 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf453 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_286], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf449, primals_537, primals_538, buf450, buf451, buf453, primals_537, primals_538, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_537
        del primals_538
        buf454 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_286, x_288], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf449, buf450, buf451, primals_224, primals_225, buf454, 3211264, grid=grid(3211264), stream=stream0)
        del primals_225
        # Source Nodes: [x_290], Original ATen: [aten.convolution]
        buf455 = extern_kernels.convolution(buf454, primals_226, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf455, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf456 = buf439; del buf439  # reuse
        buf457 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf459 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_291], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf455, primals_540, primals_541, buf456, buf457, buf459, primals_540, primals_541, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_540
        del primals_541
        buf460 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_27, x_291, x_292], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf455, buf456, buf457, primals_227, primals_228, buf442, buf460, 1605632, grid=grid(1605632), stream=stream0)
        del primals_228
        # Source Nodes: [x_294], Original ATen: [aten.convolution]
        buf461 = extern_kernels.convolution(buf460, primals_229, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf461, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf462 = buf451; del buf451  # reuse
        buf463 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf465 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_295], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf461, primals_543, primals_544, buf462, buf463, buf465, primals_543, primals_544, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_543
        del primals_544
        buf466 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_295, x_296], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf461, buf462, buf463, primals_230, primals_231, buf466, 3211264, grid=grid(3211264), stream=stream0)
        del primals_231
        # Source Nodes: [x_297], Original ATen: [aten.convolution]
        buf467 = extern_kernels.convolution(buf466, primals_232, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf467, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf468 = buf463; del buf463  # reuse
        buf469 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf471 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_298], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf467, primals_546, primals_547, buf468, buf469, buf471, primals_546, primals_547, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_546
        del primals_547
        buf472 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_298, x_300], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf467, buf468, buf469, primals_233, primals_234, buf472, 3211264, grid=grid(3211264), stream=stream0)
        del primals_234
        # Source Nodes: [x_302], Original ATen: [aten.convolution]
        buf473 = extern_kernels.convolution(buf472, primals_235, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf473, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf474 = buf457; del buf457  # reuse
        buf475 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf477 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_303], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf473, primals_549, primals_550, buf474, buf475, buf477, primals_549, primals_550, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_549
        del primals_550
        buf478 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_28, x_303, x_304], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf473, buf474, buf475, primals_236, primals_237, buf460, buf478, 1605632, grid=grid(1605632), stream=stream0)
        del primals_237
        # Source Nodes: [x_306], Original ATen: [aten.convolution]
        buf479 = extern_kernels.convolution(buf478, primals_238, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf479, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf480 = buf469; del buf469  # reuse
        buf481 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf483 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_307], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf479, primals_552, primals_553, buf480, buf481, buf483, primals_552, primals_553, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_552
        del primals_553
        buf484 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_307, x_308], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf479, buf480, buf481, primals_239, primals_240, buf484, 3211264, grid=grid(3211264), stream=stream0)
        del primals_240
        # Source Nodes: [x_309], Original ATen: [aten.convolution]
        buf485 = extern_kernels.convolution(buf484, primals_241, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf485, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf486 = buf481; del buf481  # reuse
        buf487 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf489 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_310], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf485, primals_555, primals_556, buf486, buf487, buf489, primals_555, primals_556, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_555
        del primals_556
        buf490 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_310, x_312], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf485, buf486, buf487, primals_242, primals_243, buf490, 3211264, grid=grid(3211264), stream=stream0)
        del primals_243
        # Source Nodes: [x_314], Original ATen: [aten.convolution]
        buf491 = extern_kernels.convolution(buf490, primals_244, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf491, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf492 = buf475; del buf475  # reuse
        buf493 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf495 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_315], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf491, primals_558, primals_559, buf492, buf493, buf495, primals_558, primals_559, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_558
        del primals_559
        buf496 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_29, x_315, x_316], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf491, buf492, buf493, primals_245, primals_246, buf478, buf496, 1605632, grid=grid(1605632), stream=stream0)
        del primals_246
        # Source Nodes: [x_318], Original ATen: [aten.convolution]
        buf497 = extern_kernels.convolution(buf496, primals_247, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf497, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf498 = buf487; del buf487  # reuse
        buf499 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf501 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_319], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf497, primals_561, primals_562, buf498, buf499, buf501, primals_561, primals_562, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_561
        del primals_562
        buf502 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_319, x_320], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf497, buf498, buf499, primals_248, primals_249, buf502, 3211264, grid=grid(3211264), stream=stream0)
        del primals_249
        # Source Nodes: [x_321], Original ATen: [aten.convolution]
        buf503 = extern_kernels.convolution(buf502, primals_250, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf503, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf504 = buf499; del buf499  # reuse
        buf505 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf507 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_322], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf503, primals_564, primals_565, buf504, buf505, buf507, primals_564, primals_565, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_564
        del primals_565
        buf508 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_322, x_324], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf503, buf504, buf505, primals_251, primals_252, buf508, 3211264, grid=grid(3211264), stream=stream0)
        del primals_252
        # Source Nodes: [x_326], Original ATen: [aten.convolution]
        buf509 = extern_kernels.convolution(buf508, primals_253, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf509, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf510 = buf493; del buf493  # reuse
        buf511 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf513 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_327], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf509, primals_567, primals_568, buf510, buf511, buf513, primals_567, primals_568, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_567
        del primals_568
        buf514 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_30, x_327, x_328], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf509, buf510, buf511, primals_254, primals_255, buf496, buf514, 1605632, grid=grid(1605632), stream=stream0)
        del primals_255
        # Source Nodes: [x_330], Original ATen: [aten.convolution]
        buf515 = extern_kernels.convolution(buf514, primals_256, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf515, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf516 = buf505; del buf505  # reuse
        buf517 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf519 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_331], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf515, primals_570, primals_571, buf516, buf517, buf519, primals_570, primals_571, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_570
        del primals_571
        buf520 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_331, x_332], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf515, buf516, buf517, primals_257, primals_258, buf520, 3211264, grid=grid(3211264), stream=stream0)
        del primals_258
        # Source Nodes: [x_333], Original ATen: [aten.convolution]
        buf521 = extern_kernels.convolution(buf520, primals_259, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf521, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf522 = buf517; del buf517  # reuse
        buf523 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf525 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_334], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf521, primals_573, primals_574, buf522, buf523, buf525, primals_573, primals_574, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_573
        del primals_574
        buf526 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_334, x_336], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf521, buf522, buf523, primals_260, primals_261, buf526, 3211264, grid=grid(3211264), stream=stream0)
        del primals_261
        # Source Nodes: [x_338], Original ATen: [aten.convolution]
        buf527 = extern_kernels.convolution(buf526, primals_262, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf527, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf528 = buf511; del buf511  # reuse
        buf529 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf531 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_339], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf527, primals_576, primals_577, buf528, buf529, buf531, primals_576, primals_577, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_576
        del primals_577
        buf532 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_31, x_339, x_340], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf527, buf528, buf529, primals_263, primals_264, buf514, buf532, 1605632, grid=grid(1605632), stream=stream0)
        del primals_264
        # Source Nodes: [x_342], Original ATen: [aten.convolution]
        buf533 = extern_kernels.convolution(buf532, primals_265, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf533, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf534 = buf523; del buf523  # reuse
        buf535 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf537 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_343], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf533, primals_579, primals_580, buf534, buf535, buf537, primals_579, primals_580, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_579
        del primals_580
        buf538 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_343, x_344], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf533, buf534, buf535, primals_266, primals_267, buf538, 3211264, grid=grid(3211264), stream=stream0)
        del primals_267
        # Source Nodes: [x_345], Original ATen: [aten.convolution]
        buf539 = extern_kernels.convolution(buf538, primals_268, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf539, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf540 = buf535; del buf535  # reuse
        buf541 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf543 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_346], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf539, primals_582, primals_583, buf540, buf541, buf543, primals_582, primals_583, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_582
        del primals_583
        buf544 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_346, x_348], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf539, buf540, buf541, primals_269, primals_270, buf544, 3211264, grid=grid(3211264), stream=stream0)
        del primals_270
        # Source Nodes: [x_350], Original ATen: [aten.convolution]
        buf545 = extern_kernels.convolution(buf544, primals_271, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf545, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf546 = buf529; del buf529  # reuse
        buf547 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf549 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_351], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf545, primals_585, primals_586, buf546, buf547, buf549, primals_585, primals_586, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_585
        del primals_586
        buf550 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_32, x_351, x_352], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf545, buf546, buf547, primals_272, primals_273, buf532, buf550, 1605632, grid=grid(1605632), stream=stream0)
        del primals_273
        # Source Nodes: [x_354], Original ATen: [aten.convolution]
        buf551 = extern_kernels.convolution(buf550, primals_274, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf551, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf552 = buf541; del buf541  # reuse
        buf553 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf555 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_355], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf551, primals_588, primals_589, buf552, buf553, buf555, primals_588, primals_589, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_588
        del primals_589
        buf556 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_355, x_356], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf551, buf552, buf553, primals_275, primals_276, buf556, 3211264, grid=grid(3211264), stream=stream0)
        del primals_276
        # Source Nodes: [x_357], Original ATen: [aten.convolution]
        buf557 = extern_kernels.convolution(buf556, primals_277, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf557, (8, 2048, 14, 14), (401408, 196, 14, 1))
        buf558 = buf553; del buf553  # reuse
        buf559 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf561 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_358], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf557, primals_591, primals_592, buf558, buf559, buf561, primals_591, primals_592, 2048, 1568, grid=grid(2048), stream=stream0)
        del primals_591
        del primals_592
        buf562 = empty((8, 2048, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_358, x_360], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_19.run(buf557, buf558, buf559, primals_278, primals_279, buf562, 3211264, grid=grid(3211264), stream=stream0)
        del primals_279
        # Source Nodes: [x_362], Original ATen: [aten.convolution]
        buf563 = extern_kernels.convolution(buf562, primals_280, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf563, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf564 = buf547; del buf547  # reuse
        buf565 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf567 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_363], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_20.run(buf563, primals_594, primals_595, buf564, buf565, buf567, primals_594, primals_595, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_594
        del primals_595
        buf568 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_33, x_363, x_364], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf563, buf564, buf565, primals_281, primals_282, buf550, buf568, 1605632, grid=grid(1605632), stream=stream0)
        del buf565
        del primals_282
        # Source Nodes: [x_367], Original ATen: [aten.convolution]
        buf569 = extern_kernels.convolution(buf568, primals_283, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf569, (8, 4096, 14, 14), (802816, 196, 14, 1))
        buf570 = empty_strided((1, 4096, 1, 1), (4096, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf571 = empty_strided((1, 4096, 1, 1), (4096, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf573 = empty((4096, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_368], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_23.run(buf569, primals_597, primals_598, buf570, buf571, buf573, primals_597, primals_598, 4096, 1568, grid=grid(4096), stream=stream0)
        del primals_597
        del primals_598
        buf574 = empty((8, 4096, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_368, x_369], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_24.run(buf569, buf570, buf571, primals_284, primals_285, buf574, 6422528, grid=grid(6422528), stream=stream0)
        del primals_285
        # Source Nodes: [x_370], Original ATen: [aten.convolution]
        buf575 = extern_kernels.convolution(buf574, primals_286, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf575, (8, 4096, 7, 7), (200704, 49, 7, 1))
        buf576 = buf571; del buf571  # reuse
        buf577 = empty_strided((1, 4096, 1, 1), (4096, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf579 = empty((4096, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_371], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_25.run(buf575, primals_600, primals_601, buf576, buf577, buf579, primals_600, primals_601, 4096, 392, grid=grid(4096), stream=stream0)
        del primals_600
        del primals_601
        buf580 = empty((8, 4096, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_371, x_373], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf575, buf576, buf577, primals_287, primals_288, buf580, 1605632, grid=grid(1605632), stream=stream0)
        del primals_288
        # Source Nodes: [x_375], Original ATen: [aten.convolution]
        buf581 = extern_kernels.convolution(buf580, primals_289, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf581, (8, 2048, 7, 7), (100352, 49, 7, 1))
        buf582 = buf559; del buf559  # reuse
        buf583 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf585 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_376], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf581, primals_603, primals_604, buf582, buf583, buf585, primals_603, primals_604, 2048, 392, grid=grid(2048), stream=stream0)
        del primals_603
        del primals_604
        # Source Nodes: [getattr_l__mod___layer4___0___downsample_0], Original ATen: [aten.convolution]
        buf586 = extern_kernels.convolution(buf568, primals_292, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf586, (8, 2048, 7, 7), (100352, 49, 7, 1))
        buf587 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf588 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf590 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_34], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf586, primals_606, primals_607, buf587, buf588, buf590, primals_606, primals_607, 2048, 392, grid=grid(2048), stream=stream0)
        del primals_606
        del primals_607
        buf591 = empty((8, 2048, 7, 7), device='cuda', dtype=torch.float32)
        buf592 = buf591; del buf591  # reuse
        # Source Nodes: [shortcut_34, shortcut_35, x_376, x_377], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_28.run(buf592, buf581, buf582, buf583, primals_290, primals_291, buf586, buf587, buf588, primals_293, primals_294, 802816, grid=grid(802816), stream=stream0)
        del primals_291
        del primals_294
        # Source Nodes: [x_379], Original ATen: [aten.convolution]
        buf593 = extern_kernels.convolution(buf592, primals_295, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf593, (8, 4096, 7, 7), (200704, 49, 7, 1))
        buf594 = buf577; del buf577  # reuse
        buf595 = empty_strided((1, 4096, 1, 1), (4096, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf597 = empty((4096, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_380], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_25.run(buf593, primals_609, primals_610, buf594, buf595, buf597, primals_609, primals_610, 4096, 392, grid=grid(4096), stream=stream0)
        del primals_609
        del primals_610
        buf598 = empty((8, 4096, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_380, x_381], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf593, buf594, buf595, primals_296, primals_297, buf598, 1605632, grid=grid(1605632), stream=stream0)
        del primals_297
        # Source Nodes: [x_382], Original ATen: [aten.convolution]
        buf599 = extern_kernels.convolution(buf598, primals_298, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf599, (8, 4096, 7, 7), (200704, 49, 7, 1))
        buf600 = buf595; del buf595  # reuse
        buf601 = empty_strided((1, 4096, 1, 1), (4096, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf603 = empty((4096, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_383], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_25.run(buf599, primals_612, primals_613, buf600, buf601, buf603, primals_612, primals_613, 4096, 392, grid=grid(4096), stream=stream0)
        del primals_612
        del primals_613
        buf604 = empty((8, 4096, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_383, x_385], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf599, buf600, buf601, primals_299, primals_300, buf604, 1605632, grid=grid(1605632), stream=stream0)
        del primals_300
        # Source Nodes: [x_387], Original ATen: [aten.convolution]
        buf605 = extern_kernels.convolution(buf604, primals_301, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf605, (8, 2048, 7, 7), (100352, 49, 7, 1))
        buf606 = buf588; del buf588  # reuse
        buf607 = buf583; del buf583  # reuse
        buf609 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_388], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf605, primals_615, primals_616, buf606, buf607, buf609, primals_615, primals_616, 2048, 392, grid=grid(2048), stream=stream0)
        del primals_615
        del primals_616
        buf610 = empty((8, 2048, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_36, x_388, x_389], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_29.run(buf605, buf606, buf607, primals_302, primals_303, buf592, buf610, 802816, grid=grid(802816), stream=stream0)
        del primals_303
        # Source Nodes: [x_391], Original ATen: [aten.convolution]
        buf611 = extern_kernels.convolution(buf610, primals_304, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf611, (8, 4096, 7, 7), (200704, 49, 7, 1))
        buf612 = buf601; del buf601  # reuse
        buf613 = empty_strided((1, 4096, 1, 1), (4096, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf615 = empty((4096, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_392], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_25.run(buf611, primals_618, primals_619, buf612, buf613, buf615, primals_618, primals_619, 4096, 392, grid=grid(4096), stream=stream0)
        del primals_618
        del primals_619
        buf616 = empty((8, 4096, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_392, x_393], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf611, buf612, buf613, primals_305, primals_306, buf616, 1605632, grid=grid(1605632), stream=stream0)
        del primals_306
        # Source Nodes: [x_394], Original ATen: [aten.convolution]
        buf617 = extern_kernels.convolution(buf616, primals_307, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf617, (8, 4096, 7, 7), (200704, 49, 7, 1))
        buf618 = buf613; del buf613  # reuse
        buf619 = empty_strided((1, 4096, 1, 1), (4096, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf621 = empty((4096, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_395], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_25.run(buf617, primals_621, primals_622, buf618, buf619, buf621, primals_621, primals_622, 4096, 392, grid=grid(4096), stream=stream0)
        del primals_621
        del primals_622
        buf622 = empty((8, 4096, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_395, x_397], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_26.run(buf617, buf618, buf619, primals_308, primals_309, buf622, 1605632, grid=grid(1605632), stream=stream0)
        del buf619
        del primals_309
        # Source Nodes: [x_399], Original ATen: [aten.convolution]
        buf623 = extern_kernels.convolution(buf622, primals_310, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf623, (8, 2048, 7, 7), (100352, 49, 7, 1))
        buf624 = buf607; del buf607  # reuse
        buf625 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf627 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_400], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_27.run(buf623, primals_624, primals_625, buf624, buf625, buf627, primals_624, primals_625, 2048, 392, grid=grid(2048), stream=stream0)
        del primals_624
        del primals_625
        buf632 = empty((8, 2048, 7, 7), device='cuda', dtype=torch.bool)
        buf629 = empty_strided((8, 2048, 1, 1), (2048, 1, 16384, 16384), device='cuda', dtype=torch.float32)
        buf630 = reinterpret_tensor(buf629, (8, 2048), (2048, 1), 0); del buf629  # reuse
        # Source Nodes: [x_400, x_401, x_404, x_405, x_407], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.mean, aten.relu, aten.threshold_backward, aten.view]
        triton_per_fused__native_batch_norm_legit_functional_add_mean_relu_threshold_backward_view_30.run(buf630, buf623, buf624, buf625, primals_311, primals_312, buf610, buf632, 16384, 49, grid=grid(16384), stream=stream0)
        del buf625
        del primals_312
        buf631 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_314, buf630, reinterpret_tensor(primals_313, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf631)
        del primals_314
        # Source Nodes: [x_1], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_317, primals_317, 1, grid=grid(1), stream=stream0)
        del primals_317
        # Source Nodes: [x_5], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_320, primals_320, 1, grid=grid(1), stream=stream0)
        del primals_320
        # Source Nodes: [x_8], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_323, primals_323, 1, grid=grid(1), stream=stream0)
        del primals_323
        # Source Nodes: [x_13], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_326, primals_326, 1, grid=grid(1), stream=stream0)
        del primals_326
        # Source Nodes: [shortcut_1], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_329, primals_329, 1, grid=grid(1), stream=stream0)
        del primals_329
        # Source Nodes: [x_17], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_332, primals_332, 1, grid=grid(1), stream=stream0)
        del primals_332
        # Source Nodes: [x_20], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_335, primals_335, 1, grid=grid(1), stream=stream0)
        del primals_335
        # Source Nodes: [x_25], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_338, primals_338, 1, grid=grid(1), stream=stream0)
        del primals_338
        # Source Nodes: [x_29], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_341, primals_341, 1, grid=grid(1), stream=stream0)
        del primals_341
        # Source Nodes: [x_32], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_344, primals_344, 1, grid=grid(1), stream=stream0)
        del primals_344
        # Source Nodes: [x_37], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_347, primals_347, 1, grid=grid(1), stream=stream0)
        del primals_347
        # Source Nodes: [x_42], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_350, primals_350, 1, grid=grid(1), stream=stream0)
        del primals_350
        # Source Nodes: [x_45], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_353, primals_353, 1, grid=grid(1), stream=stream0)
        del primals_353
        # Source Nodes: [x_50], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_356, primals_356, 1, grid=grid(1), stream=stream0)
        del primals_356
        # Source Nodes: [shortcut_5], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_359, primals_359, 1, grid=grid(1), stream=stream0)
        del primals_359
        # Source Nodes: [x_54], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_362, primals_362, 1, grid=grid(1), stream=stream0)
        del primals_362
        # Source Nodes: [x_57], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_365, primals_365, 1, grid=grid(1), stream=stream0)
        del primals_365
        # Source Nodes: [x_62], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_368, primals_368, 1, grid=grid(1), stream=stream0)
        del primals_368
        # Source Nodes: [x_66], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_371, primals_371, 1, grid=grid(1), stream=stream0)
        del primals_371
        # Source Nodes: [x_69], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_374, primals_374, 1, grid=grid(1), stream=stream0)
        del primals_374
        # Source Nodes: [x_74], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_377, primals_377, 1, grid=grid(1), stream=stream0)
        del primals_377
        # Source Nodes: [x_78], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_380, primals_380, 1, grid=grid(1), stream=stream0)
        del primals_380
        # Source Nodes: [x_81], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_383, primals_383, 1, grid=grid(1), stream=stream0)
        del primals_383
        # Source Nodes: [x_86], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_386, primals_386, 1, grid=grid(1), stream=stream0)
        del primals_386
        # Source Nodes: [x_91], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_389, primals_389, 1, grid=grid(1), stream=stream0)
        del primals_389
        # Source Nodes: [x_94], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_392, primals_392, 1, grid=grid(1), stream=stream0)
        del primals_392
        # Source Nodes: [x_99], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_395, primals_395, 1, grid=grid(1), stream=stream0)
        del primals_395
        # Source Nodes: [shortcut_10], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_398, primals_398, 1, grid=grid(1), stream=stream0)
        del primals_398
        # Source Nodes: [x_103], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_401, primals_401, 1, grid=grid(1), stream=stream0)
        del primals_401
        # Source Nodes: [x_106], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_404, primals_404, 1, grid=grid(1), stream=stream0)
        del primals_404
        # Source Nodes: [x_111], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_407, primals_407, 1, grid=grid(1), stream=stream0)
        del primals_407
        # Source Nodes: [x_115], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_410, primals_410, 1, grid=grid(1), stream=stream0)
        del primals_410
        # Source Nodes: [x_118], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_413, primals_413, 1, grid=grid(1), stream=stream0)
        del primals_413
        # Source Nodes: [x_123], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_416, primals_416, 1, grid=grid(1), stream=stream0)
        del primals_416
        # Source Nodes: [x_127], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_419, primals_419, 1, grid=grid(1), stream=stream0)
        del primals_419
        # Source Nodes: [x_130], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_422, primals_422, 1, grid=grid(1), stream=stream0)
        del primals_422
        # Source Nodes: [x_135], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_425, primals_425, 1, grid=grid(1), stream=stream0)
        del primals_425
        # Source Nodes: [x_139], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_428, primals_428, 1, grid=grid(1), stream=stream0)
        del primals_428
        # Source Nodes: [x_142], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_431, primals_431, 1, grid=grid(1), stream=stream0)
        del primals_431
        # Source Nodes: [x_147], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_434, primals_434, 1, grid=grid(1), stream=stream0)
        del primals_434
        # Source Nodes: [x_151], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_437, primals_437, 1, grid=grid(1), stream=stream0)
        del primals_437
        # Source Nodes: [x_154], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_440, primals_440, 1, grid=grid(1), stream=stream0)
        del primals_440
        # Source Nodes: [x_159], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_443, primals_443, 1, grid=grid(1), stream=stream0)
        del primals_443
        # Source Nodes: [x_163], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_446, primals_446, 1, grid=grid(1), stream=stream0)
        del primals_446
        # Source Nodes: [x_166], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_449, primals_449, 1, grid=grid(1), stream=stream0)
        del primals_449
        # Source Nodes: [x_171], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_452, primals_452, 1, grid=grid(1), stream=stream0)
        del primals_452
        # Source Nodes: [x_175], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_455, primals_455, 1, grid=grid(1), stream=stream0)
        del primals_455
        # Source Nodes: [x_178], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_458, primals_458, 1, grid=grid(1), stream=stream0)
        del primals_458
        # Source Nodes: [x_183], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_461, primals_461, 1, grid=grid(1), stream=stream0)
        del primals_461
        # Source Nodes: [x_187], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_464, primals_464, 1, grid=grid(1), stream=stream0)
        del primals_464
        # Source Nodes: [x_190], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_467, primals_467, 1, grid=grid(1), stream=stream0)
        del primals_467
        # Source Nodes: [x_195], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_470, primals_470, 1, grid=grid(1), stream=stream0)
        del primals_470
        # Source Nodes: [x_199], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_473, primals_473, 1, grid=grid(1), stream=stream0)
        del primals_473
        # Source Nodes: [x_202], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_476, primals_476, 1, grid=grid(1), stream=stream0)
        del primals_476
        # Source Nodes: [x_207], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_479, primals_479, 1, grid=grid(1), stream=stream0)
        del primals_479
        # Source Nodes: [x_211], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_482, primals_482, 1, grid=grid(1), stream=stream0)
        del primals_482
        # Source Nodes: [x_214], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_485, primals_485, 1, grid=grid(1), stream=stream0)
        del primals_485
        # Source Nodes: [x_219], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_488, primals_488, 1, grid=grid(1), stream=stream0)
        del primals_488
        # Source Nodes: [x_223], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_491, primals_491, 1, grid=grid(1), stream=stream0)
        del primals_491
        # Source Nodes: [x_226], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_494, primals_494, 1, grid=grid(1), stream=stream0)
        del primals_494
        # Source Nodes: [x_231], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_497, primals_497, 1, grid=grid(1), stream=stream0)
        del primals_497
        # Source Nodes: [x_235], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_500, primals_500, 1, grid=grid(1), stream=stream0)
        del primals_500
        # Source Nodes: [x_238], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_503, primals_503, 1, grid=grid(1), stream=stream0)
        del primals_503
        # Source Nodes: [x_243], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_506, primals_506, 1, grid=grid(1), stream=stream0)
        del primals_506
        # Source Nodes: [x_247], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_509, primals_509, 1, grid=grid(1), stream=stream0)
        del primals_509
        # Source Nodes: [x_250], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_512, primals_512, 1, grid=grid(1), stream=stream0)
        del primals_512
        # Source Nodes: [x_255], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_515, primals_515, 1, grid=grid(1), stream=stream0)
        del primals_515
        # Source Nodes: [x_259], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_518, primals_518, 1, grid=grid(1), stream=stream0)
        del primals_518
        # Source Nodes: [x_262], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_521, primals_521, 1, grid=grid(1), stream=stream0)
        del primals_521
        # Source Nodes: [x_267], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_524, primals_524, 1, grid=grid(1), stream=stream0)
        del primals_524
        # Source Nodes: [x_271], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_527, primals_527, 1, grid=grid(1), stream=stream0)
        del primals_527
        # Source Nodes: [x_274], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_530, primals_530, 1, grid=grid(1), stream=stream0)
        del primals_530
        # Source Nodes: [x_279], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_533, primals_533, 1, grid=grid(1), stream=stream0)
        del primals_533
        # Source Nodes: [x_283], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_536, primals_536, 1, grid=grid(1), stream=stream0)
        del primals_536
        # Source Nodes: [x_286], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_539, primals_539, 1, grid=grid(1), stream=stream0)
        del primals_539
        # Source Nodes: [x_291], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_542, primals_542, 1, grid=grid(1), stream=stream0)
        del primals_542
        # Source Nodes: [x_295], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_545, primals_545, 1, grid=grid(1), stream=stream0)
        del primals_545
        # Source Nodes: [x_298], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_548, primals_548, 1, grid=grid(1), stream=stream0)
        del primals_548
        # Source Nodes: [x_303], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_551, primals_551, 1, grid=grid(1), stream=stream0)
        del primals_551
        # Source Nodes: [x_307], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_554, primals_554, 1, grid=grid(1), stream=stream0)
        del primals_554
        # Source Nodes: [x_310], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_557, primals_557, 1, grid=grid(1), stream=stream0)
        del primals_557
        # Source Nodes: [x_315], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_560, primals_560, 1, grid=grid(1), stream=stream0)
        del primals_560
        # Source Nodes: [x_319], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_563, primals_563, 1, grid=grid(1), stream=stream0)
        del primals_563
        # Source Nodes: [x_322], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_566, primals_566, 1, grid=grid(1), stream=stream0)
        del primals_566
        # Source Nodes: [x_327], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_569, primals_569, 1, grid=grid(1), stream=stream0)
        del primals_569
        # Source Nodes: [x_331], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_572, primals_572, 1, grid=grid(1), stream=stream0)
        del primals_572
        # Source Nodes: [x_334], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_575, primals_575, 1, grid=grid(1), stream=stream0)
        del primals_575
        # Source Nodes: [x_339], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_578, primals_578, 1, grid=grid(1), stream=stream0)
        del primals_578
        # Source Nodes: [x_343], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_581, primals_581, 1, grid=grid(1), stream=stream0)
        del primals_581
        # Source Nodes: [x_346], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_584, primals_584, 1, grid=grid(1), stream=stream0)
        del primals_584
        # Source Nodes: [x_351], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_587, primals_587, 1, grid=grid(1), stream=stream0)
        del primals_587
        # Source Nodes: [x_355], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_590, primals_590, 1, grid=grid(1), stream=stream0)
        del primals_590
        # Source Nodes: [x_358], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_593, primals_593, 1, grid=grid(1), stream=stream0)
        del primals_593
        # Source Nodes: [x_363], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_596, primals_596, 1, grid=grid(1), stream=stream0)
        del primals_596
        # Source Nodes: [x_368], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_599, primals_599, 1, grid=grid(1), stream=stream0)
        del primals_599
        # Source Nodes: [x_371], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_602, primals_602, 1, grid=grid(1), stream=stream0)
        del primals_602
        # Source Nodes: [x_376], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_605, primals_605, 1, grid=grid(1), stream=stream0)
        del primals_605
        # Source Nodes: [shortcut_34], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_608, primals_608, 1, grid=grid(1), stream=stream0)
        del primals_608
        # Source Nodes: [x_380], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_611, primals_611, 1, grid=grid(1), stream=stream0)
        del primals_611
        # Source Nodes: [x_383], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_614, primals_614, 1, grid=grid(1), stream=stream0)
        del primals_614
        # Source Nodes: [x_388], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_617, primals_617, 1, grid=grid(1), stream=stream0)
        del primals_617
        # Source Nodes: [x_392], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_620, primals_620, 1, grid=grid(1), stream=stream0)
        del primals_620
        # Source Nodes: [x_395], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_623, primals_623, 1, grid=grid(1), stream=stream0)
        del primals_623
        # Source Nodes: [x_400], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(primals_626, primals_626, 1, grid=grid(1), stream=stream0)
        del primals_626
        return (buf631, primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_173, primals_175, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_191, primals_193, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_236, primals_238, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_251, primals_253, primals_254, primals_256, primals_257, primals_259, primals_260, primals_262, primals_263, primals_265, primals_266, primals_268, primals_269, primals_271, primals_272, primals_274, primals_275, primals_277, primals_278, primals_280, primals_281, primals_283, primals_284, primals_286, primals_287, primals_289, primals_290, primals_292, primals_293, primals_295, primals_296, primals_298, primals_299, primals_301, primals_302, primals_304, primals_305, primals_307, primals_308, primals_310, primals_311, primals_627, buf0, buf7, buf8, buf9, buf10, buf11, buf15, buf16, buf17, buf21, buf22, buf23, buf27, buf28, buf32, buf34, buf35, buf39, buf40, buf41, buf45, buf46, buf47, buf51, buf52, buf53, buf57, buf58, buf59, buf63, buf64, buf65, buf69, buf70, buf71, buf75, buf76, buf77, buf81, buf82, buf83, buf87, buf88, buf92, buf94, buf95, buf99, buf100, buf101, buf105, buf106, buf107, buf111, buf112, buf113, buf117, buf118, buf119, buf123, buf124, buf125, buf129, buf130, buf131, buf135, buf136, buf137, buf141, buf142, buf143, buf147, buf148, buf149, buf153, buf154, buf155, buf159, buf160, buf161, buf165, buf166, buf170, buf172, buf173, buf177, buf178, buf179, buf183, buf184, buf185, buf189, buf190, buf191, buf195, buf196, buf197, buf201, buf202, buf203, buf207, buf208, buf209, buf213, buf214, buf215, buf219, buf220, buf221, buf225, buf226, buf227, buf231, buf232, buf233, buf237, buf238, buf239, buf243, buf244, buf245, buf249, buf250, buf251, buf255, buf256, buf257, buf261, buf262, buf263, buf267, buf268, buf269, buf273, buf274, buf275, buf279, buf280, buf281, buf285, buf286, buf287, buf291, buf292, buf293, buf297, buf298, buf299, buf303, buf304, buf305, buf309, buf310, buf311, buf315, buf316, buf317, buf321, buf322, buf323, buf327, buf328, buf329, buf333, buf334, buf335, buf339, buf340, buf341, buf345, buf346, buf347, buf351, buf352, buf353, buf357, buf358, buf359, buf363, buf364, buf365, buf369, buf370, buf371, buf375, buf376, buf377, buf381, buf382, buf383, buf387, buf388, buf389, buf393, buf394, buf395, buf399, buf400, buf401, buf405, buf406, buf407, buf411, buf412, buf413, buf417, buf418, buf419, buf423, buf424, buf425, buf429, buf430, buf431, buf435, buf436, buf437, buf441, buf442, buf443, buf447, buf448, buf449, buf453, buf454, buf455, buf459, buf460, buf461, buf465, buf466, buf467, buf471, buf472, buf473, buf477, buf478, buf479, buf483, buf484, buf485, buf489, buf490, buf491, buf495, buf496, buf497, buf501, buf502, buf503, buf507, buf508, buf509, buf513, buf514, buf515, buf519, buf520, buf521, buf525, buf526, buf527, buf531, buf532, buf533, buf537, buf538, buf539, buf543, buf544, buf545, buf549, buf550, buf551, buf555, buf556, buf557, buf561, buf562, buf563, buf567, buf568, buf569, buf573, buf574, buf575, buf579, buf580, buf581, buf585, buf586, buf590, buf592, buf593, buf597, buf598, buf599, buf603, buf604, buf605, buf609, buf610, buf611, buf615, buf616, buf617, buf621, buf622, buf623, buf627, buf630, reinterpret_tensor(primals_313, (1000, 2048), (2048, 1), 0), buf632, reinterpret_tensor(buf624, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf618, (1, 4096, 1, 1), (4096, 1, 1, 1), 0), reinterpret_tensor(buf612, (1, 4096, 1, 1), (4096, 1, 1, 1), 0), reinterpret_tensor(buf606, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf600, (1, 4096, 1, 1), (4096, 1, 1, 1), 0), reinterpret_tensor(buf594, (1, 4096, 1, 1), (4096, 1, 1, 1), 0), reinterpret_tensor(buf587, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf582, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf576, (1, 4096, 1, 1), (4096, 1, 1, 1), 0), reinterpret_tensor(buf570, (1, 4096, 1, 1), (4096, 1, 1, 1), 0), reinterpret_tensor(buf564, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf558, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf552, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf546, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf540, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf534, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf528, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf522, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf516, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf510, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf504, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf498, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf492, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf486, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf480, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf474, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf468, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf462, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf456, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf450, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf444, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf438, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf432, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf426, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf420, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf414, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf408, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf402, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf396, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf390, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf384, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf378, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf372, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf366, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf360, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf354, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf348, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf342, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf336, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf330, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf324, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf318, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf312, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf306, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf300, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf294, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf288, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf282, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf276, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf270, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf264, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf258, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf252, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf246, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf240, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf234, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf228, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf222, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf216, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf210, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf204, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf198, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf192, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf186, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf180, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf174, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf167, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf162, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf156, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf150, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf144, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf138, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf132, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf126, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf120, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf114, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf108, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf102, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf96, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf89, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf84, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf78, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf72, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf66, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf60, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf54, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf48, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf42, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf36, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf29, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf24, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf18, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf12, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf4, (1, 64, 1, 1), (64, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((512, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((4096, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((4096, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((2048, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((4096, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((4096, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((2048, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((4096, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((4096, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((2048, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_318 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_321 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_324 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_327 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_330 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_333 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_336 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_339 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_342 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_345 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_348 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_351 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_354 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_357 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_360 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_363 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_366 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_369 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_372 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_375 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_378 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_381 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_384 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_387 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_390 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_393 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_396 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_399 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_402 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_405 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_408 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_411 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_414 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_417 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_420 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_423 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_426 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_429 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_432 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_435 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_438 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_441 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_444 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_447 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_450 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_453 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_456 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_459 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_462 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_465 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_468 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_471 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_474 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_477 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_480 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_483 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_486 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_489 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_492 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_495 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_498 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_501 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_504 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_507 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_510 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_513 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_515 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_516 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_518 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_519 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_521 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_522 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_524 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_525 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_527 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_528 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_530 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_531 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_533 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_534 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_536 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_537 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_539 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_540 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_542 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_543 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_545 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_546 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_548 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_549 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_551 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_552 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_554 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_555 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_557 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_558 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_560 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_561 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_563 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_564 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_566 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_567 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_569 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_570 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_572 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_573 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_575 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_576 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_578 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_579 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_581 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_582 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_584 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_585 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_587 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_588 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_590 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_591 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_593 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_594 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_596 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_597 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_599 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_600 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_602 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_603 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_605 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_606 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_608 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_609 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_611 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_612 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_614 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_615 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_617 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_618 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_620 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_621 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_623 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_624 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_626 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_627 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('swsl_resnext101_32x16d', benchmark_compiled_module)
