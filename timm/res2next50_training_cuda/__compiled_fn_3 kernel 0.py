
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


# kernel path: /tmp/torchinductor_youkaichao/ig/cigspgxggf33pekwuuojqb3q7kkinyipdrisqmzqz3c3p3j5fpon.py
# Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
# out_1 => var_mean_1
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


# kernel path: /tmp/torchinductor_youkaichao/a7/ca7sufs4sz6ncupfpphisznpipawdpkyvexz26mn6ouvgmtj5olb.py
# Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
# out_1 => add_6, add_7, add_8, mul_10, mul_11, mul_12, mul_8, mul_9, rsqrt_1, squeeze_4, var_mean_1
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_5', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/bd/cbduxtnfvcncog4kjmlgi63qm6onjksv6wmo6hc3sxggtvu54hpc.py
# Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# out_1 => add_6, add_9, mul_13, mul_7, rsqrt_1, sub_1, var_mean_1
# out_2 => relu_1
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x3), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2w/c2wt73nvbaobchoydobafr2szehm4s7tmlc6fdvwja7gamb2bf5t.py
# Source Nodes: [sp_2], Original ATen: [aten._native_batch_norm_legit_functional]
# sp_2 => var_mean_2
triton_red_fused__native_batch_norm_legit_functional_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_7', 'mutated_arg_names': []}
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


# kernel path: /tmp/torchinductor_youkaichao/2d/c2dlpki6v4esmydm4bkbfncsqoc5rvm532tavdhs3fdn6vurxwu5.py
# Source Nodes: [sp_2], Original ATen: [aten._native_batch_norm_legit_functional]
# sp_2 => add_11, add_12, add_13, mul_15, mul_16, mul_17, mul_18, mul_19, rsqrt_2, squeeze_7, var_mean_2
triton_per_fused__native_batch_norm_legit_functional_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_8', 'mutated_arg_names': ['in_ptr3', 'in_ptr4', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/p7/cp7no2unv23jvw3p7ah4dlo7sbnrxhpegvv4g2k6havmul3lif57.py
# Source Nodes: [sp_2, sp_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# sp_2 => add_11, add_14, mul_14, mul_20, rsqrt_2, sub_2, var_mean_2
# sp_3 => relu_2
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_9', 'mutated_arg_names': []},
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
    x2 = (xindex // 100352)
    x4 = xindex % 100352
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x4 + (401408*x2)), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pj/cpj5zgwhg6vj6baoavyndzesllox4lgjofxpvi2essipulxcdxoa.py
# Source Nodes: [getattr_l__mod___layer1___0___pool], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer1___0___pool => avg_pool2d
triton_poi_fused_avg_pool2d_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 56) % 56
    x0 = xindex % 56
    x3 = (xindex // 100352)
    x6 = xindex % 100352
    tmp0 = (-1) + x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (300999 + x6 + (401408*x3)), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + (301000 + x6 + (401408*x3)), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x0
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + (301001 + x6 + (401408*x3)), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + (301055 + x6 + (401408*x3)), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (301056 + x6 + (401408*x3)), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (301057 + x6 + (401408*x3)), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x1
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (301111 + x6 + (401408*x3)), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (301112 + x6 + (401408*x3)), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (301113 + x6 + (401408*x3)), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 57, tl.int64)
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
    tl.store(out_ptr0 + (x6 + (401408*x3)), tmp145, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/el/cel3pm6kmnarqlya2izf4sjzl5fxnwmkk2c6a33y5x6klkcvugwq.py
# Source Nodes: [out_5], Original ATen: [aten._native_batch_norm_legit_functional]
# out_5 => add_26, add_27, add_28, mul_36, mul_37, mul_38, mul_39, mul_40, rsqrt_5, squeeze_16, var_mean_5
triton_red_fused__native_batch_norm_legit_functional_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_11', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/mw/cmwlgwk6huyzuhysnwc2r2efbgqu2imcfqwxr7dc5x4jw5a4qysk.py
# Source Nodes: [out_5, out_6, shortcut_1, shortcut_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_5 => add_26, add_29, mul_35, mul_41, rsqrt_5, sub_5, var_mean_5
# out_6 => add_35
# shortcut_1 => add_31, add_34, mul_42, mul_48, rsqrt_6, sub_6, var_mean_6
# shortcut_2 => relu_5
triton_poi_fused__native_batch_norm_legit_functional_add_relu_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_12', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/7d/c7di64gk2wrqtrskqhzweipn6mk6n4mis7f6jp6qsqpmsiah2cm6.py
# Source Nodes: [sp_15, sp_16, sp_17], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
# sp_15 => add_42, add_45, mul_56, mul_62, rsqrt_8, sub_8, var_mean_8
# sp_16 => relu_7
# sp_17 => add_46
triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 32
    x2 = (xindex // 100352)
    x4 = xindex % 100352
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (100352 + x4 + (401408*x2)), None)
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
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = tmp14 <= tmp17
    tl.store(out_ptr0 + (x4 + (401408*x2)), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
    tl.store(out_ptr2 + (x3), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/to/ctomrgriwgsz2jc4skichjhsi3pmola3ei2r2kzrdtbz5mdglij4.py
# Source Nodes: [sp_19, sp_20, sp_21], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
# sp_19 => add_48, add_51, mul_63, mul_69, rsqrt_9, sub_9, var_mean_9
# sp_20 => relu_8
# sp_21 => add_52
triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 32
    x2 = (xindex // 100352)
    x4 = xindex % 100352
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (200704 + x4 + (401408*x2)), None)
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
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = tmp14 <= tmp17
    tl.store(out_ptr0 + (x4 + (401408*x2)), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
    tl.store(out_ptr2 + (x3), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lo/clohe3li25h33d3xy5mlrsrsj7tpjnvanhccua3aed5dlexbenm3.py
# Source Nodes: [cat_30], Original ATen: [aten.cat]
# cat_30 => cat_1
triton_poi_fused_cat_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 100352
    x1 = (xindex // 100352)
    tmp0 = tl.load(in_ptr0 + (301056 + x0 + (401408*x1)), None)
    tl.store(out_ptr0 + (x0 + (401408*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tg/ctgfkyf6usbajjwh4am23b2th254tttuodokl5ixx2tpzbfobw5j.py
# Source Nodes: [out_13, out_14, shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_13 => add_59, add_62, mul_77, mul_83, rsqrt_11, sub_11, var_mean_11
# out_14 => add_63
# shortcut_3 => relu_10
triton_poi_fused__native_batch_norm_legit_functional_add_relu_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_16', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/pu/cpuyi32qivspnbnstgapf57lu2clbjweotkad47n4a77hwjfe2hq.py
# Source Nodes: [out_25, out_26], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# out_25 => add_93, add_96, mul_119, mul_125, rsqrt_17, sub_17, var_mean_17
# out_26 => relu_16
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x3), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6s/c6sym5mlhiukvwbkt2jo3yecc5255guzrjwkypj34w52coppocbh.py
# Source Nodes: [sp_41], Original ATen: [aten._native_batch_norm_legit_functional]
# sp_41 => add_100, add_98, add_99, mul_127, mul_128, mul_129, mul_130, mul_131, rsqrt_18, squeeze_55, var_mean_18
triton_red_fused__native_batch_norm_legit_functional_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_18', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/u6/cu66l24h7j5jfm2rafzieuvdnqwia5djpnpdtvhj6txxmhjydqav.py
# Source Nodes: [sp_41, sp_42], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# sp_41 => add_101, add_98, mul_126, mul_132, rsqrt_18, sub_18, var_mean_18
# sp_42 => relu_17
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_19', 'mutated_arg_names': []},
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
    x2 = (xindex // 50176)
    x4 = xindex % 50176
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x4 + (200704*x2)), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7e/c7e4bj75tvx3mt2m2r7ha7eux7qysaf6u2p3np2qshmeflolqryp.py
# Source Nodes: [getattr_l__mod___layer2___0___pool], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer2___0___pool => avg_pool2d_1
triton_poi_fused_avg_pool2d_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 28) % 28
    x0 = xindex % 28
    x3 = (xindex // 50176)
    x6 = (xindex // 28) % 1792
    x7 = xindex % 50176
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (602055 + (2*x0) + (112*x6) + (802816*x3)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + (602056 + (2*x0) + (112*x6) + (802816*x3)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + (602057 + (2*x0) + (112*x6) + (802816*x3)), tmp27, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + (602111 + (2*x0) + (112*x6) + (802816*x3)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (602112 + (2*x0) + (112*x6) + (802816*x3)), tmp41, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (602113 + (2*x0) + (112*x6) + (802816*x3)), tmp46, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (602167 + (2*x0) + (112*x6) + (802816*x3)), tmp55, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (602168 + (2*x0) + (112*x6) + (802816*x3)), tmp60, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (602169 + (2*x0) + (112*x6) + (802816*x3)), tmp65, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 57, tl.int64)
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
    tl.store(out_ptr0 + (x7 + (200704*x3)), tmp145, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/r6/cr62a72udz4zu3qlca5tt57g5sw7xkl2i5xpu5bjsw7nv3df655v.py
# Source Nodes: [out_29], Original ATen: [aten._native_batch_norm_legit_functional]
# out_29 => add_113, add_114, add_115, mul_148, mul_149, mul_150, mul_151, mul_152, rsqrt_21, squeeze_64, var_mean_21
triton_red_fused__native_batch_norm_legit_functional_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_21', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/nt/cntmi7nyt56s3rde4llplxfetlfnt4ssquxytvpfww36jz7zhe75.py
# Source Nodes: [out_29, out_30, shortcut_5, shortcut_6], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_29 => add_113, add_116, mul_147, mul_153, rsqrt_21, sub_21, var_mean_21
# out_30 => add_122
# shortcut_5 => add_118, add_121, mul_154, mul_160, rsqrt_22, sub_22, var_mean_22
# shortcut_6 => relu_20
triton_poi_fused__native_batch_norm_legit_functional_add_relu_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_22', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/r4/cr4hgcvch3xmkwtwmetcqpwmz2rfdnxbucsi7d6yhfou7r33um4w.py
# Source Nodes: [out_33], Original ATen: [aten._native_batch_norm_legit_functional]
# out_33 => add_124, add_125, add_126, mul_162, mul_163, mul_164, mul_165, mul_166, rsqrt_23, squeeze_70, var_mean_23
triton_red_fused__native_batch_norm_legit_functional_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_23', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 256
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
        tmp0 = tl.load(in_ptr0 + (r1 + (784*x0) + (200704*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/zb/czbnaun37knmw2bouamnarbgoi4niydhbbw4ho7salov4ezhoifp.py
# Source Nodes: [out_33, out_34], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# out_33 => add_124, add_127, mul_161, mul_167, rsqrt_23, sub_23, var_mean_23
# out_34 => relu_21
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 256
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x3), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6k/c6kykccqyi5ydt6nijgdxmq7xjmnio6eha3qvv6smyqtdhw5afql.py
# Source Nodes: [sp_54, sp_55, sp_56], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
# sp_54 => add_129, add_132, mul_168, mul_174, rsqrt_24, sub_24, var_mean_24
# sp_55 => relu_22
# sp_56 => add_133
triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_25 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 64
    x2 = (xindex // 50176)
    x4 = xindex % 50176
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (50176 + x4 + (200704*x2)), None)
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
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = tmp14 <= tmp17
    tl.store(out_ptr0 + (x4 + (200704*x2)), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
    tl.store(out_ptr2 + (x3), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cd/ccdzen2upykqmfdi322xxuf6zifr6m5l5ao2d3wipr6rbohcorhl.py
# Source Nodes: [sp_58, sp_59, sp_60], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
# sp_58 => add_135, add_138, mul_175, mul_181, rsqrt_25, sub_25, var_mean_25
# sp_59 => relu_23
# sp_60 => add_139
triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 64
    x2 = (xindex // 50176)
    x4 = xindex % 50176
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (100352 + x4 + (200704*x2)), None)
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
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = tmp14 <= tmp17
    tl.store(out_ptr0 + (x4 + (200704*x2)), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
    tl.store(out_ptr2 + (x3), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bl/cbl5kl25uicxsi2m455do44qsmteywxmvxdboat2572qiepv45qo.py
# Source Nodes: [cat_27], Original ATen: [aten.cat]
# cat_27 => cat_4
triton_poi_fused_cat_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 50176
    x1 = (xindex // 50176)
    tmp0 = tl.load(in_ptr0 + (150528 + x0 + (200704*x1)), None)
    tl.store(out_ptr0 + (x0 + (200704*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/f2/cf23ok76ijqbnuzybiawfmxsb3uvtknycqret75leaqajivdqzw6.py
# Source Nodes: [out_37, out_38, shortcut_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_37 => add_146, add_149, mul_189, mul_195, rsqrt_27, sub_27, var_mean_27
# out_38 => add_150
# shortcut_7 => relu_25
triton_poi_fused__native_batch_norm_legit_functional_add_relu_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_28', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ki/ckihcalllwkud7v72xr7z4turllusiosmuwqmauxrnkiwc44u2hs.py
# Source Nodes: [out_57, out_58], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# out_57 => add_208, add_211, mul_266, mul_272, rsqrt_38, sub_38, var_mean_38
# out_58 => relu_36
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_29 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x3), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2i/c2ialsd357kfcmzcuwueoplsyr55hxc6rd7fmci3zeqi324uroah.py
# Source Nodes: [sp_93], Original ATen: [aten._native_batch_norm_legit_functional]
# sp_93 => add_213, add_214, add_215, mul_274, mul_275, mul_276, mul_277, mul_278, rsqrt_39, squeeze_118, var_mean_39
triton_red_fused__native_batch_norm_legit_functional_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_30', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/6s/c6sd73fpgihtkrwqkumx7y6qt247g7vrqrj72nbg5xcgto6tk357.py
# Source Nodes: [sp_93, sp_94], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# sp_93 => add_213, add_216, mul_273, mul_279, rsqrt_39, sub_39, var_mean_39
# sp_94 => relu_37
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 128
    x2 = (xindex // 25088)
    x4 = xindex % 25088
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x4 + (100352*x2)), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/23/c23gyztzlt5hhw74mshdhulowph77cy3hil7h7wiscpwc7fyj4gb.py
# Source Nodes: [getattr_l__mod___layer3___0___pool], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer3___0___pool => avg_pool2d_2
triton_poi_fused_avg_pool2d_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 14) % 14
    x0 = xindex % 14
    x3 = (xindex // 25088)
    x6 = (xindex // 14) % 1792
    x7 = xindex % 25088
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (301027 + (2*x0) + (56*x6) + (401408*x3)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + (301028 + (2*x0) + (56*x6) + (401408*x3)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + (301029 + (2*x0) + (56*x6) + (401408*x3)), tmp27, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + (301055 + (2*x0) + (56*x6) + (401408*x3)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (301056 + (2*x0) + (56*x6) + (401408*x3)), tmp41, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (301057 + (2*x0) + (56*x6) + (401408*x3)), tmp46, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (301083 + (2*x0) + (56*x6) + (401408*x3)), tmp55, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (301084 + (2*x0) + (56*x6) + (401408*x3)), tmp60, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (301085 + (2*x0) + (56*x6) + (401408*x3)), tmp65, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 29, tl.int64)
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
    tl.store(out_ptr0 + (x7 + (100352*x3)), tmp145, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cv/ccvdtreqhj426itpele2qa4ab2yqgbd7xgreesid3n7jyymhlhba.py
# Source Nodes: [out_61], Original ATen: [aten._native_batch_norm_legit_functional]
# out_61 => add_228, add_229, add_230, mul_295, mul_296, mul_297, mul_298, mul_299, rsqrt_42, squeeze_127, var_mean_42
triton_red_fused__native_batch_norm_legit_functional_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_33', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/ss/csswq7zzb673jj2ssqmthpl5uyqzwjrdghw6c2pyvzswm564tize.py
# Source Nodes: [out_61, out_62, shortcut_10, shortcut_11], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_61 => add_228, add_231, mul_294, mul_300, rsqrt_42, sub_42, var_mean_42
# out_62 => add_237
# shortcut_10 => add_233, add_236, mul_301, mul_307, rsqrt_43, sub_43, var_mean_43
# shortcut_11 => relu_40
triton_poi_fused__native_batch_norm_legit_functional_add_relu_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_34', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/wo/cwo4w63ctzbzcb7sugftnjoclmmh4i5qzkviba675kse523kmtqn.py
# Source Nodes: [out_65], Original ATen: [aten._native_batch_norm_legit_functional]
# out_65 => add_239, add_240, add_241, mul_309, mul_310, mul_311, mul_312, mul_313, rsqrt_44, squeeze_133, var_mean_44
triton_red_fused__native_batch_norm_legit_functional_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_red_fused__native_batch_norm_legit_functional_35', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 512
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
        tmp0 = tl.load(in_ptr0 + (r1 + (196*x0) + (100352*r2)), rmask & xmask, eviction_policy='evict_first', other=0.0)
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


# kernel path: /tmp/torchinductor_youkaichao/av/cavqdb7lvmtibwhya2xzhwffwwhcnwbcscw3b2s67f2qfo3h6a7j.py
# Source Nodes: [out_65, out_66], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# out_65 => add_239, add_242, mul_308, mul_314, rsqrt_44, sub_44, var_mean_44
# out_66 => relu_41
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 512
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x3), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/m6/cm6xtdc67rq7yv7sspknd7mverer3lvp4sbadmr2yxvosuznkvy4.py
# Source Nodes: [sp_106, sp_107, sp_108], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
# sp_106 => add_244, add_247, mul_315, mul_321, rsqrt_45, sub_45, var_mean_45
# sp_107 => relu_42
# sp_108 => add_248
triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 128
    x2 = (xindex // 25088)
    x4 = xindex % 25088
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (25088 + x4 + (100352*x2)), None)
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
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = tmp14 <= tmp17
    tl.store(out_ptr0 + (x4 + (100352*x2)), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
    tl.store(out_ptr2 + (x3), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mv/cmvjzakoud6xsapwyzv3tteous7e7yjwto7gawmu5fzv4w7dnigr.py
# Source Nodes: [sp_110, sp_111, sp_112], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
# sp_110 => add_250, add_253, mul_322, mul_328, rsqrt_46, sub_46, var_mean_46
# sp_111 => relu_43
# sp_112 => add_254
triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_38 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 128
    x2 = (xindex // 25088)
    x4 = xindex % 25088
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (50176 + x4 + (100352*x2)), None)
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
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = tmp14 <= tmp17
    tl.store(out_ptr0 + (x4 + (100352*x2)), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
    tl.store(out_ptr2 + (x3), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mo/cmojsvuwomgoiafkeonnqnueycj7igc6ief6mvjkclmvz45mm7em.py
# Source Nodes: [cat_23], Original ATen: [aten.cat]
# cat_23 => cat_8
triton_poi_fused_cat_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 25088
    x1 = (xindex // 25088)
    tmp0 = tl.load(in_ptr0 + (75264 + x0 + (100352*x1)), None)
    tl.store(out_ptr0 + (x0 + (100352*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mj/cmjf5eveynqdjdnfpo2lyqhpzv7uggreexpadqhooxylaf5vuxul.py
# Source Nodes: [out_69, out_70, shortcut_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_69 => add_261, add_264, mul_336, mul_342, rsqrt_48, sub_48, var_mean_48
# out_70 => add_265
# shortcut_12 => relu_45
triton_poi_fused__native_batch_norm_legit_functional_add_relu_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_40', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/el/cel3vledvoztyc2u4m72utkptsl4uh2j6mt7wc3bf2bviltafd4a.py
# Source Nodes: [out_105, out_106], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# out_105 => add_379, add_382, mul_483, mul_489, rsqrt_69, sub_69, var_mean_69
# out_106 => relu_66
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x3), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2w/c2w5olhgpsqvjoveibu62fllv3lkjvxqznvskaujpvesmo5ekmnt.py
# Source Nodes: [sp_171], Original ATen: [aten._native_batch_norm_legit_functional]
# sp_171 => add_384, add_385, add_386, mul_491, mul_492, mul_493, mul_494, mul_495, rsqrt_70, squeeze_211, var_mean_70
triton_per_fused__native_batch_norm_legit_functional_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_42', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel):
    xnumel = 256
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (12544*r2)), rmask & xmask, other=0.0)
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
    tl.store(out_ptr2 + (x0), tmp21, xmask)
    tl.store(out_ptr4 + (x0), tmp27, xmask)
    tl.store(out_ptr6 + (x0), tmp33, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7x/c7xhgvt3lwu7c7urhxj3oe2z5sclb7lb4lqqdhlqkcxfmc5nfykh.py
# Source Nodes: [sp_171, sp_172], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# sp_171 => add_384, add_387, mul_490, mul_496, rsqrt_70, sub_70, var_mean_70
# sp_172 => relu_67
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 256
    x2 = (xindex // 12544)
    x4 = xindex % 12544
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x4 + (50176*x2)), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bq/cbqmicrampyd3unr4hw2hvq6pfxxliufaepvwmvtzl2d2xvu2umy.py
# Source Nodes: [getattr_l__mod___layer4___0___pool], Original ATen: [aten.avg_pool2d]
# getattr_l__mod___layer4___0___pool => avg_pool2d_3
triton_poi_fused_avg_pool2d_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 7) % 7
    x0 = xindex % 7
    x3 = (xindex // 12544)
    x6 = (xindex // 7) % 1792
    x7 = xindex % 12544
    tmp0 = (-1) + (2*x1)
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x0)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + (150513 + (2*x0) + (28*x6) + (200704*x3)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + (150514 + (2*x0) + (28*x6) + (200704*x3)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + (150515 + (2*x0) + (28*x6) + (200704*x3)), tmp27, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + (150527 + (2*x0) + (28*x6) + (200704*x3)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (150528 + (2*x0) + (28*x6) + (200704*x3)), tmp41, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (150529 + (2*x0) + (28*x6) + (200704*x3)), tmp46, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (150541 + (2*x0) + (28*x6) + (200704*x3)), tmp55, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (150542 + (2*x0) + (28*x6) + (200704*x3)), tmp60, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (150543 + (2*x0) + (28*x6) + (200704*x3)), tmp65, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 15, tl.int64)
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
    tl.store(out_ptr0 + (x7 + (50176*x3)), tmp145, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cv/ccvdfiyb5wkx5vplnnees37jatdk5bfw3mgha55epxhqh2imi5tx.py
# Source Nodes: [out_109], Original ATen: [aten._native_batch_norm_legit_functional]
# out_109 => add_399, add_400, add_401, mul_512, mul_513, mul_514, mul_515, mul_516, rsqrt_73, squeeze_220, var_mean_73
triton_per_fused__native_batch_norm_legit_functional_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_45', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
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


# kernel path: /tmp/torchinductor_youkaichao/hm/chmw5x47333yjgmtyfyf2zjz7y6m23fcrdv5dmir3lbfue5xe7ix.py
# Source Nodes: [out_109, out_110, shortcut_17, shortcut_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_109 => add_399, add_402, mul_511, mul_517, rsqrt_73, sub_73, var_mean_73
# out_110 => add_408
# shortcut_17 => add_404, add_407, mul_518, mul_524, rsqrt_74, sub_74, var_mean_74
# shortcut_18 => relu_70
triton_poi_fused__native_batch_norm_legit_functional_add_relu_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_46', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/uq/cuqnubnr6x26ujlykh5acljdc5tc4scej7kmyo6ac6s2w6ww4voj.py
# Source Nodes: [out_113], Original ATen: [aten._native_batch_norm_legit_functional]
# out_113 => add_410, add_411, add_412, mul_526, mul_527, mul_528, mul_529, mul_530, rsqrt_75, squeeze_226, var_mean_75
triton_per_fused__native_batch_norm_legit_functional_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_47', 'mutated_arg_names': ['in_ptr1', 'in_ptr2', 'out_ptr4', 'out_ptr6']}
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr4, out_ptr6, xnumel, rnumel):
    xnumel = 1024
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
    tmp0 = tl.load(in_ptr0 + (r1 + (49*x0) + (50176*r2)), rmask & xmask, other=0.0)
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
    tl.store(out_ptr2 + (x0), tmp21, xmask)
    tl.store(out_ptr4 + (x0), tmp27, xmask)
    tl.store(out_ptr6 + (x0), tmp33, xmask)
    tl.store(out_ptr0 + (x0), tmp10, xmask)
    tl.store(out_ptr1 + (x0), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r2/cr22ffc4f7q4ym4kimcwraou5qs7z5xxq4of2xyaksvpikdxxqt3.py
# Source Nodes: [out_113, out_114], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
# out_113 => add_410, add_413, mul_525, mul_531, rsqrt_75, sub_75, var_mean_75
# out_114 => relu_71
triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_48 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 1024
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
    tmp15 = 0.0
    tmp16 = tmp14 <= tmp15
    tl.store(out_ptr0 + (x3), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ws/cwschjzh33fi2z75fclob4qcb626k557bbqxzutgjrtgkt243mc3.py
# Source Nodes: [sp_184, sp_185, sp_186], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
# sp_184 => add_415, add_418, mul_532, mul_538, rsqrt_76, sub_76, var_mean_76
# sp_185 => relu_72
# sp_186 => add_419
triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 256
    x2 = (xindex // 12544)
    x4 = xindex % 12544
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (12544 + x4 + (50176*x2)), None)
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
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = tmp14 <= tmp17
    tl.store(out_ptr0 + (x4 + (50176*x2)), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
    tl.store(out_ptr2 + (x3), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/iv/civ5chdog6vgoh63a4lcuammpdjiqvqcj4oxaw2auaf3iyo3h3bm.py
# Source Nodes: [sp_188, sp_189, sp_190], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
# sp_188 => add_421, add_424, mul_539, mul_545, rsqrt_77, sub_77, var_mean_77
# sp_189 => relu_73
# sp_190 => add_425
triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_50 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*i1', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 256
    x2 = (xindex // 12544)
    x4 = xindex % 12544
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp12 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (25088 + x4 + (50176*x2)), None)
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
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = tmp14 <= tmp17
    tl.store(out_ptr0 + (x4 + (50176*x2)), tmp14, None)
    tl.store(out_ptr1 + (x3), tmp16, None)
    tl.store(out_ptr2 + (x3), tmp18, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yc/cycg232p5gdryumaqlplfkz6ytppeov22m73x3cwjbb6elpjmnrw.py
# Source Nodes: [cat_17], Original ATen: [aten.cat]
# cat_17 => cat_14
triton_poi_fused_cat_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 12544
    x1 = (xindex // 12544)
    tmp0 = tl.load(in_ptr0 + (37632 + x0 + (50176*x1)), None)
    tl.store(out_ptr0 + (x0 + (50176*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/m5/cm5i7ipao6pkyod27d2rrebt6kurebfkcupc7wdpgucc6r7kijq6.py
# Source Nodes: [out_117, out_118, shortcut_19], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
# out_117 => add_432, add_435, mul_553, mul_559, rsqrt_79, sub_79, var_mean_79
# out_118 => add_436
# shortcut_19 => relu_75
triton_poi_fused__native_batch_norm_legit_functional_add_relu_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_functional_add_relu_52', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/zx/czxvrfuxutieabmpste4nrxc5skv54awcrgeaadtrklfgrqgbccl.py
# Source Nodes: [out_125, out_126, x_11, x_8, x_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.mean, aten.relu, aten.threshold_backward, aten.view]
# out_125 => add_460, add_463, mul_588, mul_594, rsqrt_84, sub_84, var_mean_84
# out_126 => add_464
# x_11 => view
# x_8 => relu_80
# x_9 => mean
triton_per_fused__native_batch_norm_legit_functional_add_mean_relu_threshold_backward_view_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_functional_add_mean_relu_threshold_backward_view_53', 'mutated_arg_names': ['in_out_ptr0']}
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


# kernel path: /tmp/torchinductor_youkaichao/5c/c5ceus3hrnaienmcvdq7hpeshy3i3eqlajbec6ujdnazi6u3pab3.py
# Source Nodes: [x_1], Original ATen: [aten.add]
# x_1 => add
triton_poi_fused_add_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_54', 'mutated_arg_names': ['in_ptr0', 'out_ptr1']},
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_6, (128, ), (1, ))
    assert_size_stride(primals_7, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_8, (32, ), (1, ))
    assert_size_stride(primals_9, (32, ), (1, ))
    assert_size_stride(primals_10, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_11, (32, ), (1, ))
    assert_size_stride(primals_12, (32, ), (1, ))
    assert_size_stride(primals_13, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_14, (32, ), (1, ))
    assert_size_stride(primals_15, (32, ), (1, ))
    assert_size_stride(primals_16, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_18, (256, ), (1, ))
    assert_size_stride(primals_19, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_20, (256, ), (1, ))
    assert_size_stride(primals_21, (256, ), (1, ))
    assert_size_stride(primals_22, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_23, (128, ), (1, ))
    assert_size_stride(primals_24, (128, ), (1, ))
    assert_size_stride(primals_25, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_26, (32, ), (1, ))
    assert_size_stride(primals_27, (32, ), (1, ))
    assert_size_stride(primals_28, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_29, (32, ), (1, ))
    assert_size_stride(primals_30, (32, ), (1, ))
    assert_size_stride(primals_31, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_32, (32, ), (1, ))
    assert_size_stride(primals_33, (32, ), (1, ))
    assert_size_stride(primals_34, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_36, (256, ), (1, ))
    assert_size_stride(primals_37, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_38, (128, ), (1, ))
    assert_size_stride(primals_39, (128, ), (1, ))
    assert_size_stride(primals_40, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_41, (32, ), (1, ))
    assert_size_stride(primals_42, (32, ), (1, ))
    assert_size_stride(primals_43, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_44, (32, ), (1, ))
    assert_size_stride(primals_45, (32, ), (1, ))
    assert_size_stride(primals_46, (32, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_47, (32, ), (1, ))
    assert_size_stride(primals_48, (32, ), (1, ))
    assert_size_stride(primals_49, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_50, (256, ), (1, ))
    assert_size_stride(primals_51, (256, ), (1, ))
    assert_size_stride(primals_52, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_53, (256, ), (1, ))
    assert_size_stride(primals_54, (256, ), (1, ))
    assert_size_stride(primals_55, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_56, (64, ), (1, ))
    assert_size_stride(primals_57, (64, ), (1, ))
    assert_size_stride(primals_58, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_59, (64, ), (1, ))
    assert_size_stride(primals_60, (64, ), (1, ))
    assert_size_stride(primals_61, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_62, (64, ), (1, ))
    assert_size_stride(primals_63, (64, ), (1, ))
    assert_size_stride(primals_64, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_65, (512, ), (1, ))
    assert_size_stride(primals_66, (512, ), (1, ))
    assert_size_stride(primals_67, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_68, (512, ), (1, ))
    assert_size_stride(primals_69, (512, ), (1, ))
    assert_size_stride(primals_70, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_71, (256, ), (1, ))
    assert_size_stride(primals_72, (256, ), (1, ))
    assert_size_stride(primals_73, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_74, (64, ), (1, ))
    assert_size_stride(primals_75, (64, ), (1, ))
    assert_size_stride(primals_76, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_77, (64, ), (1, ))
    assert_size_stride(primals_78, (64, ), (1, ))
    assert_size_stride(primals_79, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_80, (64, ), (1, ))
    assert_size_stride(primals_81, (64, ), (1, ))
    assert_size_stride(primals_82, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_83, (512, ), (1, ))
    assert_size_stride(primals_84, (512, ), (1, ))
    assert_size_stride(primals_85, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_86, (256, ), (1, ))
    assert_size_stride(primals_87, (256, ), (1, ))
    assert_size_stride(primals_88, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_89, (64, ), (1, ))
    assert_size_stride(primals_90, (64, ), (1, ))
    assert_size_stride(primals_91, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_92, (64, ), (1, ))
    assert_size_stride(primals_93, (64, ), (1, ))
    assert_size_stride(primals_94, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_95, (64, ), (1, ))
    assert_size_stride(primals_96, (64, ), (1, ))
    assert_size_stride(primals_97, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_98, (512, ), (1, ))
    assert_size_stride(primals_99, (512, ), (1, ))
    assert_size_stride(primals_100, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_101, (256, ), (1, ))
    assert_size_stride(primals_102, (256, ), (1, ))
    assert_size_stride(primals_103, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_104, (64, ), (1, ))
    assert_size_stride(primals_105, (64, ), (1, ))
    assert_size_stride(primals_106, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_107, (64, ), (1, ))
    assert_size_stride(primals_108, (64, ), (1, ))
    assert_size_stride(primals_109, (64, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_110, (64, ), (1, ))
    assert_size_stride(primals_111, (64, ), (1, ))
    assert_size_stride(primals_112, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_113, (512, ), (1, ))
    assert_size_stride(primals_114, (512, ), (1, ))
    assert_size_stride(primals_115, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_116, (512, ), (1, ))
    assert_size_stride(primals_117, (512, ), (1, ))
    assert_size_stride(primals_118, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_119, (128, ), (1, ))
    assert_size_stride(primals_120, (128, ), (1, ))
    assert_size_stride(primals_121, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_122, (128, ), (1, ))
    assert_size_stride(primals_123, (128, ), (1, ))
    assert_size_stride(primals_124, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_125, (128, ), (1, ))
    assert_size_stride(primals_126, (128, ), (1, ))
    assert_size_stride(primals_127, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_128, (1024, ), (1, ))
    assert_size_stride(primals_129, (1024, ), (1, ))
    assert_size_stride(primals_130, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_131, (1024, ), (1, ))
    assert_size_stride(primals_132, (1024, ), (1, ))
    assert_size_stride(primals_133, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_134, (512, ), (1, ))
    assert_size_stride(primals_135, (512, ), (1, ))
    assert_size_stride(primals_136, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_137, (128, ), (1, ))
    assert_size_stride(primals_138, (128, ), (1, ))
    assert_size_stride(primals_139, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_140, (128, ), (1, ))
    assert_size_stride(primals_141, (128, ), (1, ))
    assert_size_stride(primals_142, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_143, (128, ), (1, ))
    assert_size_stride(primals_144, (128, ), (1, ))
    assert_size_stride(primals_145, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_146, (1024, ), (1, ))
    assert_size_stride(primals_147, (1024, ), (1, ))
    assert_size_stride(primals_148, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_149, (512, ), (1, ))
    assert_size_stride(primals_150, (512, ), (1, ))
    assert_size_stride(primals_151, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_152, (128, ), (1, ))
    assert_size_stride(primals_153, (128, ), (1, ))
    assert_size_stride(primals_154, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_155, (128, ), (1, ))
    assert_size_stride(primals_156, (128, ), (1, ))
    assert_size_stride(primals_157, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_158, (128, ), (1, ))
    assert_size_stride(primals_159, (128, ), (1, ))
    assert_size_stride(primals_160, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_161, (1024, ), (1, ))
    assert_size_stride(primals_162, (1024, ), (1, ))
    assert_size_stride(primals_163, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_164, (512, ), (1, ))
    assert_size_stride(primals_165, (512, ), (1, ))
    assert_size_stride(primals_166, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_167, (128, ), (1, ))
    assert_size_stride(primals_168, (128, ), (1, ))
    assert_size_stride(primals_169, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_170, (128, ), (1, ))
    assert_size_stride(primals_171, (128, ), (1, ))
    assert_size_stride(primals_172, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_173, (128, ), (1, ))
    assert_size_stride(primals_174, (128, ), (1, ))
    assert_size_stride(primals_175, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_176, (1024, ), (1, ))
    assert_size_stride(primals_177, (1024, ), (1, ))
    assert_size_stride(primals_178, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_179, (512, ), (1, ))
    assert_size_stride(primals_180, (512, ), (1, ))
    assert_size_stride(primals_181, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_182, (128, ), (1, ))
    assert_size_stride(primals_183, (128, ), (1, ))
    assert_size_stride(primals_184, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_185, (128, ), (1, ))
    assert_size_stride(primals_186, (128, ), (1, ))
    assert_size_stride(primals_187, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_188, (128, ), (1, ))
    assert_size_stride(primals_189, (128, ), (1, ))
    assert_size_stride(primals_190, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_191, (1024, ), (1, ))
    assert_size_stride(primals_192, (1024, ), (1, ))
    assert_size_stride(primals_193, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_194, (512, ), (1, ))
    assert_size_stride(primals_195, (512, ), (1, ))
    assert_size_stride(primals_196, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_197, (128, ), (1, ))
    assert_size_stride(primals_198, (128, ), (1, ))
    assert_size_stride(primals_199, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_200, (128, ), (1, ))
    assert_size_stride(primals_201, (128, ), (1, ))
    assert_size_stride(primals_202, (128, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_203, (128, ), (1, ))
    assert_size_stride(primals_204, (128, ), (1, ))
    assert_size_stride(primals_205, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_206, (1024, ), (1, ))
    assert_size_stride(primals_207, (1024, ), (1, ))
    assert_size_stride(primals_208, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_209, (1024, ), (1, ))
    assert_size_stride(primals_210, (1024, ), (1, ))
    assert_size_stride(primals_211, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_212, (256, ), (1, ))
    assert_size_stride(primals_213, (256, ), (1, ))
    assert_size_stride(primals_214, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_215, (256, ), (1, ))
    assert_size_stride(primals_216, (256, ), (1, ))
    assert_size_stride(primals_217, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_218, (256, ), (1, ))
    assert_size_stride(primals_219, (256, ), (1, ))
    assert_size_stride(primals_220, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_221, (2048, ), (1, ))
    assert_size_stride(primals_222, (2048, ), (1, ))
    assert_size_stride(primals_223, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_224, (2048, ), (1, ))
    assert_size_stride(primals_225, (2048, ), (1, ))
    assert_size_stride(primals_226, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_227, (1024, ), (1, ))
    assert_size_stride(primals_228, (1024, ), (1, ))
    assert_size_stride(primals_229, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_230, (256, ), (1, ))
    assert_size_stride(primals_231, (256, ), (1, ))
    assert_size_stride(primals_232, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_233, (256, ), (1, ))
    assert_size_stride(primals_234, (256, ), (1, ))
    assert_size_stride(primals_235, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_236, (256, ), (1, ))
    assert_size_stride(primals_237, (256, ), (1, ))
    assert_size_stride(primals_238, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_239, (2048, ), (1, ))
    assert_size_stride(primals_240, (2048, ), (1, ))
    assert_size_stride(primals_241, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_242, (1024, ), (1, ))
    assert_size_stride(primals_243, (1024, ), (1, ))
    assert_size_stride(primals_244, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_245, (256, ), (1, ))
    assert_size_stride(primals_246, (256, ), (1, ))
    assert_size_stride(primals_247, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_248, (256, ), (1, ))
    assert_size_stride(primals_249, (256, ), (1, ))
    assert_size_stride(primals_250, (256, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_251, (256, ), (1, ))
    assert_size_stride(primals_252, (256, ), (1, ))
    assert_size_stride(primals_253, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_254, (2048, ), (1, ))
    assert_size_stride(primals_255, (2048, ), (1, ))
    assert_size_stride(primals_256, (1000, 2048), (2048, 1))
    assert_size_stride(primals_257, (1000, ), (1, ))
    assert_size_stride(primals_258, (64, ), (1, ))
    assert_size_stride(primals_259, (64, ), (1, ))
    assert_size_stride(primals_260, (), ())
    assert_size_stride(primals_261, (128, ), (1, ))
    assert_size_stride(primals_262, (128, ), (1, ))
    assert_size_stride(primals_263, (), ())
    assert_size_stride(primals_264, (32, ), (1, ))
    assert_size_stride(primals_265, (32, ), (1, ))
    assert_size_stride(primals_266, (), ())
    assert_size_stride(primals_267, (32, ), (1, ))
    assert_size_stride(primals_268, (32, ), (1, ))
    assert_size_stride(primals_269, (), ())
    assert_size_stride(primals_270, (32, ), (1, ))
    assert_size_stride(primals_271, (32, ), (1, ))
    assert_size_stride(primals_272, (), ())
    assert_size_stride(primals_273, (256, ), (1, ))
    assert_size_stride(primals_274, (256, ), (1, ))
    assert_size_stride(primals_275, (), ())
    assert_size_stride(primals_276, (256, ), (1, ))
    assert_size_stride(primals_277, (256, ), (1, ))
    assert_size_stride(primals_278, (), ())
    assert_size_stride(primals_279, (128, ), (1, ))
    assert_size_stride(primals_280, (128, ), (1, ))
    assert_size_stride(primals_281, (), ())
    assert_size_stride(primals_282, (32, ), (1, ))
    assert_size_stride(primals_283, (32, ), (1, ))
    assert_size_stride(primals_284, (), ())
    assert_size_stride(primals_285, (32, ), (1, ))
    assert_size_stride(primals_286, (32, ), (1, ))
    assert_size_stride(primals_287, (), ())
    assert_size_stride(primals_288, (32, ), (1, ))
    assert_size_stride(primals_289, (32, ), (1, ))
    assert_size_stride(primals_290, (), ())
    assert_size_stride(primals_291, (256, ), (1, ))
    assert_size_stride(primals_292, (256, ), (1, ))
    assert_size_stride(primals_293, (), ())
    assert_size_stride(primals_294, (128, ), (1, ))
    assert_size_stride(primals_295, (128, ), (1, ))
    assert_size_stride(primals_296, (), ())
    assert_size_stride(primals_297, (32, ), (1, ))
    assert_size_stride(primals_298, (32, ), (1, ))
    assert_size_stride(primals_299, (), ())
    assert_size_stride(primals_300, (32, ), (1, ))
    assert_size_stride(primals_301, (32, ), (1, ))
    assert_size_stride(primals_302, (), ())
    assert_size_stride(primals_303, (32, ), (1, ))
    assert_size_stride(primals_304, (32, ), (1, ))
    assert_size_stride(primals_305, (), ())
    assert_size_stride(primals_306, (256, ), (1, ))
    assert_size_stride(primals_307, (256, ), (1, ))
    assert_size_stride(primals_308, (), ())
    assert_size_stride(primals_309, (256, ), (1, ))
    assert_size_stride(primals_310, (256, ), (1, ))
    assert_size_stride(primals_311, (), ())
    assert_size_stride(primals_312, (64, ), (1, ))
    assert_size_stride(primals_313, (64, ), (1, ))
    assert_size_stride(primals_314, (), ())
    assert_size_stride(primals_315, (64, ), (1, ))
    assert_size_stride(primals_316, (64, ), (1, ))
    assert_size_stride(primals_317, (), ())
    assert_size_stride(primals_318, (64, ), (1, ))
    assert_size_stride(primals_319, (64, ), (1, ))
    assert_size_stride(primals_320, (), ())
    assert_size_stride(primals_321, (512, ), (1, ))
    assert_size_stride(primals_322, (512, ), (1, ))
    assert_size_stride(primals_323, (), ())
    assert_size_stride(primals_324, (512, ), (1, ))
    assert_size_stride(primals_325, (512, ), (1, ))
    assert_size_stride(primals_326, (), ())
    assert_size_stride(primals_327, (256, ), (1, ))
    assert_size_stride(primals_328, (256, ), (1, ))
    assert_size_stride(primals_329, (), ())
    assert_size_stride(primals_330, (64, ), (1, ))
    assert_size_stride(primals_331, (64, ), (1, ))
    assert_size_stride(primals_332, (), ())
    assert_size_stride(primals_333, (64, ), (1, ))
    assert_size_stride(primals_334, (64, ), (1, ))
    assert_size_stride(primals_335, (), ())
    assert_size_stride(primals_336, (64, ), (1, ))
    assert_size_stride(primals_337, (64, ), (1, ))
    assert_size_stride(primals_338, (), ())
    assert_size_stride(primals_339, (512, ), (1, ))
    assert_size_stride(primals_340, (512, ), (1, ))
    assert_size_stride(primals_341, (), ())
    assert_size_stride(primals_342, (256, ), (1, ))
    assert_size_stride(primals_343, (256, ), (1, ))
    assert_size_stride(primals_344, (), ())
    assert_size_stride(primals_345, (64, ), (1, ))
    assert_size_stride(primals_346, (64, ), (1, ))
    assert_size_stride(primals_347, (), ())
    assert_size_stride(primals_348, (64, ), (1, ))
    assert_size_stride(primals_349, (64, ), (1, ))
    assert_size_stride(primals_350, (), ())
    assert_size_stride(primals_351, (64, ), (1, ))
    assert_size_stride(primals_352, (64, ), (1, ))
    assert_size_stride(primals_353, (), ())
    assert_size_stride(primals_354, (512, ), (1, ))
    assert_size_stride(primals_355, (512, ), (1, ))
    assert_size_stride(primals_356, (), ())
    assert_size_stride(primals_357, (256, ), (1, ))
    assert_size_stride(primals_358, (256, ), (1, ))
    assert_size_stride(primals_359, (), ())
    assert_size_stride(primals_360, (64, ), (1, ))
    assert_size_stride(primals_361, (64, ), (1, ))
    assert_size_stride(primals_362, (), ())
    assert_size_stride(primals_363, (64, ), (1, ))
    assert_size_stride(primals_364, (64, ), (1, ))
    assert_size_stride(primals_365, (), ())
    assert_size_stride(primals_366, (64, ), (1, ))
    assert_size_stride(primals_367, (64, ), (1, ))
    assert_size_stride(primals_368, (), ())
    assert_size_stride(primals_369, (512, ), (1, ))
    assert_size_stride(primals_370, (512, ), (1, ))
    assert_size_stride(primals_371, (), ())
    assert_size_stride(primals_372, (512, ), (1, ))
    assert_size_stride(primals_373, (512, ), (1, ))
    assert_size_stride(primals_374, (), ())
    assert_size_stride(primals_375, (128, ), (1, ))
    assert_size_stride(primals_376, (128, ), (1, ))
    assert_size_stride(primals_377, (), ())
    assert_size_stride(primals_378, (128, ), (1, ))
    assert_size_stride(primals_379, (128, ), (1, ))
    assert_size_stride(primals_380, (), ())
    assert_size_stride(primals_381, (128, ), (1, ))
    assert_size_stride(primals_382, (128, ), (1, ))
    assert_size_stride(primals_383, (), ())
    assert_size_stride(primals_384, (1024, ), (1, ))
    assert_size_stride(primals_385, (1024, ), (1, ))
    assert_size_stride(primals_386, (), ())
    assert_size_stride(primals_387, (1024, ), (1, ))
    assert_size_stride(primals_388, (1024, ), (1, ))
    assert_size_stride(primals_389, (), ())
    assert_size_stride(primals_390, (512, ), (1, ))
    assert_size_stride(primals_391, (512, ), (1, ))
    assert_size_stride(primals_392, (), ())
    assert_size_stride(primals_393, (128, ), (1, ))
    assert_size_stride(primals_394, (128, ), (1, ))
    assert_size_stride(primals_395, (), ())
    assert_size_stride(primals_396, (128, ), (1, ))
    assert_size_stride(primals_397, (128, ), (1, ))
    assert_size_stride(primals_398, (), ())
    assert_size_stride(primals_399, (128, ), (1, ))
    assert_size_stride(primals_400, (128, ), (1, ))
    assert_size_stride(primals_401, (), ())
    assert_size_stride(primals_402, (1024, ), (1, ))
    assert_size_stride(primals_403, (1024, ), (1, ))
    assert_size_stride(primals_404, (), ())
    assert_size_stride(primals_405, (512, ), (1, ))
    assert_size_stride(primals_406, (512, ), (1, ))
    assert_size_stride(primals_407, (), ())
    assert_size_stride(primals_408, (128, ), (1, ))
    assert_size_stride(primals_409, (128, ), (1, ))
    assert_size_stride(primals_410, (), ())
    assert_size_stride(primals_411, (128, ), (1, ))
    assert_size_stride(primals_412, (128, ), (1, ))
    assert_size_stride(primals_413, (), ())
    assert_size_stride(primals_414, (128, ), (1, ))
    assert_size_stride(primals_415, (128, ), (1, ))
    assert_size_stride(primals_416, (), ())
    assert_size_stride(primals_417, (1024, ), (1, ))
    assert_size_stride(primals_418, (1024, ), (1, ))
    assert_size_stride(primals_419, (), ())
    assert_size_stride(primals_420, (512, ), (1, ))
    assert_size_stride(primals_421, (512, ), (1, ))
    assert_size_stride(primals_422, (), ())
    assert_size_stride(primals_423, (128, ), (1, ))
    assert_size_stride(primals_424, (128, ), (1, ))
    assert_size_stride(primals_425, (), ())
    assert_size_stride(primals_426, (128, ), (1, ))
    assert_size_stride(primals_427, (128, ), (1, ))
    assert_size_stride(primals_428, (), ())
    assert_size_stride(primals_429, (128, ), (1, ))
    assert_size_stride(primals_430, (128, ), (1, ))
    assert_size_stride(primals_431, (), ())
    assert_size_stride(primals_432, (1024, ), (1, ))
    assert_size_stride(primals_433, (1024, ), (1, ))
    assert_size_stride(primals_434, (), ())
    assert_size_stride(primals_435, (512, ), (1, ))
    assert_size_stride(primals_436, (512, ), (1, ))
    assert_size_stride(primals_437, (), ())
    assert_size_stride(primals_438, (128, ), (1, ))
    assert_size_stride(primals_439, (128, ), (1, ))
    assert_size_stride(primals_440, (), ())
    assert_size_stride(primals_441, (128, ), (1, ))
    assert_size_stride(primals_442, (128, ), (1, ))
    assert_size_stride(primals_443, (), ())
    assert_size_stride(primals_444, (128, ), (1, ))
    assert_size_stride(primals_445, (128, ), (1, ))
    assert_size_stride(primals_446, (), ())
    assert_size_stride(primals_447, (1024, ), (1, ))
    assert_size_stride(primals_448, (1024, ), (1, ))
    assert_size_stride(primals_449, (), ())
    assert_size_stride(primals_450, (512, ), (1, ))
    assert_size_stride(primals_451, (512, ), (1, ))
    assert_size_stride(primals_452, (), ())
    assert_size_stride(primals_453, (128, ), (1, ))
    assert_size_stride(primals_454, (128, ), (1, ))
    assert_size_stride(primals_455, (), ())
    assert_size_stride(primals_456, (128, ), (1, ))
    assert_size_stride(primals_457, (128, ), (1, ))
    assert_size_stride(primals_458, (), ())
    assert_size_stride(primals_459, (128, ), (1, ))
    assert_size_stride(primals_460, (128, ), (1, ))
    assert_size_stride(primals_461, (), ())
    assert_size_stride(primals_462, (1024, ), (1, ))
    assert_size_stride(primals_463, (1024, ), (1, ))
    assert_size_stride(primals_464, (), ())
    assert_size_stride(primals_465, (1024, ), (1, ))
    assert_size_stride(primals_466, (1024, ), (1, ))
    assert_size_stride(primals_467, (), ())
    assert_size_stride(primals_468, (256, ), (1, ))
    assert_size_stride(primals_469, (256, ), (1, ))
    assert_size_stride(primals_470, (), ())
    assert_size_stride(primals_471, (256, ), (1, ))
    assert_size_stride(primals_472, (256, ), (1, ))
    assert_size_stride(primals_473, (), ())
    assert_size_stride(primals_474, (256, ), (1, ))
    assert_size_stride(primals_475, (256, ), (1, ))
    assert_size_stride(primals_476, (), ())
    assert_size_stride(primals_477, (2048, ), (1, ))
    assert_size_stride(primals_478, (2048, ), (1, ))
    assert_size_stride(primals_479, (), ())
    assert_size_stride(primals_480, (2048, ), (1, ))
    assert_size_stride(primals_481, (2048, ), (1, ))
    assert_size_stride(primals_482, (), ())
    assert_size_stride(primals_483, (1024, ), (1, ))
    assert_size_stride(primals_484, (1024, ), (1, ))
    assert_size_stride(primals_485, (), ())
    assert_size_stride(primals_486, (256, ), (1, ))
    assert_size_stride(primals_487, (256, ), (1, ))
    assert_size_stride(primals_488, (), ())
    assert_size_stride(primals_489, (256, ), (1, ))
    assert_size_stride(primals_490, (256, ), (1, ))
    assert_size_stride(primals_491, (), ())
    assert_size_stride(primals_492, (256, ), (1, ))
    assert_size_stride(primals_493, (256, ), (1, ))
    assert_size_stride(primals_494, (), ())
    assert_size_stride(primals_495, (2048, ), (1, ))
    assert_size_stride(primals_496, (2048, ), (1, ))
    assert_size_stride(primals_497, (), ())
    assert_size_stride(primals_498, (1024, ), (1, ))
    assert_size_stride(primals_499, (1024, ), (1, ))
    assert_size_stride(primals_500, (), ())
    assert_size_stride(primals_501, (256, ), (1, ))
    assert_size_stride(primals_502, (256, ), (1, ))
    assert_size_stride(primals_503, (), ())
    assert_size_stride(primals_504, (256, ), (1, ))
    assert_size_stride(primals_505, (256, ), (1, ))
    assert_size_stride(primals_506, (), ())
    assert_size_stride(primals_507, (256, ), (1, ))
    assert_size_stride(primals_508, (256, ), (1, ))
    assert_size_stride(primals_509, (), ())
    assert_size_stride(primals_510, (2048, ), (1, ))
    assert_size_stride(primals_511, (2048, ), (1, ))
    assert_size_stride(primals_512, (), ())
    assert_size_stride(primals_513, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_513, primals_1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
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
        triton_per_fused__native_batch_norm_legit_functional_1.run(buf1, buf2, buf3, primals_258, primals_259, buf4, buf5, buf7, primals_258, primals_259, 64, 13, grid=grid(64), stream=stream0)
        del buf1
        del buf2
        del buf3
        del primals_258
        del primals_259
        buf8 = empty((8, 64, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_relu_2.run(buf0, buf4, buf5, primals_2, primals_3, buf8, 6422528, grid=grid(6422528), stream=stream0)
        del primals_3
        buf9 = empty((8, 64, 56, 56), device='cuda', dtype=torch.float32)
        buf10 = empty((8, 64, 56, 56), device='cuda', dtype=torch.int64)
        # Source Nodes: [shortcut], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_3.run(buf8, buf9, buf10, 1605632, grid=grid(1605632), stream=stream0)
        # Source Nodes: [out], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf9, primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf12 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        buf13 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        buf14 = empty_strided((1, 128, 1, 1, 4), (512, 1, 512, 512, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf11, buf12, buf13, buf14, 512, 6272, grid=grid(512), stream=stream0)
        buf15 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf16 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf18 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf12, buf13, buf14, primals_261, primals_262, buf15, buf16, buf18, primals_261, primals_262, 128, 4, grid=grid(128), stream=stream0)
        del primals_261
        del primals_262
        buf19 = empty((8, 128, 56, 56), device='cuda', dtype=torch.float32)
        buf674 = empty((8, 128, 56, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_6.run(buf11, buf15, buf16, primals_5, primals_6, buf19, buf674, 3211264, grid=grid(3211264), stream=stream0)
        del primals_6
        # Source Nodes: [sp_1], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(reinterpret_tensor(buf19, (8, 32, 56, 56), (401408, 3136, 56, 1), 0), primals_7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf20, (8, 32, 56, 56), (100352, 3136, 56, 1))
        buf21 = reinterpret_tensor(buf16, (1, 32, 1, 1, 4), (128, 1, 128, 128, 32), 0); del buf16  # reuse
        buf22 = empty_strided((1, 32, 1, 1, 4), (128, 1, 128, 128, 32), device='cuda', dtype=torch.float32)
        buf23 = empty_strided((1, 32, 1, 1, 4), (128, 1, 128, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_2], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf20, buf21, buf22, buf23, 128, 6272, grid=grid(128), stream=stream0)
        buf24 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf25 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf27 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_2], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_8.run(buf21, buf22, buf23, primals_264, primals_265, buf24, buf25, buf27, primals_264, primals_265, 32, 4, grid=grid(32), stream=stream0)
        del primals_264
        del primals_265
        buf48 = empty((8, 128, 56, 56), device='cuda', dtype=torch.float32)
        buf28 = reinterpret_tensor(buf48, (8, 32, 56, 56), (401408, 3136, 56, 1), 0)  # alias
        buf673 = empty((8, 32, 56, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_2, sp_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_9.run(buf20, buf24, buf25, primals_8, primals_9, buf28, buf673, 802816, grid=grid(802816), stream=stream0)
        del primals_9
        # Source Nodes: [sp_5], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(reinterpret_tensor(buf19, (8, 32, 56, 56), (401408, 3136, 56, 1), 100352), primals_10, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf29, (8, 32, 56, 56), (100352, 3136, 56, 1))
        buf30 = buf23; del buf23  # reuse
        buf31 = buf22; del buf22  # reuse
        buf32 = buf21; del buf21  # reuse
        # Source Nodes: [sp_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf29, buf30, buf31, buf32, 128, 6272, grid=grid(128), stream=stream0)
        buf33 = buf25; del buf25  # reuse
        buf34 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf36 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_6], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_8.run(buf30, buf31, buf32, primals_267, primals_268, buf33, buf34, buf36, primals_267, primals_268, 32, 4, grid=grid(32), stream=stream0)
        del primals_267
        del primals_268
        buf37 = reinterpret_tensor(buf48, (8, 32, 56, 56), (401408, 3136, 56, 1), 100352)  # alias
        buf672 = empty((8, 32, 56, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_6, sp_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_9.run(buf29, buf33, buf34, primals_11, primals_12, buf37, buf672, 802816, grid=grid(802816), stream=stream0)
        del primals_12
        # Source Nodes: [sp_9], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(reinterpret_tensor(buf19, (8, 32, 56, 56), (401408, 3136, 56, 1), 200704), primals_13, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf38, (8, 32, 56, 56), (100352, 3136, 56, 1))
        buf39 = buf32; del buf32  # reuse
        buf40 = buf31; del buf31  # reuse
        buf41 = buf30; del buf30  # reuse
        # Source Nodes: [sp_10], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf38, buf39, buf40, buf41, 128, 6272, grid=grid(128), stream=stream0)
        buf42 = buf34; del buf34  # reuse
        buf43 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf45 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_10], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_8.run(buf39, buf40, buf41, primals_270, primals_271, buf42, buf43, buf45, primals_270, primals_271, 32, 4, grid=grid(32), stream=stream0)
        del primals_270
        del primals_271
        buf46 = reinterpret_tensor(buf48, (8, 32, 56, 56), (401408, 3136, 56, 1), 200704)  # alias
        buf671 = empty((8, 32, 56, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_10, sp_11], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_9.run(buf38, buf42, buf43, primals_14, primals_15, buf46, buf671, 802816, grid=grid(802816), stream=stream0)
        del primals_15
        buf47 = reinterpret_tensor(buf48, (8, 32, 56, 56), (401408, 3136, 56, 1), 301056)  # alias
        # Source Nodes: [getattr_l__mod___layer1___0___pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_10.run(buf19, buf47, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [out_4], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (8, 256, 56, 56), (802816, 3136, 56, 1))
        buf50 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf51 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf53 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf49, primals_273, primals_274, buf50, buf51, buf53, primals_273, primals_274, 256, 25088, grid=grid(256), stream=stream0)
        del primals_273
        del primals_274
        # Source Nodes: [getattr_l__mod___layer1___0___downsample_0], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf9, primals_19, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 256, 56, 56), (802816, 3136, 56, 1))
        buf55 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf56 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf58 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_1], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf54, primals_276, primals_277, buf55, buf56, buf58, primals_276, primals_277, 256, 25088, grid=grid(256), stream=stream0)
        del primals_276
        del primals_277
        buf59 = empty((8, 256, 56, 56), device='cuda', dtype=torch.float32)
        buf60 = buf59; del buf59  # reuse
        # Source Nodes: [out_5, out_6, shortcut_1, shortcut_2], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_12.run(buf60, buf49, buf50, buf51, primals_17, primals_18, buf54, buf55, buf56, primals_20, primals_21, 6422528, grid=grid(6422528), stream=stream0)
        del primals_18
        del primals_21
        # Source Nodes: [out_8], Original ATen: [aten.convolution]
        buf61 = extern_kernels.convolution(buf60, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf62 = buf14; del buf14  # reuse
        buf63 = buf13; del buf13  # reuse
        buf64 = buf12; del buf12  # reuse
        # Source Nodes: [out_9], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf61, buf62, buf63, buf64, 512, 6272, grid=grid(512), stream=stream0)
        buf65 = reinterpret_tensor(buf41, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf41  # reuse
        buf66 = reinterpret_tensor(buf40, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf40  # reuse
        buf68 = reinterpret_tensor(buf39, (128, ), (1, ), 0); del buf39  # reuse
        # Source Nodes: [out_9], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf62, buf63, buf64, primals_279, primals_280, buf65, buf66, buf68, primals_279, primals_280, 128, 4, grid=grid(128), stream=stream0)
        del primals_279
        del primals_280
        buf69 = empty((8, 128, 56, 56), device='cuda', dtype=torch.float32)
        buf670 = empty((8, 128, 56, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_10, out_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_6.run(buf61, buf65, buf66, primals_23, primals_24, buf69, buf670, 3211264, grid=grid(3211264), stream=stream0)
        del primals_24
        # Source Nodes: [sp_14], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(reinterpret_tensor(buf69, (8, 32, 56, 56), (401408, 3136, 56, 1), 0), primals_25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf70, (8, 32, 56, 56), (100352, 3136, 56, 1))
        buf71 = reinterpret_tensor(buf66, (1, 32, 1, 1, 4), (128, 1, 128, 128, 32), 0); del buf66  # reuse
        buf72 = empty_strided((1, 32, 1, 1, 4), (128, 1, 128, 128, 32), device='cuda', dtype=torch.float32)
        buf73 = empty_strided((1, 32, 1, 1, 4), (128, 1, 128, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_15], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf70, buf71, buf72, buf73, 128, 6272, grid=grid(128), stream=stream0)
        buf74 = buf43; del buf43  # reuse
        buf75 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf77 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_15], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_8.run(buf71, buf72, buf73, primals_282, primals_283, buf74, buf75, buf77, primals_282, primals_283, 32, 4, grid=grid(32), stream=stream0)
        del primals_282
        del primals_283
        buf100 = empty((8, 128, 56, 56), device='cuda', dtype=torch.float32)
        buf78 = reinterpret_tensor(buf100, (8, 32, 56, 56), (401408, 3136, 56, 1), 0)  # alias
        buf79 = empty((8, 32, 56, 56), device='cuda', dtype=torch.float32)
        buf669 = empty((8, 32, 56, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_15, sp_16, sp_17], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_13.run(buf70, buf74, buf75, primals_26, primals_27, buf69, buf78, buf79, buf669, 802816, grid=grid(802816), stream=stream0)
        del primals_27
        # Source Nodes: [sp_18], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf80, (8, 32, 56, 56), (100352, 3136, 56, 1))
        buf81 = buf73; del buf73  # reuse
        buf82 = buf72; del buf72  # reuse
        buf83 = buf71; del buf71  # reuse
        # Source Nodes: [sp_19], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf80, buf81, buf82, buf83, 128, 6272, grid=grid(128), stream=stream0)
        buf84 = buf75; del buf75  # reuse
        buf85 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf87 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_19], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_8.run(buf81, buf82, buf83, primals_285, primals_286, buf84, buf85, buf87, primals_285, primals_286, 32, 4, grid=grid(32), stream=stream0)
        del primals_285
        del primals_286
        buf88 = reinterpret_tensor(buf100, (8, 32, 56, 56), (401408, 3136, 56, 1), 100352)  # alias
        buf89 = empty((8, 32, 56, 56), device='cuda', dtype=torch.float32)
        buf668 = empty((8, 32, 56, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_19, sp_20, sp_21], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_14.run(buf80, buf84, buf85, primals_29, primals_30, buf69, buf88, buf89, buf668, 802816, grid=grid(802816), stream=stream0)
        del primals_30
        # Source Nodes: [sp_22], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, primals_31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf90, (8, 32, 56, 56), (100352, 3136, 56, 1))
        buf91 = buf83; del buf83  # reuse
        buf92 = buf82; del buf82  # reuse
        buf93 = buf81; del buf81  # reuse
        # Source Nodes: [sp_23], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf90, buf91, buf92, buf93, 128, 6272, grid=grid(128), stream=stream0)
        buf94 = buf85; del buf85  # reuse
        buf95 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf97 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_23], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_8.run(buf91, buf92, buf93, primals_288, primals_289, buf94, buf95, buf97, primals_288, primals_289, 32, 4, grid=grid(32), stream=stream0)
        del primals_288
        del primals_289
        buf98 = reinterpret_tensor(buf100, (8, 32, 56, 56), (401408, 3136, 56, 1), 200704)  # alias
        buf667 = empty((8, 32, 56, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_23, sp_24], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_9.run(buf90, buf94, buf95, primals_32, primals_33, buf98, buf667, 802816, grid=grid(802816), stream=stream0)
        del primals_33
        buf99 = reinterpret_tensor(buf100, (8, 32, 56, 56), (401408, 3136, 56, 1), 301056)  # alias
        # Source Nodes: [cat_30], Original ATen: [aten.cat]
        triton_poi_fused_cat_15.run(buf69, buf99, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [out_12], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (8, 256, 56, 56), (802816, 3136, 56, 1))
        buf102 = buf56; del buf56  # reuse
        buf103 = buf51; del buf51  # reuse
        buf105 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_13], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf101, primals_291, primals_292, buf102, buf103, buf105, primals_291, primals_292, 256, 25088, grid=grid(256), stream=stream0)
        del primals_291
        del primals_292
        buf106 = empty((8, 256, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_13, out_14, shortcut_3], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_16.run(buf101, buf102, buf103, primals_35, primals_36, buf60, buf106, 6422528, grid=grid(6422528), stream=stream0)
        del primals_36
        # Source Nodes: [out_16], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_37, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (8, 128, 56, 56), (401408, 3136, 56, 1))
        buf108 = buf64; del buf64  # reuse
        buf109 = buf63; del buf63  # reuse
        buf110 = buf62; del buf62  # reuse
        # Source Nodes: [out_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_4.run(buf107, buf108, buf109, buf110, 512, 6272, grid=grid(512), stream=stream0)
        buf111 = reinterpret_tensor(buf93, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf93  # reuse
        buf112 = reinterpret_tensor(buf92, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf92  # reuse
        buf114 = reinterpret_tensor(buf91, (128, ), (1, ), 0); del buf91  # reuse
        # Source Nodes: [out_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_5.run(buf108, buf109, buf110, primals_294, primals_295, buf111, buf112, buf114, primals_294, primals_295, 128, 4, grid=grid(128), stream=stream0)
        del primals_294
        del primals_295
        buf115 = empty((8, 128, 56, 56), device='cuda', dtype=torch.float32)
        buf666 = empty((8, 128, 56, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_17, out_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_6.run(buf107, buf111, buf112, primals_38, primals_39, buf115, buf666, 3211264, grid=grid(3211264), stream=stream0)
        del primals_39
        # Source Nodes: [sp_27], Original ATen: [aten.convolution]
        buf116 = extern_kernels.convolution(reinterpret_tensor(buf115, (8, 32, 56, 56), (401408, 3136, 56, 1), 0), primals_40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf116, (8, 32, 56, 56), (100352, 3136, 56, 1))
        buf117 = reinterpret_tensor(buf112, (1, 32, 1, 1, 4), (128, 1, 128, 128, 32), 0); del buf112  # reuse
        buf118 = empty_strided((1, 32, 1, 1, 4), (128, 1, 128, 128, 32), device='cuda', dtype=torch.float32)
        buf119 = empty_strided((1, 32, 1, 1, 4), (128, 1, 128, 128, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_28], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf116, buf117, buf118, buf119, 128, 6272, grid=grid(128), stream=stream0)
        buf120 = buf95; del buf95  # reuse
        buf121 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf123 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_28], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_8.run(buf117, buf118, buf119, primals_297, primals_298, buf120, buf121, buf123, primals_297, primals_298, 32, 4, grid=grid(32), stream=stream0)
        del primals_297
        del primals_298
        buf146 = empty((8, 128, 56, 56), device='cuda', dtype=torch.float32)
        buf124 = reinterpret_tensor(buf146, (8, 32, 56, 56), (401408, 3136, 56, 1), 0)  # alias
        buf125 = empty((8, 32, 56, 56), device='cuda', dtype=torch.float32)
        buf665 = empty((8, 32, 56, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_28, sp_29, sp_30], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_13.run(buf116, buf120, buf121, primals_41, primals_42, buf115, buf124, buf125, buf665, 802816, grid=grid(802816), stream=stream0)
        del primals_42
        # Source Nodes: [sp_31], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf125, primals_43, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf126, (8, 32, 56, 56), (100352, 3136, 56, 1))
        buf127 = buf119; del buf119  # reuse
        buf128 = buf118; del buf118  # reuse
        buf129 = buf117; del buf117  # reuse
        # Source Nodes: [sp_32], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf126, buf127, buf128, buf129, 128, 6272, grid=grid(128), stream=stream0)
        buf130 = buf121; del buf121  # reuse
        buf131 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf133 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_32], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_8.run(buf127, buf128, buf129, primals_300, primals_301, buf130, buf131, buf133, primals_300, primals_301, 32, 4, grid=grid(32), stream=stream0)
        del primals_300
        del primals_301
        buf134 = reinterpret_tensor(buf146, (8, 32, 56, 56), (401408, 3136, 56, 1), 100352)  # alias
        buf135 = empty((8, 32, 56, 56), device='cuda', dtype=torch.float32)
        buf664 = empty((8, 32, 56, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_32, sp_33, sp_34], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_14.run(buf126, buf130, buf131, primals_44, primals_45, buf115, buf134, buf135, buf664, 802816, grid=grid(802816), stream=stream0)
        del primals_45
        # Source Nodes: [sp_35], Original ATen: [aten.convolution]
        buf136 = extern_kernels.convolution(buf135, primals_46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf136, (8, 32, 56, 56), (100352, 3136, 56, 1))
        buf137 = buf129; del buf129  # reuse
        buf138 = buf128; del buf128  # reuse
        buf139 = buf127; del buf127  # reuse
        # Source Nodes: [sp_36], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_7.run(buf136, buf137, buf138, buf139, 128, 6272, grid=grid(128), stream=stream0)
        buf140 = buf131; del buf131  # reuse
        buf141 = empty_strided((1, 32, 1, 1), (32, 1, 32, 32), device='cuda', dtype=torch.float32)
        buf143 = empty((32, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_36], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_8.run(buf137, buf138, buf139, primals_303, primals_304, buf140, buf141, buf143, primals_303, primals_304, 32, 4, grid=grid(32), stream=stream0)
        del primals_303
        del primals_304
        buf144 = reinterpret_tensor(buf146, (8, 32, 56, 56), (401408, 3136, 56, 1), 200704)  # alias
        buf663 = empty((8, 32, 56, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_36, sp_37], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_9.run(buf136, buf140, buf141, primals_47, primals_48, buf144, buf663, 802816, grid=grid(802816), stream=stream0)
        del buf141
        del primals_48
        buf145 = reinterpret_tensor(buf146, (8, 32, 56, 56), (401408, 3136, 56, 1), 301056)  # alias
        # Source Nodes: [cat_29], Original ATen: [aten.cat]
        triton_poi_fused_cat_15.run(buf115, buf145, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [out_20], Original ATen: [aten.convolution]
        buf147 = extern_kernels.convolution(buf146, primals_49, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (8, 256, 56, 56), (802816, 3136, 56, 1))
        buf148 = buf103; del buf103  # reuse
        buf149 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf151 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_21], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf147, primals_306, primals_307, buf148, buf149, buf151, primals_306, primals_307, 256, 25088, grid=grid(256), stream=stream0)
        del primals_306
        del primals_307
        buf152 = empty((8, 256, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_21, out_22, shortcut_4], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_16.run(buf147, buf148, buf149, primals_50, primals_51, buf106, buf152, 6422528, grid=grid(6422528), stream=stream0)
        del primals_51
        # Source Nodes: [out_24], Original ATen: [aten.convolution]
        buf153 = extern_kernels.convolution(buf152, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (8, 256, 56, 56), (802816, 3136, 56, 1))
        buf154 = buf149; del buf149  # reuse
        buf155 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf157 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_25], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_11.run(buf153, primals_309, primals_310, buf154, buf155, buf157, primals_309, primals_310, 256, 25088, grid=grid(256), stream=stream0)
        del primals_309
        del primals_310
        buf158 = empty((8, 256, 56, 56), device='cuda', dtype=torch.float32)
        buf662 = empty((8, 256, 56, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_25, out_26], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_17.run(buf153, buf154, buf155, primals_53, primals_54, buf158, buf662, 6422528, grid=grid(6422528), stream=stream0)
        del primals_54
        # Source Nodes: [sp_40], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(reinterpret_tensor(buf158, (8, 64, 56, 56), (802816, 3136, 56, 1), 0), primals_55, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf159, (8, 64, 28, 28), (50176, 784, 28, 1))
        buf160 = buf5; del buf5  # reuse
        buf161 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf163 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf159, primals_312, primals_313, buf160, buf161, buf163, primals_312, primals_313, 64, 6272, grid=grid(64), stream=stream0)
        del primals_312
        del primals_313
        buf178 = empty((8, 256, 28, 28), device='cuda', dtype=torch.float32)
        buf164 = reinterpret_tensor(buf178, (8, 64, 28, 28), (200704, 784, 28, 1), 0)  # alias
        buf661 = empty((8, 64, 28, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_41, sp_42], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_19.run(buf159, buf160, buf161, primals_56, primals_57, buf164, buf661, 401408, grid=grid(401408), stream=stream0)
        del primals_57
        # Source Nodes: [sp_44], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(reinterpret_tensor(buf158, (8, 64, 56, 56), (802816, 3136, 56, 1), 200704), primals_58, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf165, (8, 64, 28, 28), (50176, 784, 28, 1))
        buf166 = buf161; del buf161  # reuse
        buf167 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf169 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_45], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf165, primals_315, primals_316, buf166, buf167, buf169, primals_315, primals_316, 64, 6272, grid=grid(64), stream=stream0)
        del primals_315
        del primals_316
        buf170 = reinterpret_tensor(buf178, (8, 64, 28, 28), (200704, 784, 28, 1), 50176)  # alias
        buf660 = empty((8, 64, 28, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_45, sp_46], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_19.run(buf165, buf166, buf167, primals_59, primals_60, buf170, buf660, 401408, grid=grid(401408), stream=stream0)
        del primals_60
        # Source Nodes: [sp_48], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(reinterpret_tensor(buf158, (8, 64, 56, 56), (802816, 3136, 56, 1), 401408), primals_61, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf171, (8, 64, 28, 28), (50176, 784, 28, 1))
        buf172 = buf167; del buf167  # reuse
        buf173 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf175 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_49], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf171, primals_318, primals_319, buf172, buf173, buf175, primals_318, primals_319, 64, 6272, grid=grid(64), stream=stream0)
        del primals_318
        del primals_319
        buf176 = reinterpret_tensor(buf178, (8, 64, 28, 28), (200704, 784, 28, 1), 100352)  # alias
        buf659 = empty((8, 64, 28, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_49, sp_50], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_19.run(buf171, buf172, buf173, primals_62, primals_63, buf176, buf659, 401408, grid=grid(401408), stream=stream0)
        del primals_63
        buf177 = reinterpret_tensor(buf178, (8, 64, 28, 28), (200704, 784, 28, 1), 150528)  # alias
        # Source Nodes: [getattr_l__mod___layer2___0___pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_20.run(buf158, buf177, 401408, grid=grid(401408), stream=stream0)
        # Source Nodes: [out_28], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf178, primals_64, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (8, 512, 28, 28), (401408, 784, 28, 1))
        buf180 = reinterpret_tensor(buf110, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf110  # reuse
        buf181 = reinterpret_tensor(buf109, (1, 512, 1, 1), (512, 1, 512, 512), 0); del buf109  # reuse
        buf183 = reinterpret_tensor(buf108, (512, ), (1, ), 0); del buf108  # reuse
        # Source Nodes: [out_29], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf179, primals_321, primals_322, buf180, buf181, buf183, primals_321, primals_322, 512, 6272, grid=grid(512), stream=stream0)
        del primals_321
        del primals_322
        # Source Nodes: [getattr_l__mod___layer2___0___downsample_0], Original ATen: [aten.convolution]
        buf184 = extern_kernels.convolution(buf152, primals_67, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf184, (8, 512, 28, 28), (401408, 784, 28, 1))
        buf185 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf186 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf188 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_5], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf184, primals_324, primals_325, buf185, buf186, buf188, primals_324, primals_325, 512, 6272, grid=grid(512), stream=stream0)
        del primals_324
        del primals_325
        buf189 = empty((8, 512, 28, 28), device='cuda', dtype=torch.float32)
        buf190 = buf189; del buf189  # reuse
        # Source Nodes: [out_29, out_30, shortcut_5, shortcut_6], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_22.run(buf190, buf179, buf180, buf181, primals_65, primals_66, buf184, buf185, buf186, primals_68, primals_69, 3211264, grid=grid(3211264), stream=stream0)
        del primals_66
        del primals_69
        # Source Nodes: [out_32], Original ATen: [aten.convolution]
        buf191 = extern_kernels.convolution(buf190, primals_70, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (8, 256, 28, 28), (200704, 784, 28, 1))
        buf192 = buf155; del buf155  # reuse
        buf193 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf195 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_33], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_23.run(buf191, primals_327, primals_328, buf192, buf193, buf195, primals_327, primals_328, 256, 6272, grid=grid(256), stream=stream0)
        del primals_327
        del primals_328
        buf196 = empty((8, 256, 28, 28), device='cuda', dtype=torch.float32)
        buf658 = empty((8, 256, 28, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_33, out_34], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_24.run(buf191, buf192, buf193, primals_71, primals_72, buf196, buf658, 1605632, grid=grid(1605632), stream=stream0)
        del primals_72
        # Source Nodes: [sp_53], Original ATen: [aten.convolution]
        buf197 = extern_kernels.convolution(reinterpret_tensor(buf196, (8, 64, 28, 28), (200704, 784, 28, 1), 0), primals_73, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf197, (8, 64, 28, 28), (50176, 784, 28, 1))
        buf198 = buf173; del buf173  # reuse
        buf199 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf201 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_54], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf197, primals_330, primals_331, buf198, buf199, buf201, primals_330, primals_331, 64, 6272, grid=grid(64), stream=stream0)
        del primals_330
        del primals_331
        buf218 = empty((8, 256, 28, 28), device='cuda', dtype=torch.float32)
        buf202 = reinterpret_tensor(buf218, (8, 64, 28, 28), (200704, 784, 28, 1), 0)  # alias
        buf203 = empty((8, 64, 28, 28), device='cuda', dtype=torch.float32)
        buf657 = empty((8, 64, 28, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_54, sp_55, sp_56], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_25.run(buf197, buf198, buf199, primals_74, primals_75, buf196, buf202, buf203, buf657, 401408, grid=grid(401408), stream=stream0)
        del primals_75
        # Source Nodes: [sp_57], Original ATen: [aten.convolution]
        buf204 = extern_kernels.convolution(buf203, primals_76, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf204, (8, 64, 28, 28), (50176, 784, 28, 1))
        buf205 = buf199; del buf199  # reuse
        buf206 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf208 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_58], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf204, primals_333, primals_334, buf205, buf206, buf208, primals_333, primals_334, 64, 6272, grid=grid(64), stream=stream0)
        del primals_333
        del primals_334
        buf209 = reinterpret_tensor(buf218, (8, 64, 28, 28), (200704, 784, 28, 1), 50176)  # alias
        buf210 = empty((8, 64, 28, 28), device='cuda', dtype=torch.float32)
        buf656 = empty((8, 64, 28, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_58, sp_59, sp_60], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_26.run(buf204, buf205, buf206, primals_77, primals_78, buf196, buf209, buf210, buf656, 401408, grid=grid(401408), stream=stream0)
        del primals_78
        # Source Nodes: [sp_61], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf210, primals_79, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf211, (8, 64, 28, 28), (50176, 784, 28, 1))
        buf212 = buf206; del buf206  # reuse
        buf213 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf215 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_62], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf211, primals_336, primals_337, buf212, buf213, buf215, primals_336, primals_337, 64, 6272, grid=grid(64), stream=stream0)
        del primals_336
        del primals_337
        buf216 = reinterpret_tensor(buf218, (8, 64, 28, 28), (200704, 784, 28, 1), 100352)  # alias
        buf655 = empty((8, 64, 28, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_62, sp_63], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_19.run(buf211, buf212, buf213, primals_80, primals_81, buf216, buf655, 401408, grid=grid(401408), stream=stream0)
        del primals_81
        buf217 = reinterpret_tensor(buf218, (8, 64, 28, 28), (200704, 784, 28, 1), 150528)  # alias
        # Source Nodes: [cat_27], Original ATen: [aten.cat]
        triton_poi_fused_cat_27.run(buf196, buf217, 401408, grid=grid(401408), stream=stream0)
        # Source Nodes: [out_36], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf218, primals_82, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (8, 512, 28, 28), (401408, 784, 28, 1))
        buf220 = buf186; del buf186  # reuse
        buf221 = buf181; del buf181  # reuse
        buf223 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_37], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf219, primals_339, primals_340, buf220, buf221, buf223, primals_339, primals_340, 512, 6272, grid=grid(512), stream=stream0)
        del primals_339
        del primals_340
        buf224 = empty((8, 512, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_37, out_38, shortcut_7], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_28.run(buf219, buf220, buf221, primals_83, primals_84, buf190, buf224, 3211264, grid=grid(3211264), stream=stream0)
        del primals_84
        # Source Nodes: [out_40], Original ATen: [aten.convolution]
        buf225 = extern_kernels.convolution(buf224, primals_85, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf225, (8, 256, 28, 28), (200704, 784, 28, 1))
        buf226 = buf193; del buf193  # reuse
        buf227 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf229 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_41], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_23.run(buf225, primals_342, primals_343, buf226, buf227, buf229, primals_342, primals_343, 256, 6272, grid=grid(256), stream=stream0)
        del primals_342
        del primals_343
        buf230 = empty((8, 256, 28, 28), device='cuda', dtype=torch.float32)
        buf654 = empty((8, 256, 28, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_41, out_42], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_24.run(buf225, buf226, buf227, primals_86, primals_87, buf230, buf654, 1605632, grid=grid(1605632), stream=stream0)
        del primals_87
        # Source Nodes: [sp_66], Original ATen: [aten.convolution]
        buf231 = extern_kernels.convolution(reinterpret_tensor(buf230, (8, 64, 28, 28), (200704, 784, 28, 1), 0), primals_88, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf231, (8, 64, 28, 28), (50176, 784, 28, 1))
        buf232 = buf213; del buf213  # reuse
        buf233 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf235 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_67], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf231, primals_345, primals_346, buf232, buf233, buf235, primals_345, primals_346, 64, 6272, grid=grid(64), stream=stream0)
        del primals_345
        del primals_346
        buf252 = empty((8, 256, 28, 28), device='cuda', dtype=torch.float32)
        buf236 = reinterpret_tensor(buf252, (8, 64, 28, 28), (200704, 784, 28, 1), 0)  # alias
        buf237 = empty((8, 64, 28, 28), device='cuda', dtype=torch.float32)
        buf653 = empty((8, 64, 28, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_67, sp_68, sp_69], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_25.run(buf231, buf232, buf233, primals_89, primals_90, buf230, buf236, buf237, buf653, 401408, grid=grid(401408), stream=stream0)
        del primals_90
        # Source Nodes: [sp_70], Original ATen: [aten.convolution]
        buf238 = extern_kernels.convolution(buf237, primals_91, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf238, (8, 64, 28, 28), (50176, 784, 28, 1))
        buf239 = buf233; del buf233  # reuse
        buf240 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf242 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_71], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf238, primals_348, primals_349, buf239, buf240, buf242, primals_348, primals_349, 64, 6272, grid=grid(64), stream=stream0)
        del primals_348
        del primals_349
        buf243 = reinterpret_tensor(buf252, (8, 64, 28, 28), (200704, 784, 28, 1), 50176)  # alias
        buf244 = empty((8, 64, 28, 28), device='cuda', dtype=torch.float32)
        buf652 = empty((8, 64, 28, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_71, sp_72, sp_73], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_26.run(buf238, buf239, buf240, primals_92, primals_93, buf230, buf243, buf244, buf652, 401408, grid=grid(401408), stream=stream0)
        del primals_93
        # Source Nodes: [sp_74], Original ATen: [aten.convolution]
        buf245 = extern_kernels.convolution(buf244, primals_94, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf245, (8, 64, 28, 28), (50176, 784, 28, 1))
        buf246 = buf240; del buf240  # reuse
        buf247 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf249 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_75], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf245, primals_351, primals_352, buf246, buf247, buf249, primals_351, primals_352, 64, 6272, grid=grid(64), stream=stream0)
        del primals_351
        del primals_352
        buf250 = reinterpret_tensor(buf252, (8, 64, 28, 28), (200704, 784, 28, 1), 100352)  # alias
        buf651 = empty((8, 64, 28, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_75, sp_76], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_19.run(buf245, buf246, buf247, primals_95, primals_96, buf250, buf651, 401408, grid=grid(401408), stream=stream0)
        del primals_96
        buf251 = reinterpret_tensor(buf252, (8, 64, 28, 28), (200704, 784, 28, 1), 150528)  # alias
        # Source Nodes: [cat_26], Original ATen: [aten.cat]
        triton_poi_fused_cat_27.run(buf230, buf251, 401408, grid=grid(401408), stream=stream0)
        # Source Nodes: [out_44], Original ATen: [aten.convolution]
        buf253 = extern_kernels.convolution(buf252, primals_97, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf253, (8, 512, 28, 28), (401408, 784, 28, 1))
        buf254 = buf221; del buf221  # reuse
        buf255 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf257 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_45], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf253, primals_354, primals_355, buf254, buf255, buf257, primals_354, primals_355, 512, 6272, grid=grid(512), stream=stream0)
        del primals_354
        del primals_355
        buf258 = empty((8, 512, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_45, out_46, shortcut_8], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_28.run(buf253, buf254, buf255, primals_98, primals_99, buf224, buf258, 3211264, grid=grid(3211264), stream=stream0)
        del primals_99
        # Source Nodes: [out_48], Original ATen: [aten.convolution]
        buf259 = extern_kernels.convolution(buf258, primals_100, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf259, (8, 256, 28, 28), (200704, 784, 28, 1))
        buf260 = buf227; del buf227  # reuse
        buf261 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf263 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_49], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_23.run(buf259, primals_357, primals_358, buf260, buf261, buf263, primals_357, primals_358, 256, 6272, grid=grid(256), stream=stream0)
        del primals_357
        del primals_358
        buf264 = empty((8, 256, 28, 28), device='cuda', dtype=torch.float32)
        buf650 = empty((8, 256, 28, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_49, out_50], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_24.run(buf259, buf260, buf261, primals_101, primals_102, buf264, buf650, 1605632, grid=grid(1605632), stream=stream0)
        del primals_102
        # Source Nodes: [sp_79], Original ATen: [aten.convolution]
        buf265 = extern_kernels.convolution(reinterpret_tensor(buf264, (8, 64, 28, 28), (200704, 784, 28, 1), 0), primals_103, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf265, (8, 64, 28, 28), (50176, 784, 28, 1))
        buf266 = buf247; del buf247  # reuse
        buf267 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf269 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_80], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf265, primals_360, primals_361, buf266, buf267, buf269, primals_360, primals_361, 64, 6272, grid=grid(64), stream=stream0)
        del primals_360
        del primals_361
        buf286 = empty((8, 256, 28, 28), device='cuda', dtype=torch.float32)
        buf270 = reinterpret_tensor(buf286, (8, 64, 28, 28), (200704, 784, 28, 1), 0)  # alias
        buf271 = empty((8, 64, 28, 28), device='cuda', dtype=torch.float32)
        buf649 = empty((8, 64, 28, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_80, sp_81, sp_82], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_25.run(buf265, buf266, buf267, primals_104, primals_105, buf264, buf270, buf271, buf649, 401408, grid=grid(401408), stream=stream0)
        del primals_105
        # Source Nodes: [sp_83], Original ATen: [aten.convolution]
        buf272 = extern_kernels.convolution(buf271, primals_106, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf272, (8, 64, 28, 28), (50176, 784, 28, 1))
        buf273 = buf267; del buf267  # reuse
        buf274 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf276 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_84], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf272, primals_363, primals_364, buf273, buf274, buf276, primals_363, primals_364, 64, 6272, grid=grid(64), stream=stream0)
        del primals_363
        del primals_364
        buf277 = reinterpret_tensor(buf286, (8, 64, 28, 28), (200704, 784, 28, 1), 50176)  # alias
        buf278 = empty((8, 64, 28, 28), device='cuda', dtype=torch.float32)
        buf648 = empty((8, 64, 28, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_84, sp_85, sp_86], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_26.run(buf272, buf273, buf274, primals_107, primals_108, buf264, buf277, buf278, buf648, 401408, grid=grid(401408), stream=stream0)
        del primals_108
        # Source Nodes: [sp_87], Original ATen: [aten.convolution]
        buf279 = extern_kernels.convolution(buf278, primals_109, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf279, (8, 64, 28, 28), (50176, 784, 28, 1))
        buf280 = buf274; del buf274  # reuse
        buf281 = empty_strided((1, 64, 1, 1), (64, 1, 64, 64), device='cuda', dtype=torch.float32)
        buf283 = empty((64, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_88], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_18.run(buf279, primals_366, primals_367, buf280, buf281, buf283, primals_366, primals_367, 64, 6272, grid=grid(64), stream=stream0)
        del primals_366
        del primals_367
        buf284 = reinterpret_tensor(buf286, (8, 64, 28, 28), (200704, 784, 28, 1), 100352)  # alias
        buf647 = empty((8, 64, 28, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_88, sp_89], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_19.run(buf279, buf280, buf281, primals_110, primals_111, buf284, buf647, 401408, grid=grid(401408), stream=stream0)
        del buf281
        del primals_111
        buf285 = reinterpret_tensor(buf286, (8, 64, 28, 28), (200704, 784, 28, 1), 150528)  # alias
        # Source Nodes: [cat_25], Original ATen: [aten.cat]
        triton_poi_fused_cat_27.run(buf264, buf285, 401408, grid=grid(401408), stream=stream0)
        # Source Nodes: [out_52], Original ATen: [aten.convolution]
        buf287 = extern_kernels.convolution(buf286, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf287, (8, 512, 28, 28), (401408, 784, 28, 1))
        buf288 = buf255; del buf255  # reuse
        buf289 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf291 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_53], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf287, primals_369, primals_370, buf288, buf289, buf291, primals_369, primals_370, 512, 6272, grid=grid(512), stream=stream0)
        del primals_369
        del primals_370
        buf292 = empty((8, 512, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_53, out_54, shortcut_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_28.run(buf287, buf288, buf289, primals_113, primals_114, buf258, buf292, 3211264, grid=grid(3211264), stream=stream0)
        del primals_114
        # Source Nodes: [out_56], Original ATen: [aten.convolution]
        buf293 = extern_kernels.convolution(buf292, primals_115, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf293, (8, 512, 28, 28), (401408, 784, 28, 1))
        buf294 = buf289; del buf289  # reuse
        buf295 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf297 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_57], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_21.run(buf293, primals_372, primals_373, buf294, buf295, buf297, primals_372, primals_373, 512, 6272, grid=grid(512), stream=stream0)
        del primals_372
        del primals_373
        buf298 = empty((8, 512, 28, 28), device='cuda', dtype=torch.float32)
        buf646 = empty((8, 512, 28, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_57, out_58], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_29.run(buf293, buf294, buf295, primals_116, primals_117, buf298, buf646, 3211264, grid=grid(3211264), stream=stream0)
        del primals_117
        # Source Nodes: [sp_92], Original ATen: [aten.convolution]
        buf299 = extern_kernels.convolution(reinterpret_tensor(buf298, (8, 128, 28, 28), (401408, 784, 28, 1), 0), primals_118, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf299, (8, 128, 14, 14), (25088, 196, 14, 1))
        buf300 = reinterpret_tensor(buf139, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf139  # reuse
        buf301 = reinterpret_tensor(buf138, (1, 128, 1, 1), (128, 1, 128, 128), 0); del buf138  # reuse
        buf303 = reinterpret_tensor(buf137, (128, ), (1, ), 0); del buf137  # reuse
        # Source Nodes: [sp_93], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf299, primals_375, primals_376, buf300, buf301, buf303, primals_375, primals_376, 128, 1568, grid=grid(128), stream=stream0)
        del primals_375
        del primals_376
        buf318 = empty((8, 512, 14, 14), device='cuda', dtype=torch.float32)
        buf304 = reinterpret_tensor(buf318, (8, 128, 14, 14), (100352, 196, 14, 1), 0)  # alias
        buf645 = empty((8, 128, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_93, sp_94], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_31.run(buf299, buf300, buf301, primals_119, primals_120, buf304, buf645, 200704, grid=grid(200704), stream=stream0)
        del primals_120
        # Source Nodes: [sp_96], Original ATen: [aten.convolution]
        buf305 = extern_kernels.convolution(reinterpret_tensor(buf298, (8, 128, 28, 28), (401408, 784, 28, 1), 100352), primals_121, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf305, (8, 128, 14, 14), (25088, 196, 14, 1))
        buf306 = buf301; del buf301  # reuse
        buf307 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf309 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_97], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf305, primals_378, primals_379, buf306, buf307, buf309, primals_378, primals_379, 128, 1568, grid=grid(128), stream=stream0)
        del primals_378
        del primals_379
        buf310 = reinterpret_tensor(buf318, (8, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        buf644 = empty((8, 128, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_97, sp_98], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_31.run(buf305, buf306, buf307, primals_122, primals_123, buf310, buf644, 200704, grid=grid(200704), stream=stream0)
        del primals_123
        # Source Nodes: [sp_100], Original ATen: [aten.convolution]
        buf311 = extern_kernels.convolution(reinterpret_tensor(buf298, (8, 128, 28, 28), (401408, 784, 28, 1), 200704), primals_124, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf311, (8, 128, 14, 14), (25088, 196, 14, 1))
        buf312 = buf307; del buf307  # reuse
        buf313 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf315 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_101], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf311, primals_381, primals_382, buf312, buf313, buf315, primals_381, primals_382, 128, 1568, grid=grid(128), stream=stream0)
        del primals_381
        del primals_382
        buf316 = reinterpret_tensor(buf318, (8, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        buf643 = empty((8, 128, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_101, sp_102], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_31.run(buf311, buf312, buf313, primals_125, primals_126, buf316, buf643, 200704, grid=grid(200704), stream=stream0)
        del primals_126
        buf317 = reinterpret_tensor(buf318, (8, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        # Source Nodes: [getattr_l__mod___layer3___0___pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_32.run(buf298, buf317, 200704, grid=grid(200704), stream=stream0)
        # Source Nodes: [out_60], Original ATen: [aten.convolution]
        buf319 = extern_kernels.convolution(buf318, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf319, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf320 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf321 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf323 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_61], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf319, primals_384, primals_385, buf320, buf321, buf323, primals_384, primals_385, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_384
        del primals_385
        # Source Nodes: [getattr_l__mod___layer3___0___downsample_0], Original ATen: [aten.convolution]
        buf324 = extern_kernels.convolution(buf292, primals_130, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf324, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf325 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf326 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf328 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_10], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf324, primals_387, primals_388, buf325, buf326, buf328, primals_387, primals_388, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_387
        del primals_388
        buf329 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        buf330 = buf329; del buf329  # reuse
        # Source Nodes: [out_61, out_62, shortcut_10, shortcut_11], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_34.run(buf330, buf319, buf320, buf321, primals_128, primals_129, buf324, buf325, buf326, primals_131, primals_132, 1605632, grid=grid(1605632), stream=stream0)
        del primals_129
        del primals_132
        # Source Nodes: [out_64], Original ATen: [aten.convolution]
        buf331 = extern_kernels.convolution(buf330, primals_133, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf331, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf332 = buf295; del buf295  # reuse
        buf333 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf335 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_65], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf331, primals_390, primals_391, buf332, buf333, buf335, primals_390, primals_391, 512, 1568, grid=grid(512), stream=stream0)
        del primals_390
        del primals_391
        buf336 = empty((8, 512, 14, 14), device='cuda', dtype=torch.float32)
        buf642 = empty((8, 512, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_65, out_66], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_36.run(buf331, buf332, buf333, primals_134, primals_135, buf336, buf642, 802816, grid=grid(802816), stream=stream0)
        del primals_135
        # Source Nodes: [sp_105], Original ATen: [aten.convolution]
        buf337 = extern_kernels.convolution(reinterpret_tensor(buf336, (8, 128, 14, 14), (100352, 196, 14, 1), 0), primals_136, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf337, (8, 128, 14, 14), (25088, 196, 14, 1))
        buf338 = buf313; del buf313  # reuse
        buf339 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf341 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_106], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf337, primals_393, primals_394, buf338, buf339, buf341, primals_393, primals_394, 128, 1568, grid=grid(128), stream=stream0)
        del primals_393
        del primals_394
        buf358 = empty((8, 512, 14, 14), device='cuda', dtype=torch.float32)
        buf342 = reinterpret_tensor(buf358, (8, 128, 14, 14), (100352, 196, 14, 1), 0)  # alias
        buf343 = empty((8, 128, 14, 14), device='cuda', dtype=torch.float32)
        buf641 = empty((8, 128, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_106, sp_107, sp_108], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_37.run(buf337, buf338, buf339, primals_137, primals_138, buf336, buf342, buf343, buf641, 200704, grid=grid(200704), stream=stream0)
        del primals_138
        # Source Nodes: [sp_109], Original ATen: [aten.convolution]
        buf344 = extern_kernels.convolution(buf343, primals_139, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf344, (8, 128, 14, 14), (25088, 196, 14, 1))
        buf345 = buf339; del buf339  # reuse
        buf346 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf348 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_110], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf344, primals_396, primals_397, buf345, buf346, buf348, primals_396, primals_397, 128, 1568, grid=grid(128), stream=stream0)
        del primals_396
        del primals_397
        buf349 = reinterpret_tensor(buf358, (8, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        buf350 = empty((8, 128, 14, 14), device='cuda', dtype=torch.float32)
        buf640 = empty((8, 128, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_110, sp_111, sp_112], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_38.run(buf344, buf345, buf346, primals_140, primals_141, buf336, buf349, buf350, buf640, 200704, grid=grid(200704), stream=stream0)
        del primals_141
        # Source Nodes: [sp_113], Original ATen: [aten.convolution]
        buf351 = extern_kernels.convolution(buf350, primals_142, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf351, (8, 128, 14, 14), (25088, 196, 14, 1))
        buf352 = buf346; del buf346  # reuse
        buf353 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf355 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_114], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf351, primals_399, primals_400, buf352, buf353, buf355, primals_399, primals_400, 128, 1568, grid=grid(128), stream=stream0)
        del primals_399
        del primals_400
        buf356 = reinterpret_tensor(buf358, (8, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        buf639 = empty((8, 128, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_114, sp_115], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_31.run(buf351, buf352, buf353, primals_143, primals_144, buf356, buf639, 200704, grid=grid(200704), stream=stream0)
        del primals_144
        buf357 = reinterpret_tensor(buf358, (8, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        # Source Nodes: [cat_23], Original ATen: [aten.cat]
        triton_poi_fused_cat_39.run(buf336, buf357, 200704, grid=grid(200704), stream=stream0)
        # Source Nodes: [out_68], Original ATen: [aten.convolution]
        buf359 = extern_kernels.convolution(buf358, primals_145, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf359, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf360 = buf326; del buf326  # reuse
        buf361 = buf321; del buf321  # reuse
        buf363 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_69], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf359, primals_402, primals_403, buf360, buf361, buf363, primals_402, primals_403, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_402
        del primals_403
        buf364 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_69, out_70, shortcut_12], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_40.run(buf359, buf360, buf361, primals_146, primals_147, buf330, buf364, 1605632, grid=grid(1605632), stream=stream0)
        del primals_147
        # Source Nodes: [out_72], Original ATen: [aten.convolution]
        buf365 = extern_kernels.convolution(buf364, primals_148, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf365, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf366 = buf333; del buf333  # reuse
        buf367 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf369 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_73], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf365, primals_405, primals_406, buf366, buf367, buf369, primals_405, primals_406, 512, 1568, grid=grid(512), stream=stream0)
        del primals_405
        del primals_406
        buf370 = empty((8, 512, 14, 14), device='cuda', dtype=torch.float32)
        buf638 = empty((8, 512, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_73, out_74], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_36.run(buf365, buf366, buf367, primals_149, primals_150, buf370, buf638, 802816, grid=grid(802816), stream=stream0)
        del primals_150
        # Source Nodes: [sp_118], Original ATen: [aten.convolution]
        buf371 = extern_kernels.convolution(reinterpret_tensor(buf370, (8, 128, 14, 14), (100352, 196, 14, 1), 0), primals_151, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf371, (8, 128, 14, 14), (25088, 196, 14, 1))
        buf372 = buf353; del buf353  # reuse
        buf373 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf375 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_119], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf371, primals_408, primals_409, buf372, buf373, buf375, primals_408, primals_409, 128, 1568, grid=grid(128), stream=stream0)
        del primals_408
        del primals_409
        buf392 = empty((8, 512, 14, 14), device='cuda', dtype=torch.float32)
        buf376 = reinterpret_tensor(buf392, (8, 128, 14, 14), (100352, 196, 14, 1), 0)  # alias
        buf377 = empty((8, 128, 14, 14), device='cuda', dtype=torch.float32)
        buf637 = empty((8, 128, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_119, sp_120, sp_121], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_37.run(buf371, buf372, buf373, primals_152, primals_153, buf370, buf376, buf377, buf637, 200704, grid=grid(200704), stream=stream0)
        del primals_153
        # Source Nodes: [sp_122], Original ATen: [aten.convolution]
        buf378 = extern_kernels.convolution(buf377, primals_154, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf378, (8, 128, 14, 14), (25088, 196, 14, 1))
        buf379 = buf373; del buf373  # reuse
        buf380 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf382 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_123], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf378, primals_411, primals_412, buf379, buf380, buf382, primals_411, primals_412, 128, 1568, grid=grid(128), stream=stream0)
        del primals_411
        del primals_412
        buf383 = reinterpret_tensor(buf392, (8, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        buf384 = empty((8, 128, 14, 14), device='cuda', dtype=torch.float32)
        buf636 = empty((8, 128, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_123, sp_124, sp_125], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_38.run(buf378, buf379, buf380, primals_155, primals_156, buf370, buf383, buf384, buf636, 200704, grid=grid(200704), stream=stream0)
        del primals_156
        # Source Nodes: [sp_126], Original ATen: [aten.convolution]
        buf385 = extern_kernels.convolution(buf384, primals_157, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf385, (8, 128, 14, 14), (25088, 196, 14, 1))
        buf386 = buf380; del buf380  # reuse
        buf387 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf389 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_127], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf385, primals_414, primals_415, buf386, buf387, buf389, primals_414, primals_415, 128, 1568, grid=grid(128), stream=stream0)
        del primals_414
        del primals_415
        buf390 = reinterpret_tensor(buf392, (8, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        buf635 = empty((8, 128, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_127, sp_128], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_31.run(buf385, buf386, buf387, primals_158, primals_159, buf390, buf635, 200704, grid=grid(200704), stream=stream0)
        del primals_159
        buf391 = reinterpret_tensor(buf392, (8, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        # Source Nodes: [cat_22], Original ATen: [aten.cat]
        triton_poi_fused_cat_39.run(buf370, buf391, 200704, grid=grid(200704), stream=stream0)
        # Source Nodes: [out_76], Original ATen: [aten.convolution]
        buf393 = extern_kernels.convolution(buf392, primals_160, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf393, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf394 = buf361; del buf361  # reuse
        buf395 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf397 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_77], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf393, primals_417, primals_418, buf394, buf395, buf397, primals_417, primals_418, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_417
        del primals_418
        buf398 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_77, out_78, shortcut_13], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_40.run(buf393, buf394, buf395, primals_161, primals_162, buf364, buf398, 1605632, grid=grid(1605632), stream=stream0)
        del primals_162
        # Source Nodes: [out_80], Original ATen: [aten.convolution]
        buf399 = extern_kernels.convolution(buf398, primals_163, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf399, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf400 = buf367; del buf367  # reuse
        buf401 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf403 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_81], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf399, primals_420, primals_421, buf400, buf401, buf403, primals_420, primals_421, 512, 1568, grid=grid(512), stream=stream0)
        del primals_420
        del primals_421
        buf404 = empty((8, 512, 14, 14), device='cuda', dtype=torch.float32)
        buf634 = empty((8, 512, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_81, out_82], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_36.run(buf399, buf400, buf401, primals_164, primals_165, buf404, buf634, 802816, grid=grid(802816), stream=stream0)
        del primals_165
        # Source Nodes: [sp_131], Original ATen: [aten.convolution]
        buf405 = extern_kernels.convolution(reinterpret_tensor(buf404, (8, 128, 14, 14), (100352, 196, 14, 1), 0), primals_166, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf405, (8, 128, 14, 14), (25088, 196, 14, 1))
        buf406 = buf387; del buf387  # reuse
        buf407 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf409 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_132], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf405, primals_423, primals_424, buf406, buf407, buf409, primals_423, primals_424, 128, 1568, grid=grid(128), stream=stream0)
        del primals_423
        del primals_424
        buf426 = empty((8, 512, 14, 14), device='cuda', dtype=torch.float32)
        buf410 = reinterpret_tensor(buf426, (8, 128, 14, 14), (100352, 196, 14, 1), 0)  # alias
        buf411 = empty((8, 128, 14, 14), device='cuda', dtype=torch.float32)
        buf633 = empty((8, 128, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_132, sp_133, sp_134], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_37.run(buf405, buf406, buf407, primals_167, primals_168, buf404, buf410, buf411, buf633, 200704, grid=grid(200704), stream=stream0)
        del primals_168
        # Source Nodes: [sp_135], Original ATen: [aten.convolution]
        buf412 = extern_kernels.convolution(buf411, primals_169, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf412, (8, 128, 14, 14), (25088, 196, 14, 1))
        buf413 = buf407; del buf407  # reuse
        buf414 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf416 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_136], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf412, primals_426, primals_427, buf413, buf414, buf416, primals_426, primals_427, 128, 1568, grid=grid(128), stream=stream0)
        del primals_426
        del primals_427
        buf417 = reinterpret_tensor(buf426, (8, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        buf418 = empty((8, 128, 14, 14), device='cuda', dtype=torch.float32)
        buf632 = empty((8, 128, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_136, sp_137, sp_138], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_38.run(buf412, buf413, buf414, primals_170, primals_171, buf404, buf417, buf418, buf632, 200704, grid=grid(200704), stream=stream0)
        del primals_171
        # Source Nodes: [sp_139], Original ATen: [aten.convolution]
        buf419 = extern_kernels.convolution(buf418, primals_172, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf419, (8, 128, 14, 14), (25088, 196, 14, 1))
        buf420 = buf414; del buf414  # reuse
        buf421 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf423 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_140], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf419, primals_429, primals_430, buf420, buf421, buf423, primals_429, primals_430, 128, 1568, grid=grid(128), stream=stream0)
        del primals_429
        del primals_430
        buf424 = reinterpret_tensor(buf426, (8, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        buf631 = empty((8, 128, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_140, sp_141], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_31.run(buf419, buf420, buf421, primals_173, primals_174, buf424, buf631, 200704, grid=grid(200704), stream=stream0)
        del primals_174
        buf425 = reinterpret_tensor(buf426, (8, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        # Source Nodes: [cat_21], Original ATen: [aten.cat]
        triton_poi_fused_cat_39.run(buf404, buf425, 200704, grid=grid(200704), stream=stream0)
        # Source Nodes: [out_84], Original ATen: [aten.convolution]
        buf427 = extern_kernels.convolution(buf426, primals_175, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf427, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf428 = buf395; del buf395  # reuse
        buf429 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf431 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_85], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf427, primals_432, primals_433, buf428, buf429, buf431, primals_432, primals_433, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_432
        del primals_433
        buf432 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_85, out_86, shortcut_14], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_40.run(buf427, buf428, buf429, primals_176, primals_177, buf398, buf432, 1605632, grid=grid(1605632), stream=stream0)
        del primals_177
        # Source Nodes: [out_88], Original ATen: [aten.convolution]
        buf433 = extern_kernels.convolution(buf432, primals_178, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf433, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf434 = buf401; del buf401  # reuse
        buf435 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf437 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_89], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf433, primals_435, primals_436, buf434, buf435, buf437, primals_435, primals_436, 512, 1568, grid=grid(512), stream=stream0)
        del primals_435
        del primals_436
        buf438 = empty((8, 512, 14, 14), device='cuda', dtype=torch.float32)
        buf630 = empty((8, 512, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_89, out_90], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_36.run(buf433, buf434, buf435, primals_179, primals_180, buf438, buf630, 802816, grid=grid(802816), stream=stream0)
        del primals_180
        # Source Nodes: [sp_144], Original ATen: [aten.convolution]
        buf439 = extern_kernels.convolution(reinterpret_tensor(buf438, (8, 128, 14, 14), (100352, 196, 14, 1), 0), primals_181, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf439, (8, 128, 14, 14), (25088, 196, 14, 1))
        buf440 = buf421; del buf421  # reuse
        buf441 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf443 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_145], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf439, primals_438, primals_439, buf440, buf441, buf443, primals_438, primals_439, 128, 1568, grid=grid(128), stream=stream0)
        del primals_438
        del primals_439
        buf460 = empty((8, 512, 14, 14), device='cuda', dtype=torch.float32)
        buf444 = reinterpret_tensor(buf460, (8, 128, 14, 14), (100352, 196, 14, 1), 0)  # alias
        buf445 = empty((8, 128, 14, 14), device='cuda', dtype=torch.float32)
        buf629 = empty((8, 128, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_145, sp_146, sp_147], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_37.run(buf439, buf440, buf441, primals_182, primals_183, buf438, buf444, buf445, buf629, 200704, grid=grid(200704), stream=stream0)
        del primals_183
        # Source Nodes: [sp_148], Original ATen: [aten.convolution]
        buf446 = extern_kernels.convolution(buf445, primals_184, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf446, (8, 128, 14, 14), (25088, 196, 14, 1))
        buf447 = buf441; del buf441  # reuse
        buf448 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf450 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_149], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf446, primals_441, primals_442, buf447, buf448, buf450, primals_441, primals_442, 128, 1568, grid=grid(128), stream=stream0)
        del primals_441
        del primals_442
        buf451 = reinterpret_tensor(buf460, (8, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        buf452 = empty((8, 128, 14, 14), device='cuda', dtype=torch.float32)
        buf628 = empty((8, 128, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_149, sp_150, sp_151], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_38.run(buf446, buf447, buf448, primals_185, primals_186, buf438, buf451, buf452, buf628, 200704, grid=grid(200704), stream=stream0)
        del primals_186
        # Source Nodes: [sp_152], Original ATen: [aten.convolution]
        buf453 = extern_kernels.convolution(buf452, primals_187, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf453, (8, 128, 14, 14), (25088, 196, 14, 1))
        buf454 = buf448; del buf448  # reuse
        buf455 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf457 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_153], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf453, primals_444, primals_445, buf454, buf455, buf457, primals_444, primals_445, 128, 1568, grid=grid(128), stream=stream0)
        del primals_444
        del primals_445
        buf458 = reinterpret_tensor(buf460, (8, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        buf627 = empty((8, 128, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_153, sp_154], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_31.run(buf453, buf454, buf455, primals_188, primals_189, buf458, buf627, 200704, grid=grid(200704), stream=stream0)
        del primals_189
        buf459 = reinterpret_tensor(buf460, (8, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        # Source Nodes: [cat_20], Original ATen: [aten.cat]
        triton_poi_fused_cat_39.run(buf438, buf459, 200704, grid=grid(200704), stream=stream0)
        # Source Nodes: [out_92], Original ATen: [aten.convolution]
        buf461 = extern_kernels.convolution(buf460, primals_190, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf461, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf462 = buf429; del buf429  # reuse
        buf463 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf465 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_93], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf461, primals_447, primals_448, buf462, buf463, buf465, primals_447, primals_448, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_447
        del primals_448
        buf466 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_93, out_94, shortcut_15], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_40.run(buf461, buf462, buf463, primals_191, primals_192, buf432, buf466, 1605632, grid=grid(1605632), stream=stream0)
        del primals_192
        # Source Nodes: [out_96], Original ATen: [aten.convolution]
        buf467 = extern_kernels.convolution(buf466, primals_193, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf467, (8, 512, 14, 14), (100352, 196, 14, 1))
        buf468 = buf435; del buf435  # reuse
        buf469 = empty_strided((1, 512, 1, 1), (512, 1, 512, 512), device='cuda', dtype=torch.float32)
        buf471 = empty((512, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_97], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_35.run(buf467, primals_450, primals_451, buf468, buf469, buf471, primals_450, primals_451, 512, 1568, grid=grid(512), stream=stream0)
        del primals_450
        del primals_451
        buf472 = empty((8, 512, 14, 14), device='cuda', dtype=torch.float32)
        buf626 = empty((8, 512, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_97, out_98], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_36.run(buf467, buf468, buf469, primals_194, primals_195, buf472, buf626, 802816, grid=grid(802816), stream=stream0)
        del buf469
        del primals_195
        # Source Nodes: [sp_157], Original ATen: [aten.convolution]
        buf473 = extern_kernels.convolution(reinterpret_tensor(buf472, (8, 128, 14, 14), (100352, 196, 14, 1), 0), primals_196, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf473, (8, 128, 14, 14), (25088, 196, 14, 1))
        buf474 = buf455; del buf455  # reuse
        buf475 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf477 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_158], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf473, primals_453, primals_454, buf474, buf475, buf477, primals_453, primals_454, 128, 1568, grid=grid(128), stream=stream0)
        del primals_453
        del primals_454
        buf494 = empty((8, 512, 14, 14), device='cuda', dtype=torch.float32)
        buf478 = reinterpret_tensor(buf494, (8, 128, 14, 14), (100352, 196, 14, 1), 0)  # alias
        buf479 = empty((8, 128, 14, 14), device='cuda', dtype=torch.float32)
        buf625 = empty((8, 128, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_158, sp_159, sp_160], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_37.run(buf473, buf474, buf475, primals_197, primals_198, buf472, buf478, buf479, buf625, 200704, grid=grid(200704), stream=stream0)
        del primals_198
        # Source Nodes: [sp_161], Original ATen: [aten.convolution]
        buf480 = extern_kernels.convolution(buf479, primals_199, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf480, (8, 128, 14, 14), (25088, 196, 14, 1))
        buf481 = buf475; del buf475  # reuse
        buf482 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf484 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_162], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf480, primals_456, primals_457, buf481, buf482, buf484, primals_456, primals_457, 128, 1568, grid=grid(128), stream=stream0)
        del primals_456
        del primals_457
        buf485 = reinterpret_tensor(buf494, (8, 128, 14, 14), (100352, 196, 14, 1), 25088)  # alias
        buf486 = empty((8, 128, 14, 14), device='cuda', dtype=torch.float32)
        buf624 = empty((8, 128, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_162, sp_163, sp_164], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_38.run(buf480, buf481, buf482, primals_200, primals_201, buf472, buf485, buf486, buf624, 200704, grid=grid(200704), stream=stream0)
        del primals_201
        # Source Nodes: [sp_165], Original ATen: [aten.convolution]
        buf487 = extern_kernels.convolution(buf486, primals_202, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf487, (8, 128, 14, 14), (25088, 196, 14, 1))
        buf488 = buf482; del buf482  # reuse
        buf489 = empty_strided((1, 128, 1, 1), (128, 1, 128, 128), device='cuda', dtype=torch.float32)
        buf491 = empty((128, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_166], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_30.run(buf487, primals_459, primals_460, buf488, buf489, buf491, primals_459, primals_460, 128, 1568, grid=grid(128), stream=stream0)
        del primals_459
        del primals_460
        buf492 = reinterpret_tensor(buf494, (8, 128, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        buf623 = empty((8, 128, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_166, sp_167], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_31.run(buf487, buf488, buf489, primals_203, primals_204, buf492, buf623, 200704, grid=grid(200704), stream=stream0)
        del buf489
        del primals_204
        buf493 = reinterpret_tensor(buf494, (8, 128, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        # Source Nodes: [cat_19], Original ATen: [aten.cat]
        triton_poi_fused_cat_39.run(buf472, buf493, 200704, grid=grid(200704), stream=stream0)
        # Source Nodes: [out_100], Original ATen: [aten.convolution]
        buf495 = extern_kernels.convolution(buf494, primals_205, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf495, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf496 = buf463; del buf463  # reuse
        buf497 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf499 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_101], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf495, primals_462, primals_463, buf496, buf497, buf499, primals_462, primals_463, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_462
        del primals_463
        buf500 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_101, out_102, shortcut_16], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_40.run(buf495, buf496, buf497, primals_206, primals_207, buf466, buf500, 1605632, grid=grid(1605632), stream=stream0)
        del primals_207
        # Source Nodes: [out_104], Original ATen: [aten.convolution]
        buf501 = extern_kernels.convolution(buf500, primals_208, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf501, (8, 1024, 14, 14), (200704, 196, 14, 1))
        buf502 = buf497; del buf497  # reuse
        buf503 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf505 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_105], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_red_fused__native_batch_norm_legit_functional_33.run(buf501, primals_465, primals_466, buf502, buf503, buf505, primals_465, primals_466, 1024, 1568, grid=grid(1024), stream=stream0)
        del primals_465
        del primals_466
        buf506 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.float32)
        buf622 = empty((8, 1024, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_105, out_106], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_41.run(buf501, buf502, buf503, primals_209, primals_210, buf506, buf622, 1605632, grid=grid(1605632), stream=stream0)
        del primals_210
        # Source Nodes: [sp_170], Original ATen: [aten.convolution]
        buf507 = extern_kernels.convolution(reinterpret_tensor(buf506, (8, 256, 14, 14), (200704, 196, 14, 1), 0), primals_211, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf507, (8, 256, 7, 7), (12544, 49, 7, 1))
        buf508 = buf261; del buf261  # reuse
        buf509 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf511 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_171], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_42.run(buf507, primals_468, primals_469, buf508, buf509, buf511, primals_468, primals_469, 256, 392, grid=grid(256), stream=stream0)
        del primals_468
        del primals_469
        buf526 = empty((8, 1024, 7, 7), device='cuda', dtype=torch.float32)
        buf512 = reinterpret_tensor(buf526, (8, 256, 7, 7), (50176, 49, 7, 1), 0)  # alias
        buf621 = empty((8, 256, 7, 7), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_171, sp_172], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_43.run(buf507, buf508, buf509, primals_212, primals_213, buf512, buf621, 100352, grid=grid(100352), stream=stream0)
        del primals_213
        # Source Nodes: [sp_174], Original ATen: [aten.convolution]
        buf513 = extern_kernels.convolution(reinterpret_tensor(buf506, (8, 256, 14, 14), (200704, 196, 14, 1), 50176), primals_214, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf513, (8, 256, 7, 7), (12544, 49, 7, 1))
        buf514 = buf509; del buf509  # reuse
        buf515 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf517 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_175], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_42.run(buf513, primals_471, primals_472, buf514, buf515, buf517, primals_471, primals_472, 256, 392, grid=grid(256), stream=stream0)
        del primals_471
        del primals_472
        buf518 = reinterpret_tensor(buf526, (8, 256, 7, 7), (50176, 49, 7, 1), 12544)  # alias
        buf620 = empty((8, 256, 7, 7), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_175, sp_176], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_43.run(buf513, buf514, buf515, primals_215, primals_216, buf518, buf620, 100352, grid=grid(100352), stream=stream0)
        del primals_216
        # Source Nodes: [sp_178], Original ATen: [aten.convolution]
        buf519 = extern_kernels.convolution(reinterpret_tensor(buf506, (8, 256, 14, 14), (200704, 196, 14, 1), 100352), primals_217, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf519, (8, 256, 7, 7), (12544, 49, 7, 1))
        buf520 = buf515; del buf515  # reuse
        buf521 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf523 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_179], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_42.run(buf519, primals_474, primals_475, buf520, buf521, buf523, primals_474, primals_475, 256, 392, grid=grid(256), stream=stream0)
        del primals_474
        del primals_475
        buf524 = reinterpret_tensor(buf526, (8, 256, 7, 7), (50176, 49, 7, 1), 25088)  # alias
        buf619 = empty((8, 256, 7, 7), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_179, sp_180], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_43.run(buf519, buf520, buf521, primals_218, primals_219, buf524, buf619, 100352, grid=grid(100352), stream=stream0)
        del primals_219
        buf525 = reinterpret_tensor(buf526, (8, 256, 7, 7), (50176, 49, 7, 1), 37632)  # alias
        # Source Nodes: [getattr_l__mod___layer4___0___pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_44.run(buf506, buf525, 100352, grid=grid(100352), stream=stream0)
        # Source Nodes: [out_108], Original ATen: [aten.convolution]
        buf527 = extern_kernels.convolution(buf526, primals_220, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf527, (8, 2048, 7, 7), (100352, 49, 7, 1))
        buf528 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf529 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf531 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_109], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf527, primals_477, primals_478, buf528, buf529, buf531, primals_477, primals_478, 2048, 392, grid=grid(2048), stream=stream0)
        del primals_477
        del primals_478
        # Source Nodes: [getattr_l__mod___layer4___0___downsample_0], Original ATen: [aten.convolution]
        buf532 = extern_kernels.convolution(buf500, primals_223, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf532, (8, 2048, 7, 7), (100352, 49, 7, 1))
        buf533 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf534 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf536 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_17], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf532, primals_480, primals_481, buf533, buf534, buf536, primals_480, primals_481, 2048, 392, grid=grid(2048), stream=stream0)
        del primals_480
        del primals_481
        buf537 = empty((8, 2048, 7, 7), device='cuda', dtype=torch.float32)
        buf538 = buf537; del buf537  # reuse
        # Source Nodes: [out_109, out_110, shortcut_17, shortcut_18], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_46.run(buf538, buf527, buf528, buf529, primals_221, primals_222, buf532, buf533, buf534, primals_224, primals_225, 802816, grid=grid(802816), stream=stream0)
        del primals_222
        del primals_225
        # Source Nodes: [out_112], Original ATen: [aten.convolution]
        buf539 = extern_kernels.convolution(buf538, primals_226, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf539, (8, 1024, 7, 7), (50176, 49, 7, 1))
        buf540 = buf503; del buf503  # reuse
        buf541 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf543 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_113], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_47.run(buf539, primals_483, primals_484, buf540, buf541, buf543, primals_483, primals_484, 1024, 392, grid=grid(1024), stream=stream0)
        del primals_483
        del primals_484
        buf544 = empty((8, 1024, 7, 7), device='cuda', dtype=torch.float32)
        buf618 = empty((8, 1024, 7, 7), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_113, out_114], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_48.run(buf539, buf540, buf541, primals_227, primals_228, buf544, buf618, 401408, grid=grid(401408), stream=stream0)
        del primals_228
        # Source Nodes: [sp_183], Original ATen: [aten.convolution]
        buf545 = extern_kernels.convolution(reinterpret_tensor(buf544, (8, 256, 7, 7), (50176, 49, 7, 1), 0), primals_229, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf545, (8, 256, 7, 7), (12544, 49, 7, 1))
        buf546 = buf521; del buf521  # reuse
        buf547 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf549 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_184], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_42.run(buf545, primals_486, primals_487, buf546, buf547, buf549, primals_486, primals_487, 256, 392, grid=grid(256), stream=stream0)
        del primals_486
        del primals_487
        buf566 = empty((8, 1024, 7, 7), device='cuda', dtype=torch.float32)
        buf550 = reinterpret_tensor(buf566, (8, 256, 7, 7), (50176, 49, 7, 1), 0)  # alias
        buf551 = empty((8, 256, 7, 7), device='cuda', dtype=torch.float32)
        buf617 = empty((8, 256, 7, 7), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_184, sp_185, sp_186], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_49.run(buf545, buf546, buf547, primals_230, primals_231, buf544, buf550, buf551, buf617, 100352, grid=grid(100352), stream=stream0)
        del primals_231
        # Source Nodes: [sp_187], Original ATen: [aten.convolution]
        buf552 = extern_kernels.convolution(buf551, primals_232, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf552, (8, 256, 7, 7), (12544, 49, 7, 1))
        buf553 = buf547; del buf547  # reuse
        buf554 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf556 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_188], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_42.run(buf552, primals_489, primals_490, buf553, buf554, buf556, primals_489, primals_490, 256, 392, grid=grid(256), stream=stream0)
        del primals_489
        del primals_490
        buf557 = reinterpret_tensor(buf566, (8, 256, 7, 7), (50176, 49, 7, 1), 12544)  # alias
        buf558 = empty((8, 256, 7, 7), device='cuda', dtype=torch.float32)
        buf616 = empty((8, 256, 7, 7), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_188, sp_189, sp_190], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_50.run(buf552, buf553, buf554, primals_233, primals_234, buf544, buf557, buf558, buf616, 100352, grid=grid(100352), stream=stream0)
        del primals_234
        # Source Nodes: [sp_191], Original ATen: [aten.convolution]
        buf559 = extern_kernels.convolution(buf558, primals_235, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf559, (8, 256, 7, 7), (12544, 49, 7, 1))
        buf560 = buf554; del buf554  # reuse
        buf561 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf563 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_192], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_42.run(buf559, primals_492, primals_493, buf560, buf561, buf563, primals_492, primals_493, 256, 392, grid=grid(256), stream=stream0)
        del primals_492
        del primals_493
        buf564 = reinterpret_tensor(buf566, (8, 256, 7, 7), (50176, 49, 7, 1), 25088)  # alias
        buf615 = empty((8, 256, 7, 7), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_192, sp_193], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_43.run(buf559, buf560, buf561, primals_236, primals_237, buf564, buf615, 100352, grid=grid(100352), stream=stream0)
        del primals_237
        buf565 = reinterpret_tensor(buf566, (8, 256, 7, 7), (50176, 49, 7, 1), 37632)  # alias
        # Source Nodes: [cat_17], Original ATen: [aten.cat]
        triton_poi_fused_cat_51.run(buf544, buf565, 100352, grid=grid(100352), stream=stream0)
        # Source Nodes: [out_116], Original ATen: [aten.convolution]
        buf567 = extern_kernels.convolution(buf566, primals_238, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf567, (8, 2048, 7, 7), (100352, 49, 7, 1))
        buf568 = buf534; del buf534  # reuse
        buf569 = buf529; del buf529  # reuse
        buf571 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_117], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf567, primals_495, primals_496, buf568, buf569, buf571, primals_495, primals_496, 2048, 392, grid=grid(2048), stream=stream0)
        del primals_495
        del primals_496
        buf572 = empty((8, 2048, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_117, out_118, shortcut_19], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_52.run(buf567, buf568, buf569, primals_239, primals_240, buf538, buf572, 802816, grid=grid(802816), stream=stream0)
        del primals_240
        # Source Nodes: [out_120], Original ATen: [aten.convolution]
        buf573 = extern_kernels.convolution(buf572, primals_241, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf573, (8, 1024, 7, 7), (50176, 49, 7, 1))
        buf574 = buf541; del buf541  # reuse
        buf575 = empty_strided((1, 1024, 1, 1), (1024, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf577 = empty((1024, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_121], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_47.run(buf573, primals_498, primals_499, buf574, buf575, buf577, primals_498, primals_499, 1024, 392, grid=grid(1024), stream=stream0)
        del primals_498
        del primals_499
        buf578 = empty((8, 1024, 7, 7), device='cuda', dtype=torch.float32)
        buf614 = empty((8, 1024, 7, 7), device='cuda', dtype=torch.bool)
        # Source Nodes: [out_121, out_122], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_48.run(buf573, buf574, buf575, primals_242, primals_243, buf578, buf614, 401408, grid=grid(401408), stream=stream0)
        del buf575
        del primals_243
        # Source Nodes: [sp_196], Original ATen: [aten.convolution]
        buf579 = extern_kernels.convolution(reinterpret_tensor(buf578, (8, 256, 7, 7), (50176, 49, 7, 1), 0), primals_244, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf579, (8, 256, 7, 7), (12544, 49, 7, 1))
        buf580 = buf561; del buf561  # reuse
        buf581 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf583 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_197], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_42.run(buf579, primals_501, primals_502, buf580, buf581, buf583, primals_501, primals_502, 256, 392, grid=grid(256), stream=stream0)
        del primals_501
        del primals_502
        buf600 = empty((8, 1024, 7, 7), device='cuda', dtype=torch.float32)
        buf584 = reinterpret_tensor(buf600, (8, 256, 7, 7), (50176, 49, 7, 1), 0)  # alias
        buf585 = empty((8, 256, 7, 7), device='cuda', dtype=torch.float32)
        buf613 = empty((8, 256, 7, 7), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_197, sp_198, sp_199], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_49.run(buf579, buf580, buf581, primals_245, primals_246, buf578, buf584, buf585, buf613, 100352, grid=grid(100352), stream=stream0)
        del primals_246
        # Source Nodes: [sp_200], Original ATen: [aten.convolution]
        buf586 = extern_kernels.convolution(buf585, primals_247, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf586, (8, 256, 7, 7), (12544, 49, 7, 1))
        buf587 = buf581; del buf581  # reuse
        buf588 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf590 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_201], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_42.run(buf586, primals_504, primals_505, buf587, buf588, buf590, primals_504, primals_505, 256, 392, grid=grid(256), stream=stream0)
        del primals_504
        del primals_505
        buf591 = reinterpret_tensor(buf600, (8, 256, 7, 7), (50176, 49, 7, 1), 12544)  # alias
        buf592 = empty((8, 256, 7, 7), device='cuda', dtype=torch.float32)
        buf612 = empty((8, 256, 7, 7), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_201, sp_202, sp_203], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_add_relu_threshold_backward_50.run(buf586, buf587, buf588, primals_248, primals_249, buf578, buf591, buf592, buf612, 100352, grid=grid(100352), stream=stream0)
        del primals_249
        # Source Nodes: [sp_204], Original ATen: [aten.convolution]
        buf593 = extern_kernels.convolution(buf592, primals_250, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf593, (8, 256, 7, 7), (12544, 49, 7, 1))
        buf594 = buf588; del buf588  # reuse
        buf595 = empty_strided((1, 256, 1, 1), (256, 1, 256, 256), device='cuda', dtype=torch.float32)
        buf597 = empty((256, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [sp_205], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_42.run(buf593, primals_507, primals_508, buf594, buf595, buf597, primals_507, primals_508, 256, 392, grid=grid(256), stream=stream0)
        del primals_507
        del primals_508
        buf598 = reinterpret_tensor(buf600, (8, 256, 7, 7), (50176, 49, 7, 1), 25088)  # alias
        buf611 = empty((8, 256, 7, 7), device='cuda', dtype=torch.bool)
        # Source Nodes: [sp_205, sp_206], Original ATen: [aten._native_batch_norm_legit_functional, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_functional_relu_threshold_backward_43.run(buf593, buf594, buf595, primals_251, primals_252, buf598, buf611, 100352, grid=grid(100352), stream=stream0)
        del buf595
        del primals_252
        buf599 = reinterpret_tensor(buf600, (8, 256, 7, 7), (50176, 49, 7, 1), 37632)  # alias
        # Source Nodes: [cat_16], Original ATen: [aten.cat]
        triton_poi_fused_cat_51.run(buf578, buf599, 100352, grid=grid(100352), stream=stream0)
        # Source Nodes: [out_124], Original ATen: [aten.convolution]
        buf601 = extern_kernels.convolution(buf600, primals_253, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf601, (8, 2048, 7, 7), (100352, 49, 7, 1))
        buf602 = buf569; del buf569  # reuse
        buf603 = empty_strided((1, 2048, 1, 1), (2048, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf605 = empty((2048, ), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_125], Original ATen: [aten._native_batch_norm_legit_functional]
        triton_per_fused__native_batch_norm_legit_functional_45.run(buf601, primals_510, primals_511, buf602, buf603, buf605, primals_510, primals_511, 2048, 392, grid=grid(2048), stream=stream0)
        del primals_510
        del primals_511
        buf610 = empty((8, 2048, 7, 7), device='cuda', dtype=torch.bool)
        buf607 = empty_strided((8, 2048, 1, 1), (2048, 1, 16384, 16384), device='cuda', dtype=torch.float32)
        buf608 = reinterpret_tensor(buf607, (8, 2048), (2048, 1), 0); del buf607  # reuse
        # Source Nodes: [out_125, out_126, x_11, x_8, x_9], Original ATen: [aten._native_batch_norm_legit_functional, aten.add, aten.mean, aten.relu, aten.threshold_backward, aten.view]
        triton_per_fused__native_batch_norm_legit_functional_add_mean_relu_threshold_backward_view_53.run(buf608, buf601, buf602, buf603, primals_254, primals_255, buf572, buf610, 16384, 49, grid=grid(16384), stream=stream0)
        del buf603
        del primals_255
        buf609 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_257, buf608, reinterpret_tensor(primals_256, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf609)
        del primals_257
        # Source Nodes: [x_1], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_260, primals_260, 1, grid=grid(1), stream=stream0)
        del primals_260
        # Source Nodes: [out_1], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_263, primals_263, 1, grid=grid(1), stream=stream0)
        del primals_263
        # Source Nodes: [sp_2], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_266, primals_266, 1, grid=grid(1), stream=stream0)
        del primals_266
        # Source Nodes: [sp_6], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_269, primals_269, 1, grid=grid(1), stream=stream0)
        del primals_269
        # Source Nodes: [sp_10], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_272, primals_272, 1, grid=grid(1), stream=stream0)
        del primals_272
        # Source Nodes: [out_5], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_275, primals_275, 1, grid=grid(1), stream=stream0)
        del primals_275
        # Source Nodes: [shortcut_1], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_278, primals_278, 1, grid=grid(1), stream=stream0)
        del primals_278
        # Source Nodes: [out_9], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_281, primals_281, 1, grid=grid(1), stream=stream0)
        del primals_281
        # Source Nodes: [sp_15], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_284, primals_284, 1, grid=grid(1), stream=stream0)
        del primals_284
        # Source Nodes: [sp_19], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_287, primals_287, 1, grid=grid(1), stream=stream0)
        del primals_287
        # Source Nodes: [sp_23], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_290, primals_290, 1, grid=grid(1), stream=stream0)
        del primals_290
        # Source Nodes: [out_13], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_293, primals_293, 1, grid=grid(1), stream=stream0)
        del primals_293
        # Source Nodes: [out_17], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_296, primals_296, 1, grid=grid(1), stream=stream0)
        del primals_296
        # Source Nodes: [sp_28], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_299, primals_299, 1, grid=grid(1), stream=stream0)
        del primals_299
        # Source Nodes: [sp_32], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_302, primals_302, 1, grid=grid(1), stream=stream0)
        del primals_302
        # Source Nodes: [sp_36], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_305, primals_305, 1, grid=grid(1), stream=stream0)
        del primals_305
        # Source Nodes: [out_21], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_308, primals_308, 1, grid=grid(1), stream=stream0)
        del primals_308
        # Source Nodes: [out_25], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_311, primals_311, 1, grid=grid(1), stream=stream0)
        del primals_311
        # Source Nodes: [sp_41], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_314, primals_314, 1, grid=grid(1), stream=stream0)
        del primals_314
        # Source Nodes: [sp_45], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_317, primals_317, 1, grid=grid(1), stream=stream0)
        del primals_317
        # Source Nodes: [sp_49], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_320, primals_320, 1, grid=grid(1), stream=stream0)
        del primals_320
        # Source Nodes: [out_29], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_323, primals_323, 1, grid=grid(1), stream=stream0)
        del primals_323
        # Source Nodes: [shortcut_5], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_326, primals_326, 1, grid=grid(1), stream=stream0)
        del primals_326
        # Source Nodes: [out_33], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_329, primals_329, 1, grid=grid(1), stream=stream0)
        del primals_329
        # Source Nodes: [sp_54], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_332, primals_332, 1, grid=grid(1), stream=stream0)
        del primals_332
        # Source Nodes: [sp_58], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_335, primals_335, 1, grid=grid(1), stream=stream0)
        del primals_335
        # Source Nodes: [sp_62], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_338, primals_338, 1, grid=grid(1), stream=stream0)
        del primals_338
        # Source Nodes: [out_37], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_341, primals_341, 1, grid=grid(1), stream=stream0)
        del primals_341
        # Source Nodes: [out_41], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_344, primals_344, 1, grid=grid(1), stream=stream0)
        del primals_344
        # Source Nodes: [sp_67], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_347, primals_347, 1, grid=grid(1), stream=stream0)
        del primals_347
        # Source Nodes: [sp_71], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_350, primals_350, 1, grid=grid(1), stream=stream0)
        del primals_350
        # Source Nodes: [sp_75], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_353, primals_353, 1, grid=grid(1), stream=stream0)
        del primals_353
        # Source Nodes: [out_45], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_356, primals_356, 1, grid=grid(1), stream=stream0)
        del primals_356
        # Source Nodes: [out_49], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_359, primals_359, 1, grid=grid(1), stream=stream0)
        del primals_359
        # Source Nodes: [sp_80], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_362, primals_362, 1, grid=grid(1), stream=stream0)
        del primals_362
        # Source Nodes: [sp_84], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_365, primals_365, 1, grid=grid(1), stream=stream0)
        del primals_365
        # Source Nodes: [sp_88], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_368, primals_368, 1, grid=grid(1), stream=stream0)
        del primals_368
        # Source Nodes: [out_53], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_371, primals_371, 1, grid=grid(1), stream=stream0)
        del primals_371
        # Source Nodes: [out_57], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_374, primals_374, 1, grid=grid(1), stream=stream0)
        del primals_374
        # Source Nodes: [sp_93], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_377, primals_377, 1, grid=grid(1), stream=stream0)
        del primals_377
        # Source Nodes: [sp_97], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_380, primals_380, 1, grid=grid(1), stream=stream0)
        del primals_380
        # Source Nodes: [sp_101], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_383, primals_383, 1, grid=grid(1), stream=stream0)
        del primals_383
        # Source Nodes: [out_61], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_386, primals_386, 1, grid=grid(1), stream=stream0)
        del primals_386
        # Source Nodes: [shortcut_10], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_389, primals_389, 1, grid=grid(1), stream=stream0)
        del primals_389
        # Source Nodes: [out_65], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_392, primals_392, 1, grid=grid(1), stream=stream0)
        del primals_392
        # Source Nodes: [sp_106], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_395, primals_395, 1, grid=grid(1), stream=stream0)
        del primals_395
        # Source Nodes: [sp_110], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_398, primals_398, 1, grid=grid(1), stream=stream0)
        del primals_398
        # Source Nodes: [sp_114], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_401, primals_401, 1, grid=grid(1), stream=stream0)
        del primals_401
        # Source Nodes: [out_69], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_404, primals_404, 1, grid=grid(1), stream=stream0)
        del primals_404
        # Source Nodes: [out_73], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_407, primals_407, 1, grid=grid(1), stream=stream0)
        del primals_407
        # Source Nodes: [sp_119], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_410, primals_410, 1, grid=grid(1), stream=stream0)
        del primals_410
        # Source Nodes: [sp_123], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_413, primals_413, 1, grid=grid(1), stream=stream0)
        del primals_413
        # Source Nodes: [sp_127], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_416, primals_416, 1, grid=grid(1), stream=stream0)
        del primals_416
        # Source Nodes: [out_77], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_419, primals_419, 1, grid=grid(1), stream=stream0)
        del primals_419
        # Source Nodes: [out_81], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_422, primals_422, 1, grid=grid(1), stream=stream0)
        del primals_422
        # Source Nodes: [sp_132], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_425, primals_425, 1, grid=grid(1), stream=stream0)
        del primals_425
        # Source Nodes: [sp_136], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_428, primals_428, 1, grid=grid(1), stream=stream0)
        del primals_428
        # Source Nodes: [sp_140], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_431, primals_431, 1, grid=grid(1), stream=stream0)
        del primals_431
        # Source Nodes: [out_85], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_434, primals_434, 1, grid=grid(1), stream=stream0)
        del primals_434
        # Source Nodes: [out_89], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_437, primals_437, 1, grid=grid(1), stream=stream0)
        del primals_437
        # Source Nodes: [sp_145], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_440, primals_440, 1, grid=grid(1), stream=stream0)
        del primals_440
        # Source Nodes: [sp_149], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_443, primals_443, 1, grid=grid(1), stream=stream0)
        del primals_443
        # Source Nodes: [sp_153], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_446, primals_446, 1, grid=grid(1), stream=stream0)
        del primals_446
        # Source Nodes: [out_93], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_449, primals_449, 1, grid=grid(1), stream=stream0)
        del primals_449
        # Source Nodes: [out_97], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_452, primals_452, 1, grid=grid(1), stream=stream0)
        del primals_452
        # Source Nodes: [sp_158], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_455, primals_455, 1, grid=grid(1), stream=stream0)
        del primals_455
        # Source Nodes: [sp_162], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_458, primals_458, 1, grid=grid(1), stream=stream0)
        del primals_458
        # Source Nodes: [sp_166], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_461, primals_461, 1, grid=grid(1), stream=stream0)
        del primals_461
        # Source Nodes: [out_101], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_464, primals_464, 1, grid=grid(1), stream=stream0)
        del primals_464
        # Source Nodes: [out_105], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_467, primals_467, 1, grid=grid(1), stream=stream0)
        del primals_467
        # Source Nodes: [sp_171], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_470, primals_470, 1, grid=grid(1), stream=stream0)
        del primals_470
        # Source Nodes: [sp_175], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_473, primals_473, 1, grid=grid(1), stream=stream0)
        del primals_473
        # Source Nodes: [sp_179], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_476, primals_476, 1, grid=grid(1), stream=stream0)
        del primals_476
        # Source Nodes: [out_109], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_479, primals_479, 1, grid=grid(1), stream=stream0)
        del primals_479
        # Source Nodes: [shortcut_17], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_482, primals_482, 1, grid=grid(1), stream=stream0)
        del primals_482
        # Source Nodes: [out_113], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_485, primals_485, 1, grid=grid(1), stream=stream0)
        del primals_485
        # Source Nodes: [sp_184], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_488, primals_488, 1, grid=grid(1), stream=stream0)
        del primals_488
        # Source Nodes: [sp_188], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_491, primals_491, 1, grid=grid(1), stream=stream0)
        del primals_491
        # Source Nodes: [sp_192], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_494, primals_494, 1, grid=grid(1), stream=stream0)
        del primals_494
        # Source Nodes: [out_117], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_497, primals_497, 1, grid=grid(1), stream=stream0)
        del primals_497
        # Source Nodes: [out_121], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_500, primals_500, 1, grid=grid(1), stream=stream0)
        del primals_500
        # Source Nodes: [sp_197], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_503, primals_503, 1, grid=grid(1), stream=stream0)
        del primals_503
        # Source Nodes: [sp_201], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_506, primals_506, 1, grid=grid(1), stream=stream0)
        del primals_506
        # Source Nodes: [sp_205], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_509, primals_509, 1, grid=grid(1), stream=stream0)
        del primals_509
        # Source Nodes: [out_125], Original ATen: [aten.add]
        triton_poi_fused_add_54.run(primals_512, primals_512, 1, grid=grid(1), stream=stream0)
        del primals_512
        return (buf609, primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_160, primals_161, primals_163, primals_164, primals_166, primals_167, primals_169, primals_170, primals_172, primals_173, primals_175, primals_176, primals_178, primals_179, primals_181, primals_182, primals_184, primals_185, primals_187, primals_188, primals_190, primals_191, primals_193, primals_194, primals_196, primals_197, primals_199, primals_200, primals_202, primals_203, primals_205, primals_206, primals_208, primals_209, primals_211, primals_212, primals_214, primals_215, primals_217, primals_218, primals_220, primals_221, primals_223, primals_224, primals_226, primals_227, primals_229, primals_230, primals_232, primals_233, primals_235, primals_236, primals_238, primals_239, primals_241, primals_242, primals_244, primals_245, primals_247, primals_248, primals_250, primals_251, primals_253, primals_254, primals_513, buf0, buf7, buf8, buf9, buf10, buf11, buf18, reinterpret_tensor(buf19, (8, 32, 56, 56), (401408, 3136, 56, 1), 0), buf20, buf27, reinterpret_tensor(buf19, (8, 32, 56, 56), (401408, 3136, 56, 1), 100352), buf29, buf36, reinterpret_tensor(buf19, (8, 32, 56, 56), (401408, 3136, 56, 1), 200704), buf38, buf45, reinterpret_tensor(buf19, (8, 32, 56, 56), (401408, 3136, 56, 1), 301056), buf48, buf49, buf53, buf54, buf58, buf60, buf61, buf68, reinterpret_tensor(buf69, (8, 32, 56, 56), (401408, 3136, 56, 1), 0), buf70, buf77, buf79, buf80, buf87, buf89, buf90, buf97, buf100, buf101, buf105, buf106, buf107, buf114, reinterpret_tensor(buf115, (8, 32, 56, 56), (401408, 3136, 56, 1), 0), buf116, buf123, buf125, buf126, buf133, buf135, buf136, buf143, buf146, buf147, buf151, buf152, buf153, buf157, reinterpret_tensor(buf158, (8, 64, 56, 56), (802816, 3136, 56, 1), 0), buf159, buf163, reinterpret_tensor(buf158, (8, 64, 56, 56), (802816, 3136, 56, 1), 200704), buf165, buf169, reinterpret_tensor(buf158, (8, 64, 56, 56), (802816, 3136, 56, 1), 401408), buf171, buf175, reinterpret_tensor(buf158, (8, 64, 56, 56), (802816, 3136, 56, 1), 602112), buf178, buf179, buf183, buf184, buf188, buf190, buf191, buf195, reinterpret_tensor(buf196, (8, 64, 28, 28), (200704, 784, 28, 1), 0), buf197, buf201, buf203, buf204, buf208, buf210, buf211, buf215, buf218, buf219, buf223, buf224, buf225, buf229, reinterpret_tensor(buf230, (8, 64, 28, 28), (200704, 784, 28, 1), 0), buf231, buf235, buf237, buf238, buf242, buf244, buf245, buf249, buf252, buf253, buf257, buf258, buf259, buf263, reinterpret_tensor(buf264, (8, 64, 28, 28), (200704, 784, 28, 1), 0), buf265, buf269, buf271, buf272, buf276, buf278, buf279, buf283, buf286, buf287, buf291, buf292, buf293, buf297, reinterpret_tensor(buf298, (8, 128, 28, 28), (401408, 784, 28, 1), 0), buf299, buf303, reinterpret_tensor(buf298, (8, 128, 28, 28), (401408, 784, 28, 1), 100352), buf305, buf309, reinterpret_tensor(buf298, (8, 128, 28, 28), (401408, 784, 28, 1), 200704), buf311, buf315, reinterpret_tensor(buf298, (8, 128, 28, 28), (401408, 784, 28, 1), 301056), buf318, buf319, buf323, buf324, buf328, buf330, buf331, buf335, reinterpret_tensor(buf336, (8, 128, 14, 14), (100352, 196, 14, 1), 0), buf337, buf341, buf343, buf344, buf348, buf350, buf351, buf355, buf358, buf359, buf363, buf364, buf365, buf369, reinterpret_tensor(buf370, (8, 128, 14, 14), (100352, 196, 14, 1), 0), buf371, buf375, buf377, buf378, buf382, buf384, buf385, buf389, buf392, buf393, buf397, buf398, buf399, buf403, reinterpret_tensor(buf404, (8, 128, 14, 14), (100352, 196, 14, 1), 0), buf405, buf409, buf411, buf412, buf416, buf418, buf419, buf423, buf426, buf427, buf431, buf432, buf433, buf437, reinterpret_tensor(buf438, (8, 128, 14, 14), (100352, 196, 14, 1), 0), buf439, buf443, buf445, buf446, buf450, buf452, buf453, buf457, buf460, buf461, buf465, buf466, buf467, buf471, reinterpret_tensor(buf472, (8, 128, 14, 14), (100352, 196, 14, 1), 0), buf473, buf477, buf479, buf480, buf484, buf486, buf487, buf491, buf494, buf495, buf499, buf500, buf501, buf505, reinterpret_tensor(buf506, (8, 256, 14, 14), (200704, 196, 14, 1), 0), buf507, buf511, reinterpret_tensor(buf506, (8, 256, 14, 14), (200704, 196, 14, 1), 50176), buf513, buf517, reinterpret_tensor(buf506, (8, 256, 14, 14), (200704, 196, 14, 1), 100352), buf519, buf523, reinterpret_tensor(buf506, (8, 256, 14, 14), (200704, 196, 14, 1), 150528), buf526, buf527, buf531, buf532, buf536, buf538, buf539, buf543, reinterpret_tensor(buf544, (8, 256, 7, 7), (50176, 49, 7, 1), 0), buf545, buf549, buf551, buf552, buf556, buf558, buf559, buf563, buf566, buf567, buf571, buf572, buf573, buf577, reinterpret_tensor(buf578, (8, 256, 7, 7), (50176, 49, 7, 1), 0), buf579, buf583, buf585, buf586, buf590, buf592, buf593, buf597, buf600, buf601, buf605, buf608, reinterpret_tensor(primals_256, (1000, 2048), (2048, 1), 0), buf610, reinterpret_tensor(buf602, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), buf611, reinterpret_tensor(buf594, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf612, reinterpret_tensor(buf587, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf613, reinterpret_tensor(buf580, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf614, reinterpret_tensor(buf574, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf568, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), buf615, reinterpret_tensor(buf560, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf616, reinterpret_tensor(buf553, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf617, reinterpret_tensor(buf546, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf618, reinterpret_tensor(buf540, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf533, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), reinterpret_tensor(buf528, (1, 2048, 1, 1), (2048, 1, 1, 1), 0), buf619, reinterpret_tensor(buf520, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf620, reinterpret_tensor(buf514, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf621, reinterpret_tensor(buf508, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf622, reinterpret_tensor(buf502, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf496, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf623, reinterpret_tensor(buf488, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf624, reinterpret_tensor(buf481, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf625, reinterpret_tensor(buf474, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf626, reinterpret_tensor(buf468, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf462, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf627, reinterpret_tensor(buf454, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf628, reinterpret_tensor(buf447, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf629, reinterpret_tensor(buf440, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf630, reinterpret_tensor(buf434, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf428, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf631, reinterpret_tensor(buf420, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf632, reinterpret_tensor(buf413, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf633, reinterpret_tensor(buf406, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf634, reinterpret_tensor(buf400, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf394, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf635, reinterpret_tensor(buf386, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf636, reinterpret_tensor(buf379, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf637, reinterpret_tensor(buf372, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf638, reinterpret_tensor(buf366, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf360, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf639, reinterpret_tensor(buf352, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf640, reinterpret_tensor(buf345, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf641, reinterpret_tensor(buf338, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf642, reinterpret_tensor(buf332, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf325, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), reinterpret_tensor(buf320, (1, 1024, 1, 1), (1024, 1, 1, 1), 0), buf643, reinterpret_tensor(buf312, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf644, reinterpret_tensor(buf306, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf645, reinterpret_tensor(buf300, (1, 128, 1, 1), (128, 1, 1, 1), 0), buf646, reinterpret_tensor(buf294, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf288, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf647, reinterpret_tensor(buf280, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf648, reinterpret_tensor(buf273, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf649, reinterpret_tensor(buf266, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf650, reinterpret_tensor(buf260, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf254, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf651, reinterpret_tensor(buf246, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf652, reinterpret_tensor(buf239, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf653, reinterpret_tensor(buf232, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf654, reinterpret_tensor(buf226, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf220, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf655, reinterpret_tensor(buf212, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf656, reinterpret_tensor(buf205, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf657, reinterpret_tensor(buf198, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf658, reinterpret_tensor(buf192, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf185, (1, 512, 1, 1), (512, 1, 1, 1), 0), reinterpret_tensor(buf180, (1, 512, 1, 1), (512, 1, 1, 1), 0), buf659, reinterpret_tensor(buf172, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf660, reinterpret_tensor(buf166, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf661, reinterpret_tensor(buf160, (1, 64, 1, 1), (64, 1, 1, 1), 0), buf662, reinterpret_tensor(buf154, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf148, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf663, reinterpret_tensor(buf140, (1, 32, 1, 1), (32, 1, 1, 1), 0), buf664, reinterpret_tensor(buf130, (1, 32, 1, 1), (32, 1, 1, 1), 0), buf665, reinterpret_tensor(buf120, (1, 32, 1, 1), (32, 1, 1, 1), 0), buf666, reinterpret_tensor(buf111, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf102, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf667, reinterpret_tensor(buf94, (1, 32, 1, 1), (32, 1, 1, 1), 0), buf668, reinterpret_tensor(buf84, (1, 32, 1, 1), (32, 1, 1, 1), 0), buf669, reinterpret_tensor(buf74, (1, 32, 1, 1), (32, 1, 1, 1), 0), buf670, reinterpret_tensor(buf65, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf55, (1, 256, 1, 1), (256, 1, 1, 1), 0), reinterpret_tensor(buf50, (1, 256, 1, 1), (256, 1, 1, 1), 0), buf671, reinterpret_tensor(buf42, (1, 32, 1, 1), (32, 1, 1, 1), 0), buf672, reinterpret_tensor(buf33, (1, 32, 1, 1), (32, 1, 1, 1), 0), buf673, reinterpret_tensor(buf24, (1, 32, 1, 1), (32, 1, 1, 1), 0), buf674, reinterpret_tensor(buf15, (1, 128, 1, 1), (128, 1, 1, 1), 0), reinterpret_tensor(buf4, (1, 64, 1, 1), (64, 1, 1, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((32, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((64, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((128, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((256, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_261 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_264 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_267 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_270 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_273 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_276 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_279 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_282 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_285 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_288 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_291 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_294 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_297 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_300 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_303 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_306 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_309 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_312 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_315 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_318 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_321 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_324 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_327 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_330 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_333 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_336 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_339 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_342 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_345 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_348 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_351 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_354 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_357 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_360 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_363 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_366 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_368 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_369 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_371 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_372 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_374 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_375 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_377 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_378 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_380 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_381 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_383 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_384 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_386 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_387 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_389 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_390 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_392 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_393 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_395 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_396 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_398 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_399 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_401 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_402 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_404 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_405 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_407 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_408 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_410 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_411 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_413 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_414 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_416 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_417 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_419 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_420 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_422 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_423 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_425 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_426 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_428 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_429 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_431 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_432 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_434 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_435 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_437 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_438 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_440 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_441 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_443 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_444 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_446 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_447 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_449 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_450 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_452 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_453 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_455 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_456 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_458 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_459 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_461 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_462 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_464 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_465 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_467 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_468 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_470 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_471 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_473 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_474 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_476 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_477 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_479 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_480 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_482 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_483 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_485 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_486 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_488 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_489 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_491 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_492 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_494 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_495 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_497 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_498 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_500 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_501 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_503 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_504 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_506 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_507 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_509 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_510 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_512 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_513 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('res2next50', benchmark_compiled_module)
