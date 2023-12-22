
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


# kernel path: /tmp/torchinductor_youkaichao/ir/cirhf24lyvogopcweijlynbtsjvxjmsvvzjalzkgpmr6f6f7tjc6.py
# Source Nodes: [l__mod___features_norm0, l__mod___features_relu0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_norm0 => add_1, mul_1, mul_2, sub
# l__mod___features_relu0 => relu
triton_poi_fused__native_batch_norm_legit_no_training_relu_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 64
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/a4/ca4w42iqvxuqa2266enfhej53tn6ehifmscf7r4cabbp4hvjdwnw.py
# Source Nodes: [cat_117, cat_118, cat_119, l__mod___features_denseblock1_denselayer1_norm1, l__mod___features_denseblock1_denselayer1_relu1, l__mod___features_pool0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.max_pool2d_with_indices, aten.native_batch_norm_backward, aten.relu]
# cat_117 => cat_5
# cat_118 => cat_4
# cat_119 => cat_3
# l__mod___features_denseblock1_denselayer1_norm1 => add_3, mul_4, mul_5, sub_1
# l__mod___features_denseblock1_denselayer1_relu1 => relu_1
# l__mod___features_pool0 => getitem_1, max_pool2d_with_indices
triton_poi_fused__native_batch_norm_legit_no_training_cat_max_pool2d_with_indices_native_batch_norm_backward_relu_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i64', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_max_pool2d_with_indices_native_batch_norm_backward_relu_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 56) % 56
    x0 = xindex % 56
    x5 = (xindex // 56)
    x3 = (xindex // 200704)
    x6 = xindex % 200704
    x7 = xindex
    x2 = (xindex // 3136) % 64
    tmp95 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp97 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp105 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp107 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
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
    tmp11 = tl.load(in_ptr0 + ((-113) + (2*x0) + (224*x5)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x0
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-112) + (2*x0) + (224*x5)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-111) + (2*x0) + (224*x5)), tmp27, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 2*x1
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + (2*x0) + (224*x5)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + ((2*x0) + (224*x5)), tmp41, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + (2*x0) + (224*x5)), tmp46, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 1 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (111 + (2*x0) + (224*x5)), tmp55, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (112 + (2*x0) + (224*x5)), tmp60, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (113 + (2*x0) + (224*x5)), tmp65, eviction_policy='evict_last', other=0.0)
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
    tmp96 = tmp69 - tmp95
    tmp98 = 1e-05
    tmp99 = tmp97 + tmp98
    tmp100 = tl.sqrt(tmp99)
    tmp101 = 1 / tmp100
    tmp102 = 1.0
    tmp103 = tmp101 * tmp102
    tmp104 = tmp96 * tmp103
    tmp106 = tmp104 * tmp105
    tmp108 = tmp106 + tmp107
    tmp109 = triton_helpers.maximum(0, tmp108)
    tl.store(out_ptr0 + (x6 + (301056*x3)), tmp69, None)
    tl.store(out_ptr1 + (x7), tmp94, None)
    tl.store(out_ptr2 + (x7), tmp109, None)
    tl.store(out_ptr3 + (x6 + (602112*x3)), tmp69, None)
    tl.store(out_ptr4 + (x6 + (702464*x3)), tmp69, None)
    tl.store(out_ptr5 + (x6 + (802816*x3)), tmp69, None)
    tl.store(out_ptr6 + (x7), tmp96, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5w/c5wojlu5g63zdr6jjohtdwe2v52hytiohfaaa37djlgiljiuqafu.py
# Source Nodes: [l__mod___features_denseblock1_denselayer1_norm2, l__mod___features_denseblock1_denselayer1_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock1_denselayer1_norm2 => add_5, mul_7, mul_8, sub_2
# l__mod___features_denseblock1_denselayer1_relu2 => relu_2
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/r4/cr47kr2wv5rt3wtiqu7kypbnydoymtlpoulsw2ic4rmi2ezyfgfk.py
# Source Nodes: [cat_117, cat_118, cat_119, cat_122], Original ATen: [aten.cat]
# cat_117 => cat_5
# cat_118 => cat_4
# cat_119 => cat_3
# cat_122 => cat
triton_poi_fused_cat_3 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 100352
    x1 = (xindex // 100352)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + (301056*x1)), tmp0, None)
    tl.store(out_ptr1 + (x0 + (602112*x1)), tmp0, None)
    tl.store(out_ptr2 + (x0 + (702464*x1)), tmp0, None)
    tl.store(out_ptr3 + (x0 + (802816*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rc/crc6etcc6r2upodo3bfryhpv6ge2jehxplife6hovz2j3i3zaxoi.py
# Source Nodes: [l__mod___features_denseblock1_denselayer2_norm1, l__mod___features_denseblock1_denselayer2_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock1_denselayer2_norm1 => add_7, mul_10, mul_11, sub_3
# l__mod___features_denseblock1_denselayer2_relu1 => relu_3
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 96
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/px/cpx4bphjzlop2zfsn4pncffvvjmbtofqxddsaya47nmb377sgilz.py
# Source Nodes: [cat_121, l__mod___features_denseblock1_denselayer3_norm1, l__mod___features_denseblock1_denselayer3_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_121 => cat_1
# l__mod___features_denseblock1_denselayer3_norm1 => add_11, mul_16, mul_17, sub_5
# l__mod___features_denseblock1_denselayer3_relu1 => relu_5
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3136) % 128
    x2 = (xindex // 401408)
    x3 = xindex % 401408
    x4 = xindex
    tmp23 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (301056*x2)), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 96, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-200704) + x3 + (100352*x2)), tmp11, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 128, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-301056) + x3 + (100352*x2)), tmp15, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tmp24 = tmp22 - tmp23
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = tl.sqrt(tmp27)
    tmp29 = 1 / tmp28
    tmp30 = 1.0
    tmp31 = tmp29 * tmp30
    tmp32 = tmp24 * tmp31
    tmp34 = tmp32 * tmp33
    tmp36 = tmp34 + tmp35
    tmp37 = triton_helpers.maximum(0, tmp36)
    tl.store(out_ptr0 + (x4), tmp22, None)
    tl.store(out_ptr1 + (x4), tmp37, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rz/crz4t6v7tybqp4wdwpux4iqzzn26xwhv57jejhg7m2c4nnlosp6e.py
# Source Nodes: [cat_120, l__mod___features_denseblock1_denselayer4_norm1, l__mod___features_denseblock1_denselayer4_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_120 => cat_2
# l__mod___features_denseblock1_denselayer4_norm1 => add_15, mul_22, mul_23, sub_7
# l__mod___features_denseblock1_denselayer4_relu1 => relu_7
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2007040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3136) % 160
    x2 = (xindex // 501760)
    x3 = xindex % 501760
    x4 = xindex
    tmp31 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (301056*x2)), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 96, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-200704) + x3 + (100352*x2)), tmp11, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 128, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-301056) + x3 + (100352*x2)), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1], 160, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr3 + ((-401408) + x3 + (100352*x2)), tmp22, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tl.where(tmp18, tmp21, tmp27)
    tmp29 = tl.where(tmp11, tmp14, tmp28)
    tmp30 = tl.where(tmp4, tmp7, tmp29)
    tmp32 = tmp30 - tmp31
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.sqrt(tmp35)
    tmp37 = 1 / tmp36
    tmp38 = 1.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp32 * tmp39
    tmp42 = tmp40 * tmp41
    tmp44 = tmp42 + tmp43
    tmp45 = triton_helpers.maximum(0, tmp44)
    tl.store(out_ptr0 + (x4), tmp30, None)
    tl.store(out_ptr1 + (x4), tmp45, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qr/cqrncmnq52dsy2utmrr23kff5bbbx3i5tawuvffmdxuffijog6jx.py
# Source Nodes: [cat_117, cat_118, cat_119], Original ATen: [aten.cat]
# cat_117 => cat_5
# cat_118 => cat_4
# cat_119 => cat_3
triton_poi_fused_cat_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 100352
    x1 = (xindex // 100352)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + (602112*x1)), tmp0, None)
    tl.store(out_ptr1 + (x0 + (702464*x1)), tmp0, None)
    tl.store(out_ptr2 + (x0 + (802816*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ff/cffcq5nvdechji6epwfkfdiczzqg3hdmf3fxchowck3nyum76dt3.py
# Source Nodes: [l__mod___features_denseblock1_denselayer5_norm1, l__mod___features_denseblock1_denselayer5_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock1_denselayer5_norm1 => add_19, mul_28, mul_29, sub_9
# l__mod___features_denseblock1_denselayer5_relu1 => relu_9
triton_poi_fused__native_batch_norm_legit_no_training_relu_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 192
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zo/czogiixsmnj3vx6vgjpj22563jnlg6zxcebzfmu6hwbcm6stwkmz.py
# Source Nodes: [cat_117, cat_118], Original ATen: [aten.cat]
# cat_117 => cat_5
# cat_118 => cat_4
triton_poi_fused_cat_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 100352
    x1 = (xindex // 100352)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + (702464*x1)), tmp0, None)
    tl.store(out_ptr1 + (x0 + (802816*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/za/czasfrjjse3mr4sygagkbny4pkpj7gydwzyoinn5vofnldwaqcqq.py
# Source Nodes: [l__mod___features_denseblock1_denselayer6_norm1, l__mod___features_denseblock1_denselayer6_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock1_denselayer6_norm1 => add_23, mul_34, mul_35, sub_11
# l__mod___features_denseblock1_denselayer6_relu1 => relu_11
triton_poi_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2809856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 224
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jr/cjr5dagoysmdslrbl7lyqbe6awjq6puuqi3q7lmcj5iohpafsapa.py
# Source Nodes: [cat_117], Original ATen: [aten.cat]
# cat_117 => cat_5
triton_poi_fused_cat_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 100352
    x1 = (xindex // 100352)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + (802816*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ew/cewjs5bv2a4meioeda24wuztlfzb4auhcukjiy2r5lcp7ioomhfq.py
# Source Nodes: [l__mod___features_transition1_norm, l__mod___features_transition1_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_transition1_norm => add_27, mul_40, mul_41, sub_13
# l__mod___features_transition1_relu => relu_13
triton_poi_fused__native_batch_norm_legit_no_training_relu_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yt/cytapynx7ozyztg7b6z22d7w5d2s3ylly4qo3jke5cfh4u64t52t.py
# Source Nodes: [cat_104, cat_105, cat_106, cat_107, cat_108, cat_109, cat_110, cat_111, cat_112, l__mod___features_denseblock2_denselayer1_norm1, l__mod___features_denseblock2_denselayer1_relu1, l__mod___features_transition1_pool], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d, aten.cat, aten.relu]
# cat_104 => cat_17
# cat_105 => cat_16
# cat_106 => cat_15
# cat_107 => cat_14
# cat_108 => cat_13
# cat_109 => cat_12
# cat_110 => cat_11
# cat_111 => cat_10
# cat_112 => cat_9
# l__mod___features_denseblock2_denselayer1_norm1 => add_29, mul_43, mul_44, sub_14
# l__mod___features_denseblock2_denselayer1_relu1 => relu_14
# l__mod___features_transition1_pool => avg_pool2d
triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 28
    x1 = (xindex // 28)
    x2 = xindex
    x4 = (xindex // 784) % 128
    x5 = (xindex // 100352)
    x6 = xindex % 100352
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (112*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (112*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (56 + (2*x0) + (112*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (57 + (2*x0) + (112*x1)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr3 + (x4), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x4), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sqrt(tmp13)
    tmp15 = 1 / tmp14
    tmp16 = 1.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp10 * tmp17
    tmp20 = tmp18 * tmp19
    tmp22 = tmp20 + tmp21
    tmp23 = triton_helpers.maximum(0, tmp22)
    tl.store(out_ptr0 + (x2), tmp8, None)
    tl.store(out_ptr1 + (x2), tmp23, None)
    tl.store(out_ptr2 + (x6 + (200704*x5)), tmp8, None)
    tl.store(out_ptr3 + (x6 + (225792*x5)), tmp8, None)
    tl.store(out_ptr4 + (x6 + (250880*x5)), tmp8, None)
    tl.store(out_ptr5 + (x6 + (275968*x5)), tmp8, None)
    tl.store(out_ptr6 + (x6 + (301056*x5)), tmp8, None)
    tl.store(out_ptr7 + (x6 + (326144*x5)), tmp8, None)
    tl.store(out_ptr8 + (x6 + (351232*x5)), tmp8, None)
    tl.store(out_ptr9 + (x6 + (376320*x5)), tmp8, None)
    tl.store(out_ptr10 + (x6 + (401408*x5)), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/r3/cr3xdtgnrqspq5tzcppjsylg4fd2lwg36ytyf5kubffzxavtpdzx.py
# Source Nodes: [l__mod___features_denseblock2_denselayer1_norm2, l__mod___features_denseblock2_denselayer1_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock2_denselayer1_norm2 => add_31, mul_46, mul_47, sub_15
# l__mod___features_denseblock2_denselayer1_relu2 => relu_15
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rc/crcni76pmzbrkk7p5lddpibleg5wes5jcxucx2tkuzxdy4ijiser.py
# Source Nodes: [cat_115, l__mod___features_denseblock2_denselayer2_norm1, l__mod___features_denseblock2_denselayer2_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_115 => cat_6
# l__mod___features_denseblock2_denselayer2_norm1 => add_33, mul_49, mul_50, sub_16
# l__mod___features_denseblock2_denselayer2_relu1 => relu_16
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 160
    x2 = (xindex // 125440)
    x3 = xindex % 125440
    x4 = xindex
    tmp15 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (100352*x2)), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 160, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-100352) + x3 + (25088*x2)), tmp8, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tmp14 - tmp15
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = 1.0
    tmp23 = tmp21 * tmp22
    tmp24 = tmp16 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(out_ptr0 + (x4), tmp14, None)
    tl.store(out_ptr1 + (x4), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yx/cyxcwsskkmcslbbnbdokqvhzl5nftsfmul7cdbqpyxqlyyy6elsp.py
# Source Nodes: [cat_114, l__mod___features_denseblock2_denselayer3_norm1, l__mod___features_denseblock2_denselayer3_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_114 => cat_7
# l__mod___features_denseblock2_denselayer3_norm1 => add_37, mul_55, mul_56, sub_18
# l__mod___features_denseblock2_denselayer3_relu1 => relu_18
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 192
    x2 = (xindex // 150528)
    x3 = xindex % 150528
    x4 = xindex
    tmp23 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (100352*x2)), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 160, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-100352) + x3 + (25088*x2)), tmp11, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 192, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-125440) + x3 + (25088*x2)), tmp15, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tmp24 = tmp22 - tmp23
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = tl.sqrt(tmp27)
    tmp29 = 1 / tmp28
    tmp30 = 1.0
    tmp31 = tmp29 * tmp30
    tmp32 = tmp24 * tmp31
    tmp34 = tmp32 * tmp33
    tmp36 = tmp34 + tmp35
    tmp37 = triton_helpers.maximum(0, tmp36)
    tl.store(out_ptr0 + (x4), tmp22, None)
    tl.store(out_ptr1 + (x4), tmp37, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/an/cantcouvgh7est37qvwd7ic5wj5ddfcu3hzd26k6sosdk2mzmr6p.py
# Source Nodes: [cat_113, l__mod___features_denseblock2_denselayer4_norm1, l__mod___features_denseblock2_denselayer4_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_113 => cat_8
# l__mod___features_denseblock2_denselayer4_norm1 => add_41, mul_61, mul_62, sub_20
# l__mod___features_denseblock2_denselayer4_relu1 => relu_20
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 702464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 224
    x2 = (xindex // 175616)
    x3 = xindex % 175616
    x4 = xindex
    tmp31 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (100352*x2)), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 160, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-100352) + x3 + (25088*x2)), tmp11, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 192, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-125440) + x3 + (25088*x2)), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1], 224, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr3 + ((-150528) + x3 + (25088*x2)), tmp22, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tl.where(tmp18, tmp21, tmp27)
    tmp29 = tl.where(tmp11, tmp14, tmp28)
    tmp30 = tl.where(tmp4, tmp7, tmp29)
    tmp32 = tmp30 - tmp31
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.sqrt(tmp35)
    tmp37 = 1 / tmp36
    tmp38 = 1.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp32 * tmp39
    tmp42 = tmp40 * tmp41
    tmp44 = tmp42 + tmp43
    tmp45 = triton_helpers.maximum(0, tmp44)
    tl.store(out_ptr0 + (x4), tmp30, None)
    tl.store(out_ptr1 + (x4), tmp45, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/a4/ca43dkzzj4s66lt3sr42dsvrodwju4jpi24dn6axot54j27eycyw.py
# Source Nodes: [cat_107, cat_108, cat_109, cat_110, cat_111, cat_112], Original ATen: [aten.cat]
# cat_107 => cat_14
# cat_108 => cat_13
# cat_109 => cat_12
# cat_110 => cat_11
# cat_111 => cat_10
# cat_112 => cat_9
triton_poi_fused_cat_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 25088
    x1 = (xindex // 25088)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + (200704*x1)), tmp0, None)
    tl.store(out_ptr1 + (x0 + (225792*x1)), tmp0, None)
    tl.store(out_ptr2 + (x0 + (250880*x1)), tmp0, None)
    tl.store(out_ptr3 + (x0 + (275968*x1)), tmp0, None)
    tl.store(out_ptr4 + (x0 + (301056*x1)), tmp0, None)
    tl.store(out_ptr5 + (x0 + (326144*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/y4/cy4mhs65rmu3odpsq2muxxt5jxshdsohrmxt77n35ozcl7wy7ays.py
# Source Nodes: [l__mod___features_denseblock2_denselayer5_norm1, l__mod___features_denseblock2_denselayer5_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock2_denselayer5_norm1 => add_45, mul_67, mul_68, sub_22
# l__mod___features_denseblock2_denselayer5_relu1 => relu_22
triton_poi_fused__native_batch_norm_legit_no_training_relu_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 256
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nm/cnmfvqlwfkfwcgycyg35remwecjxom3eupxyvr4wxdvwgjnyezmx.py
# Source Nodes: [cat_107, cat_108, cat_109, cat_110, cat_111], Original ATen: [aten.cat]
# cat_107 => cat_14
# cat_108 => cat_13
# cat_109 => cat_12
# cat_110 => cat_11
# cat_111 => cat_10
triton_poi_fused_cat_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 25088
    x1 = (xindex // 25088)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + (225792*x1)), tmp0, None)
    tl.store(out_ptr1 + (x0 + (250880*x1)), tmp0, None)
    tl.store(out_ptr2 + (x0 + (275968*x1)), tmp0, None)
    tl.store(out_ptr3 + (x0 + (301056*x1)), tmp0, None)
    tl.store(out_ptr4 + (x0 + (326144*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xx/cxxi3iahbkivssf7tvvua5z72r34ybpssx4rnyev4rfys7xfdimj.py
# Source Nodes: [l__mod___features_denseblock2_denselayer6_norm1, l__mod___features_denseblock2_denselayer6_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock2_denselayer6_norm1 => add_49, mul_73, mul_74, sub_24
# l__mod___features_denseblock2_denselayer6_relu1 => relu_24
triton_poi_fused__native_batch_norm_legit_no_training_relu_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 903168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 288
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5y/c5yy2ig57daicsyzy5gm663hfjihl5absewfj3o6fhvsp7asflny.py
# Source Nodes: [cat_106, cat_107, cat_108, cat_109, cat_110], Original ATen: [aten.cat]
# cat_106 => cat_15
# cat_107 => cat_14
# cat_108 => cat_13
# cat_109 => cat_12
# cat_110 => cat_11
triton_poi_fused_cat_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 25088
    x1 = (xindex // 25088)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + (250880*x1)), tmp0, None)
    tl.store(out_ptr1 + (x0 + (275968*x1)), tmp0, None)
    tl.store(out_ptr2 + (x0 + (301056*x1)), tmp0, None)
    tl.store(out_ptr3 + (x0 + (326144*x1)), tmp0, None)
    tl.store(out_ptr4 + (x0 + (351232*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pf/cpfdicq6lqpgxllgggfbeoqfjgpoje6rovoe7dzdh4opo3kl75a3.py
# Source Nodes: [l__mod___features_denseblock2_denselayer7_norm1, l__mod___features_denseblock2_denselayer7_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock2_denselayer7_norm1 => add_53, mul_79, mul_80, sub_26
# l__mod___features_denseblock2_denselayer7_relu1 => relu_26
triton_poi_fused__native_batch_norm_legit_no_training_relu_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 320
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/m7/cm7er7lt66rzhgjvkxw6n5by2c6td66htscrkt53enmhfl2kfsbx.py
# Source Nodes: [cat_105, cat_106, cat_107, cat_108, cat_109], Original ATen: [aten.cat]
# cat_105 => cat_16
# cat_106 => cat_15
# cat_107 => cat_14
# cat_108 => cat_13
# cat_109 => cat_12
triton_poi_fused_cat_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 25088
    x1 = (xindex // 25088)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + (275968*x1)), tmp0, None)
    tl.store(out_ptr1 + (x0 + (301056*x1)), tmp0, None)
    tl.store(out_ptr2 + (x0 + (326144*x1)), tmp0, None)
    tl.store(out_ptr3 + (x0 + (351232*x1)), tmp0, None)
    tl.store(out_ptr4 + (x0 + (376320*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/td/ctdizla3or7efg4apwiqu26rwqugyjmqfpuvmqoaosa4lkunvnih.py
# Source Nodes: [l__mod___features_denseblock2_denselayer8_norm1, l__mod___features_denseblock2_denselayer8_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock2_denselayer8_norm1 => add_57, mul_85, mul_86, sub_28
# l__mod___features_denseblock2_denselayer8_relu1 => relu_28
triton_poi_fused__native_batch_norm_legit_no_training_relu_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1103872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 352
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2l/c2l5irepnmkahbk3sc4to57qgcyp5aakgmemoiedflzbo7wc2n6h.py
# Source Nodes: [cat_104, cat_105, cat_106, cat_107, cat_108], Original ATen: [aten.cat]
# cat_104 => cat_17
# cat_105 => cat_16
# cat_106 => cat_15
# cat_107 => cat_14
# cat_108 => cat_13
triton_poi_fused_cat_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 25088
    x1 = (xindex // 25088)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + (301056*x1)), tmp0, None)
    tl.store(out_ptr1 + (x0 + (326144*x1)), tmp0, None)
    tl.store(out_ptr2 + (x0 + (351232*x1)), tmp0, None)
    tl.store(out_ptr3 + (x0 + (376320*x1)), tmp0, None)
    tl.store(out_ptr4 + (x0 + (401408*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bs/cbs6uslh2syk54qkhkpclyj6bz2w4ezjysopxdjx3xkztpxu4enc.py
# Source Nodes: [l__mod___features_denseblock2_denselayer9_norm1, l__mod___features_denseblock2_denselayer9_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock2_denselayer9_norm1 => add_61, mul_91, mul_92, sub_30
# l__mod___features_denseblock2_denselayer9_relu1 => relu_30
triton_poi_fused__native_batch_norm_legit_no_training_relu_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 384
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/on/cong6oe3vqeetnu4wch7ejju5wcb4qgsszd7rsqkxs6ip42vlsrn.py
# Source Nodes: [cat_104, cat_105, cat_106, cat_107], Original ATen: [aten.cat]
# cat_104 => cat_17
# cat_105 => cat_16
# cat_106 => cat_15
# cat_107 => cat_14
triton_poi_fused_cat_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 25088
    x1 = (xindex // 25088)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + (326144*x1)), tmp0, None)
    tl.store(out_ptr1 + (x0 + (351232*x1)), tmp0, None)
    tl.store(out_ptr2 + (x0 + (376320*x1)), tmp0, None)
    tl.store(out_ptr3 + (x0 + (401408*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rm/crmjaq3hfepa56sgkt6vcslatwlfqcy3i7jsimlw6t5ttb6mxihx.py
# Source Nodes: [l__mod___features_denseblock2_denselayer10_norm1, l__mod___features_denseblock2_denselayer10_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock2_denselayer10_norm1 => add_65, mul_97, mul_98, sub_32
# l__mod___features_denseblock2_denselayer10_relu1 => relu_32
triton_poi_fused__native_batch_norm_legit_no_training_relu_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1304576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 416
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cy/ccy4e7bfiohtr5xcddi2likfkkkybmfa7uknc6r7pdgtfkqoljex.py
# Source Nodes: [cat_104, cat_105, cat_106], Original ATen: [aten.cat]
# cat_104 => cat_17
# cat_105 => cat_16
# cat_106 => cat_15
triton_poi_fused_cat_30 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 25088
    x1 = (xindex // 25088)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + (351232*x1)), tmp0, None)
    tl.store(out_ptr1 + (x0 + (376320*x1)), tmp0, None)
    tl.store(out_ptr2 + (x0 + (401408*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/c6/cc62gu2ngrbntutfw6mkcofh6vovxq5frg7rd4bthmenieawlwkn.py
# Source Nodes: [l__mod___features_denseblock2_denselayer11_norm1, l__mod___features_denseblock2_denselayer11_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock2_denselayer11_norm1 => add_69, mul_103, mul_104, sub_34
# l__mod___features_denseblock2_denselayer11_relu1 => relu_34
triton_poi_fused__native_batch_norm_legit_no_training_relu_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 448
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rj/crjffay33246obkgvllpcvqtaddzzx5g6grkuecticzokw32xzq6.py
# Source Nodes: [cat_104, cat_105], Original ATen: [aten.cat]
# cat_104 => cat_17
# cat_105 => cat_16
triton_poi_fused_cat_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 25088
    x1 = (xindex // 25088)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + (376320*x1)), tmp0, None)
    tl.store(out_ptr1 + (x0 + (401408*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xd/cxdzvcf7s2gqbsodrl2aqnwippwi6ngugxo26spwa2dsjj7eil3p.py
# Source Nodes: [l__mod___features_denseblock2_denselayer12_norm1, l__mod___features_denseblock2_denselayer12_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock2_denselayer12_norm1 => add_73, mul_109, mul_110, sub_36
# l__mod___features_denseblock2_denselayer12_relu1 => relu_36
triton_poi_fused__native_batch_norm_legit_no_training_relu_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 480
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jq/cjqx6ainpx4q43rvdgjdesfzwoqwsfjoxdzugbkye7d52zlhbqwf.py
# Source Nodes: [cat_104], Original ATen: [aten.cat]
# cat_104 => cat_17
triton_poi_fused_cat_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 25088
    x1 = (xindex // 25088)
    tmp0 = tl.load(in_ptr0 + (x2), None)
    tl.store(out_ptr0 + (x0 + (401408*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5b/c5bafhnr34wdz3e5ruj3vyu54nkafhwjielf42nd553kaojedscc.py
# Source Nodes: [l__mod___features_transition2_norm, l__mod___features_transition2_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_transition2_norm => add_77, mul_115, mul_116, sub_38
# l__mod___features_transition2_relu => relu_38
triton_poi_fused__native_batch_norm_legit_no_training_relu_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vg/cvgibabdkpw6nhdz4bnmutcbbouhkdufyojfl4eau5rm3oy22426.py
# Source Nodes: [cat_79, cat_80, cat_81, cat_82, cat_83, cat_84, cat_85, cat_86, cat_87, cat_88, cat_89, cat_90, cat_91, cat_92, cat_93, cat_94, cat_95, cat_96, cat_97, cat_98, cat_99, l__mod___features_denseblock3_denselayer1_norm1, l__mod___features_denseblock3_denselayer1_relu1, l__mod___features_transition2_pool], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d, aten.cat, aten.relu]
# cat_79 => cat_41
# cat_80 => cat_40
# cat_81 => cat_39
# cat_82 => cat_38
# cat_83 => cat_37
# cat_84 => cat_36
# cat_85 => cat_35
# cat_86 => cat_34
# cat_87 => cat_33
# cat_88 => cat_32
# cat_89 => cat_31
# cat_90 => cat_30
# cat_91 => cat_29
# cat_92 => cat_28
# cat_93 => cat_27
# cat_94 => cat_26
# cat_95 => cat_25
# cat_96 => cat_24
# cat_97 => cat_23
# cat_98 => cat_22
# cat_99 => cat_21
# l__mod___features_denseblock3_denselayer1_norm1 => add_79, mul_118, mul_119, sub_39
# l__mod___features_denseblock3_denselayer1_relu1 => relu_39
# l__mod___features_transition2_pool => avg_pool2d_1
triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: '*fp32', 28: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(28,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 14
    x1 = (xindex // 14)
    x2 = xindex
    x4 = (xindex // 196) % 256
    x5 = (xindex // 50176)
    x6 = xindex % 50176
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (56*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (56*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (28 + (2*x0) + (56*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (29 + (2*x0) + (56*x1)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr3 + (x4), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x4), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sqrt(tmp13)
    tmp15 = 1 / tmp14
    tmp16 = 1.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp10 * tmp17
    tmp20 = tmp18 * tmp19
    tmp22 = tmp20 + tmp21
    tmp23 = triton_helpers.maximum(0, tmp22)
    tl.store(out_ptr0 + (x2), tmp8, None)
    tl.store(out_ptr1 + (x2), tmp23, None)
    tl.store(out_ptr2 + (x6 + (75264*x5)), tmp8, None)
    tl.store(out_ptr3 + (x6 + (81536*x5)), tmp8, None)
    tl.store(out_ptr4 + (x6 + (87808*x5)), tmp8, None)
    tl.store(out_ptr5 + (x6 + (94080*x5)), tmp8, None)
    tl.store(out_ptr6 + (x6 + (100352*x5)), tmp8, None)
    tl.store(out_ptr7 + (x6 + (106624*x5)), tmp8, None)
    tl.store(out_ptr8 + (x6 + (112896*x5)), tmp8, None)
    tl.store(out_ptr9 + (x6 + (119168*x5)), tmp8, None)
    tl.store(out_ptr10 + (x6 + (125440*x5)), tmp8, None)
    tl.store(out_ptr11 + (x6 + (131712*x5)), tmp8, None)
    tl.store(out_ptr12 + (x6 + (137984*x5)), tmp8, None)
    tl.store(out_ptr13 + (x6 + (144256*x5)), tmp8, None)
    tl.store(out_ptr14 + (x6 + (150528*x5)), tmp8, None)
    tl.store(out_ptr15 + (x6 + (156800*x5)), tmp8, None)
    tl.store(out_ptr16 + (x6 + (163072*x5)), tmp8, None)
    tl.store(out_ptr17 + (x6 + (169344*x5)), tmp8, None)
    tl.store(out_ptr18 + (x6 + (175616*x5)), tmp8, None)
    tl.store(out_ptr19 + (x6 + (181888*x5)), tmp8, None)
    tl.store(out_ptr20 + (x6 + (188160*x5)), tmp8, None)
    tl.store(out_ptr21 + (x6 + (194432*x5)), tmp8, None)
    tl.store(out_ptr22 + (x6 + (200704*x5)), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/lc/clctws3feabog3bj7pixry2xrqdc2fln2x2jwcy7tw3dlyfdnqil.py
# Source Nodes: [l__mod___features_denseblock3_denselayer1_norm2, l__mod___features_denseblock3_denselayer1_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock3_denselayer1_norm2 => add_81, mul_121, mul_122, sub_40
# l__mod___features_denseblock3_denselayer1_relu2 => relu_40
triton_poi_fused__native_batch_norm_legit_no_training_relu_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 128
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mx/cmxazeb656v3q3qpc7lfnfjmxri3lr5j7fborbtug45mfe6xce6i.py
# Source Nodes: [cat_102, l__mod___features_denseblock3_denselayer2_norm1, l__mod___features_denseblock3_denselayer2_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_102 => cat_18
# l__mod___features_denseblock3_denselayer2_norm1 => add_83, mul_124, mul_125, sub_41
# l__mod___features_denseblock3_denselayer2_relu1 => relu_41
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_38 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 225792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 288
    x2 = (xindex // 56448)
    x3 = xindex % 56448
    x4 = xindex
    tmp15 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (50176*x2)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 288, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-50176) + x3 + (6272*x2)), tmp8 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tmp14 - tmp15
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = 1.0
    tmp23 = tmp21 * tmp22
    tmp24 = tmp16 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(out_ptr0 + (x4), tmp14, xmask)
    tl.store(out_ptr1 + (x4), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wu/cwuhjprzuzlh63eti4adhlcd6vroxkkinm7n2neyigthgskqytyh.py
# Source Nodes: [cat_101, l__mod___features_denseblock3_denselayer3_norm1, l__mod___features_denseblock3_denselayer3_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_101 => cat_19
# l__mod___features_denseblock3_denselayer3_norm1 => add_87, mul_130, mul_131, sub_43
# l__mod___features_denseblock3_denselayer3_relu1 => relu_43
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 320
    x2 = (xindex // 62720)
    x3 = xindex % 62720
    x4 = xindex
    tmp23 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (50176*x2)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 288, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-50176) + x3 + (6272*x2)), tmp11 & xmask, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 320, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-56448) + x3 + (6272*x2)), tmp15 & xmask, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tmp24 = tmp22 - tmp23
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = tl.sqrt(tmp27)
    tmp29 = 1 / tmp28
    tmp30 = 1.0
    tmp31 = tmp29 * tmp30
    tmp32 = tmp24 * tmp31
    tmp34 = tmp32 * tmp33
    tmp36 = tmp34 + tmp35
    tmp37 = triton_helpers.maximum(0, tmp36)
    tl.store(out_ptr0 + (x4), tmp22, xmask)
    tl.store(out_ptr1 + (x4), tmp37, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hv/chv5du45qc7l67oirqgzshbmtt66byklw4svyaeju3wc5mblhzzj.py
# Source Nodes: [cat_100, l__mod___features_denseblock3_denselayer4_norm1, l__mod___features_denseblock3_denselayer4_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_100 => cat_20
# l__mod___features_denseblock3_denselayer4_norm1 => add_91, mul_136, mul_137, sub_45
# l__mod___features_denseblock3_denselayer4_relu1 => relu_45
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_40 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 275968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 352
    x2 = (xindex // 68992)
    x3 = xindex % 68992
    x4 = xindex
    tmp31 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (50176*x2)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 288, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-50176) + x3 + (6272*x2)), tmp11 & xmask, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 320, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-56448) + x3 + (6272*x2)), tmp18 & xmask, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1], 352, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr3 + ((-62720) + x3 + (6272*x2)), tmp22 & xmask, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tl.where(tmp18, tmp21, tmp27)
    tmp29 = tl.where(tmp11, tmp14, tmp28)
    tmp30 = tl.where(tmp4, tmp7, tmp29)
    tmp32 = tmp30 - tmp31
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.sqrt(tmp35)
    tmp37 = 1 / tmp36
    tmp38 = 1.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp32 * tmp39
    tmp42 = tmp40 * tmp41
    tmp44 = tmp42 + tmp43
    tmp45 = triton_helpers.maximum(0, tmp44)
    tl.store(out_ptr0 + (x4), tmp30, xmask)
    tl.store(out_ptr1 + (x4), tmp45, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/za/czado4zwrsdqaswto3cmns4n2u6ojj2vrsnyilltppu6ujvrg4qg.py
# Source Nodes: [cat_94, cat_95, cat_96, cat_97, cat_98, cat_99], Original ATen: [aten.cat]
# cat_94 => cat_26
# cat_95 => cat_25
# cat_96 => cat_24
# cat_97 => cat_23
# cat_98 => cat_22
# cat_99 => cat_21
triton_poi_fused_cat_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6272
    x1 = (xindex // 6272)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (75264*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (81536*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (87808*x1)), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + (94080*x1)), tmp0, xmask)
    tl.store(out_ptr4 + (x0 + (100352*x1)), tmp0, xmask)
    tl.store(out_ptr5 + (x0 + (106624*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t5/ct5kb7ocxbxof7xtwb2yh4c4inzjiniswkvnit4y73ng2rsof5gl.py
# Source Nodes: [l__mod___features_denseblock3_denselayer5_norm1, l__mod___features_denseblock3_denselayer5_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock3_denselayer5_norm1 => add_95, mul_142, mul_143, sub_47
# l__mod___features_denseblock3_denselayer5_relu1 => relu_47
triton_poi_fused__native_batch_norm_legit_no_training_relu_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 384
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ru/cruy2cs536cp3u3dmyqwcx3nyvah5bw2s3un4y76u3vfmhwcqcme.py
# Source Nodes: [cat_94, cat_95, cat_96, cat_97, cat_98], Original ATen: [aten.cat]
# cat_94 => cat_26
# cat_95 => cat_25
# cat_96 => cat_24
# cat_97 => cat_23
# cat_98 => cat_22
triton_poi_fused_cat_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6272
    x1 = (xindex // 6272)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (81536*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (87808*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (94080*x1)), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + (100352*x1)), tmp0, xmask)
    tl.store(out_ptr4 + (x0 + (106624*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mn/cmnvyp3s3r3r3dj3h2tbbt2vanz56jgyhy4hu5fcvukfyhovcegs.py
# Source Nodes: [l__mod___features_denseblock3_denselayer6_norm1, l__mod___features_denseblock3_denselayer6_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock3_denselayer6_norm1 => add_99, mul_148, mul_149, sub_49
# l__mod___features_denseblock3_denselayer6_relu1 => relu_49
triton_poi_fused__native_batch_norm_legit_no_training_relu_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 326144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 416
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t4/ct4kdjjt4lrqeabprecaja5tuf67kyraapl6mpaz6b7czijuc2wt.py
# Source Nodes: [cat_93, cat_94, cat_95, cat_96, cat_97], Original ATen: [aten.cat]
# cat_93 => cat_27
# cat_94 => cat_26
# cat_95 => cat_25
# cat_96 => cat_24
# cat_97 => cat_23
triton_poi_fused_cat_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6272
    x1 = (xindex // 6272)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (87808*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (94080*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (100352*x1)), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + (106624*x1)), tmp0, xmask)
    tl.store(out_ptr4 + (x0 + (112896*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dg/cdgb6h4tex5acjw2tgiuhz2vq7hggpaqg7qhigqjc62t5farag24.py
# Source Nodes: [l__mod___features_denseblock3_denselayer7_norm1, l__mod___features_denseblock3_denselayer7_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock3_denselayer7_norm1 => add_103, mul_154, mul_155, sub_51
# l__mod___features_denseblock3_denselayer7_relu1 => relu_51
triton_poi_fused__native_batch_norm_legit_no_training_relu_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 448
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bl/cblb4yvigbnqjfug5bioqprlz7c3utq32eajwmgpm3qu6ayjjn64.py
# Source Nodes: [cat_92, cat_93, cat_94, cat_95, cat_96], Original ATen: [aten.cat]
# cat_92 => cat_28
# cat_93 => cat_27
# cat_94 => cat_26
# cat_95 => cat_25
# cat_96 => cat_24
triton_poi_fused_cat_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6272
    x1 = (xindex // 6272)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (94080*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (100352*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (106624*x1)), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + (112896*x1)), tmp0, xmask)
    tl.store(out_ptr4 + (x0 + (119168*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o2/co2fb4pneyatypwbzuvcrurvlyywkr22yrz4dapyr67pu6zbfsa7.py
# Source Nodes: [l__mod___features_denseblock3_denselayer8_norm1, l__mod___features_denseblock3_denselayer8_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock3_denselayer8_norm1 => add_107, mul_160, mul_161, sub_53
# l__mod___features_denseblock3_denselayer8_relu1 => relu_53
triton_poi_fused__native_batch_norm_legit_no_training_relu_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 480
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cs/ccswqw73c7o74r3tbuyawgvudajh6zjrbfxjjp6wdrlvxkg4ha3e.py
# Source Nodes: [cat_91, cat_92, cat_93, cat_94, cat_95], Original ATen: [aten.cat]
# cat_91 => cat_29
# cat_92 => cat_28
# cat_93 => cat_27
# cat_94 => cat_26
# cat_95 => cat_25
triton_poi_fused_cat_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6272
    x1 = (xindex // 6272)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (100352*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (106624*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (112896*x1)), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + (119168*x1)), tmp0, xmask)
    tl.store(out_ptr4 + (x0 + (125440*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fu/cfupzakcqsjiqc6bej2emkzsdgnzpwoc2cab6zzxlz3gzr2pfh7v.py
# Source Nodes: [l__mod___features_denseblock3_denselayer9_norm1, l__mod___features_denseblock3_denselayer9_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock3_denselayer9_norm1 => add_111, mul_166, mul_167, sub_55
# l__mod___features_denseblock3_denselayer9_relu1 => relu_55
triton_poi_fused__native_batch_norm_legit_no_training_relu_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 512
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/la/clan4n5ped5rw4fyk4ewalxkr4gsovhmtwe7ml2s7p7mc3aiku5l.py
# Source Nodes: [cat_91, cat_92, cat_93, cat_94], Original ATen: [aten.cat]
# cat_91 => cat_29
# cat_92 => cat_28
# cat_93 => cat_27
# cat_94 => cat_26
triton_poi_fused_cat_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6272
    x1 = (xindex // 6272)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (106624*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (112896*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (119168*x1)), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + (125440*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o2/co2nxz6jcnnjwj5tyatmsd7pbxzwxlnqrdd4oik5urrryqgmp4z4.py
# Source Nodes: [l__mod___features_denseblock3_denselayer10_norm1, l__mod___features_denseblock3_denselayer10_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock3_denselayer10_norm1 => add_115, mul_172, mul_173, sub_57
# l__mod___features_denseblock3_denselayer10_relu1 => relu_57
triton_poi_fused__native_batch_norm_legit_no_training_relu_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_52', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 426496
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 544
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rw/crw5rbe7w7h7xiqdhkd5ty4l4xxkmuz75sbsyajjxpgisyd6rlvu.py
# Source Nodes: [cat_90, cat_91, cat_92, cat_93], Original ATen: [aten.cat]
# cat_90 => cat_30
# cat_91 => cat_29
# cat_92 => cat_28
# cat_93 => cat_27
triton_poi_fused_cat_53 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_53', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6272
    x1 = (xindex // 6272)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (112896*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (119168*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (125440*x1)), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + (131712*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bs/cbsudtlutj2tpfzrqd2hbpm7dacqc53gbnnq3h47hmc5xac3d6gn.py
# Source Nodes: [l__mod___features_denseblock3_denselayer11_norm1, l__mod___features_denseblock3_denselayer11_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock3_denselayer11_norm1 => add_119, mul_178, mul_179, sub_59
# l__mod___features_denseblock3_denselayer11_relu1 => relu_59
triton_poi_fused__native_batch_norm_legit_no_training_relu_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 576
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k3/ck3mmfvn6xt56j25bmsulgch5mr32cdu23ch2kjelaegmq5ffmbu.py
# Source Nodes: [cat_89, cat_90, cat_91, cat_92], Original ATen: [aten.cat]
# cat_89 => cat_31
# cat_90 => cat_30
# cat_91 => cat_29
# cat_92 => cat_28
triton_poi_fused_cat_55 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_55', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6272
    x1 = (xindex // 6272)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (119168*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (125440*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (131712*x1)), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + (137984*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/r7/cr7umaje5lzkqe6ybti44tfn4mouko24r7cd4uav44pbjenueern.py
# Source Nodes: [l__mod___features_denseblock3_denselayer12_norm1, l__mod___features_denseblock3_denselayer12_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock3_denselayer12_norm1 => add_123, mul_184, mul_185, sub_61
# l__mod___features_denseblock3_denselayer12_relu1 => relu_61
triton_poi_fused__native_batch_norm_legit_no_training_relu_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_56', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 476672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 608
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rd/crdni2voucir4cuy22xgpk4qye4htpchmcuyq7swfybrvcqgfgnc.py
# Source Nodes: [cat_88, cat_89, cat_90, cat_91], Original ATen: [aten.cat]
# cat_88 => cat_32
# cat_89 => cat_31
# cat_90 => cat_30
# cat_91 => cat_29
triton_poi_fused_cat_57 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_57', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6272
    x1 = (xindex // 6272)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (125440*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (131712*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (137984*x1)), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + (144256*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zg/czgspvqnpngds3ugsyqz3qrpplaicoajhwouz33bwstsydye32jc.py
# Source Nodes: [l__mod___features_denseblock3_denselayer13_norm1, l__mod___features_denseblock3_denselayer13_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock3_denselayer13_norm1 => add_127, mul_190, mul_191, sub_63
# l__mod___features_denseblock3_denselayer13_relu1 => relu_63
triton_poi_fused__native_batch_norm_legit_no_training_relu_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_58', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 640
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/45/c45rdme4yy47gun3flx5nd4suhvujtvgyefaqfatahixvknt5mwe.py
# Source Nodes: [cat_87, cat_88, cat_89, cat_90], Original ATen: [aten.cat]
# cat_87 => cat_33
# cat_88 => cat_32
# cat_89 => cat_31
# cat_90 => cat_30
triton_poi_fused_cat_59 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_59', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6272
    x1 = (xindex // 6272)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (131712*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (137984*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (144256*x1)), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + (150528*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hg/chg4zuzovvnfrh2ubo5nkxfozkseux2ybolrzx22nud23kqdzpqp.py
# Source Nodes: [l__mod___features_denseblock3_denselayer14_norm1, l__mod___features_denseblock3_denselayer14_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock3_denselayer14_norm1 => add_131, mul_196, mul_197, sub_65
# l__mod___features_denseblock3_denselayer14_relu1 => relu_65
triton_poi_fused__native_batch_norm_legit_no_training_relu_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_60', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 526848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 672
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rk/crk652urltxaocveyhxthatdvuhpqhaa42wzizqn3m4nngt7fvgm.py
# Source Nodes: [cat_86, cat_87, cat_88, cat_89], Original ATen: [aten.cat]
# cat_86 => cat_34
# cat_87 => cat_33
# cat_88 => cat_32
# cat_89 => cat_31
triton_poi_fused_cat_61 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_61', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6272
    x1 = (xindex // 6272)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (137984*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (144256*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (150528*x1)), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + (156800*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/el/cel4qrot2zes3tn55677qlc3r5tkv5t453naeovnqe3yhfegtzt5.py
# Source Nodes: [l__mod___features_denseblock3_denselayer15_norm1, l__mod___features_denseblock3_denselayer15_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock3_denselayer15_norm1 => add_135, mul_202, mul_203, sub_67
# l__mod___features_denseblock3_denselayer15_relu1 => relu_67
triton_poi_fused__native_batch_norm_legit_no_training_relu_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_62', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 551936
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 704
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lx/clxvu7f2b5hzguwxnu47busuvgrcwu6yyheceourbppzr2ypq4zt.py
# Source Nodes: [cat_86, cat_87, cat_88], Original ATen: [aten.cat]
# cat_86 => cat_34
# cat_87 => cat_33
# cat_88 => cat_32
triton_poi_fused_cat_63 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_63', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6272
    x1 = (xindex // 6272)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (144256*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (150528*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (156800*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cu/ccuexpsdx2qr3tktj6thh52jl5ere33a3sjz4hq7utbpfbyaul7m.py
# Source Nodes: [l__mod___features_denseblock3_denselayer16_norm1, l__mod___features_denseblock3_denselayer16_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock3_denselayer16_norm1 => add_139, mul_208, mul_209, sub_69
# l__mod___features_denseblock3_denselayer16_relu1 => relu_69
triton_poi_fused__native_batch_norm_legit_no_training_relu_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_64', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 577024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 736
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uu/cuu44ijxyfwryqokefxgpvhrwwhs7moejp7yht7e53bbxajp3nyf.py
# Source Nodes: [cat_85, cat_86, cat_87], Original ATen: [aten.cat]
# cat_85 => cat_35
# cat_86 => cat_34
# cat_87 => cat_33
triton_poi_fused_cat_65 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_65', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6272
    x1 = (xindex // 6272)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (150528*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (156800*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (163072*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bk/cbkf76dspj3smnvnaadhtqd2za4izvsg7p7zpw3ot6euyrk7br4e.py
# Source Nodes: [l__mod___features_denseblock3_denselayer17_norm1, l__mod___features_denseblock3_denselayer17_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock3_denselayer17_norm1 => add_143, mul_214, mul_215, sub_71
# l__mod___features_denseblock3_denselayer17_relu1 => relu_71
triton_poi_fused__native_batch_norm_legit_no_training_relu_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_66', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 768
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ef/cefh5nynitdoxci4rhn5mrnriu4vwhokzy35kiosxz33m6pd5f5y.py
# Source Nodes: [cat_84, cat_85, cat_86], Original ATen: [aten.cat]
# cat_84 => cat_36
# cat_85 => cat_35
# cat_86 => cat_34
triton_poi_fused_cat_67 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_67', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6272
    x1 = (xindex // 6272)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (156800*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (163072*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (169344*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/32/c32zsi2hhrl2xehf4vmqj3hizhv5dbamzp45txqdrrylsofzianm.py
# Source Nodes: [l__mod___features_denseblock3_denselayer18_norm1, l__mod___features_denseblock3_denselayer18_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock3_denselayer18_norm1 => add_147, mul_220, mul_221, sub_73
# l__mod___features_denseblock3_denselayer18_relu1 => relu_73
triton_poi_fused__native_batch_norm_legit_no_training_relu_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_68', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 627200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 800
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dv/cdvhnaeirifvc6n5hv2jcgusw4fyql375tpdiwe6euvwgydlcqoj.py
# Source Nodes: [cat_83, cat_84, cat_85], Original ATen: [aten.cat]
# cat_83 => cat_37
# cat_84 => cat_36
# cat_85 => cat_35
triton_poi_fused_cat_69 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_69', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6272
    x1 = (xindex // 6272)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (163072*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (169344*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (175616*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jz/cjz6uo64uwtp2rognmgrri6ue6nnqsj326wne2b3dohnopor7ezr.py
# Source Nodes: [l__mod___features_denseblock3_denselayer19_norm1, l__mod___features_denseblock3_denselayer19_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock3_denselayer19_norm1 => add_151, mul_226, mul_227, sub_75
# l__mod___features_denseblock3_denselayer19_relu1 => relu_75
triton_poi_fused__native_batch_norm_legit_no_training_relu_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_70', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 652288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 832
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bg/cbgqxq4nukr3ufq6bmy7tcywfp7mqrb6j5o3vzvhjjsj442zyqfb.py
# Source Nodes: [cat_82, cat_83, cat_84], Original ATen: [aten.cat]
# cat_82 => cat_38
# cat_83 => cat_37
# cat_84 => cat_36
triton_poi_fused_cat_71 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_71', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6272
    x1 = (xindex // 6272)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (169344*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (175616*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (181888*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/al/calrqvj4w5qtdszrcsm77j5lxklmwlbiflq3scihmlyzejyqfznu.py
# Source Nodes: [l__mod___features_denseblock3_denselayer20_norm1, l__mod___features_denseblock3_denselayer20_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock3_denselayer20_norm1 => add_155, mul_232, mul_233, sub_77
# l__mod___features_denseblock3_denselayer20_relu1 => relu_77
triton_poi_fused__native_batch_norm_legit_no_training_relu_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_72', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 677376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 864
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4z/c4z7p3hpvdpr7janfd57oidd3llryeyyywrk57hazvrf27w6rwwx.py
# Source Nodes: [cat_81, cat_82, cat_83], Original ATen: [aten.cat]
# cat_81 => cat_39
# cat_82 => cat_38
# cat_83 => cat_37
triton_poi_fused_cat_73 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_73', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6272
    x1 = (xindex // 6272)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (175616*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (181888*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (188160*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lt/cltdpcd7i3hh35g5raq5cosexmh5ine7uivpxkbvkjidld6w4tlp.py
# Source Nodes: [l__mod___features_denseblock3_denselayer21_norm1, l__mod___features_denseblock3_denselayer21_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock3_denselayer21_norm1 => add_159, mul_238, mul_239, sub_79
# l__mod___features_denseblock3_denselayer21_relu1 => relu_79
triton_poi_fused__native_batch_norm_legit_no_training_relu_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_74', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 702464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 896
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kn/ckn3qmlwscakg6ehs7qwfg4fq46gabic3fyn55htq6c3xivz3n33.py
# Source Nodes: [cat_80, cat_81, cat_82], Original ATen: [aten.cat]
# cat_80 => cat_40
# cat_81 => cat_39
# cat_82 => cat_38
triton_poi_fused_cat_75 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_75', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6272
    x1 = (xindex // 6272)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (181888*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (188160*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (194432*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6a/c6acazlq62ldmbxa3j4thvrv4v7t2jpemecx7ykqmrazyly4eo5g.py
# Source Nodes: [l__mod___features_denseblock3_denselayer22_norm1, l__mod___features_denseblock3_denselayer22_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock3_denselayer22_norm1 => add_163, mul_244, mul_245, sub_81
# l__mod___features_denseblock3_denselayer22_relu1 => relu_81
triton_poi_fused__native_batch_norm_legit_no_training_relu_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_76', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 727552
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 928
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wp/cwpl5xq6g7ivc7z2vyjlcgtm45j3gndalswxhvrcivspsmqeduqv.py
# Source Nodes: [cat_79, cat_80, cat_81], Original ATen: [aten.cat]
# cat_79 => cat_41
# cat_80 => cat_40
# cat_81 => cat_39
triton_poi_fused_cat_77 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_77', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6272
    x1 = (xindex // 6272)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (188160*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (194432*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (200704*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/er/cer3zkbwkf7m7iwfy4ejna7lihy5gpnx4nn5h2dcxnopw5iesjrt.py
# Source Nodes: [l__mod___features_denseblock3_denselayer23_norm1, l__mod___features_denseblock3_denselayer23_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock3_denselayer23_norm1 => add_167, mul_250, mul_251, sub_83
# l__mod___features_denseblock3_denselayer23_relu1 => relu_83
triton_poi_fused__native_batch_norm_legit_no_training_relu_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_78', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 960
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fs/cfsuomp455e5o7m7nmwnzjheh7kb5kfud3g7ehtppjogntdumavh.py
# Source Nodes: [cat_79, cat_80], Original ATen: [aten.cat]
# cat_79 => cat_41
# cat_80 => cat_40
triton_poi_fused_cat_79 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_79', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6272
    x1 = (xindex // 6272)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (194432*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (200704*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mb/cmbb4fe4oum3zprx3lffe3mgfrf6x2p6x24db2zpf4sfcsoo56ff.py
# Source Nodes: [l__mod___features_denseblock3_denselayer24_norm1, l__mod___features_denseblock3_denselayer24_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock3_denselayer24_norm1 => add_171, mul_256, mul_257, sub_85
# l__mod___features_denseblock3_denselayer24_relu1 => relu_85
triton_poi_fused__native_batch_norm_legit_no_training_relu_80 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_80', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 777728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 992
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pi/cpig75d4kmix657u5fde4ukyconhl3dcsxwzadbrxys7ynbuyz4c.py
# Source Nodes: [cat_79], Original ATen: [aten.cat]
# cat_79 => cat_41
triton_poi_fused_cat_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_81', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 6272
    x1 = (xindex // 6272)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (200704*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6o/c6ojib6d5twecdsl3snszpshso2aqvmlt5j5musmosig52fbl2ub.py
# Source Nodes: [l__mod___features_transition3_norm, l__mod___features_transition3_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_transition3_norm => add_175, mul_262, mul_263, sub_87
# l__mod___features_transition3_relu => relu_87
triton_poi_fused__native_batch_norm_legit_no_training_relu_82 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_82', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1024
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4u/c4ujio4hopwa6fwuowj7sujoyzwtdiz6gi3ecan3etvw6rjyj7z6.py
# Source Nodes: [cat_62, cat_63, cat_64, cat_65, cat_66, cat_67, cat_68, cat_69, cat_70, cat_71, cat_72, cat_73, cat_74, l__mod___features_denseblock4_denselayer1_norm1, l__mod___features_denseblock4_denselayer1_relu1, l__mod___features_transition3_pool], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d, aten.cat, aten.relu]
# cat_62 => cat_57
# cat_63 => cat_56
# cat_64 => cat_55
# cat_65 => cat_54
# cat_66 => cat_53
# cat_67 => cat_52
# cat_68 => cat_51
# cat_69 => cat_50
# cat_70 => cat_49
# cat_71 => cat_48
# cat_72 => cat_47
# cat_73 => cat_46
# cat_74 => cat_45
# l__mod___features_denseblock4_denselayer1_norm1 => add_177, mul_265, mul_266, sub_88
# l__mod___features_denseblock4_denselayer1_relu1 => relu_88
# l__mod___features_transition3_pool => avg_pool2d_2
triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_83 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(20,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_83', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 7
    x1 = (xindex // 7)
    x2 = xindex
    x4 = (xindex // 49) % 512
    x5 = (xindex // 25088)
    x6 = xindex % 25088
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (28*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (28*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (14 + (2*x0) + (28*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (15 + (2*x0) + (28*x1)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x4), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x4), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr3 + (x4), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x4), None, eviction_policy='evict_last')
    tmp2 = tmp1 + tmp0
    tmp4 = tmp3 + tmp2
    tmp6 = tmp5 + tmp4
    tmp7 = 0.25
    tmp8 = tmp6 * tmp7
    tmp10 = tmp8 - tmp9
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.sqrt(tmp13)
    tmp15 = 1 / tmp14
    tmp16 = 1.0
    tmp17 = tmp15 * tmp16
    tmp18 = tmp10 * tmp17
    tmp20 = tmp18 * tmp19
    tmp22 = tmp20 + tmp21
    tmp23 = triton_helpers.maximum(0, tmp22)
    tl.store(out_ptr0 + (x2), tmp8, None)
    tl.store(out_ptr1 + (x2), tmp23, None)
    tl.store(out_ptr2 + (x6 + (31360*x5)), tmp8, None)
    tl.store(out_ptr3 + (x6 + (32928*x5)), tmp8, None)
    tl.store(out_ptr4 + (x6 + (34496*x5)), tmp8, None)
    tl.store(out_ptr5 + (x6 + (36064*x5)), tmp8, None)
    tl.store(out_ptr6 + (x6 + (37632*x5)), tmp8, None)
    tl.store(out_ptr7 + (x6 + (39200*x5)), tmp8, None)
    tl.store(out_ptr8 + (x6 + (40768*x5)), tmp8, None)
    tl.store(out_ptr9 + (x6 + (42336*x5)), tmp8, None)
    tl.store(out_ptr10 + (x6 + (43904*x5)), tmp8, None)
    tl.store(out_ptr11 + (x6 + (45472*x5)), tmp8, None)
    tl.store(out_ptr12 + (x6 + (47040*x5)), tmp8, None)
    tl.store(out_ptr13 + (x6 + (48608*x5)), tmp8, None)
    tl.store(out_ptr14 + (x6 + (50176*x5)), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/65/c65edjcrhsu2oy4xgdq4yqt4q5wt5vrbbxd6sspnwyrpcb5z3f4g.py
# Source Nodes: [l__mod___features_denseblock4_denselayer1_norm2, l__mod___features_denseblock4_denselayer1_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock4_denselayer1_norm2 => add_179, mul_268, mul_269, sub_89
# l__mod___features_denseblock4_denselayer1_relu2 => relu_89
triton_poi_fused__native_batch_norm_legit_no_training_relu_84 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_84', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 128
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hv/chvzitdj65c5sltd7ofz3wb7bnzyttkjvuplxmgtjohjpok7lbyq.py
# Source Nodes: [cat_77, l__mod___features_denseblock4_denselayer2_norm1, l__mod___features_denseblock4_denselayer2_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_77 => cat_42
# l__mod___features_denseblock4_denselayer2_norm1 => add_181, mul_271, mul_272, sub_90
# l__mod___features_denseblock4_denselayer2_relu1 => relu_90
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_85 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_85', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 106624
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 49) % 544
    x2 = (xindex // 26656)
    x3 = xindex % 26656
    x4 = xindex
    tmp15 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (25088*x2)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 544, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-25088) + x3 + (1568*x2)), tmp8 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp8, tmp11, tmp12)
    tmp14 = tl.where(tmp4, tmp7, tmp13)
    tmp16 = tmp14 - tmp15
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = 1.0
    tmp23 = tmp21 * tmp22
    tmp24 = tmp16 * tmp23
    tmp26 = tmp24 * tmp25
    tmp28 = tmp26 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(out_ptr0 + (x4), tmp14, xmask)
    tl.store(out_ptr1 + (x4), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oc/cocshy6snkn53y5s4ol2y4eiv5umhagcnltyga6lexsuxdi43oxj.py
# Source Nodes: [cat_76, l__mod___features_denseblock4_denselayer3_norm1, l__mod___features_denseblock4_denselayer3_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_76 => cat_43
# l__mod___features_denseblock4_denselayer3_norm1 => add_185, mul_277, mul_278, sub_92
# l__mod___features_denseblock4_denselayer3_relu1 => relu_92
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_86 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_86', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 49) % 576
    x2 = (xindex // 28224)
    x3 = xindex % 28224
    x4 = xindex
    tmp23 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (25088*x2)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 544, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-25088) + x3 + (1568*x2)), tmp11 & xmask, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 576, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-26656) + x3 + (1568*x2)), tmp15 & xmask, other=0.0)
    tmp19 = tl.full(tmp18.shape, 0.0, tmp18.dtype)
    tmp20 = tl.where(tmp15, tmp18, tmp19)
    tmp21 = tl.where(tmp11, tmp14, tmp20)
    tmp22 = tl.where(tmp4, tmp7, tmp21)
    tmp24 = tmp22 - tmp23
    tmp26 = 1e-05
    tmp27 = tmp25 + tmp26
    tmp28 = tl.sqrt(tmp27)
    tmp29 = 1 / tmp28
    tmp30 = 1.0
    tmp31 = tmp29 * tmp30
    tmp32 = tmp24 * tmp31
    tmp34 = tmp32 * tmp33
    tmp36 = tmp34 + tmp35
    tmp37 = triton_helpers.maximum(0, tmp36)
    tl.store(out_ptr0 + (x4), tmp22, xmask)
    tl.store(out_ptr1 + (x4), tmp37, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i7/ci7kg4hfwxq2dlypzfa35r4uzhavgnzba3owa3cm46f3f2xfnoch.py
# Source Nodes: [cat_75, l__mod___features_denseblock4_denselayer4_norm1, l__mod___features_denseblock4_denselayer4_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_75 => cat_44
# l__mod___features_denseblock4_denselayer4_norm1 => add_189, mul_283, mul_284, sub_94
# l__mod___features_denseblock4_denselayer4_relu1 => relu_94
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_87 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_87', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 119168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 49) % 608
    x2 = (xindex // 29792)
    x3 = xindex % 29792
    x4 = xindex
    tmp31 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr6 + (x1), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr7 + (x1), xmask, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (25088*x2)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 544, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-25088) + x3 + (1568*x2)), tmp11 & xmask, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 576, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-26656) + x3 + (1568*x2)), tmp18 & xmask, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp0 >= tmp16
    tmp23 = tl.full([1], 608, tl.int64)
    tmp24 = tmp0 < tmp23
    tmp25 = tl.load(in_ptr3 + ((-28224) + x3 + (1568*x2)), tmp22 & xmask, other=0.0)
    tmp26 = tl.full(tmp25.shape, 0.0, tmp25.dtype)
    tmp27 = tl.where(tmp22, tmp25, tmp26)
    tmp28 = tl.where(tmp18, tmp21, tmp27)
    tmp29 = tl.where(tmp11, tmp14, tmp28)
    tmp30 = tl.where(tmp4, tmp7, tmp29)
    tmp32 = tmp30 - tmp31
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.sqrt(tmp35)
    tmp37 = 1 / tmp36
    tmp38 = 1.0
    tmp39 = tmp37 * tmp38
    tmp40 = tmp32 * tmp39
    tmp42 = tmp40 * tmp41
    tmp44 = tmp42 + tmp43
    tmp45 = triton_helpers.maximum(0, tmp44)
    tl.store(out_ptr0 + (x4), tmp30, xmask)
    tl.store(out_ptr1 + (x4), tmp45, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k5/ck56yxmnghflpurnzhm53jtz3avptermrjcqwdgns7m3e3biehv7.py
# Source Nodes: [cat_69, cat_70, cat_71, cat_72, cat_73, cat_74], Original ATen: [aten.cat]
# cat_69 => cat_50
# cat_70 => cat_49
# cat_71 => cat_48
# cat_72 => cat_47
# cat_73 => cat_46
# cat_74 => cat_45
triton_poi_fused_cat_88 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_88', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1568
    x1 = (xindex // 1568)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (31360*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (32928*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (34496*x1)), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + (36064*x1)), tmp0, xmask)
    tl.store(out_ptr4 + (x0 + (37632*x1)), tmp0, xmask)
    tl.store(out_ptr5 + (x0 + (39200*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bm/cbm2afj7rdkpcieefgkpvfwvokxr36hs42mzdhpdlv7sfmu4oaic.py
# Source Nodes: [l__mod___features_denseblock4_denselayer5_norm1, l__mod___features_denseblock4_denselayer5_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock4_denselayer5_norm1 => add_193, mul_289, mul_290, sub_96
# l__mod___features_denseblock4_denselayer5_relu1 => relu_96
triton_poi_fused__native_batch_norm_legit_no_training_relu_89 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_89', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 640
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ai/cailczygkr23qvl62y32gs34u4cog44wxt5uqhcbuhlxt4emjhx4.py
# Source Nodes: [cat_69, cat_70, cat_71, cat_72, cat_73], Original ATen: [aten.cat]
# cat_69 => cat_50
# cat_70 => cat_49
# cat_71 => cat_48
# cat_72 => cat_47
# cat_73 => cat_46
triton_poi_fused_cat_90 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_90', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1568
    x1 = (xindex // 1568)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (32928*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (34496*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (36064*x1)), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + (37632*x1)), tmp0, xmask)
    tl.store(out_ptr4 + (x0 + (39200*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7y/c7ykapyj6f3w4ok7w75qaidkqqe7r3szb6qadknohn3pfcg2iw6p.py
# Source Nodes: [l__mod___features_denseblock4_denselayer6_norm1, l__mod___features_denseblock4_denselayer6_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock4_denselayer6_norm1 => add_197, mul_295, mul_296, sub_98
# l__mod___features_denseblock4_denselayer6_relu1 => relu_98
triton_poi_fused__native_batch_norm_legit_no_training_relu_91 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_91', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131712
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 672
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/23/c23xvygbx2nkprcammovqsfpioys7wp2444kvztagl4g2cjigv7x.py
# Source Nodes: [cat_68, cat_69, cat_70, cat_71, cat_72], Original ATen: [aten.cat]
# cat_68 => cat_51
# cat_69 => cat_50
# cat_70 => cat_49
# cat_71 => cat_48
# cat_72 => cat_47
triton_poi_fused_cat_92 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_92', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1568
    x1 = (xindex // 1568)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (34496*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (36064*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (37632*x1)), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + (39200*x1)), tmp0, xmask)
    tl.store(out_ptr4 + (x0 + (40768*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/a2/ca2flbdphtgslalcza7a2zfrqwdigm4nqcx4m2f5w6aenh5wjex4.py
# Source Nodes: [l__mod___features_denseblock4_denselayer7_norm1, l__mod___features_denseblock4_denselayer7_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock4_denselayer7_norm1 => add_201, mul_301, mul_302, sub_100
# l__mod___features_denseblock4_denselayer7_relu1 => relu_100
triton_poi_fused__native_batch_norm_legit_no_training_relu_93 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_93', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 137984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 704
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yp/cyp5gqpw27ho3b6x4g5d45ua6vgyolzewlpnbgew6bjxw4bvlbms.py
# Source Nodes: [cat_67, cat_68, cat_69, cat_70, cat_71], Original ATen: [aten.cat]
# cat_67 => cat_52
# cat_68 => cat_51
# cat_69 => cat_50
# cat_70 => cat_49
# cat_71 => cat_48
triton_poi_fused_cat_94 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_94', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1568
    x1 = (xindex // 1568)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (36064*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (37632*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (39200*x1)), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + (40768*x1)), tmp0, xmask)
    tl.store(out_ptr4 + (x0 + (42336*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5b/c5bmtyny7zkw5ja6df65jhvvhmyxrv22bdpj73teufrdu4thhwvz.py
# Source Nodes: [l__mod___features_denseblock4_denselayer8_norm1, l__mod___features_denseblock4_denselayer8_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock4_denselayer8_norm1 => add_205, mul_307, mul_308, sub_102
# l__mod___features_denseblock4_denselayer8_relu1 => relu_102
triton_poi_fused__native_batch_norm_legit_no_training_relu_95 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_95', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 736
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6s/c6sl3heljk6uh2e3pa7dare37p5udv65wvb7bq37dugomjoahuae.py
# Source Nodes: [cat_66, cat_67, cat_68, cat_69, cat_70], Original ATen: [aten.cat]
# cat_66 => cat_53
# cat_67 => cat_52
# cat_68 => cat_51
# cat_69 => cat_50
# cat_70 => cat_49
triton_poi_fused_cat_96 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_96', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1568
    x1 = (xindex // 1568)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (37632*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (39200*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (40768*x1)), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + (42336*x1)), tmp0, xmask)
    tl.store(out_ptr4 + (x0 + (43904*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/77/c77lodcjjnrpjm35nqu32nvfx25qtcs3wh237g5tjlekzquxwvxq.py
# Source Nodes: [l__mod___features_denseblock4_denselayer9_norm1, l__mod___features_denseblock4_denselayer9_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock4_denselayer9_norm1 => add_209, mul_313, mul_314, sub_104
# l__mod___features_denseblock4_denselayer9_relu1 => relu_104
triton_poi_fused__native_batch_norm_legit_no_training_relu_97 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_97', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 768
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v2/cv2j3qv5kfmhtznzb5bishpxqioizmg37usnpj6ygudrqifwmtyp.py
# Source Nodes: [cat_66, cat_67, cat_68, cat_69], Original ATen: [aten.cat]
# cat_66 => cat_53
# cat_67 => cat_52
# cat_68 => cat_51
# cat_69 => cat_50
triton_poi_fused_cat_98 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_98', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1568
    x1 = (xindex // 1568)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (39200*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (40768*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (42336*x1)), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + (43904*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w3/cw3bsjfusot72iok3r5mbdromuld5amwh7rnwuzpgbliai4rgfio.py
# Source Nodes: [l__mod___features_denseblock4_denselayer10_norm1, l__mod___features_denseblock4_denselayer10_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock4_denselayer10_norm1 => add_213, mul_319, mul_320, sub_106
# l__mod___features_denseblock4_denselayer10_relu1 => relu_106
triton_poi_fused__native_batch_norm_legit_no_training_relu_99 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_99', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 156800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 800
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ko/ckow7qhwa44knfpumuccmmyy6m2mjk3ehnipwwbpqztl5tepjhpj.py
# Source Nodes: [cat_65, cat_66, cat_67, cat_68], Original ATen: [aten.cat]
# cat_65 => cat_54
# cat_66 => cat_53
# cat_67 => cat_52
# cat_68 => cat_51
triton_poi_fused_cat_100 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_100', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1568
    x1 = (xindex // 1568)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (40768*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (42336*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (43904*x1)), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + (45472*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xk/cxkv2lzwcihlpool7zze2jkjdgdb7q3l7qyqeok63my6xx3el4b4.py
# Source Nodes: [l__mod___features_denseblock4_denselayer11_norm1, l__mod___features_denseblock4_denselayer11_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock4_denselayer11_norm1 => add_217, mul_325, mul_326, sub_108
# l__mod___features_denseblock4_denselayer11_relu1 => relu_108
triton_poi_fused__native_batch_norm_legit_no_training_relu_101 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_101', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 832
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xg/cxgbyrbzo4yghxgq2bqlpsrascuxmcjkbl4pcquwfzehxikvqcpf.py
# Source Nodes: [cat_64, cat_65, cat_66, cat_67], Original ATen: [aten.cat]
# cat_64 => cat_55
# cat_65 => cat_54
# cat_66 => cat_53
# cat_67 => cat_52
triton_poi_fused_cat_102 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_102', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1568
    x1 = (xindex // 1568)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (42336*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (43904*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (45472*x1)), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + (47040*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/md/cmdsaok5k626rvwezvcng73l43huctja2dnmbv6t7qlkg3h4dyy2.py
# Source Nodes: [l__mod___features_denseblock4_denselayer12_norm1, l__mod___features_denseblock4_denselayer12_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock4_denselayer12_norm1 => add_221, mul_331, mul_332, sub_110
# l__mod___features_denseblock4_denselayer12_relu1 => relu_110
triton_poi_fused__native_batch_norm_legit_no_training_relu_103 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_103', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 169344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 864
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mz/cmzbkqfatzz6dm76k7rmdu6rtg45xhmbgnmmy4q7d2om434zlssi.py
# Source Nodes: [cat_63, cat_64, cat_65, cat_66], Original ATen: [aten.cat]
# cat_63 => cat_56
# cat_64 => cat_55
# cat_65 => cat_54
# cat_66 => cat_53
triton_poi_fused_cat_104 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_104', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1568
    x1 = (xindex // 1568)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (43904*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (45472*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (47040*x1)), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + (48608*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hd/chdhm7ssyw3rzjqqyoexaeb3cmn74mxql7fo4wdfxliuxewvky5s.py
# Source Nodes: [l__mod___features_denseblock4_denselayer13_norm1, l__mod___features_denseblock4_denselayer13_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock4_denselayer13_norm1 => add_225, mul_337, mul_338, sub_112
# l__mod___features_denseblock4_denselayer13_relu1 => relu_112
triton_poi_fused__native_batch_norm_legit_no_training_relu_105 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_105', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 175616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 896
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7t/c7tb7qxc3s3pmaqyb6xvaq42mwcrnbteci2sm25mcqcf7pp7d2xs.py
# Source Nodes: [cat_62, cat_63, cat_64, cat_65], Original ATen: [aten.cat]
# cat_62 => cat_57
# cat_63 => cat_56
# cat_64 => cat_55
# cat_65 => cat_54
triton_poi_fused_cat_106 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_106', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1568
    x1 = (xindex // 1568)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (45472*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (47040*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (48608*x1)), tmp0, xmask)
    tl.store(out_ptr3 + (x0 + (50176*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yz/cyzpg3gmsrqvidcrs4akmx5scy264wxlakoiqwbo2yaze26ibk4g.py
# Source Nodes: [l__mod___features_denseblock4_denselayer14_norm1, l__mod___features_denseblock4_denselayer14_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock4_denselayer14_norm1 => add_229, mul_343, mul_344, sub_114
# l__mod___features_denseblock4_denselayer14_relu1 => relu_114
triton_poi_fused__native_batch_norm_legit_no_training_relu_107 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_107', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 181888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 928
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/63/c63n5pb37xqrvzk5w67kq4efq422bfh4mxnsgkizrc4izpvyb3fr.py
# Source Nodes: [cat_62, cat_63, cat_64], Original ATen: [aten.cat]
# cat_62 => cat_57
# cat_63 => cat_56
# cat_64 => cat_55
triton_poi_fused_cat_108 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_108', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1568
    x1 = (xindex // 1568)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (47040*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (48608*x1)), tmp0, xmask)
    tl.store(out_ptr2 + (x0 + (50176*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cq/ccqhzbkipauk7bphkiml3lw2qez7pwz5iivpjtfntlxz4gbac7l4.py
# Source Nodes: [l__mod___features_denseblock4_denselayer15_norm1, l__mod___features_denseblock4_denselayer15_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock4_denselayer15_norm1 => add_233, mul_349, mul_350, sub_116
# l__mod___features_denseblock4_denselayer15_relu1 => relu_116
triton_poi_fused__native_batch_norm_legit_no_training_relu_109 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_109', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 188160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 960
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x6/cx6exn3652dzityhvcocyowr4rm5etz5r3wd5ibfonmwuc3ptn3q.py
# Source Nodes: [cat_62, cat_63], Original ATen: [aten.cat]
# cat_62 => cat_57
# cat_63 => cat_56
triton_poi_fused_cat_110 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_110', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1568
    x1 = (xindex // 1568)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (48608*x1)), tmp0, xmask)
    tl.store(out_ptr1 + (x0 + (50176*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7l/c7lbj4j3xj2piz7ajrljh53gfkt7i74um6sv222qcyulwe7r4lah.py
# Source Nodes: [l__mod___features_denseblock4_denselayer16_norm1, l__mod___features_denseblock4_denselayer16_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___features_denseblock4_denselayer16_norm1 => add_237, mul_355, mul_356, sub_118
# l__mod___features_denseblock4_denselayer16_relu1 => relu_118
triton_poi_fused__native_batch_norm_legit_no_training_relu_111 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_111', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 194432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 992
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fp/cfp2ddnnk5z5f6qipwqs43y3sci3xuoagvlgted4elqu5noxoccf.py
# Source Nodes: [cat_62], Original ATen: [aten.cat]
# cat_62 => cat_57
triton_poi_fused_cat_112 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_112', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6272
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1568
    x1 = (xindex // 1568)
    tmp0 = tl.load(in_ptr0 + (x2), xmask)
    tl.store(out_ptr0 + (x0 + (50176*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fp/cfphezdukavkeoedypmbf7don4zjma5lqkxrnlqct35znflze7xe.py
# Source Nodes: [features, out, out_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu, aten.threshold_backward, aten.view]
# features => add_241, mul_361, mul_362, sub_120
# out => relu_120
# out_1 => mean
# out_2 => view
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_threshold_backward_view_113 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_threshold_backward_view_113', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 1e-05
    tmp5 = tmp3 + tmp4
    tmp6 = tl.sqrt(tmp5)
    tmp7 = 1 / tmp6
    tmp8 = 1.0
    tmp9 = tmp7 * tmp8
    tmp10 = tmp2 * tmp9
    tmp12 = tmp10 * tmp11
    tmp14 = tmp12 + tmp13
    tmp15 = triton_helpers.maximum(0, tmp14)
    tmp16 = 0.0
    tmp17 = tmp15 <= tmp16
    tmp18 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp22 = 49.0
    tmp23 = tmp21 / tmp22
    tl.store(out_ptr1 + (r2 + (49*x3)), tmp17, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp23, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (64, ), (1, ))
    assert_size_stride(primals_6, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_9, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_10, (96, ), (1, ))
    assert_size_stride(primals_11, (96, ), (1, ))
    assert_size_stride(primals_12, (128, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_15, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_16, (128, ), (1, ))
    assert_size_stride(primals_17, (128, ), (1, ))
    assert_size_stride(primals_18, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_19, (128, ), (1, ))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_21, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_22, (160, ), (1, ))
    assert_size_stride(primals_23, (160, ), (1, ))
    assert_size_stride(primals_24, (128, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_25, (128, ), (1, ))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_27, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_28, (192, ), (1, ))
    assert_size_stride(primals_29, (192, ), (1, ))
    assert_size_stride(primals_30, (128, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_31, (128, ), (1, ))
    assert_size_stride(primals_32, (128, ), (1, ))
    assert_size_stride(primals_33, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_34, (224, ), (1, ))
    assert_size_stride(primals_35, (224, ), (1, ))
    assert_size_stride(primals_36, (128, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_37, (128, ), (1, ))
    assert_size_stride(primals_38, (128, ), (1, ))
    assert_size_stride(primals_39, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_40, (256, ), (1, ))
    assert_size_stride(primals_41, (256, ), (1, ))
    assert_size_stride(primals_42, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_43, (128, ), (1, ))
    assert_size_stride(primals_44, (128, ), (1, ))
    assert_size_stride(primals_45, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_46, (128, ), (1, ))
    assert_size_stride(primals_47, (128, ), (1, ))
    assert_size_stride(primals_48, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_49, (160, ), (1, ))
    assert_size_stride(primals_50, (160, ), (1, ))
    assert_size_stride(primals_51, (128, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(primals_52, (128, ), (1, ))
    assert_size_stride(primals_53, (128, ), (1, ))
    assert_size_stride(primals_54, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_55, (192, ), (1, ))
    assert_size_stride(primals_56, (192, ), (1, ))
    assert_size_stride(primals_57, (128, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(primals_58, (128, ), (1, ))
    assert_size_stride(primals_59, (128, ), (1, ))
    assert_size_stride(primals_60, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_61, (224, ), (1, ))
    assert_size_stride(primals_62, (224, ), (1, ))
    assert_size_stride(primals_63, (128, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(primals_64, (128, ), (1, ))
    assert_size_stride(primals_65, (128, ), (1, ))
    assert_size_stride(primals_66, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_67, (256, ), (1, ))
    assert_size_stride(primals_68, (256, ), (1, ))
    assert_size_stride(primals_69, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_70, (128, ), (1, ))
    assert_size_stride(primals_71, (128, ), (1, ))
    assert_size_stride(primals_72, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_73, (288, ), (1, ))
    assert_size_stride(primals_74, (288, ), (1, ))
    assert_size_stride(primals_75, (128, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_76, (128, ), (1, ))
    assert_size_stride(primals_77, (128, ), (1, ))
    assert_size_stride(primals_78, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_79, (320, ), (1, ))
    assert_size_stride(primals_80, (320, ), (1, ))
    assert_size_stride(primals_81, (128, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_82, (128, ), (1, ))
    assert_size_stride(primals_83, (128, ), (1, ))
    assert_size_stride(primals_84, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_85, (352, ), (1, ))
    assert_size_stride(primals_86, (352, ), (1, ))
    assert_size_stride(primals_87, (128, 352, 1, 1), (352, 1, 1, 1))
    assert_size_stride(primals_88, (128, ), (1, ))
    assert_size_stride(primals_89, (128, ), (1, ))
    assert_size_stride(primals_90, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_91, (384, ), (1, ))
    assert_size_stride(primals_92, (384, ), (1, ))
    assert_size_stride(primals_93, (128, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_94, (128, ), (1, ))
    assert_size_stride(primals_95, (128, ), (1, ))
    assert_size_stride(primals_96, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_97, (416, ), (1, ))
    assert_size_stride(primals_98, (416, ), (1, ))
    assert_size_stride(primals_99, (128, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_100, (128, ), (1, ))
    assert_size_stride(primals_101, (128, ), (1, ))
    assert_size_stride(primals_102, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_103, (448, ), (1, ))
    assert_size_stride(primals_104, (448, ), (1, ))
    assert_size_stride(primals_105, (128, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_106, (128, ), (1, ))
    assert_size_stride(primals_107, (128, ), (1, ))
    assert_size_stride(primals_108, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_109, (480, ), (1, ))
    assert_size_stride(primals_110, (480, ), (1, ))
    assert_size_stride(primals_111, (128, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_112, (128, ), (1, ))
    assert_size_stride(primals_113, (128, ), (1, ))
    assert_size_stride(primals_114, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_115, (512, ), (1, ))
    assert_size_stride(primals_116, (512, ), (1, ))
    assert_size_stride(primals_117, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_118, (256, ), (1, ))
    assert_size_stride(primals_119, (256, ), (1, ))
    assert_size_stride(primals_120, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_121, (128, ), (1, ))
    assert_size_stride(primals_122, (128, ), (1, ))
    assert_size_stride(primals_123, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_124, (288, ), (1, ))
    assert_size_stride(primals_125, (288, ), (1, ))
    assert_size_stride(primals_126, (128, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(primals_127, (128, ), (1, ))
    assert_size_stride(primals_128, (128, ), (1, ))
    assert_size_stride(primals_129, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_130, (320, ), (1, ))
    assert_size_stride(primals_131, (320, ), (1, ))
    assert_size_stride(primals_132, (128, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(primals_133, (128, ), (1, ))
    assert_size_stride(primals_134, (128, ), (1, ))
    assert_size_stride(primals_135, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_136, (352, ), (1, ))
    assert_size_stride(primals_137, (352, ), (1, ))
    assert_size_stride(primals_138, (128, 352, 1, 1), (352, 1, 1, 1))
    assert_size_stride(primals_139, (128, ), (1, ))
    assert_size_stride(primals_140, (128, ), (1, ))
    assert_size_stride(primals_141, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_142, (384, ), (1, ))
    assert_size_stride(primals_143, (384, ), (1, ))
    assert_size_stride(primals_144, (128, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_145, (128, ), (1, ))
    assert_size_stride(primals_146, (128, ), (1, ))
    assert_size_stride(primals_147, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_148, (416, ), (1, ))
    assert_size_stride(primals_149, (416, ), (1, ))
    assert_size_stride(primals_150, (128, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(primals_151, (128, ), (1, ))
    assert_size_stride(primals_152, (128, ), (1, ))
    assert_size_stride(primals_153, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_154, (448, ), (1, ))
    assert_size_stride(primals_155, (448, ), (1, ))
    assert_size_stride(primals_156, (128, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(primals_157, (128, ), (1, ))
    assert_size_stride(primals_158, (128, ), (1, ))
    assert_size_stride(primals_159, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_160, (480, ), (1, ))
    assert_size_stride(primals_161, (480, ), (1, ))
    assert_size_stride(primals_162, (128, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(primals_163, (128, ), (1, ))
    assert_size_stride(primals_164, (128, ), (1, ))
    assert_size_stride(primals_165, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_166, (512, ), (1, ))
    assert_size_stride(primals_167, (512, ), (1, ))
    assert_size_stride(primals_168, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_169, (128, ), (1, ))
    assert_size_stride(primals_170, (128, ), (1, ))
    assert_size_stride(primals_171, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_172, (544, ), (1, ))
    assert_size_stride(primals_173, (544, ), (1, ))
    assert_size_stride(primals_174, (128, 544, 1, 1), (544, 1, 1, 1))
    assert_size_stride(primals_175, (128, ), (1, ))
    assert_size_stride(primals_176, (128, ), (1, ))
    assert_size_stride(primals_177, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_178, (576, ), (1, ))
    assert_size_stride(primals_179, (576, ), (1, ))
    assert_size_stride(primals_180, (128, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_181, (128, ), (1, ))
    assert_size_stride(primals_182, (128, ), (1, ))
    assert_size_stride(primals_183, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_184, (608, ), (1, ))
    assert_size_stride(primals_185, (608, ), (1, ))
    assert_size_stride(primals_186, (128, 608, 1, 1), (608, 1, 1, 1))
    assert_size_stride(primals_187, (128, ), (1, ))
    assert_size_stride(primals_188, (128, ), (1, ))
    assert_size_stride(primals_189, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_190, (640, ), (1, ))
    assert_size_stride(primals_191, (640, ), (1, ))
    assert_size_stride(primals_192, (128, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_193, (128, ), (1, ))
    assert_size_stride(primals_194, (128, ), (1, ))
    assert_size_stride(primals_195, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_196, (672, ), (1, ))
    assert_size_stride(primals_197, (672, ), (1, ))
    assert_size_stride(primals_198, (128, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_199, (128, ), (1, ))
    assert_size_stride(primals_200, (128, ), (1, ))
    assert_size_stride(primals_201, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_202, (704, ), (1, ))
    assert_size_stride(primals_203, (704, ), (1, ))
    assert_size_stride(primals_204, (128, 704, 1, 1), (704, 1, 1, 1))
    assert_size_stride(primals_205, (128, ), (1, ))
    assert_size_stride(primals_206, (128, ), (1, ))
    assert_size_stride(primals_207, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_208, (736, ), (1, ))
    assert_size_stride(primals_209, (736, ), (1, ))
    assert_size_stride(primals_210, (128, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_211, (128, ), (1, ))
    assert_size_stride(primals_212, (128, ), (1, ))
    assert_size_stride(primals_213, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_214, (768, ), (1, ))
    assert_size_stride(primals_215, (768, ), (1, ))
    assert_size_stride(primals_216, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_217, (128, ), (1, ))
    assert_size_stride(primals_218, (128, ), (1, ))
    assert_size_stride(primals_219, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_220, (800, ), (1, ))
    assert_size_stride(primals_221, (800, ), (1, ))
    assert_size_stride(primals_222, (128, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_223, (128, ), (1, ))
    assert_size_stride(primals_224, (128, ), (1, ))
    assert_size_stride(primals_225, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_226, (832, ), (1, ))
    assert_size_stride(primals_227, (832, ), (1, ))
    assert_size_stride(primals_228, (128, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(primals_229, (128, ), (1, ))
    assert_size_stride(primals_230, (128, ), (1, ))
    assert_size_stride(primals_231, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_232, (864, ), (1, ))
    assert_size_stride(primals_233, (864, ), (1, ))
    assert_size_stride(primals_234, (128, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(primals_235, (128, ), (1, ))
    assert_size_stride(primals_236, (128, ), (1, ))
    assert_size_stride(primals_237, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_238, (896, ), (1, ))
    assert_size_stride(primals_239, (896, ), (1, ))
    assert_size_stride(primals_240, (128, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_241, (128, ), (1, ))
    assert_size_stride(primals_242, (128, ), (1, ))
    assert_size_stride(primals_243, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_244, (928, ), (1, ))
    assert_size_stride(primals_245, (928, ), (1, ))
    assert_size_stride(primals_246, (128, 928, 1, 1), (928, 1, 1, 1))
    assert_size_stride(primals_247, (128, ), (1, ))
    assert_size_stride(primals_248, (128, ), (1, ))
    assert_size_stride(primals_249, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_250, (960, ), (1, ))
    assert_size_stride(primals_251, (960, ), (1, ))
    assert_size_stride(primals_252, (128, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_253, (128, ), (1, ))
    assert_size_stride(primals_254, (128, ), (1, ))
    assert_size_stride(primals_255, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_256, (992, ), (1, ))
    assert_size_stride(primals_257, (992, ), (1, ))
    assert_size_stride(primals_258, (128, 992, 1, 1), (992, 1, 1, 1))
    assert_size_stride(primals_259, (128, ), (1, ))
    assert_size_stride(primals_260, (128, ), (1, ))
    assert_size_stride(primals_261, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_262, (1024, ), (1, ))
    assert_size_stride(primals_263, (1024, ), (1, ))
    assert_size_stride(primals_264, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_265, (512, ), (1, ))
    assert_size_stride(primals_266, (512, ), (1, ))
    assert_size_stride(primals_267, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_268, (128, ), (1, ))
    assert_size_stride(primals_269, (128, ), (1, ))
    assert_size_stride(primals_270, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_271, (544, ), (1, ))
    assert_size_stride(primals_272, (544, ), (1, ))
    assert_size_stride(primals_273, (128, 544, 1, 1), (544, 1, 1, 1))
    assert_size_stride(primals_274, (128, ), (1, ))
    assert_size_stride(primals_275, (128, ), (1, ))
    assert_size_stride(primals_276, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_277, (576, ), (1, ))
    assert_size_stride(primals_278, (576, ), (1, ))
    assert_size_stride(primals_279, (128, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(primals_280, (128, ), (1, ))
    assert_size_stride(primals_281, (128, ), (1, ))
    assert_size_stride(primals_282, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_283, (608, ), (1, ))
    assert_size_stride(primals_284, (608, ), (1, ))
    assert_size_stride(primals_285, (128, 608, 1, 1), (608, 1, 1, 1))
    assert_size_stride(primals_286, (128, ), (1, ))
    assert_size_stride(primals_287, (128, ), (1, ))
    assert_size_stride(primals_288, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_289, (640, ), (1, ))
    assert_size_stride(primals_290, (640, ), (1, ))
    assert_size_stride(primals_291, (128, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(primals_292, (128, ), (1, ))
    assert_size_stride(primals_293, (128, ), (1, ))
    assert_size_stride(primals_294, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_295, (672, ), (1, ))
    assert_size_stride(primals_296, (672, ), (1, ))
    assert_size_stride(primals_297, (128, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(primals_298, (128, ), (1, ))
    assert_size_stride(primals_299, (128, ), (1, ))
    assert_size_stride(primals_300, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_301, (704, ), (1, ))
    assert_size_stride(primals_302, (704, ), (1, ))
    assert_size_stride(primals_303, (128, 704, 1, 1), (704, 1, 1, 1))
    assert_size_stride(primals_304, (128, ), (1, ))
    assert_size_stride(primals_305, (128, ), (1, ))
    assert_size_stride(primals_306, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_307, (736, ), (1, ))
    assert_size_stride(primals_308, (736, ), (1, ))
    assert_size_stride(primals_309, (128, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(primals_310, (128, ), (1, ))
    assert_size_stride(primals_311, (128, ), (1, ))
    assert_size_stride(primals_312, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_313, (768, ), (1, ))
    assert_size_stride(primals_314, (768, ), (1, ))
    assert_size_stride(primals_315, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_316, (128, ), (1, ))
    assert_size_stride(primals_317, (128, ), (1, ))
    assert_size_stride(primals_318, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_319, (800, ), (1, ))
    assert_size_stride(primals_320, (800, ), (1, ))
    assert_size_stride(primals_321, (128, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(primals_322, (128, ), (1, ))
    assert_size_stride(primals_323, (128, ), (1, ))
    assert_size_stride(primals_324, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_325, (832, ), (1, ))
    assert_size_stride(primals_326, (832, ), (1, ))
    assert_size_stride(primals_327, (128, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(primals_328, (128, ), (1, ))
    assert_size_stride(primals_329, (128, ), (1, ))
    assert_size_stride(primals_330, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_331, (864, ), (1, ))
    assert_size_stride(primals_332, (864, ), (1, ))
    assert_size_stride(primals_333, (128, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(primals_334, (128, ), (1, ))
    assert_size_stride(primals_335, (128, ), (1, ))
    assert_size_stride(primals_336, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_337, (896, ), (1, ))
    assert_size_stride(primals_338, (896, ), (1, ))
    assert_size_stride(primals_339, (128, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(primals_340, (128, ), (1, ))
    assert_size_stride(primals_341, (128, ), (1, ))
    assert_size_stride(primals_342, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_343, (928, ), (1, ))
    assert_size_stride(primals_344, (928, ), (1, ))
    assert_size_stride(primals_345, (128, 928, 1, 1), (928, 1, 1, 1))
    assert_size_stride(primals_346, (128, ), (1, ))
    assert_size_stride(primals_347, (128, ), (1, ))
    assert_size_stride(primals_348, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_349, (960, ), (1, ))
    assert_size_stride(primals_350, (960, ), (1, ))
    assert_size_stride(primals_351, (128, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(primals_352, (128, ), (1, ))
    assert_size_stride(primals_353, (128, ), (1, ))
    assert_size_stride(primals_354, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_355, (992, ), (1, ))
    assert_size_stride(primals_356, (992, ), (1, ))
    assert_size_stride(primals_357, (128, 992, 1, 1), (992, 1, 1, 1))
    assert_size_stride(primals_358, (128, ), (1, ))
    assert_size_stride(primals_359, (128, ), (1, ))
    assert_size_stride(primals_360, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_361, (1024, ), (1, ))
    assert_size_stride(primals_362, (1024, ), (1, ))
    assert_size_stride(primals_363, (1000, 1024), (1024, 1))
    assert_size_stride(primals_364, (1000, ), (1, ))
    assert_size_stride(primals_365, (64, ), (1, ))
    assert_size_stride(primals_366, (64, ), (1, ))
    assert_size_stride(primals_367, (), ())
    assert_size_stride(primals_368, (64, ), (1, ))
    assert_size_stride(primals_369, (64, ), (1, ))
    assert_size_stride(primals_370, (), ())
    assert_size_stride(primals_371, (128, ), (1, ))
    assert_size_stride(primals_372, (128, ), (1, ))
    assert_size_stride(primals_373, (), ())
    assert_size_stride(primals_374, (96, ), (1, ))
    assert_size_stride(primals_375, (96, ), (1, ))
    assert_size_stride(primals_376, (), ())
    assert_size_stride(primals_377, (128, ), (1, ))
    assert_size_stride(primals_378, (128, ), (1, ))
    assert_size_stride(primals_379, (), ())
    assert_size_stride(primals_380, (128, ), (1, ))
    assert_size_stride(primals_381, (128, ), (1, ))
    assert_size_stride(primals_382, (), ())
    assert_size_stride(primals_383, (128, ), (1, ))
    assert_size_stride(primals_384, (128, ), (1, ))
    assert_size_stride(primals_385, (), ())
    assert_size_stride(primals_386, (160, ), (1, ))
    assert_size_stride(primals_387, (160, ), (1, ))
    assert_size_stride(primals_388, (), ())
    assert_size_stride(primals_389, (128, ), (1, ))
    assert_size_stride(primals_390, (128, ), (1, ))
    assert_size_stride(primals_391, (), ())
    assert_size_stride(primals_392, (192, ), (1, ))
    assert_size_stride(primals_393, (192, ), (1, ))
    assert_size_stride(primals_394, (), ())
    assert_size_stride(primals_395, (128, ), (1, ))
    assert_size_stride(primals_396, (128, ), (1, ))
    assert_size_stride(primals_397, (), ())
    assert_size_stride(primals_398, (224, ), (1, ))
    assert_size_stride(primals_399, (224, ), (1, ))
    assert_size_stride(primals_400, (), ())
    assert_size_stride(primals_401, (128, ), (1, ))
    assert_size_stride(primals_402, (128, ), (1, ))
    assert_size_stride(primals_403, (), ())
    assert_size_stride(primals_404, (256, ), (1, ))
    assert_size_stride(primals_405, (256, ), (1, ))
    assert_size_stride(primals_406, (), ())
    assert_size_stride(primals_407, (128, ), (1, ))
    assert_size_stride(primals_408, (128, ), (1, ))
    assert_size_stride(primals_409, (), ())
    assert_size_stride(primals_410, (128, ), (1, ))
    assert_size_stride(primals_411, (128, ), (1, ))
    assert_size_stride(primals_412, (), ())
    assert_size_stride(primals_413, (160, ), (1, ))
    assert_size_stride(primals_414, (160, ), (1, ))
    assert_size_stride(primals_415, (), ())
    assert_size_stride(primals_416, (128, ), (1, ))
    assert_size_stride(primals_417, (128, ), (1, ))
    assert_size_stride(primals_418, (), ())
    assert_size_stride(primals_419, (192, ), (1, ))
    assert_size_stride(primals_420, (192, ), (1, ))
    assert_size_stride(primals_421, (), ())
    assert_size_stride(primals_422, (128, ), (1, ))
    assert_size_stride(primals_423, (128, ), (1, ))
    assert_size_stride(primals_424, (), ())
    assert_size_stride(primals_425, (224, ), (1, ))
    assert_size_stride(primals_426, (224, ), (1, ))
    assert_size_stride(primals_427, (), ())
    assert_size_stride(primals_428, (128, ), (1, ))
    assert_size_stride(primals_429, (128, ), (1, ))
    assert_size_stride(primals_430, (), ())
    assert_size_stride(primals_431, (256, ), (1, ))
    assert_size_stride(primals_432, (256, ), (1, ))
    assert_size_stride(primals_433, (), ())
    assert_size_stride(primals_434, (128, ), (1, ))
    assert_size_stride(primals_435, (128, ), (1, ))
    assert_size_stride(primals_436, (), ())
    assert_size_stride(primals_437, (288, ), (1, ))
    assert_size_stride(primals_438, (288, ), (1, ))
    assert_size_stride(primals_439, (), ())
    assert_size_stride(primals_440, (128, ), (1, ))
    assert_size_stride(primals_441, (128, ), (1, ))
    assert_size_stride(primals_442, (), ())
    assert_size_stride(primals_443, (320, ), (1, ))
    assert_size_stride(primals_444, (320, ), (1, ))
    assert_size_stride(primals_445, (), ())
    assert_size_stride(primals_446, (128, ), (1, ))
    assert_size_stride(primals_447, (128, ), (1, ))
    assert_size_stride(primals_448, (), ())
    assert_size_stride(primals_449, (352, ), (1, ))
    assert_size_stride(primals_450, (352, ), (1, ))
    assert_size_stride(primals_451, (), ())
    assert_size_stride(primals_452, (128, ), (1, ))
    assert_size_stride(primals_453, (128, ), (1, ))
    assert_size_stride(primals_454, (), ())
    assert_size_stride(primals_455, (384, ), (1, ))
    assert_size_stride(primals_456, (384, ), (1, ))
    assert_size_stride(primals_457, (), ())
    assert_size_stride(primals_458, (128, ), (1, ))
    assert_size_stride(primals_459, (128, ), (1, ))
    assert_size_stride(primals_460, (), ())
    assert_size_stride(primals_461, (416, ), (1, ))
    assert_size_stride(primals_462, (416, ), (1, ))
    assert_size_stride(primals_463, (), ())
    assert_size_stride(primals_464, (128, ), (1, ))
    assert_size_stride(primals_465, (128, ), (1, ))
    assert_size_stride(primals_466, (), ())
    assert_size_stride(primals_467, (448, ), (1, ))
    assert_size_stride(primals_468, (448, ), (1, ))
    assert_size_stride(primals_469, (), ())
    assert_size_stride(primals_470, (128, ), (1, ))
    assert_size_stride(primals_471, (128, ), (1, ))
    assert_size_stride(primals_472, (), ())
    assert_size_stride(primals_473, (480, ), (1, ))
    assert_size_stride(primals_474, (480, ), (1, ))
    assert_size_stride(primals_475, (), ())
    assert_size_stride(primals_476, (128, ), (1, ))
    assert_size_stride(primals_477, (128, ), (1, ))
    assert_size_stride(primals_478, (), ())
    assert_size_stride(primals_479, (512, ), (1, ))
    assert_size_stride(primals_480, (512, ), (1, ))
    assert_size_stride(primals_481, (), ())
    assert_size_stride(primals_482, (256, ), (1, ))
    assert_size_stride(primals_483, (256, ), (1, ))
    assert_size_stride(primals_484, (), ())
    assert_size_stride(primals_485, (128, ), (1, ))
    assert_size_stride(primals_486, (128, ), (1, ))
    assert_size_stride(primals_487, (), ())
    assert_size_stride(primals_488, (288, ), (1, ))
    assert_size_stride(primals_489, (288, ), (1, ))
    assert_size_stride(primals_490, (), ())
    assert_size_stride(primals_491, (128, ), (1, ))
    assert_size_stride(primals_492, (128, ), (1, ))
    assert_size_stride(primals_493, (), ())
    assert_size_stride(primals_494, (320, ), (1, ))
    assert_size_stride(primals_495, (320, ), (1, ))
    assert_size_stride(primals_496, (), ())
    assert_size_stride(primals_497, (128, ), (1, ))
    assert_size_stride(primals_498, (128, ), (1, ))
    assert_size_stride(primals_499, (), ())
    assert_size_stride(primals_500, (352, ), (1, ))
    assert_size_stride(primals_501, (352, ), (1, ))
    assert_size_stride(primals_502, (), ())
    assert_size_stride(primals_503, (128, ), (1, ))
    assert_size_stride(primals_504, (128, ), (1, ))
    assert_size_stride(primals_505, (), ())
    assert_size_stride(primals_506, (384, ), (1, ))
    assert_size_stride(primals_507, (384, ), (1, ))
    assert_size_stride(primals_508, (), ())
    assert_size_stride(primals_509, (128, ), (1, ))
    assert_size_stride(primals_510, (128, ), (1, ))
    assert_size_stride(primals_511, (), ())
    assert_size_stride(primals_512, (416, ), (1, ))
    assert_size_stride(primals_513, (416, ), (1, ))
    assert_size_stride(primals_514, (), ())
    assert_size_stride(primals_515, (128, ), (1, ))
    assert_size_stride(primals_516, (128, ), (1, ))
    assert_size_stride(primals_517, (), ())
    assert_size_stride(primals_518, (448, ), (1, ))
    assert_size_stride(primals_519, (448, ), (1, ))
    assert_size_stride(primals_520, (), ())
    assert_size_stride(primals_521, (128, ), (1, ))
    assert_size_stride(primals_522, (128, ), (1, ))
    assert_size_stride(primals_523, (), ())
    assert_size_stride(primals_524, (480, ), (1, ))
    assert_size_stride(primals_525, (480, ), (1, ))
    assert_size_stride(primals_526, (), ())
    assert_size_stride(primals_527, (128, ), (1, ))
    assert_size_stride(primals_528, (128, ), (1, ))
    assert_size_stride(primals_529, (), ())
    assert_size_stride(primals_530, (512, ), (1, ))
    assert_size_stride(primals_531, (512, ), (1, ))
    assert_size_stride(primals_532, (), ())
    assert_size_stride(primals_533, (128, ), (1, ))
    assert_size_stride(primals_534, (128, ), (1, ))
    assert_size_stride(primals_535, (), ())
    assert_size_stride(primals_536, (544, ), (1, ))
    assert_size_stride(primals_537, (544, ), (1, ))
    assert_size_stride(primals_538, (), ())
    assert_size_stride(primals_539, (128, ), (1, ))
    assert_size_stride(primals_540, (128, ), (1, ))
    assert_size_stride(primals_541, (), ())
    assert_size_stride(primals_542, (576, ), (1, ))
    assert_size_stride(primals_543, (576, ), (1, ))
    assert_size_stride(primals_544, (), ())
    assert_size_stride(primals_545, (128, ), (1, ))
    assert_size_stride(primals_546, (128, ), (1, ))
    assert_size_stride(primals_547, (), ())
    assert_size_stride(primals_548, (608, ), (1, ))
    assert_size_stride(primals_549, (608, ), (1, ))
    assert_size_stride(primals_550, (), ())
    assert_size_stride(primals_551, (128, ), (1, ))
    assert_size_stride(primals_552, (128, ), (1, ))
    assert_size_stride(primals_553, (), ())
    assert_size_stride(primals_554, (640, ), (1, ))
    assert_size_stride(primals_555, (640, ), (1, ))
    assert_size_stride(primals_556, (), ())
    assert_size_stride(primals_557, (128, ), (1, ))
    assert_size_stride(primals_558, (128, ), (1, ))
    assert_size_stride(primals_559, (), ())
    assert_size_stride(primals_560, (672, ), (1, ))
    assert_size_stride(primals_561, (672, ), (1, ))
    assert_size_stride(primals_562, (), ())
    assert_size_stride(primals_563, (128, ), (1, ))
    assert_size_stride(primals_564, (128, ), (1, ))
    assert_size_stride(primals_565, (), ())
    assert_size_stride(primals_566, (704, ), (1, ))
    assert_size_stride(primals_567, (704, ), (1, ))
    assert_size_stride(primals_568, (), ())
    assert_size_stride(primals_569, (128, ), (1, ))
    assert_size_stride(primals_570, (128, ), (1, ))
    assert_size_stride(primals_571, (), ())
    assert_size_stride(primals_572, (736, ), (1, ))
    assert_size_stride(primals_573, (736, ), (1, ))
    assert_size_stride(primals_574, (), ())
    assert_size_stride(primals_575, (128, ), (1, ))
    assert_size_stride(primals_576, (128, ), (1, ))
    assert_size_stride(primals_577, (), ())
    assert_size_stride(primals_578, (768, ), (1, ))
    assert_size_stride(primals_579, (768, ), (1, ))
    assert_size_stride(primals_580, (), ())
    assert_size_stride(primals_581, (128, ), (1, ))
    assert_size_stride(primals_582, (128, ), (1, ))
    assert_size_stride(primals_583, (), ())
    assert_size_stride(primals_584, (800, ), (1, ))
    assert_size_stride(primals_585, (800, ), (1, ))
    assert_size_stride(primals_586, (), ())
    assert_size_stride(primals_587, (128, ), (1, ))
    assert_size_stride(primals_588, (128, ), (1, ))
    assert_size_stride(primals_589, (), ())
    assert_size_stride(primals_590, (832, ), (1, ))
    assert_size_stride(primals_591, (832, ), (1, ))
    assert_size_stride(primals_592, (), ())
    assert_size_stride(primals_593, (128, ), (1, ))
    assert_size_stride(primals_594, (128, ), (1, ))
    assert_size_stride(primals_595, (), ())
    assert_size_stride(primals_596, (864, ), (1, ))
    assert_size_stride(primals_597, (864, ), (1, ))
    assert_size_stride(primals_598, (), ())
    assert_size_stride(primals_599, (128, ), (1, ))
    assert_size_stride(primals_600, (128, ), (1, ))
    assert_size_stride(primals_601, (), ())
    assert_size_stride(primals_602, (896, ), (1, ))
    assert_size_stride(primals_603, (896, ), (1, ))
    assert_size_stride(primals_604, (), ())
    assert_size_stride(primals_605, (128, ), (1, ))
    assert_size_stride(primals_606, (128, ), (1, ))
    assert_size_stride(primals_607, (), ())
    assert_size_stride(primals_608, (928, ), (1, ))
    assert_size_stride(primals_609, (928, ), (1, ))
    assert_size_stride(primals_610, (), ())
    assert_size_stride(primals_611, (128, ), (1, ))
    assert_size_stride(primals_612, (128, ), (1, ))
    assert_size_stride(primals_613, (), ())
    assert_size_stride(primals_614, (960, ), (1, ))
    assert_size_stride(primals_615, (960, ), (1, ))
    assert_size_stride(primals_616, (), ())
    assert_size_stride(primals_617, (128, ), (1, ))
    assert_size_stride(primals_618, (128, ), (1, ))
    assert_size_stride(primals_619, (), ())
    assert_size_stride(primals_620, (992, ), (1, ))
    assert_size_stride(primals_621, (992, ), (1, ))
    assert_size_stride(primals_622, (), ())
    assert_size_stride(primals_623, (128, ), (1, ))
    assert_size_stride(primals_624, (128, ), (1, ))
    assert_size_stride(primals_625, (), ())
    assert_size_stride(primals_626, (1024, ), (1, ))
    assert_size_stride(primals_627, (1024, ), (1, ))
    assert_size_stride(primals_628, (), ())
    assert_size_stride(primals_629, (512, ), (1, ))
    assert_size_stride(primals_630, (512, ), (1, ))
    assert_size_stride(primals_631, (), ())
    assert_size_stride(primals_632, (128, ), (1, ))
    assert_size_stride(primals_633, (128, ), (1, ))
    assert_size_stride(primals_634, (), ())
    assert_size_stride(primals_635, (544, ), (1, ))
    assert_size_stride(primals_636, (544, ), (1, ))
    assert_size_stride(primals_637, (), ())
    assert_size_stride(primals_638, (128, ), (1, ))
    assert_size_stride(primals_639, (128, ), (1, ))
    assert_size_stride(primals_640, (), ())
    assert_size_stride(primals_641, (576, ), (1, ))
    assert_size_stride(primals_642, (576, ), (1, ))
    assert_size_stride(primals_643, (), ())
    assert_size_stride(primals_644, (128, ), (1, ))
    assert_size_stride(primals_645, (128, ), (1, ))
    assert_size_stride(primals_646, (), ())
    assert_size_stride(primals_647, (608, ), (1, ))
    assert_size_stride(primals_648, (608, ), (1, ))
    assert_size_stride(primals_649, (), ())
    assert_size_stride(primals_650, (128, ), (1, ))
    assert_size_stride(primals_651, (128, ), (1, ))
    assert_size_stride(primals_652, (), ())
    assert_size_stride(primals_653, (640, ), (1, ))
    assert_size_stride(primals_654, (640, ), (1, ))
    assert_size_stride(primals_655, (), ())
    assert_size_stride(primals_656, (128, ), (1, ))
    assert_size_stride(primals_657, (128, ), (1, ))
    assert_size_stride(primals_658, (), ())
    assert_size_stride(primals_659, (672, ), (1, ))
    assert_size_stride(primals_660, (672, ), (1, ))
    assert_size_stride(primals_661, (), ())
    assert_size_stride(primals_662, (128, ), (1, ))
    assert_size_stride(primals_663, (128, ), (1, ))
    assert_size_stride(primals_664, (), ())
    assert_size_stride(primals_665, (704, ), (1, ))
    assert_size_stride(primals_666, (704, ), (1, ))
    assert_size_stride(primals_667, (), ())
    assert_size_stride(primals_668, (128, ), (1, ))
    assert_size_stride(primals_669, (128, ), (1, ))
    assert_size_stride(primals_670, (), ())
    assert_size_stride(primals_671, (736, ), (1, ))
    assert_size_stride(primals_672, (736, ), (1, ))
    assert_size_stride(primals_673, (), ())
    assert_size_stride(primals_674, (128, ), (1, ))
    assert_size_stride(primals_675, (128, ), (1, ))
    assert_size_stride(primals_676, (), ())
    assert_size_stride(primals_677, (768, ), (1, ))
    assert_size_stride(primals_678, (768, ), (1, ))
    assert_size_stride(primals_679, (), ())
    assert_size_stride(primals_680, (128, ), (1, ))
    assert_size_stride(primals_681, (128, ), (1, ))
    assert_size_stride(primals_682, (), ())
    assert_size_stride(primals_683, (800, ), (1, ))
    assert_size_stride(primals_684, (800, ), (1, ))
    assert_size_stride(primals_685, (), ())
    assert_size_stride(primals_686, (128, ), (1, ))
    assert_size_stride(primals_687, (128, ), (1, ))
    assert_size_stride(primals_688, (), ())
    assert_size_stride(primals_689, (832, ), (1, ))
    assert_size_stride(primals_690, (832, ), (1, ))
    assert_size_stride(primals_691, (), ())
    assert_size_stride(primals_692, (128, ), (1, ))
    assert_size_stride(primals_693, (128, ), (1, ))
    assert_size_stride(primals_694, (), ())
    assert_size_stride(primals_695, (864, ), (1, ))
    assert_size_stride(primals_696, (864, ), (1, ))
    assert_size_stride(primals_697, (), ())
    assert_size_stride(primals_698, (128, ), (1, ))
    assert_size_stride(primals_699, (128, ), (1, ))
    assert_size_stride(primals_700, (), ())
    assert_size_stride(primals_701, (896, ), (1, ))
    assert_size_stride(primals_702, (896, ), (1, ))
    assert_size_stride(primals_703, (), ())
    assert_size_stride(primals_704, (128, ), (1, ))
    assert_size_stride(primals_705, (128, ), (1, ))
    assert_size_stride(primals_706, (), ())
    assert_size_stride(primals_707, (928, ), (1, ))
    assert_size_stride(primals_708, (928, ), (1, ))
    assert_size_stride(primals_709, (), ())
    assert_size_stride(primals_710, (128, ), (1, ))
    assert_size_stride(primals_711, (128, ), (1, ))
    assert_size_stride(primals_712, (), ())
    assert_size_stride(primals_713, (960, ), (1, ))
    assert_size_stride(primals_714, (960, ), (1, ))
    assert_size_stride(primals_715, (), ())
    assert_size_stride(primals_716, (128, ), (1, ))
    assert_size_stride(primals_717, (128, ), (1, ))
    assert_size_stride(primals_718, (), ())
    assert_size_stride(primals_719, (992, ), (1, ))
    assert_size_stride(primals_720, (992, ), (1, ))
    assert_size_stride(primals_721, (), ())
    assert_size_stride(primals_722, (128, ), (1, ))
    assert_size_stride(primals_723, (128, ), (1, ))
    assert_size_stride(primals_724, (), ())
    assert_size_stride(primals_725, (1024, ), (1, ))
    assert_size_stride(primals_726, (1024, ), (1, ))
    assert_size_stride(primals_727, (), ())
    assert_size_stride(primals_728, (4, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [l__mod___features_conv0], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_728, primals_1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 112, 112), (802816, 12544, 112, 1))
        buf1 = empty((4, 64, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_norm0, l__mod___features_relu0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf0, primals_365, primals_366, primals_2, primals_3, buf1, 3211264, grid=grid(3211264), stream=stream0)
        del primals_3
        buf9 = empty((4, 96, 56, 56), device='cuda', dtype=torch.float32)
        buf2 = reinterpret_tensor(buf9, (4, 64, 56, 56), (301056, 3136, 56, 1), 0)  # alias
        buf3 = empty((4, 64, 56, 56), device='cuda', dtype=torch.int64)
        buf4 = empty((4, 64, 56, 56), device='cuda', dtype=torch.float32)
        buf29 = empty((4, 192, 56, 56), device='cuda', dtype=torch.float32)
        buf24 = reinterpret_tensor(buf29, (4, 64, 56, 56), (602112, 3136, 56, 1), 0)  # alias
        buf40 = empty((4, 224, 56, 56), device='cuda', dtype=torch.float32)
        buf34 = reinterpret_tensor(buf40, (4, 64, 56, 56), (702464, 3136, 56, 1), 0)  # alias
        buf52 = empty((4, 256, 56, 56), device='cuda', dtype=torch.float32)
        buf45 = reinterpret_tensor(buf52, (4, 64, 56, 56), (802816, 3136, 56, 1), 0)  # alias
        buf866 = empty((4, 64, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_117, cat_118, cat_119, l__mod___features_denseblock1_denselayer1_norm1, l__mod___features_denseblock1_denselayer1_relu1, l__mod___features_pool0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.max_pool2d_with_indices, aten.native_batch_norm_backward, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_max_pool2d_with_indices_native_batch_norm_backward_relu_1.run(buf1, primals_368, primals_369, primals_4, primals_5, buf2, buf3, buf4, buf24, buf34, buf45, buf866, 802816, grid=grid(802816), stream=stream0)
        del primals_368
        del primals_5
        # Source Nodes: [bottleneck_output], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, primals_6, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 128, 56, 56), (401408, 3136, 56, 1))
        buf6 = empty((4, 128, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock1_denselayer1_norm2, l__mod___features_denseblock1_denselayer1_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf5, primals_371, primals_372, primals_7, primals_8, buf6, 1605632, grid=grid(1605632), stream=stream0)
        del primals_8
        # Source Nodes: [new_features], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, primals_9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 32, 56, 56), (100352, 3136, 56, 1))
        buf8 = reinterpret_tensor(buf9, (4, 32, 56, 56), (301056, 3136, 56, 1), 200704)  # alias
        buf25 = reinterpret_tensor(buf29, (4, 32, 56, 56), (602112, 3136, 56, 1), 200704)  # alias
        buf35 = reinterpret_tensor(buf40, (4, 32, 56, 56), (702464, 3136, 56, 1), 200704)  # alias
        buf46 = reinterpret_tensor(buf52, (4, 32, 56, 56), (802816, 3136, 56, 1), 200704)  # alias
        # Source Nodes: [cat_117, cat_118, cat_119, cat_122], Original ATen: [aten.cat]
        triton_poi_fused_cat_3.run(buf7, buf8, buf25, buf35, buf46, 401408, grid=grid(401408), stream=stream0)
        buf10 = empty((4, 96, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock1_denselayer2_norm1, l__mod___features_denseblock1_denselayer2_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf9, primals_374, primals_375, primals_10, primals_11, buf10, 1204224, grid=grid(1204224), stream=stream0)
        del primals_11
        # Source Nodes: [bottleneck_output_2], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, primals_12, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 128, 56, 56), (401408, 3136, 56, 1))
        buf12 = empty((4, 128, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock1_denselayer2_norm2, l__mod___features_denseblock1_denselayer2_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf11, primals_377, primals_378, primals_13, primals_14, buf12, 1605632, grid=grid(1605632), stream=stream0)
        del primals_14
        # Source Nodes: [new_features_2], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, primals_15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 32, 56, 56), (100352, 3136, 56, 1))
        buf14 = empty((4, 128, 56, 56), device='cuda', dtype=torch.float32)
        buf15 = empty((4, 128, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_121, l__mod___features_denseblock1_denselayer3_norm1, l__mod___features_denseblock1_denselayer3_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5.run(buf2, buf7, buf13, primals_380, primals_381, primals_16, primals_17, buf14, buf15, 1605632, grid=grid(1605632), stream=stream0)
        del primals_17
        # Source Nodes: [bottleneck_output_4], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_18, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 128, 56, 56), (401408, 3136, 56, 1))
        buf17 = empty((4, 128, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock1_denselayer3_norm2, l__mod___features_denseblock1_denselayer3_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf16, primals_383, primals_384, primals_19, primals_20, buf17, 1605632, grid=grid(1605632), stream=stream0)
        del primals_20
        # Source Nodes: [new_features_4], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 32, 56, 56), (100352, 3136, 56, 1))
        buf19 = empty((4, 160, 56, 56), device='cuda', dtype=torch.float32)
        buf20 = empty((4, 160, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_120, l__mod___features_denseblock1_denselayer4_norm1, l__mod___features_denseblock1_denselayer4_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_6.run(buf2, buf7, buf13, buf18, primals_386, primals_387, primals_22, primals_23, buf19, buf20, 2007040, grid=grid(2007040), stream=stream0)
        del primals_23
        # Source Nodes: [bottleneck_output_6], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_24, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 128, 56, 56), (401408, 3136, 56, 1))
        buf22 = empty((4, 128, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock1_denselayer4_norm2, l__mod___features_denseblock1_denselayer4_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf21, primals_389, primals_390, primals_25, primals_26, buf22, 1605632, grid=grid(1605632), stream=stream0)
        del primals_26
        # Source Nodes: [new_features_6], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, primals_27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 32, 56, 56), (100352, 3136, 56, 1))
        buf26 = reinterpret_tensor(buf29, (4, 32, 56, 56), (602112, 3136, 56, 1), 301056)  # alias
        buf36 = reinterpret_tensor(buf40, (4, 32, 56, 56), (702464, 3136, 56, 1), 301056)  # alias
        buf47 = reinterpret_tensor(buf52, (4, 32, 56, 56), (802816, 3136, 56, 1), 301056)  # alias
        # Source Nodes: [cat_117, cat_118, cat_119], Original ATen: [aten.cat]
        triton_poi_fused_cat_7.run(buf13, buf26, buf36, buf47, 401408, grid=grid(401408), stream=stream0)
        buf27 = reinterpret_tensor(buf29, (4, 32, 56, 56), (602112, 3136, 56, 1), 401408)  # alias
        buf37 = reinterpret_tensor(buf40, (4, 32, 56, 56), (702464, 3136, 56, 1), 401408)  # alias
        buf48 = reinterpret_tensor(buf52, (4, 32, 56, 56), (802816, 3136, 56, 1), 401408)  # alias
        # Source Nodes: [cat_117, cat_118, cat_119], Original ATen: [aten.cat]
        triton_poi_fused_cat_7.run(buf18, buf27, buf37, buf48, 401408, grid=grid(401408), stream=stream0)
        buf28 = reinterpret_tensor(buf29, (4, 32, 56, 56), (602112, 3136, 56, 1), 501760)  # alias
        buf38 = reinterpret_tensor(buf40, (4, 32, 56, 56), (702464, 3136, 56, 1), 501760)  # alias
        buf49 = reinterpret_tensor(buf52, (4, 32, 56, 56), (802816, 3136, 56, 1), 501760)  # alias
        # Source Nodes: [cat_117, cat_118, cat_119], Original ATen: [aten.cat]
        triton_poi_fused_cat_7.run(buf23, buf28, buf38, buf49, 401408, grid=grid(401408), stream=stream0)
        buf30 = empty((4, 192, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock1_denselayer5_norm1, l__mod___features_denseblock1_denselayer5_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf29, primals_392, primals_393, primals_28, primals_29, buf30, 2408448, grid=grid(2408448), stream=stream0)
        del primals_29
        # Source Nodes: [bottleneck_output_8], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_30, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 128, 56, 56), (401408, 3136, 56, 1))
        buf32 = empty((4, 128, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock1_denselayer5_norm2, l__mod___features_denseblock1_denselayer5_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf31, primals_395, primals_396, primals_31, primals_32, buf32, 1605632, grid=grid(1605632), stream=stream0)
        del primals_32
        # Source Nodes: [new_features_8], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_33, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 32, 56, 56), (100352, 3136, 56, 1))
        buf39 = reinterpret_tensor(buf40, (4, 32, 56, 56), (702464, 3136, 56, 1), 602112)  # alias
        buf50 = reinterpret_tensor(buf52, (4, 32, 56, 56), (802816, 3136, 56, 1), 602112)  # alias
        # Source Nodes: [cat_117, cat_118], Original ATen: [aten.cat]
        triton_poi_fused_cat_9.run(buf33, buf39, buf50, 401408, grid=grid(401408), stream=stream0)
        buf41 = empty((4, 224, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock1_denselayer6_norm1, l__mod___features_denseblock1_denselayer6_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf40, primals_398, primals_399, primals_34, primals_35, buf41, 2809856, grid=grid(2809856), stream=stream0)
        del primals_35
        # Source Nodes: [bottleneck_output_10], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_36, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 128, 56, 56), (401408, 3136, 56, 1))
        buf43 = empty((4, 128, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock1_denselayer6_norm2, l__mod___features_denseblock1_denselayer6_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf42, primals_401, primals_402, primals_37, primals_38, buf43, 1605632, grid=grid(1605632), stream=stream0)
        del primals_38
        # Source Nodes: [new_features_10], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_39, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 32, 56, 56), (100352, 3136, 56, 1))
        buf51 = reinterpret_tensor(buf52, (4, 32, 56, 56), (802816, 3136, 56, 1), 702464)  # alias
        # Source Nodes: [cat_117], Original ATen: [aten.cat]
        triton_poi_fused_cat_11.run(buf44, buf51, 401408, grid=grid(401408), stream=stream0)
        buf53 = empty((4, 256, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_transition1_norm, l__mod___features_transition1_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf52, primals_404, primals_405, primals_40, primals_41, buf53, 3211264, grid=grid(3211264), stream=stream0)
        del primals_41
        # Source Nodes: [l__mod___features_transition1_conv], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, primals_42, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 128, 56, 56), (401408, 3136, 56, 1))
        buf55 = reinterpret_tensor(buf44, (4, 128, 28, 28), (100352, 784, 28, 1), 0); del buf44  # reuse
        buf56 = reinterpret_tensor(buf33, (4, 128, 28, 28), (100352, 784, 28, 1), 0); del buf33  # reuse
        buf80 = empty((4, 256, 28, 28), device='cuda', dtype=torch.float32)
        buf75 = reinterpret_tensor(buf80, (4, 128, 28, 28), (200704, 784, 28, 1), 0)  # alias
        buf91 = empty((4, 288, 28, 28), device='cuda', dtype=torch.float32)
        buf85 = reinterpret_tensor(buf91, (4, 128, 28, 28), (225792, 784, 28, 1), 0)  # alias
        buf103 = empty((4, 320, 28, 28), device='cuda', dtype=torch.float32)
        buf96 = reinterpret_tensor(buf103, (4, 128, 28, 28), (250880, 784, 28, 1), 0)  # alias
        buf116 = empty((4, 352, 28, 28), device='cuda', dtype=torch.float32)
        buf108 = reinterpret_tensor(buf116, (4, 128, 28, 28), (275968, 784, 28, 1), 0)  # alias
        buf130 = empty((4, 384, 28, 28), device='cuda', dtype=torch.float32)
        buf121 = reinterpret_tensor(buf130, (4, 128, 28, 28), (301056, 784, 28, 1), 0)  # alias
        buf145 = empty((4, 416, 28, 28), device='cuda', dtype=torch.float32)
        buf135 = reinterpret_tensor(buf145, (4, 128, 28, 28), (326144, 784, 28, 1), 0)  # alias
        buf161 = empty((4, 448, 28, 28), device='cuda', dtype=torch.float32)
        buf150 = reinterpret_tensor(buf161, (4, 128, 28, 28), (351232, 784, 28, 1), 0)  # alias
        buf178 = empty((4, 480, 28, 28), device='cuda', dtype=torch.float32)
        buf166 = reinterpret_tensor(buf178, (4, 128, 28, 28), (376320, 784, 28, 1), 0)  # alias
        buf196 = empty((4, 512, 28, 28), device='cuda', dtype=torch.float32)
        buf183 = reinterpret_tensor(buf196, (4, 128, 28, 28), (401408, 784, 28, 1), 0)  # alias
        # Source Nodes: [cat_104, cat_105, cat_106, cat_107, cat_108, cat_109, cat_110, cat_111, cat_112, l__mod___features_denseblock2_denselayer1_norm1, l__mod___features_denseblock2_denselayer1_relu1, l__mod___features_transition1_pool], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_13.run(buf54, primals_407, primals_408, primals_43, primals_44, buf55, buf56, buf75, buf85, buf96, buf108, buf121, buf135, buf150, buf166, buf183, 401408, grid=grid(401408), stream=stream0)
        del primals_44
        # Source Nodes: [bottleneck_output_12], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, primals_45, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (4, 128, 28, 28), (100352, 784, 28, 1))
        buf58 = reinterpret_tensor(buf23, (4, 128, 28, 28), (100352, 784, 28, 1), 0); del buf23  # reuse
        # Source Nodes: [l__mod___features_denseblock2_denselayer1_norm2, l__mod___features_denseblock2_denselayer1_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf57, primals_410, primals_411, primals_46, primals_47, buf58, 401408, grid=grid(401408), stream=stream0)
        del primals_47
        # Source Nodes: [new_features_12], Original ATen: [aten.convolution]
        buf59 = extern_kernels.convolution(buf58, primals_48, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf59, (4, 32, 28, 28), (25088, 784, 28, 1))
        buf60 = empty((4, 160, 28, 28), device='cuda', dtype=torch.float32)
        buf61 = empty((4, 160, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_115, l__mod___features_denseblock2_denselayer2_norm1, l__mod___features_denseblock2_denselayer2_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_15.run(buf55, buf59, primals_413, primals_414, primals_49, primals_50, buf60, buf61, 501760, grid=grid(501760), stream=stream0)
        del primals_50
        # Source Nodes: [bottleneck_output_14], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_51, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 128, 28, 28), (100352, 784, 28, 1))
        buf63 = reinterpret_tensor(buf18, (4, 128, 28, 28), (100352, 784, 28, 1), 0); del buf18  # reuse
        # Source Nodes: [l__mod___features_denseblock2_denselayer2_norm2, l__mod___features_denseblock2_denselayer2_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf62, primals_416, primals_417, primals_52, primals_53, buf63, 401408, grid=grid(401408), stream=stream0)
        del primals_53
        # Source Nodes: [new_features_14], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_54, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 32, 28, 28), (25088, 784, 28, 1))
        buf65 = empty((4, 192, 28, 28), device='cuda', dtype=torch.float32)
        buf66 = empty((4, 192, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_114, l__mod___features_denseblock2_denselayer3_norm1, l__mod___features_denseblock2_denselayer3_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16.run(buf55, buf59, buf64, primals_419, primals_420, primals_55, primals_56, buf65, buf66, 602112, grid=grid(602112), stream=stream0)
        del primals_56
        # Source Nodes: [bottleneck_output_16], Original ATen: [aten.convolution]
        buf67 = extern_kernels.convolution(buf66, primals_57, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (4, 128, 28, 28), (100352, 784, 28, 1))
        buf68 = reinterpret_tensor(buf13, (4, 128, 28, 28), (100352, 784, 28, 1), 0); del buf13  # reuse
        # Source Nodes: [l__mod___features_denseblock2_denselayer3_norm2, l__mod___features_denseblock2_denselayer3_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf67, primals_422, primals_423, primals_58, primals_59, buf68, 401408, grid=grid(401408), stream=stream0)
        del primals_59
        # Source Nodes: [new_features_16], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, primals_60, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (4, 32, 28, 28), (25088, 784, 28, 1))
        buf70 = empty((4, 224, 28, 28), device='cuda', dtype=torch.float32)
        buf71 = empty((4, 224, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_113, l__mod___features_denseblock2_denselayer4_norm1, l__mod___features_denseblock2_denselayer4_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_17.run(buf55, buf59, buf64, buf69, primals_425, primals_426, primals_61, primals_62, buf70, buf71, 702464, grid=grid(702464), stream=stream0)
        del primals_62
        # Source Nodes: [bottleneck_output_18], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, primals_63, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 128, 28, 28), (100352, 784, 28, 1))
        buf73 = reinterpret_tensor(buf7, (4, 128, 28, 28), (100352, 784, 28, 1), 0); del buf7  # reuse
        # Source Nodes: [l__mod___features_denseblock2_denselayer4_norm2, l__mod___features_denseblock2_denselayer4_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf72, primals_428, primals_429, primals_64, primals_65, buf73, 401408, grid=grid(401408), stream=stream0)
        del primals_65
        # Source Nodes: [new_features_18], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_66, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 32, 28, 28), (25088, 784, 28, 1))
        buf76 = reinterpret_tensor(buf80, (4, 32, 28, 28), (200704, 784, 28, 1), 100352)  # alias
        buf86 = reinterpret_tensor(buf91, (4, 32, 28, 28), (225792, 784, 28, 1), 100352)  # alias
        buf97 = reinterpret_tensor(buf103, (4, 32, 28, 28), (250880, 784, 28, 1), 100352)  # alias
        buf109 = reinterpret_tensor(buf116, (4, 32, 28, 28), (275968, 784, 28, 1), 100352)  # alias
        buf122 = reinterpret_tensor(buf130, (4, 32, 28, 28), (301056, 784, 28, 1), 100352)  # alias
        buf136 = reinterpret_tensor(buf145, (4, 32, 28, 28), (326144, 784, 28, 1), 100352)  # alias
        # Source Nodes: [cat_107, cat_108, cat_109, cat_110, cat_111, cat_112], Original ATen: [aten.cat]
        triton_poi_fused_cat_18.run(buf59, buf76, buf86, buf97, buf109, buf122, buf136, 100352, grid=grid(100352), stream=stream0)
        buf77 = reinterpret_tensor(buf80, (4, 32, 28, 28), (200704, 784, 28, 1), 125440)  # alias
        buf87 = reinterpret_tensor(buf91, (4, 32, 28, 28), (225792, 784, 28, 1), 125440)  # alias
        buf98 = reinterpret_tensor(buf103, (4, 32, 28, 28), (250880, 784, 28, 1), 125440)  # alias
        buf110 = reinterpret_tensor(buf116, (4, 32, 28, 28), (275968, 784, 28, 1), 125440)  # alias
        buf123 = reinterpret_tensor(buf130, (4, 32, 28, 28), (301056, 784, 28, 1), 125440)  # alias
        buf137 = reinterpret_tensor(buf145, (4, 32, 28, 28), (326144, 784, 28, 1), 125440)  # alias
        # Source Nodes: [cat_107, cat_108, cat_109, cat_110, cat_111, cat_112], Original ATen: [aten.cat]
        triton_poi_fused_cat_18.run(buf64, buf77, buf87, buf98, buf110, buf123, buf137, 100352, grid=grid(100352), stream=stream0)
        buf78 = reinterpret_tensor(buf80, (4, 32, 28, 28), (200704, 784, 28, 1), 150528)  # alias
        buf88 = reinterpret_tensor(buf91, (4, 32, 28, 28), (225792, 784, 28, 1), 150528)  # alias
        buf99 = reinterpret_tensor(buf103, (4, 32, 28, 28), (250880, 784, 28, 1), 150528)  # alias
        buf111 = reinterpret_tensor(buf116, (4, 32, 28, 28), (275968, 784, 28, 1), 150528)  # alias
        buf124 = reinterpret_tensor(buf130, (4, 32, 28, 28), (301056, 784, 28, 1), 150528)  # alias
        buf138 = reinterpret_tensor(buf145, (4, 32, 28, 28), (326144, 784, 28, 1), 150528)  # alias
        # Source Nodes: [cat_107, cat_108, cat_109, cat_110, cat_111, cat_112], Original ATen: [aten.cat]
        triton_poi_fused_cat_18.run(buf69, buf78, buf88, buf99, buf111, buf124, buf138, 100352, grid=grid(100352), stream=stream0)
        buf79 = reinterpret_tensor(buf80, (4, 32, 28, 28), (200704, 784, 28, 1), 175616)  # alias
        buf89 = reinterpret_tensor(buf91, (4, 32, 28, 28), (225792, 784, 28, 1), 175616)  # alias
        buf100 = reinterpret_tensor(buf103, (4, 32, 28, 28), (250880, 784, 28, 1), 175616)  # alias
        buf112 = reinterpret_tensor(buf116, (4, 32, 28, 28), (275968, 784, 28, 1), 175616)  # alias
        buf125 = reinterpret_tensor(buf130, (4, 32, 28, 28), (301056, 784, 28, 1), 175616)  # alias
        buf139 = reinterpret_tensor(buf145, (4, 32, 28, 28), (326144, 784, 28, 1), 175616)  # alias
        # Source Nodes: [cat_107, cat_108, cat_109, cat_110, cat_111, cat_112], Original ATen: [aten.cat]
        triton_poi_fused_cat_18.run(buf74, buf79, buf89, buf100, buf112, buf125, buf139, 100352, grid=grid(100352), stream=stream0)
        buf81 = empty((4, 256, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock2_denselayer5_norm1, l__mod___features_denseblock2_denselayer5_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf80, primals_431, primals_432, primals_67, primals_68, buf81, 802816, grid=grid(802816), stream=stream0)
        del primals_68
        # Source Nodes: [bottleneck_output_20], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, primals_69, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 128, 28, 28), (100352, 784, 28, 1))
        buf83 = empty((4, 128, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock2_denselayer5_norm2, l__mod___features_denseblock2_denselayer5_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf82, primals_434, primals_435, primals_70, primals_71, buf83, 401408, grid=grid(401408), stream=stream0)
        del primals_71
        # Source Nodes: [new_features_20], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_72, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 32, 28, 28), (25088, 784, 28, 1))
        buf90 = reinterpret_tensor(buf91, (4, 32, 28, 28), (225792, 784, 28, 1), 200704)  # alias
        buf101 = reinterpret_tensor(buf103, (4, 32, 28, 28), (250880, 784, 28, 1), 200704)  # alias
        buf113 = reinterpret_tensor(buf116, (4, 32, 28, 28), (275968, 784, 28, 1), 200704)  # alias
        buf126 = reinterpret_tensor(buf130, (4, 32, 28, 28), (301056, 784, 28, 1), 200704)  # alias
        buf140 = reinterpret_tensor(buf145, (4, 32, 28, 28), (326144, 784, 28, 1), 200704)  # alias
        # Source Nodes: [cat_107, cat_108, cat_109, cat_110, cat_111], Original ATen: [aten.cat]
        triton_poi_fused_cat_20.run(buf84, buf90, buf101, buf113, buf126, buf140, 100352, grid=grid(100352), stream=stream0)
        buf92 = empty((4, 288, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock2_denselayer6_norm1, l__mod___features_denseblock2_denselayer6_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf91, primals_437, primals_438, primals_73, primals_74, buf92, 903168, grid=grid(903168), stream=stream0)
        del primals_74
        # Source Nodes: [bottleneck_output_22], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, primals_75, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (4, 128, 28, 28), (100352, 784, 28, 1))
        buf94 = empty((4, 128, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock2_denselayer6_norm2, l__mod___features_denseblock2_denselayer6_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf93, primals_440, primals_441, primals_76, primals_77, buf94, 401408, grid=grid(401408), stream=stream0)
        del primals_77
        # Source Nodes: [new_features_22], Original ATen: [aten.convolution]
        buf95 = extern_kernels.convolution(buf94, primals_78, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (4, 32, 28, 28), (25088, 784, 28, 1))
        buf102 = reinterpret_tensor(buf103, (4, 32, 28, 28), (250880, 784, 28, 1), 225792)  # alias
        buf114 = reinterpret_tensor(buf116, (4, 32, 28, 28), (275968, 784, 28, 1), 225792)  # alias
        buf127 = reinterpret_tensor(buf130, (4, 32, 28, 28), (301056, 784, 28, 1), 225792)  # alias
        buf141 = reinterpret_tensor(buf145, (4, 32, 28, 28), (326144, 784, 28, 1), 225792)  # alias
        buf156 = reinterpret_tensor(buf161, (4, 32, 28, 28), (351232, 784, 28, 1), 225792)  # alias
        # Source Nodes: [cat_106, cat_107, cat_108, cat_109, cat_110], Original ATen: [aten.cat]
        triton_poi_fused_cat_22.run(buf95, buf102, buf114, buf127, buf141, buf156, 100352, grid=grid(100352), stream=stream0)
        buf104 = empty((4, 320, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock2_denselayer7_norm1, l__mod___features_denseblock2_denselayer7_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf103, primals_443, primals_444, primals_79, primals_80, buf104, 1003520, grid=grid(1003520), stream=stream0)
        del primals_80
        # Source Nodes: [bottleneck_output_24], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, primals_81, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (4, 128, 28, 28), (100352, 784, 28, 1))
        buf106 = empty((4, 128, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock2_denselayer7_norm2, l__mod___features_denseblock2_denselayer7_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf105, primals_446, primals_447, primals_82, primals_83, buf106, 401408, grid=grid(401408), stream=stream0)
        del primals_83
        # Source Nodes: [new_features_24], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, primals_84, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 32, 28, 28), (25088, 784, 28, 1))
        buf115 = reinterpret_tensor(buf116, (4, 32, 28, 28), (275968, 784, 28, 1), 250880)  # alias
        buf128 = reinterpret_tensor(buf130, (4, 32, 28, 28), (301056, 784, 28, 1), 250880)  # alias
        buf142 = reinterpret_tensor(buf145, (4, 32, 28, 28), (326144, 784, 28, 1), 250880)  # alias
        buf157 = reinterpret_tensor(buf161, (4, 32, 28, 28), (351232, 784, 28, 1), 250880)  # alias
        buf173 = reinterpret_tensor(buf178, (4, 32, 28, 28), (376320, 784, 28, 1), 250880)  # alias
        # Source Nodes: [cat_105, cat_106, cat_107, cat_108, cat_109], Original ATen: [aten.cat]
        triton_poi_fused_cat_24.run(buf107, buf115, buf128, buf142, buf157, buf173, 100352, grid=grid(100352), stream=stream0)
        buf117 = empty((4, 352, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock2_denselayer8_norm1, l__mod___features_denseblock2_denselayer8_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf116, primals_449, primals_450, primals_85, primals_86, buf117, 1103872, grid=grid(1103872), stream=stream0)
        del primals_86
        # Source Nodes: [bottleneck_output_26], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, primals_87, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (4, 128, 28, 28), (100352, 784, 28, 1))
        buf119 = empty((4, 128, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock2_denselayer8_norm2, l__mod___features_denseblock2_denselayer8_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf118, primals_452, primals_453, primals_88, primals_89, buf119, 401408, grid=grid(401408), stream=stream0)
        del primals_89
        # Source Nodes: [new_features_26], Original ATen: [aten.convolution]
        buf120 = extern_kernels.convolution(buf119, primals_90, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf120, (4, 32, 28, 28), (25088, 784, 28, 1))
        buf129 = reinterpret_tensor(buf130, (4, 32, 28, 28), (301056, 784, 28, 1), 275968)  # alias
        buf143 = reinterpret_tensor(buf145, (4, 32, 28, 28), (326144, 784, 28, 1), 275968)  # alias
        buf158 = reinterpret_tensor(buf161, (4, 32, 28, 28), (351232, 784, 28, 1), 275968)  # alias
        buf174 = reinterpret_tensor(buf178, (4, 32, 28, 28), (376320, 784, 28, 1), 275968)  # alias
        buf191 = reinterpret_tensor(buf196, (4, 32, 28, 28), (401408, 784, 28, 1), 275968)  # alias
        # Source Nodes: [cat_104, cat_105, cat_106, cat_107, cat_108], Original ATen: [aten.cat]
        triton_poi_fused_cat_26.run(buf120, buf129, buf143, buf158, buf174, buf191, 100352, grid=grid(100352), stream=stream0)
        buf131 = empty((4, 384, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock2_denselayer9_norm1, l__mod___features_denseblock2_denselayer9_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_27.run(buf130, primals_455, primals_456, primals_91, primals_92, buf131, 1204224, grid=grid(1204224), stream=stream0)
        del primals_92
        # Source Nodes: [bottleneck_output_28], Original ATen: [aten.convolution]
        buf132 = extern_kernels.convolution(buf131, primals_93, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf132, (4, 128, 28, 28), (100352, 784, 28, 1))
        buf133 = empty((4, 128, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock2_denselayer9_norm2, l__mod___features_denseblock2_denselayer9_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf132, primals_458, primals_459, primals_94, primals_95, buf133, 401408, grid=grid(401408), stream=stream0)
        del primals_95
        # Source Nodes: [new_features_28], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf133, primals_96, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (4, 32, 28, 28), (25088, 784, 28, 1))
        buf144 = reinterpret_tensor(buf145, (4, 32, 28, 28), (326144, 784, 28, 1), 301056)  # alias
        buf159 = reinterpret_tensor(buf161, (4, 32, 28, 28), (351232, 784, 28, 1), 301056)  # alias
        buf175 = reinterpret_tensor(buf178, (4, 32, 28, 28), (376320, 784, 28, 1), 301056)  # alias
        buf192 = reinterpret_tensor(buf196, (4, 32, 28, 28), (401408, 784, 28, 1), 301056)  # alias
        # Source Nodes: [cat_104, cat_105, cat_106, cat_107], Original ATen: [aten.cat]
        triton_poi_fused_cat_28.run(buf134, buf144, buf159, buf175, buf192, 100352, grid=grid(100352), stream=stream0)
        buf146 = empty((4, 416, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock2_denselayer10_norm1, l__mod___features_denseblock2_denselayer10_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_29.run(buf145, primals_461, primals_462, primals_97, primals_98, buf146, 1304576, grid=grid(1304576), stream=stream0)
        del primals_98
        # Source Nodes: [bottleneck_output_30], Original ATen: [aten.convolution]
        buf147 = extern_kernels.convolution(buf146, primals_99, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (4, 128, 28, 28), (100352, 784, 28, 1))
        buf148 = empty((4, 128, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock2_denselayer10_norm2, l__mod___features_denseblock2_denselayer10_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf147, primals_464, primals_465, primals_100, primals_101, buf148, 401408, grid=grid(401408), stream=stream0)
        del primals_101
        # Source Nodes: [new_features_30], Original ATen: [aten.convolution]
        buf149 = extern_kernels.convolution(buf148, primals_102, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf149, (4, 32, 28, 28), (25088, 784, 28, 1))
        buf151 = reinterpret_tensor(buf161, (4, 32, 28, 28), (351232, 784, 28, 1), 100352)  # alias
        buf167 = reinterpret_tensor(buf178, (4, 32, 28, 28), (376320, 784, 28, 1), 100352)  # alias
        buf184 = reinterpret_tensor(buf196, (4, 32, 28, 28), (401408, 784, 28, 1), 100352)  # alias
        # Source Nodes: [cat_104, cat_105, cat_106], Original ATen: [aten.cat]
        triton_poi_fused_cat_30.run(buf59, buf151, buf167, buf184, 100352, grid=grid(100352), stream=stream0)
        buf152 = reinterpret_tensor(buf161, (4, 32, 28, 28), (351232, 784, 28, 1), 125440)  # alias
        buf168 = reinterpret_tensor(buf178, (4, 32, 28, 28), (376320, 784, 28, 1), 125440)  # alias
        buf185 = reinterpret_tensor(buf196, (4, 32, 28, 28), (401408, 784, 28, 1), 125440)  # alias
        # Source Nodes: [cat_104, cat_105, cat_106], Original ATen: [aten.cat]
        triton_poi_fused_cat_30.run(buf64, buf152, buf168, buf185, 100352, grid=grid(100352), stream=stream0)
        buf153 = reinterpret_tensor(buf161, (4, 32, 28, 28), (351232, 784, 28, 1), 150528)  # alias
        buf169 = reinterpret_tensor(buf178, (4, 32, 28, 28), (376320, 784, 28, 1), 150528)  # alias
        buf186 = reinterpret_tensor(buf196, (4, 32, 28, 28), (401408, 784, 28, 1), 150528)  # alias
        # Source Nodes: [cat_104, cat_105, cat_106], Original ATen: [aten.cat]
        triton_poi_fused_cat_30.run(buf69, buf153, buf169, buf186, 100352, grid=grid(100352), stream=stream0)
        buf154 = reinterpret_tensor(buf161, (4, 32, 28, 28), (351232, 784, 28, 1), 175616)  # alias
        buf170 = reinterpret_tensor(buf178, (4, 32, 28, 28), (376320, 784, 28, 1), 175616)  # alias
        buf187 = reinterpret_tensor(buf196, (4, 32, 28, 28), (401408, 784, 28, 1), 175616)  # alias
        # Source Nodes: [cat_104, cat_105, cat_106], Original ATen: [aten.cat]
        triton_poi_fused_cat_30.run(buf74, buf154, buf170, buf187, 100352, grid=grid(100352), stream=stream0)
        buf155 = reinterpret_tensor(buf161, (4, 32, 28, 28), (351232, 784, 28, 1), 200704)  # alias
        buf171 = reinterpret_tensor(buf178, (4, 32, 28, 28), (376320, 784, 28, 1), 200704)  # alias
        buf188 = reinterpret_tensor(buf196, (4, 32, 28, 28), (401408, 784, 28, 1), 200704)  # alias
        # Source Nodes: [cat_104, cat_105, cat_106], Original ATen: [aten.cat]
        triton_poi_fused_cat_30.run(buf84, buf155, buf171, buf188, 100352, grid=grid(100352), stream=stream0)
        buf160 = reinterpret_tensor(buf161, (4, 32, 28, 28), (351232, 784, 28, 1), 326144)  # alias
        buf176 = reinterpret_tensor(buf178, (4, 32, 28, 28), (376320, 784, 28, 1), 326144)  # alias
        buf193 = reinterpret_tensor(buf196, (4, 32, 28, 28), (401408, 784, 28, 1), 326144)  # alias
        # Source Nodes: [cat_104, cat_105, cat_106], Original ATen: [aten.cat]
        triton_poi_fused_cat_30.run(buf149, buf160, buf176, buf193, 100352, grid=grid(100352), stream=stream0)
        buf162 = empty((4, 448, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock2_denselayer11_norm1, l__mod___features_denseblock2_denselayer11_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_31.run(buf161, primals_467, primals_468, primals_103, primals_104, buf162, 1404928, grid=grid(1404928), stream=stream0)
        del primals_104
        # Source Nodes: [bottleneck_output_32], Original ATen: [aten.convolution]
        buf163 = extern_kernels.convolution(buf162, primals_105, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (4, 128, 28, 28), (100352, 784, 28, 1))
        buf164 = empty((4, 128, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock2_denselayer11_norm2, l__mod___features_denseblock2_denselayer11_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf163, primals_470, primals_471, primals_106, primals_107, buf164, 401408, grid=grid(401408), stream=stream0)
        del primals_107
        # Source Nodes: [new_features_32], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(buf164, primals_108, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (4, 32, 28, 28), (25088, 784, 28, 1))
        buf172 = reinterpret_tensor(buf178, (4, 32, 28, 28), (376320, 784, 28, 1), 225792)  # alias
        buf189 = reinterpret_tensor(buf196, (4, 32, 28, 28), (401408, 784, 28, 1), 225792)  # alias
        # Source Nodes: [cat_104, cat_105], Original ATen: [aten.cat]
        triton_poi_fused_cat_32.run(buf95, buf172, buf189, 100352, grid=grid(100352), stream=stream0)
        buf177 = reinterpret_tensor(buf178, (4, 32, 28, 28), (376320, 784, 28, 1), 351232)  # alias
        buf194 = reinterpret_tensor(buf196, (4, 32, 28, 28), (401408, 784, 28, 1), 351232)  # alias
        # Source Nodes: [cat_104, cat_105], Original ATen: [aten.cat]
        triton_poi_fused_cat_32.run(buf165, buf177, buf194, 100352, grid=grid(100352), stream=stream0)
        buf179 = empty((4, 480, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock2_denselayer12_norm1, l__mod___features_denseblock2_denselayer12_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_33.run(buf178, primals_473, primals_474, primals_109, primals_110, buf179, 1505280, grid=grid(1505280), stream=stream0)
        del primals_110
        # Source Nodes: [bottleneck_output_34], Original ATen: [aten.convolution]
        buf180 = extern_kernels.convolution(buf179, primals_111, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf180, (4, 128, 28, 28), (100352, 784, 28, 1))
        buf181 = empty((4, 128, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock2_denselayer12_norm2, l__mod___features_denseblock2_denselayer12_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf180, primals_476, primals_477, primals_112, primals_113, buf181, 401408, grid=grid(401408), stream=stream0)
        del primals_113
        # Source Nodes: [new_features_34], Original ATen: [aten.convolution]
        buf182 = extern_kernels.convolution(buf181, primals_114, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (4, 32, 28, 28), (25088, 784, 28, 1))
        buf190 = reinterpret_tensor(buf196, (4, 32, 28, 28), (401408, 784, 28, 1), 250880)  # alias
        # Source Nodes: [cat_104], Original ATen: [aten.cat]
        triton_poi_fused_cat_34.run(buf107, buf190, 100352, grid=grid(100352), stream=stream0)
        buf195 = reinterpret_tensor(buf196, (4, 32, 28, 28), (401408, 784, 28, 1), 376320)  # alias
        # Source Nodes: [cat_104], Original ATen: [aten.cat]
        triton_poi_fused_cat_34.run(buf182, buf195, 100352, grid=grid(100352), stream=stream0)
        buf197 = empty((4, 512, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_transition2_norm, l__mod___features_transition2_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_35.run(buf196, primals_479, primals_480, primals_115, primals_116, buf197, 1605632, grid=grid(1605632), stream=stream0)
        del primals_116
        # Source Nodes: [l__mod___features_transition2_conv], Original ATen: [aten.convolution]
        buf198 = extern_kernels.convolution(buf197, primals_117, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (4, 256, 28, 28), (200704, 784, 28, 1))
        buf199 = empty((4, 256, 14, 14), device='cuda', dtype=torch.float32)
        buf200 = empty((4, 256, 14, 14), device='cuda', dtype=torch.float32)
        buf224 = empty((4, 384, 14, 14), device='cuda', dtype=torch.float32)
        buf219 = reinterpret_tensor(buf224, (4, 256, 14, 14), (75264, 196, 14, 1), 0)  # alias
        buf235 = empty((4, 416, 14, 14), device='cuda', dtype=torch.float32)
        buf229 = reinterpret_tensor(buf235, (4, 256, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf247 = empty((4, 448, 14, 14), device='cuda', dtype=torch.float32)
        buf240 = reinterpret_tensor(buf247, (4, 256, 14, 14), (87808, 196, 14, 1), 0)  # alias
        buf260 = empty((4, 480, 14, 14), device='cuda', dtype=torch.float32)
        buf252 = reinterpret_tensor(buf260, (4, 256, 14, 14), (94080, 196, 14, 1), 0)  # alias
        buf274 = empty((4, 512, 14, 14), device='cuda', dtype=torch.float32)
        buf265 = reinterpret_tensor(buf274, (4, 256, 14, 14), (100352, 196, 14, 1), 0)  # alias
        buf289 = empty((4, 544, 14, 14), device='cuda', dtype=torch.float32)
        buf279 = reinterpret_tensor(buf289, (4, 256, 14, 14), (106624, 196, 14, 1), 0)  # alias
        buf305 = empty((4, 576, 14, 14), device='cuda', dtype=torch.float32)
        buf294 = reinterpret_tensor(buf305, (4, 256, 14, 14), (112896, 196, 14, 1), 0)  # alias
        buf322 = empty((4, 608, 14, 14), device='cuda', dtype=torch.float32)
        buf310 = reinterpret_tensor(buf322, (4, 256, 14, 14), (119168, 196, 14, 1), 0)  # alias
        buf340 = empty((4, 640, 14, 14), device='cuda', dtype=torch.float32)
        buf327 = reinterpret_tensor(buf340, (4, 256, 14, 14), (125440, 196, 14, 1), 0)  # alias
        buf359 = empty((4, 672, 14, 14), device='cuda', dtype=torch.float32)
        buf345 = reinterpret_tensor(buf359, (4, 256, 14, 14), (131712, 196, 14, 1), 0)  # alias
        buf379 = empty((4, 704, 14, 14), device='cuda', dtype=torch.float32)
        buf364 = reinterpret_tensor(buf379, (4, 256, 14, 14), (137984, 196, 14, 1), 0)  # alias
        buf400 = empty((4, 736, 14, 14), device='cuda', dtype=torch.float32)
        buf384 = reinterpret_tensor(buf400, (4, 256, 14, 14), (144256, 196, 14, 1), 0)  # alias
        buf422 = empty((4, 768, 14, 14), device='cuda', dtype=torch.float32)
        buf405 = reinterpret_tensor(buf422, (4, 256, 14, 14), (150528, 196, 14, 1), 0)  # alias
        buf445 = empty((4, 800, 14, 14), device='cuda', dtype=torch.float32)
        buf427 = reinterpret_tensor(buf445, (4, 256, 14, 14), (156800, 196, 14, 1), 0)  # alias
        buf469 = empty((4, 832, 14, 14), device='cuda', dtype=torch.float32)
        buf450 = reinterpret_tensor(buf469, (4, 256, 14, 14), (163072, 196, 14, 1), 0)  # alias
        buf494 = empty((4, 864, 14, 14), device='cuda', dtype=torch.float32)
        buf474 = reinterpret_tensor(buf494, (4, 256, 14, 14), (169344, 196, 14, 1), 0)  # alias
        buf520 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        buf499 = reinterpret_tensor(buf520, (4, 256, 14, 14), (175616, 196, 14, 1), 0)  # alias
        buf547 = empty((4, 928, 14, 14), device='cuda', dtype=torch.float32)
        buf525 = reinterpret_tensor(buf547, (4, 256, 14, 14), (181888, 196, 14, 1), 0)  # alias
        buf575 = empty((4, 960, 14, 14), device='cuda', dtype=torch.float32)
        buf552 = reinterpret_tensor(buf575, (4, 256, 14, 14), (188160, 196, 14, 1), 0)  # alias
        buf604 = empty((4, 992, 14, 14), device='cuda', dtype=torch.float32)
        buf580 = reinterpret_tensor(buf604, (4, 256, 14, 14), (194432, 196, 14, 1), 0)  # alias
        buf634 = empty((4, 1024, 14, 14), device='cuda', dtype=torch.float32)
        buf609 = reinterpret_tensor(buf634, (4, 256, 14, 14), (200704, 196, 14, 1), 0)  # alias
        # Source Nodes: [cat_79, cat_80, cat_81, cat_82, cat_83, cat_84, cat_85, cat_86, cat_87, cat_88, cat_89, cat_90, cat_91, cat_92, cat_93, cat_94, cat_95, cat_96, cat_97, cat_98, cat_99, l__mod___features_denseblock3_denselayer1_norm1, l__mod___features_denseblock3_denselayer1_relu1, l__mod___features_transition2_pool], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_36.run(buf198, primals_482, primals_483, primals_118, primals_119, buf199, buf200, buf219, buf229, buf240, buf252, buf265, buf279, buf294, buf310, buf327, buf345, buf364, buf384, buf405, buf427, buf450, buf474, buf499, buf525, buf552, buf580, buf609, 200704, grid=grid(200704), stream=stream0)
        del primals_119
        # Source Nodes: [bottleneck_output_36], Original ATen: [aten.convolution]
        buf201 = extern_kernels.convolution(buf200, primals_120, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (4, 128, 14, 14), (25088, 196, 14, 1))
        buf202 = reinterpret_tensor(buf182, (4, 128, 14, 14), (25088, 196, 14, 1), 0); del buf182  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer1_norm2, l__mod___features_denseblock3_denselayer1_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf201, primals_485, primals_486, primals_121, primals_122, buf202, 100352, grid=grid(100352), stream=stream0)
        del primals_122
        # Source Nodes: [new_features_36], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, primals_123, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (4, 32, 14, 14), (6272, 196, 14, 1))
        buf204 = empty((4, 288, 14, 14), device='cuda', dtype=torch.float32)
        buf205 = empty((4, 288, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_102, l__mod___features_denseblock3_denselayer2_norm1, l__mod___features_denseblock3_denselayer2_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_38.run(buf199, buf203, primals_488, primals_489, primals_124, primals_125, buf204, buf205, 225792, grid=grid(225792), stream=stream0)
        del primals_125
        # Source Nodes: [bottleneck_output_38], Original ATen: [aten.convolution]
        buf206 = extern_kernels.convolution(buf205, primals_126, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (4, 128, 14, 14), (25088, 196, 14, 1))
        buf207 = reinterpret_tensor(buf107, (4, 128, 14, 14), (25088, 196, 14, 1), 0); del buf107  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer2_norm2, l__mod___features_denseblock3_denselayer2_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf206, primals_491, primals_492, primals_127, primals_128, buf207, 100352, grid=grid(100352), stream=stream0)
        del primals_128
        # Source Nodes: [new_features_38], Original ATen: [aten.convolution]
        buf208 = extern_kernels.convolution(buf207, primals_129, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (4, 32, 14, 14), (6272, 196, 14, 1))
        buf209 = empty((4, 320, 14, 14), device='cuda', dtype=torch.float32)
        buf210 = empty((4, 320, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_101, l__mod___features_denseblock3_denselayer3_norm1, l__mod___features_denseblock3_denselayer3_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_39.run(buf199, buf203, buf208, primals_494, primals_495, primals_130, primals_131, buf209, buf210, 250880, grid=grid(250880), stream=stream0)
        del primals_131
        # Source Nodes: [bottleneck_output_40], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf210, primals_132, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (4, 128, 14, 14), (25088, 196, 14, 1))
        buf212 = reinterpret_tensor(buf165, (4, 128, 14, 14), (25088, 196, 14, 1), 0); del buf165  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer3_norm2, l__mod___features_denseblock3_denselayer3_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf211, primals_497, primals_498, primals_133, primals_134, buf212, 100352, grid=grid(100352), stream=stream0)
        del primals_134
        # Source Nodes: [new_features_40], Original ATen: [aten.convolution]
        buf213 = extern_kernels.convolution(buf212, primals_135, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf213, (4, 32, 14, 14), (6272, 196, 14, 1))
        buf214 = empty((4, 352, 14, 14), device='cuda', dtype=torch.float32)
        buf215 = empty((4, 352, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_100, l__mod___features_denseblock3_denselayer4_norm1, l__mod___features_denseblock3_denselayer4_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_40.run(buf199, buf203, buf208, buf213, primals_500, primals_501, primals_136, primals_137, buf214, buf215, 275968, grid=grid(275968), stream=stream0)
        del primals_137
        # Source Nodes: [bottleneck_output_42], Original ATen: [aten.convolution]
        buf216 = extern_kernels.convolution(buf215, primals_138, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf216, (4, 128, 14, 14), (25088, 196, 14, 1))
        buf217 = reinterpret_tensor(buf95, (4, 128, 14, 14), (25088, 196, 14, 1), 0); del buf95  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer4_norm2, l__mod___features_denseblock3_denselayer4_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf216, primals_503, primals_504, primals_139, primals_140, buf217, 100352, grid=grid(100352), stream=stream0)
        del primals_140
        # Source Nodes: [new_features_42], Original ATen: [aten.convolution]
        buf218 = extern_kernels.convolution(buf217, primals_141, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (4, 32, 14, 14), (6272, 196, 14, 1))
        buf220 = reinterpret_tensor(buf224, (4, 32, 14, 14), (75264, 196, 14, 1), 50176)  # alias
        buf230 = reinterpret_tensor(buf235, (4, 32, 14, 14), (81536, 196, 14, 1), 50176)  # alias
        buf241 = reinterpret_tensor(buf247, (4, 32, 14, 14), (87808, 196, 14, 1), 50176)  # alias
        buf253 = reinterpret_tensor(buf260, (4, 32, 14, 14), (94080, 196, 14, 1), 50176)  # alias
        buf266 = reinterpret_tensor(buf274, (4, 32, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        buf280 = reinterpret_tensor(buf289, (4, 32, 14, 14), (106624, 196, 14, 1), 50176)  # alias
        # Source Nodes: [cat_94, cat_95, cat_96, cat_97, cat_98, cat_99], Original ATen: [aten.cat]
        triton_poi_fused_cat_41.run(buf203, buf220, buf230, buf241, buf253, buf266, buf280, 25088, grid=grid(25088), stream=stream0)
        buf221 = reinterpret_tensor(buf224, (4, 32, 14, 14), (75264, 196, 14, 1), 56448)  # alias
        buf231 = reinterpret_tensor(buf235, (4, 32, 14, 14), (81536, 196, 14, 1), 56448)  # alias
        buf242 = reinterpret_tensor(buf247, (4, 32, 14, 14), (87808, 196, 14, 1), 56448)  # alias
        buf254 = reinterpret_tensor(buf260, (4, 32, 14, 14), (94080, 196, 14, 1), 56448)  # alias
        buf267 = reinterpret_tensor(buf274, (4, 32, 14, 14), (100352, 196, 14, 1), 56448)  # alias
        buf281 = reinterpret_tensor(buf289, (4, 32, 14, 14), (106624, 196, 14, 1), 56448)  # alias
        # Source Nodes: [cat_94, cat_95, cat_96, cat_97, cat_98, cat_99], Original ATen: [aten.cat]
        triton_poi_fused_cat_41.run(buf208, buf221, buf231, buf242, buf254, buf267, buf281, 25088, grid=grid(25088), stream=stream0)
        buf222 = reinterpret_tensor(buf224, (4, 32, 14, 14), (75264, 196, 14, 1), 62720)  # alias
        buf232 = reinterpret_tensor(buf235, (4, 32, 14, 14), (81536, 196, 14, 1), 62720)  # alias
        buf243 = reinterpret_tensor(buf247, (4, 32, 14, 14), (87808, 196, 14, 1), 62720)  # alias
        buf255 = reinterpret_tensor(buf260, (4, 32, 14, 14), (94080, 196, 14, 1), 62720)  # alias
        buf268 = reinterpret_tensor(buf274, (4, 32, 14, 14), (100352, 196, 14, 1), 62720)  # alias
        buf282 = reinterpret_tensor(buf289, (4, 32, 14, 14), (106624, 196, 14, 1), 62720)  # alias
        # Source Nodes: [cat_94, cat_95, cat_96, cat_97, cat_98, cat_99], Original ATen: [aten.cat]
        triton_poi_fused_cat_41.run(buf213, buf222, buf232, buf243, buf255, buf268, buf282, 25088, grid=grid(25088), stream=stream0)
        buf223 = reinterpret_tensor(buf224, (4, 32, 14, 14), (75264, 196, 14, 1), 68992)  # alias
        buf233 = reinterpret_tensor(buf235, (4, 32, 14, 14), (81536, 196, 14, 1), 68992)  # alias
        buf244 = reinterpret_tensor(buf247, (4, 32, 14, 14), (87808, 196, 14, 1), 68992)  # alias
        buf256 = reinterpret_tensor(buf260, (4, 32, 14, 14), (94080, 196, 14, 1), 68992)  # alias
        buf269 = reinterpret_tensor(buf274, (4, 32, 14, 14), (100352, 196, 14, 1), 68992)  # alias
        buf283 = reinterpret_tensor(buf289, (4, 32, 14, 14), (106624, 196, 14, 1), 68992)  # alias
        # Source Nodes: [cat_94, cat_95, cat_96, cat_97, cat_98, cat_99], Original ATen: [aten.cat]
        triton_poi_fused_cat_41.run(buf218, buf223, buf233, buf244, buf256, buf269, buf283, 25088, grid=grid(25088), stream=stream0)
        buf225 = empty((4, 384, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer5_norm1, l__mod___features_denseblock3_denselayer5_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_42.run(buf224, primals_506, primals_507, primals_142, primals_143, buf225, 301056, grid=grid(301056), stream=stream0)
        del primals_143
        # Source Nodes: [bottleneck_output_44], Original ATen: [aten.convolution]
        buf226 = extern_kernels.convolution(buf225, primals_144, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (4, 128, 14, 14), (25088, 196, 14, 1))
        buf227 = reinterpret_tensor(buf149, (4, 128, 14, 14), (25088, 196, 14, 1), 0); del buf149  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer5_norm2, l__mod___features_denseblock3_denselayer5_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf226, primals_509, primals_510, primals_145, primals_146, buf227, 100352, grid=grid(100352), stream=stream0)
        del primals_146
        # Source Nodes: [new_features_44], Original ATen: [aten.convolution]
        buf228 = extern_kernels.convolution(buf227, primals_147, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (4, 32, 14, 14), (6272, 196, 14, 1))
        buf234 = reinterpret_tensor(buf235, (4, 32, 14, 14), (81536, 196, 14, 1), 75264)  # alias
        buf245 = reinterpret_tensor(buf247, (4, 32, 14, 14), (87808, 196, 14, 1), 75264)  # alias
        buf257 = reinterpret_tensor(buf260, (4, 32, 14, 14), (94080, 196, 14, 1), 75264)  # alias
        buf270 = reinterpret_tensor(buf274, (4, 32, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        buf284 = reinterpret_tensor(buf289, (4, 32, 14, 14), (106624, 196, 14, 1), 75264)  # alias
        # Source Nodes: [cat_94, cat_95, cat_96, cat_97, cat_98], Original ATen: [aten.cat]
        triton_poi_fused_cat_43.run(buf228, buf234, buf245, buf257, buf270, buf284, 25088, grid=grid(25088), stream=stream0)
        buf236 = empty((4, 416, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer6_norm1, l__mod___features_denseblock3_denselayer6_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_44.run(buf235, primals_512, primals_513, primals_148, primals_149, buf236, 326144, grid=grid(326144), stream=stream0)
        del primals_149
        # Source Nodes: [bottleneck_output_46], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf236, primals_150, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (4, 128, 14, 14), (25088, 196, 14, 1))
        buf238 = reinterpret_tensor(buf84, (4, 128, 14, 14), (25088, 196, 14, 1), 0); del buf84  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer6_norm2, l__mod___features_denseblock3_denselayer6_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf237, primals_515, primals_516, primals_151, primals_152, buf238, 100352, grid=grid(100352), stream=stream0)
        del primals_152
        # Source Nodes: [new_features_46], Original ATen: [aten.convolution]
        buf239 = extern_kernels.convolution(buf238, primals_153, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (4, 32, 14, 14), (6272, 196, 14, 1))
        buf246 = reinterpret_tensor(buf247, (4, 32, 14, 14), (87808, 196, 14, 1), 81536)  # alias
        buf258 = reinterpret_tensor(buf260, (4, 32, 14, 14), (94080, 196, 14, 1), 81536)  # alias
        buf271 = reinterpret_tensor(buf274, (4, 32, 14, 14), (100352, 196, 14, 1), 81536)  # alias
        buf285 = reinterpret_tensor(buf289, (4, 32, 14, 14), (106624, 196, 14, 1), 81536)  # alias
        buf300 = reinterpret_tensor(buf305, (4, 32, 14, 14), (112896, 196, 14, 1), 81536)  # alias
        # Source Nodes: [cat_93, cat_94, cat_95, cat_96, cat_97], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf239, buf246, buf258, buf271, buf285, buf300, 25088, grid=grid(25088), stream=stream0)
        buf248 = empty((4, 448, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer7_norm1, l__mod___features_denseblock3_denselayer7_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_46.run(buf247, primals_518, primals_519, primals_154, primals_155, buf248, 351232, grid=grid(351232), stream=stream0)
        del primals_155
        # Source Nodes: [bottleneck_output_48], Original ATen: [aten.convolution]
        buf249 = extern_kernels.convolution(buf248, primals_156, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf249, (4, 128, 14, 14), (25088, 196, 14, 1))
        buf250 = reinterpret_tensor(buf74, (4, 128, 14, 14), (25088, 196, 14, 1), 0); del buf74  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer7_norm2, l__mod___features_denseblock3_denselayer7_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf249, primals_521, primals_522, primals_157, primals_158, buf250, 100352, grid=grid(100352), stream=stream0)
        del primals_158
        # Source Nodes: [new_features_48], Original ATen: [aten.convolution]
        buf251 = extern_kernels.convolution(buf250, primals_159, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf251, (4, 32, 14, 14), (6272, 196, 14, 1))
        buf259 = reinterpret_tensor(buf260, (4, 32, 14, 14), (94080, 196, 14, 1), 87808)  # alias
        buf272 = reinterpret_tensor(buf274, (4, 32, 14, 14), (100352, 196, 14, 1), 87808)  # alias
        buf286 = reinterpret_tensor(buf289, (4, 32, 14, 14), (106624, 196, 14, 1), 87808)  # alias
        buf301 = reinterpret_tensor(buf305, (4, 32, 14, 14), (112896, 196, 14, 1), 87808)  # alias
        buf317 = reinterpret_tensor(buf322, (4, 32, 14, 14), (119168, 196, 14, 1), 87808)  # alias
        # Source Nodes: [cat_92, cat_93, cat_94, cat_95, cat_96], Original ATen: [aten.cat]
        triton_poi_fused_cat_47.run(buf251, buf259, buf272, buf286, buf301, buf317, 25088, grid=grid(25088), stream=stream0)
        buf261 = empty((4, 480, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer8_norm1, l__mod___features_denseblock3_denselayer8_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_48.run(buf260, primals_524, primals_525, primals_160, primals_161, buf261, 376320, grid=grid(376320), stream=stream0)
        del primals_161
        # Source Nodes: [bottleneck_output_50], Original ATen: [aten.convolution]
        buf262 = extern_kernels.convolution(buf261, primals_162, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (4, 128, 14, 14), (25088, 196, 14, 1))
        buf263 = reinterpret_tensor(buf69, (4, 128, 14, 14), (25088, 196, 14, 1), 0); del buf69  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer8_norm2, l__mod___features_denseblock3_denselayer8_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf262, primals_527, primals_528, primals_163, primals_164, buf263, 100352, grid=grid(100352), stream=stream0)
        del primals_164
        # Source Nodes: [new_features_50], Original ATen: [aten.convolution]
        buf264 = extern_kernels.convolution(buf263, primals_165, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf264, (4, 32, 14, 14), (6272, 196, 14, 1))
        buf273 = reinterpret_tensor(buf274, (4, 32, 14, 14), (100352, 196, 14, 1), 94080)  # alias
        buf287 = reinterpret_tensor(buf289, (4, 32, 14, 14), (106624, 196, 14, 1), 94080)  # alias
        buf302 = reinterpret_tensor(buf305, (4, 32, 14, 14), (112896, 196, 14, 1), 94080)  # alias
        buf318 = reinterpret_tensor(buf322, (4, 32, 14, 14), (119168, 196, 14, 1), 94080)  # alias
        buf335 = reinterpret_tensor(buf340, (4, 32, 14, 14), (125440, 196, 14, 1), 94080)  # alias
        # Source Nodes: [cat_91, cat_92, cat_93, cat_94, cat_95], Original ATen: [aten.cat]
        triton_poi_fused_cat_49.run(buf264, buf273, buf287, buf302, buf318, buf335, 25088, grid=grid(25088), stream=stream0)
        buf275 = empty((4, 512, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer9_norm1, l__mod___features_denseblock3_denselayer9_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf274, primals_530, primals_531, primals_166, primals_167, buf275, 401408, grid=grid(401408), stream=stream0)
        del primals_167
        # Source Nodes: [bottleneck_output_52], Original ATen: [aten.convolution]
        buf276 = extern_kernels.convolution(buf275, primals_168, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf276, (4, 128, 14, 14), (25088, 196, 14, 1))
        buf277 = reinterpret_tensor(buf64, (4, 128, 14, 14), (25088, 196, 14, 1), 0); del buf64  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer9_norm2, l__mod___features_denseblock3_denselayer9_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf276, primals_533, primals_534, primals_169, primals_170, buf277, 100352, grid=grid(100352), stream=stream0)
        del primals_170
        # Source Nodes: [new_features_52], Original ATen: [aten.convolution]
        buf278 = extern_kernels.convolution(buf277, primals_171, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf278, (4, 32, 14, 14), (6272, 196, 14, 1))
        buf288 = reinterpret_tensor(buf289, (4, 32, 14, 14), (106624, 196, 14, 1), 100352)  # alias
        buf303 = reinterpret_tensor(buf305, (4, 32, 14, 14), (112896, 196, 14, 1), 100352)  # alias
        buf319 = reinterpret_tensor(buf322, (4, 32, 14, 14), (119168, 196, 14, 1), 100352)  # alias
        buf336 = reinterpret_tensor(buf340, (4, 32, 14, 14), (125440, 196, 14, 1), 100352)  # alias
        # Source Nodes: [cat_91, cat_92, cat_93, cat_94], Original ATen: [aten.cat]
        triton_poi_fused_cat_51.run(buf278, buf288, buf303, buf319, buf336, 25088, grid=grid(25088), stream=stream0)
        buf290 = empty((4, 544, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer10_norm1, l__mod___features_denseblock3_denselayer10_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_52.run(buf289, primals_536, primals_537, primals_172, primals_173, buf290, 426496, grid=grid(426496), stream=stream0)
        del primals_173
        # Source Nodes: [bottleneck_output_54], Original ATen: [aten.convolution]
        buf291 = extern_kernels.convolution(buf290, primals_174, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf291, (4, 128, 14, 14), (25088, 196, 14, 1))
        buf292 = reinterpret_tensor(buf59, (4, 128, 14, 14), (25088, 196, 14, 1), 0); del buf59  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer10_norm2, l__mod___features_denseblock3_denselayer10_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf291, primals_539, primals_540, primals_175, primals_176, buf292, 100352, grid=grid(100352), stream=stream0)
        del primals_176
        # Source Nodes: [new_features_54], Original ATen: [aten.convolution]
        buf293 = extern_kernels.convolution(buf292, primals_177, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf293, (4, 32, 14, 14), (6272, 196, 14, 1))
        buf295 = reinterpret_tensor(buf305, (4, 32, 14, 14), (112896, 196, 14, 1), 50176)  # alias
        buf311 = reinterpret_tensor(buf322, (4, 32, 14, 14), (119168, 196, 14, 1), 50176)  # alias
        buf328 = reinterpret_tensor(buf340, (4, 32, 14, 14), (125440, 196, 14, 1), 50176)  # alias
        buf346 = reinterpret_tensor(buf359, (4, 32, 14, 14), (131712, 196, 14, 1), 50176)  # alias
        # Source Nodes: [cat_90, cat_91, cat_92, cat_93], Original ATen: [aten.cat]
        triton_poi_fused_cat_53.run(buf203, buf295, buf311, buf328, buf346, 25088, grid=grid(25088), stream=stream0)
        buf296 = reinterpret_tensor(buf305, (4, 32, 14, 14), (112896, 196, 14, 1), 56448)  # alias
        buf312 = reinterpret_tensor(buf322, (4, 32, 14, 14), (119168, 196, 14, 1), 56448)  # alias
        buf329 = reinterpret_tensor(buf340, (4, 32, 14, 14), (125440, 196, 14, 1), 56448)  # alias
        buf347 = reinterpret_tensor(buf359, (4, 32, 14, 14), (131712, 196, 14, 1), 56448)  # alias
        # Source Nodes: [cat_90, cat_91, cat_92, cat_93], Original ATen: [aten.cat]
        triton_poi_fused_cat_53.run(buf208, buf296, buf312, buf329, buf347, 25088, grid=grid(25088), stream=stream0)
        buf297 = reinterpret_tensor(buf305, (4, 32, 14, 14), (112896, 196, 14, 1), 62720)  # alias
        buf313 = reinterpret_tensor(buf322, (4, 32, 14, 14), (119168, 196, 14, 1), 62720)  # alias
        buf330 = reinterpret_tensor(buf340, (4, 32, 14, 14), (125440, 196, 14, 1), 62720)  # alias
        buf348 = reinterpret_tensor(buf359, (4, 32, 14, 14), (131712, 196, 14, 1), 62720)  # alias
        # Source Nodes: [cat_90, cat_91, cat_92, cat_93], Original ATen: [aten.cat]
        triton_poi_fused_cat_53.run(buf213, buf297, buf313, buf330, buf348, 25088, grid=grid(25088), stream=stream0)
        buf298 = reinterpret_tensor(buf305, (4, 32, 14, 14), (112896, 196, 14, 1), 68992)  # alias
        buf314 = reinterpret_tensor(buf322, (4, 32, 14, 14), (119168, 196, 14, 1), 68992)  # alias
        buf331 = reinterpret_tensor(buf340, (4, 32, 14, 14), (125440, 196, 14, 1), 68992)  # alias
        buf349 = reinterpret_tensor(buf359, (4, 32, 14, 14), (131712, 196, 14, 1), 68992)  # alias
        # Source Nodes: [cat_90, cat_91, cat_92, cat_93], Original ATen: [aten.cat]
        triton_poi_fused_cat_53.run(buf218, buf298, buf314, buf331, buf349, 25088, grid=grid(25088), stream=stream0)
        buf299 = reinterpret_tensor(buf305, (4, 32, 14, 14), (112896, 196, 14, 1), 75264)  # alias
        buf315 = reinterpret_tensor(buf322, (4, 32, 14, 14), (119168, 196, 14, 1), 75264)  # alias
        buf332 = reinterpret_tensor(buf340, (4, 32, 14, 14), (125440, 196, 14, 1), 75264)  # alias
        buf350 = reinterpret_tensor(buf359, (4, 32, 14, 14), (131712, 196, 14, 1), 75264)  # alias
        # Source Nodes: [cat_90, cat_91, cat_92, cat_93], Original ATen: [aten.cat]
        triton_poi_fused_cat_53.run(buf228, buf299, buf315, buf332, buf350, 25088, grid=grid(25088), stream=stream0)
        buf304 = reinterpret_tensor(buf305, (4, 32, 14, 14), (112896, 196, 14, 1), 106624)  # alias
        buf320 = reinterpret_tensor(buf322, (4, 32, 14, 14), (119168, 196, 14, 1), 106624)  # alias
        buf337 = reinterpret_tensor(buf340, (4, 32, 14, 14), (125440, 196, 14, 1), 106624)  # alias
        buf355 = reinterpret_tensor(buf359, (4, 32, 14, 14), (131712, 196, 14, 1), 106624)  # alias
        # Source Nodes: [cat_90, cat_91, cat_92, cat_93], Original ATen: [aten.cat]
        triton_poi_fused_cat_53.run(buf293, buf304, buf320, buf337, buf355, 25088, grid=grid(25088), stream=stream0)
        buf306 = empty((4, 576, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer11_norm1, l__mod___features_denseblock3_denselayer11_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_54.run(buf305, primals_542, primals_543, primals_178, primals_179, buf306, 451584, grid=grid(451584), stream=stream0)
        del primals_179
        # Source Nodes: [bottleneck_output_56], Original ATen: [aten.convolution]
        buf307 = extern_kernels.convolution(buf306, primals_180, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf307, (4, 128, 14, 14), (25088, 196, 14, 1))
        buf308 = reinterpret_tensor(buf134, (4, 128, 14, 14), (25088, 196, 14, 1), 0); del buf134  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer11_norm2, l__mod___features_denseblock3_denselayer11_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf307, primals_545, primals_546, primals_181, primals_182, buf308, 100352, grid=grid(100352), stream=stream0)
        del primals_182
        # Source Nodes: [new_features_56], Original ATen: [aten.convolution]
        buf309 = extern_kernels.convolution(buf308, primals_183, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf309, (4, 32, 14, 14), (6272, 196, 14, 1))
        buf316 = reinterpret_tensor(buf322, (4, 32, 14, 14), (119168, 196, 14, 1), 81536)  # alias
        buf333 = reinterpret_tensor(buf340, (4, 32, 14, 14), (125440, 196, 14, 1), 81536)  # alias
        buf351 = reinterpret_tensor(buf359, (4, 32, 14, 14), (131712, 196, 14, 1), 81536)  # alias
        buf370 = reinterpret_tensor(buf379, (4, 32, 14, 14), (137984, 196, 14, 1), 81536)  # alias
        # Source Nodes: [cat_89, cat_90, cat_91, cat_92], Original ATen: [aten.cat]
        triton_poi_fused_cat_55.run(buf239, buf316, buf333, buf351, buf370, 25088, grid=grid(25088), stream=stream0)
        buf321 = reinterpret_tensor(buf322, (4, 32, 14, 14), (119168, 196, 14, 1), 112896)  # alias
        buf338 = reinterpret_tensor(buf340, (4, 32, 14, 14), (125440, 196, 14, 1), 112896)  # alias
        buf356 = reinterpret_tensor(buf359, (4, 32, 14, 14), (131712, 196, 14, 1), 112896)  # alias
        buf375 = reinterpret_tensor(buf379, (4, 32, 14, 14), (137984, 196, 14, 1), 112896)  # alias
        # Source Nodes: [cat_89, cat_90, cat_91, cat_92], Original ATen: [aten.cat]
        triton_poi_fused_cat_55.run(buf309, buf321, buf338, buf356, buf375, 25088, grid=grid(25088), stream=stream0)
        buf323 = empty((4, 608, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer12_norm1, l__mod___features_denseblock3_denselayer12_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_56.run(buf322, primals_548, primals_549, primals_184, primals_185, buf323, 476672, grid=grid(476672), stream=stream0)
        del primals_185
        # Source Nodes: [bottleneck_output_58], Original ATen: [aten.convolution]
        buf324 = extern_kernels.convolution(buf323, primals_186, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf324, (4, 128, 14, 14), (25088, 196, 14, 1))
        buf325 = reinterpret_tensor(buf120, (4, 128, 14, 14), (25088, 196, 14, 1), 0); del buf120  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer12_norm2, l__mod___features_denseblock3_denselayer12_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf324, primals_551, primals_552, primals_187, primals_188, buf325, 100352, grid=grid(100352), stream=stream0)
        del primals_188
        # Source Nodes: [new_features_58], Original ATen: [aten.convolution]
        buf326 = extern_kernels.convolution(buf325, primals_189, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf326, (4, 32, 14, 14), (6272, 196, 14, 1))
        buf334 = reinterpret_tensor(buf340, (4, 32, 14, 14), (125440, 196, 14, 1), 87808)  # alias
        buf352 = reinterpret_tensor(buf359, (4, 32, 14, 14), (131712, 196, 14, 1), 87808)  # alias
        buf371 = reinterpret_tensor(buf379, (4, 32, 14, 14), (137984, 196, 14, 1), 87808)  # alias
        buf391 = reinterpret_tensor(buf400, (4, 32, 14, 14), (144256, 196, 14, 1), 87808)  # alias
        # Source Nodes: [cat_88, cat_89, cat_90, cat_91], Original ATen: [aten.cat]
        triton_poi_fused_cat_57.run(buf251, buf334, buf352, buf371, buf391, 25088, grid=grid(25088), stream=stream0)
        buf339 = reinterpret_tensor(buf340, (4, 32, 14, 14), (125440, 196, 14, 1), 119168)  # alias
        buf357 = reinterpret_tensor(buf359, (4, 32, 14, 14), (131712, 196, 14, 1), 119168)  # alias
        buf376 = reinterpret_tensor(buf379, (4, 32, 14, 14), (137984, 196, 14, 1), 119168)  # alias
        buf396 = reinterpret_tensor(buf400, (4, 32, 14, 14), (144256, 196, 14, 1), 119168)  # alias
        # Source Nodes: [cat_88, cat_89, cat_90, cat_91], Original ATen: [aten.cat]
        triton_poi_fused_cat_57.run(buf326, buf339, buf357, buf376, buf396, 25088, grid=grid(25088), stream=stream0)
        buf341 = empty((4, 640, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer13_norm1, l__mod___features_denseblock3_denselayer13_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_58.run(buf340, primals_554, primals_555, primals_190, primals_191, buf341, 501760, grid=grid(501760), stream=stream0)
        del primals_191
        # Source Nodes: [bottleneck_output_60], Original ATen: [aten.convolution]
        buf342 = extern_kernels.convolution(buf341, primals_192, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf342, (4, 128, 14, 14), (25088, 196, 14, 1))
        buf343 = empty((4, 128, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer13_norm2, l__mod___features_denseblock3_denselayer13_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf342, primals_557, primals_558, primals_193, primals_194, buf343, 100352, grid=grid(100352), stream=stream0)
        del primals_194
        # Source Nodes: [new_features_60], Original ATen: [aten.convolution]
        buf344 = extern_kernels.convolution(buf343, primals_195, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf344, (4, 32, 14, 14), (6272, 196, 14, 1))
        buf353 = reinterpret_tensor(buf359, (4, 32, 14, 14), (131712, 196, 14, 1), 94080)  # alias
        buf372 = reinterpret_tensor(buf379, (4, 32, 14, 14), (137984, 196, 14, 1), 94080)  # alias
        buf392 = reinterpret_tensor(buf400, (4, 32, 14, 14), (144256, 196, 14, 1), 94080)  # alias
        buf413 = reinterpret_tensor(buf422, (4, 32, 14, 14), (150528, 196, 14, 1), 94080)  # alias
        # Source Nodes: [cat_87, cat_88, cat_89, cat_90], Original ATen: [aten.cat]
        triton_poi_fused_cat_59.run(buf264, buf353, buf372, buf392, buf413, 25088, grid=grid(25088), stream=stream0)
        buf354 = reinterpret_tensor(buf359, (4, 32, 14, 14), (131712, 196, 14, 1), 100352)  # alias
        buf373 = reinterpret_tensor(buf379, (4, 32, 14, 14), (137984, 196, 14, 1), 100352)  # alias
        buf393 = reinterpret_tensor(buf400, (4, 32, 14, 14), (144256, 196, 14, 1), 100352)  # alias
        buf414 = reinterpret_tensor(buf422, (4, 32, 14, 14), (150528, 196, 14, 1), 100352)  # alias
        # Source Nodes: [cat_87, cat_88, cat_89, cat_90], Original ATen: [aten.cat]
        triton_poi_fused_cat_59.run(buf278, buf354, buf373, buf393, buf414, 25088, grid=grid(25088), stream=stream0)
        buf358 = reinterpret_tensor(buf359, (4, 32, 14, 14), (131712, 196, 14, 1), 125440)  # alias
        buf377 = reinterpret_tensor(buf379, (4, 32, 14, 14), (137984, 196, 14, 1), 125440)  # alias
        buf397 = reinterpret_tensor(buf400, (4, 32, 14, 14), (144256, 196, 14, 1), 125440)  # alias
        buf418 = reinterpret_tensor(buf422, (4, 32, 14, 14), (150528, 196, 14, 1), 125440)  # alias
        # Source Nodes: [cat_87, cat_88, cat_89, cat_90], Original ATen: [aten.cat]
        triton_poi_fused_cat_59.run(buf344, buf358, buf377, buf397, buf418, 25088, grid=grid(25088), stream=stream0)
        buf360 = empty((4, 672, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer14_norm1, l__mod___features_denseblock3_denselayer14_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_60.run(buf359, primals_560, primals_561, primals_196, primals_197, buf360, 526848, grid=grid(526848), stream=stream0)
        del primals_197
        # Source Nodes: [bottleneck_output_62], Original ATen: [aten.convolution]
        buf361 = extern_kernels.convolution(buf360, primals_198, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf361, (4, 128, 14, 14), (25088, 196, 14, 1))
        buf362 = empty((4, 128, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer14_norm2, l__mod___features_denseblock3_denselayer14_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf361, primals_563, primals_564, primals_199, primals_200, buf362, 100352, grid=grid(100352), stream=stream0)
        del primals_200
        # Source Nodes: [new_features_62], Original ATen: [aten.convolution]
        buf363 = extern_kernels.convolution(buf362, primals_201, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf363, (4, 32, 14, 14), (6272, 196, 14, 1))
        buf365 = reinterpret_tensor(buf379, (4, 32, 14, 14), (137984, 196, 14, 1), 50176)  # alias
        buf385 = reinterpret_tensor(buf400, (4, 32, 14, 14), (144256, 196, 14, 1), 50176)  # alias
        buf406 = reinterpret_tensor(buf422, (4, 32, 14, 14), (150528, 196, 14, 1), 50176)  # alias
        buf428 = reinterpret_tensor(buf445, (4, 32, 14, 14), (156800, 196, 14, 1), 50176)  # alias
        # Source Nodes: [cat_86, cat_87, cat_88, cat_89], Original ATen: [aten.cat]
        triton_poi_fused_cat_61.run(buf203, buf365, buf385, buf406, buf428, 25088, grid=grid(25088), stream=stream0)
        buf366 = reinterpret_tensor(buf379, (4, 32, 14, 14), (137984, 196, 14, 1), 56448)  # alias
        buf386 = reinterpret_tensor(buf400, (4, 32, 14, 14), (144256, 196, 14, 1), 56448)  # alias
        buf407 = reinterpret_tensor(buf422, (4, 32, 14, 14), (150528, 196, 14, 1), 56448)  # alias
        buf429 = reinterpret_tensor(buf445, (4, 32, 14, 14), (156800, 196, 14, 1), 56448)  # alias
        # Source Nodes: [cat_86, cat_87, cat_88, cat_89], Original ATen: [aten.cat]
        triton_poi_fused_cat_61.run(buf208, buf366, buf386, buf407, buf429, 25088, grid=grid(25088), stream=stream0)
        buf367 = reinterpret_tensor(buf379, (4, 32, 14, 14), (137984, 196, 14, 1), 62720)  # alias
        buf387 = reinterpret_tensor(buf400, (4, 32, 14, 14), (144256, 196, 14, 1), 62720)  # alias
        buf408 = reinterpret_tensor(buf422, (4, 32, 14, 14), (150528, 196, 14, 1), 62720)  # alias
        buf430 = reinterpret_tensor(buf445, (4, 32, 14, 14), (156800, 196, 14, 1), 62720)  # alias
        # Source Nodes: [cat_86, cat_87, cat_88, cat_89], Original ATen: [aten.cat]
        triton_poi_fused_cat_61.run(buf213, buf367, buf387, buf408, buf430, 25088, grid=grid(25088), stream=stream0)
        buf368 = reinterpret_tensor(buf379, (4, 32, 14, 14), (137984, 196, 14, 1), 68992)  # alias
        buf388 = reinterpret_tensor(buf400, (4, 32, 14, 14), (144256, 196, 14, 1), 68992)  # alias
        buf409 = reinterpret_tensor(buf422, (4, 32, 14, 14), (150528, 196, 14, 1), 68992)  # alias
        buf431 = reinterpret_tensor(buf445, (4, 32, 14, 14), (156800, 196, 14, 1), 68992)  # alias
        # Source Nodes: [cat_86, cat_87, cat_88, cat_89], Original ATen: [aten.cat]
        triton_poi_fused_cat_61.run(buf218, buf368, buf388, buf409, buf431, 25088, grid=grid(25088), stream=stream0)
        buf369 = reinterpret_tensor(buf379, (4, 32, 14, 14), (137984, 196, 14, 1), 75264)  # alias
        buf389 = reinterpret_tensor(buf400, (4, 32, 14, 14), (144256, 196, 14, 1), 75264)  # alias
        buf410 = reinterpret_tensor(buf422, (4, 32, 14, 14), (150528, 196, 14, 1), 75264)  # alias
        buf432 = reinterpret_tensor(buf445, (4, 32, 14, 14), (156800, 196, 14, 1), 75264)  # alias
        # Source Nodes: [cat_86, cat_87, cat_88, cat_89], Original ATen: [aten.cat]
        triton_poi_fused_cat_61.run(buf228, buf369, buf389, buf410, buf432, 25088, grid=grid(25088), stream=stream0)
        buf374 = reinterpret_tensor(buf379, (4, 32, 14, 14), (137984, 196, 14, 1), 106624)  # alias
        buf394 = reinterpret_tensor(buf400, (4, 32, 14, 14), (144256, 196, 14, 1), 106624)  # alias
        buf415 = reinterpret_tensor(buf422, (4, 32, 14, 14), (150528, 196, 14, 1), 106624)  # alias
        buf437 = reinterpret_tensor(buf445, (4, 32, 14, 14), (156800, 196, 14, 1), 106624)  # alias
        # Source Nodes: [cat_86, cat_87, cat_88, cat_89], Original ATen: [aten.cat]
        triton_poi_fused_cat_61.run(buf293, buf374, buf394, buf415, buf437, 25088, grid=grid(25088), stream=stream0)
        buf378 = reinterpret_tensor(buf379, (4, 32, 14, 14), (137984, 196, 14, 1), 131712)  # alias
        buf398 = reinterpret_tensor(buf400, (4, 32, 14, 14), (144256, 196, 14, 1), 131712)  # alias
        buf419 = reinterpret_tensor(buf422, (4, 32, 14, 14), (150528, 196, 14, 1), 131712)  # alias
        buf441 = reinterpret_tensor(buf445, (4, 32, 14, 14), (156800, 196, 14, 1), 131712)  # alias
        # Source Nodes: [cat_86, cat_87, cat_88, cat_89], Original ATen: [aten.cat]
        triton_poi_fused_cat_61.run(buf363, buf378, buf398, buf419, buf441, 25088, grid=grid(25088), stream=stream0)
        buf380 = empty((4, 704, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer15_norm1, l__mod___features_denseblock3_denselayer15_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_62.run(buf379, primals_566, primals_567, primals_202, primals_203, buf380, 551936, grid=grid(551936), stream=stream0)
        del primals_203
        # Source Nodes: [bottleneck_output_64], Original ATen: [aten.convolution]
        buf381 = extern_kernels.convolution(buf380, primals_204, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf381, (4, 128, 14, 14), (25088, 196, 14, 1))
        buf382 = empty((4, 128, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer15_norm2, l__mod___features_denseblock3_denselayer15_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf381, primals_569, primals_570, primals_205, primals_206, buf382, 100352, grid=grid(100352), stream=stream0)
        del primals_206
        # Source Nodes: [new_features_64], Original ATen: [aten.convolution]
        buf383 = extern_kernels.convolution(buf382, primals_207, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf383, (4, 32, 14, 14), (6272, 196, 14, 1))
        buf390 = reinterpret_tensor(buf400, (4, 32, 14, 14), (144256, 196, 14, 1), 81536)  # alias
        buf411 = reinterpret_tensor(buf422, (4, 32, 14, 14), (150528, 196, 14, 1), 81536)  # alias
        buf433 = reinterpret_tensor(buf445, (4, 32, 14, 14), (156800, 196, 14, 1), 81536)  # alias
        # Source Nodes: [cat_86, cat_87, cat_88], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf239, buf390, buf411, buf433, 25088, grid=grid(25088), stream=stream0)
        buf395 = reinterpret_tensor(buf400, (4, 32, 14, 14), (144256, 196, 14, 1), 112896)  # alias
        buf416 = reinterpret_tensor(buf422, (4, 32, 14, 14), (150528, 196, 14, 1), 112896)  # alias
        buf438 = reinterpret_tensor(buf445, (4, 32, 14, 14), (156800, 196, 14, 1), 112896)  # alias
        # Source Nodes: [cat_86, cat_87, cat_88], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf309, buf395, buf416, buf438, 25088, grid=grid(25088), stream=stream0)
        buf399 = reinterpret_tensor(buf400, (4, 32, 14, 14), (144256, 196, 14, 1), 137984)  # alias
        buf420 = reinterpret_tensor(buf422, (4, 32, 14, 14), (150528, 196, 14, 1), 137984)  # alias
        buf442 = reinterpret_tensor(buf445, (4, 32, 14, 14), (156800, 196, 14, 1), 137984)  # alias
        # Source Nodes: [cat_86, cat_87, cat_88], Original ATen: [aten.cat]
        triton_poi_fused_cat_63.run(buf383, buf399, buf420, buf442, 25088, grid=grid(25088), stream=stream0)
        buf401 = empty((4, 736, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer16_norm1, l__mod___features_denseblock3_denselayer16_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_64.run(buf400, primals_572, primals_573, primals_208, primals_209, buf401, 577024, grid=grid(577024), stream=stream0)
        del primals_209
        # Source Nodes: [bottleneck_output_66], Original ATen: [aten.convolution]
        buf402 = extern_kernels.convolution(buf401, primals_210, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf402, (4, 128, 14, 14), (25088, 196, 14, 1))
        buf403 = empty((4, 128, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer16_norm2, l__mod___features_denseblock3_denselayer16_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf402, primals_575, primals_576, primals_211, primals_212, buf403, 100352, grid=grid(100352), stream=stream0)
        del primals_212
        # Source Nodes: [new_features_66], Original ATen: [aten.convolution]
        buf404 = extern_kernels.convolution(buf403, primals_213, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf404, (4, 32, 14, 14), (6272, 196, 14, 1))
        buf412 = reinterpret_tensor(buf422, (4, 32, 14, 14), (150528, 196, 14, 1), 87808)  # alias
        buf434 = reinterpret_tensor(buf445, (4, 32, 14, 14), (156800, 196, 14, 1), 87808)  # alias
        buf457 = reinterpret_tensor(buf469, (4, 32, 14, 14), (163072, 196, 14, 1), 87808)  # alias
        # Source Nodes: [cat_85, cat_86, cat_87], Original ATen: [aten.cat]
        triton_poi_fused_cat_65.run(buf251, buf412, buf434, buf457, 25088, grid=grid(25088), stream=stream0)
        buf417 = reinterpret_tensor(buf422, (4, 32, 14, 14), (150528, 196, 14, 1), 119168)  # alias
        buf439 = reinterpret_tensor(buf445, (4, 32, 14, 14), (156800, 196, 14, 1), 119168)  # alias
        buf462 = reinterpret_tensor(buf469, (4, 32, 14, 14), (163072, 196, 14, 1), 119168)  # alias
        # Source Nodes: [cat_85, cat_86, cat_87], Original ATen: [aten.cat]
        triton_poi_fused_cat_65.run(buf326, buf417, buf439, buf462, 25088, grid=grid(25088), stream=stream0)
        buf421 = reinterpret_tensor(buf422, (4, 32, 14, 14), (150528, 196, 14, 1), 144256)  # alias
        buf443 = reinterpret_tensor(buf445, (4, 32, 14, 14), (156800, 196, 14, 1), 144256)  # alias
        buf466 = reinterpret_tensor(buf469, (4, 32, 14, 14), (163072, 196, 14, 1), 144256)  # alias
        # Source Nodes: [cat_85, cat_86, cat_87], Original ATen: [aten.cat]
        triton_poi_fused_cat_65.run(buf404, buf421, buf443, buf466, 25088, grid=grid(25088), stream=stream0)
        buf423 = empty((4, 768, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer17_norm1, l__mod___features_denseblock3_denselayer17_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_66.run(buf422, primals_578, primals_579, primals_214, primals_215, buf423, 602112, grid=grid(602112), stream=stream0)
        del primals_215
        # Source Nodes: [bottleneck_output_68], Original ATen: [aten.convolution]
        buf424 = extern_kernels.convolution(buf423, primals_216, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf424, (4, 128, 14, 14), (25088, 196, 14, 1))
        buf425 = empty((4, 128, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer17_norm2, l__mod___features_denseblock3_denselayer17_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf424, primals_581, primals_582, primals_217, primals_218, buf425, 100352, grid=grid(100352), stream=stream0)
        del primals_218
        # Source Nodes: [new_features_68], Original ATen: [aten.convolution]
        buf426 = extern_kernels.convolution(buf425, primals_219, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf426, (4, 32, 14, 14), (6272, 196, 14, 1))
        buf435 = reinterpret_tensor(buf445, (4, 32, 14, 14), (156800, 196, 14, 1), 94080)  # alias
        buf458 = reinterpret_tensor(buf469, (4, 32, 14, 14), (163072, 196, 14, 1), 94080)  # alias
        buf482 = reinterpret_tensor(buf494, (4, 32, 14, 14), (169344, 196, 14, 1), 94080)  # alias
        # Source Nodes: [cat_84, cat_85, cat_86], Original ATen: [aten.cat]
        triton_poi_fused_cat_67.run(buf264, buf435, buf458, buf482, 25088, grid=grid(25088), stream=stream0)
        buf436 = reinterpret_tensor(buf445, (4, 32, 14, 14), (156800, 196, 14, 1), 100352)  # alias
        buf459 = reinterpret_tensor(buf469, (4, 32, 14, 14), (163072, 196, 14, 1), 100352)  # alias
        buf483 = reinterpret_tensor(buf494, (4, 32, 14, 14), (169344, 196, 14, 1), 100352)  # alias
        # Source Nodes: [cat_84, cat_85, cat_86], Original ATen: [aten.cat]
        triton_poi_fused_cat_67.run(buf278, buf436, buf459, buf483, 25088, grid=grid(25088), stream=stream0)
        buf440 = reinterpret_tensor(buf445, (4, 32, 14, 14), (156800, 196, 14, 1), 125440)  # alias
        buf463 = reinterpret_tensor(buf469, (4, 32, 14, 14), (163072, 196, 14, 1), 125440)  # alias
        buf487 = reinterpret_tensor(buf494, (4, 32, 14, 14), (169344, 196, 14, 1), 125440)  # alias
        # Source Nodes: [cat_84, cat_85, cat_86], Original ATen: [aten.cat]
        triton_poi_fused_cat_67.run(buf344, buf440, buf463, buf487, 25088, grid=grid(25088), stream=stream0)
        buf444 = reinterpret_tensor(buf445, (4, 32, 14, 14), (156800, 196, 14, 1), 150528)  # alias
        buf467 = reinterpret_tensor(buf469, (4, 32, 14, 14), (163072, 196, 14, 1), 150528)  # alias
        buf491 = reinterpret_tensor(buf494, (4, 32, 14, 14), (169344, 196, 14, 1), 150528)  # alias
        # Source Nodes: [cat_84, cat_85, cat_86], Original ATen: [aten.cat]
        triton_poi_fused_cat_67.run(buf426, buf444, buf467, buf491, 25088, grid=grid(25088), stream=stream0)
        buf446 = empty((4, 800, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer18_norm1, l__mod___features_denseblock3_denselayer18_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_68.run(buf445, primals_584, primals_585, primals_220, primals_221, buf446, 627200, grid=grid(627200), stream=stream0)
        del primals_221
        # Source Nodes: [bottleneck_output_70], Original ATen: [aten.convolution]
        buf447 = extern_kernels.convolution(buf446, primals_222, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf447, (4, 128, 14, 14), (25088, 196, 14, 1))
        buf448 = empty((4, 128, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer18_norm2, l__mod___features_denseblock3_denselayer18_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf447, primals_587, primals_588, primals_223, primals_224, buf448, 100352, grid=grid(100352), stream=stream0)
        del primals_224
        # Source Nodes: [new_features_70], Original ATen: [aten.convolution]
        buf449 = extern_kernels.convolution(buf448, primals_225, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf449, (4, 32, 14, 14), (6272, 196, 14, 1))
        buf451 = reinterpret_tensor(buf469, (4, 32, 14, 14), (163072, 196, 14, 1), 50176)  # alias
        buf475 = reinterpret_tensor(buf494, (4, 32, 14, 14), (169344, 196, 14, 1), 50176)  # alias
        buf500 = reinterpret_tensor(buf520, (4, 32, 14, 14), (175616, 196, 14, 1), 50176)  # alias
        # Source Nodes: [cat_83, cat_84, cat_85], Original ATen: [aten.cat]
        triton_poi_fused_cat_69.run(buf203, buf451, buf475, buf500, 25088, grid=grid(25088), stream=stream0)
        buf452 = reinterpret_tensor(buf469, (4, 32, 14, 14), (163072, 196, 14, 1), 56448)  # alias
        buf476 = reinterpret_tensor(buf494, (4, 32, 14, 14), (169344, 196, 14, 1), 56448)  # alias
        buf501 = reinterpret_tensor(buf520, (4, 32, 14, 14), (175616, 196, 14, 1), 56448)  # alias
        # Source Nodes: [cat_83, cat_84, cat_85], Original ATen: [aten.cat]
        triton_poi_fused_cat_69.run(buf208, buf452, buf476, buf501, 25088, grid=grid(25088), stream=stream0)
        buf453 = reinterpret_tensor(buf469, (4, 32, 14, 14), (163072, 196, 14, 1), 62720)  # alias
        buf477 = reinterpret_tensor(buf494, (4, 32, 14, 14), (169344, 196, 14, 1), 62720)  # alias
        buf502 = reinterpret_tensor(buf520, (4, 32, 14, 14), (175616, 196, 14, 1), 62720)  # alias
        # Source Nodes: [cat_83, cat_84, cat_85], Original ATen: [aten.cat]
        triton_poi_fused_cat_69.run(buf213, buf453, buf477, buf502, 25088, grid=grid(25088), stream=stream0)
        buf454 = reinterpret_tensor(buf469, (4, 32, 14, 14), (163072, 196, 14, 1), 68992)  # alias
        buf478 = reinterpret_tensor(buf494, (4, 32, 14, 14), (169344, 196, 14, 1), 68992)  # alias
        buf503 = reinterpret_tensor(buf520, (4, 32, 14, 14), (175616, 196, 14, 1), 68992)  # alias
        # Source Nodes: [cat_83, cat_84, cat_85], Original ATen: [aten.cat]
        triton_poi_fused_cat_69.run(buf218, buf454, buf478, buf503, 25088, grid=grid(25088), stream=stream0)
        buf455 = reinterpret_tensor(buf469, (4, 32, 14, 14), (163072, 196, 14, 1), 75264)  # alias
        buf479 = reinterpret_tensor(buf494, (4, 32, 14, 14), (169344, 196, 14, 1), 75264)  # alias
        buf504 = reinterpret_tensor(buf520, (4, 32, 14, 14), (175616, 196, 14, 1), 75264)  # alias
        # Source Nodes: [cat_83, cat_84, cat_85], Original ATen: [aten.cat]
        triton_poi_fused_cat_69.run(buf228, buf455, buf479, buf504, 25088, grid=grid(25088), stream=stream0)
        buf456 = reinterpret_tensor(buf469, (4, 32, 14, 14), (163072, 196, 14, 1), 81536)  # alias
        buf480 = reinterpret_tensor(buf494, (4, 32, 14, 14), (169344, 196, 14, 1), 81536)  # alias
        buf505 = reinterpret_tensor(buf520, (4, 32, 14, 14), (175616, 196, 14, 1), 81536)  # alias
        # Source Nodes: [cat_83, cat_84, cat_85], Original ATen: [aten.cat]
        triton_poi_fused_cat_69.run(buf239, buf456, buf480, buf505, 25088, grid=grid(25088), stream=stream0)
        buf460 = reinterpret_tensor(buf469, (4, 32, 14, 14), (163072, 196, 14, 1), 106624)  # alias
        buf484 = reinterpret_tensor(buf494, (4, 32, 14, 14), (169344, 196, 14, 1), 106624)  # alias
        buf509 = reinterpret_tensor(buf520, (4, 32, 14, 14), (175616, 196, 14, 1), 106624)  # alias
        # Source Nodes: [cat_83, cat_84, cat_85], Original ATen: [aten.cat]
        triton_poi_fused_cat_69.run(buf293, buf460, buf484, buf509, 25088, grid=grid(25088), stream=stream0)
        buf461 = reinterpret_tensor(buf469, (4, 32, 14, 14), (163072, 196, 14, 1), 112896)  # alias
        buf485 = reinterpret_tensor(buf494, (4, 32, 14, 14), (169344, 196, 14, 1), 112896)  # alias
        buf510 = reinterpret_tensor(buf520, (4, 32, 14, 14), (175616, 196, 14, 1), 112896)  # alias
        # Source Nodes: [cat_83, cat_84, cat_85], Original ATen: [aten.cat]
        triton_poi_fused_cat_69.run(buf309, buf461, buf485, buf510, 25088, grid=grid(25088), stream=stream0)
        buf464 = reinterpret_tensor(buf469, (4, 32, 14, 14), (163072, 196, 14, 1), 131712)  # alias
        buf488 = reinterpret_tensor(buf494, (4, 32, 14, 14), (169344, 196, 14, 1), 131712)  # alias
        buf513 = reinterpret_tensor(buf520, (4, 32, 14, 14), (175616, 196, 14, 1), 131712)  # alias
        # Source Nodes: [cat_83, cat_84, cat_85], Original ATen: [aten.cat]
        triton_poi_fused_cat_69.run(buf363, buf464, buf488, buf513, 25088, grid=grid(25088), stream=stream0)
        buf465 = reinterpret_tensor(buf469, (4, 32, 14, 14), (163072, 196, 14, 1), 137984)  # alias
        buf489 = reinterpret_tensor(buf494, (4, 32, 14, 14), (169344, 196, 14, 1), 137984)  # alias
        buf514 = reinterpret_tensor(buf520, (4, 32, 14, 14), (175616, 196, 14, 1), 137984)  # alias
        # Source Nodes: [cat_83, cat_84, cat_85], Original ATen: [aten.cat]
        triton_poi_fused_cat_69.run(buf383, buf465, buf489, buf514, 25088, grid=grid(25088), stream=stream0)
        buf468 = reinterpret_tensor(buf469, (4, 32, 14, 14), (163072, 196, 14, 1), 156800)  # alias
        buf492 = reinterpret_tensor(buf494, (4, 32, 14, 14), (169344, 196, 14, 1), 156800)  # alias
        buf517 = reinterpret_tensor(buf520, (4, 32, 14, 14), (175616, 196, 14, 1), 156800)  # alias
        # Source Nodes: [cat_83, cat_84, cat_85], Original ATen: [aten.cat]
        triton_poi_fused_cat_69.run(buf449, buf468, buf492, buf517, 25088, grid=grid(25088), stream=stream0)
        buf470 = empty((4, 832, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer19_norm1, l__mod___features_denseblock3_denselayer19_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_70.run(buf469, primals_590, primals_591, primals_226, primals_227, buf470, 652288, grid=grid(652288), stream=stream0)
        del primals_227
        # Source Nodes: [bottleneck_output_72], Original ATen: [aten.convolution]
        buf471 = extern_kernels.convolution(buf470, primals_228, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf471, (4, 128, 14, 14), (25088, 196, 14, 1))
        buf472 = empty((4, 128, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer19_norm2, l__mod___features_denseblock3_denselayer19_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf471, primals_593, primals_594, primals_229, primals_230, buf472, 100352, grid=grid(100352), stream=stream0)
        del primals_230
        # Source Nodes: [new_features_72], Original ATen: [aten.convolution]
        buf473 = extern_kernels.convolution(buf472, primals_231, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf473, (4, 32, 14, 14), (6272, 196, 14, 1))
        buf481 = reinterpret_tensor(buf494, (4, 32, 14, 14), (169344, 196, 14, 1), 87808)  # alias
        buf506 = reinterpret_tensor(buf520, (4, 32, 14, 14), (175616, 196, 14, 1), 87808)  # alias
        buf532 = reinterpret_tensor(buf547, (4, 32, 14, 14), (181888, 196, 14, 1), 87808)  # alias
        # Source Nodes: [cat_82, cat_83, cat_84], Original ATen: [aten.cat]
        triton_poi_fused_cat_71.run(buf251, buf481, buf506, buf532, 25088, grid=grid(25088), stream=stream0)
        buf486 = reinterpret_tensor(buf494, (4, 32, 14, 14), (169344, 196, 14, 1), 119168)  # alias
        buf511 = reinterpret_tensor(buf520, (4, 32, 14, 14), (175616, 196, 14, 1), 119168)  # alias
        buf537 = reinterpret_tensor(buf547, (4, 32, 14, 14), (181888, 196, 14, 1), 119168)  # alias
        # Source Nodes: [cat_82, cat_83, cat_84], Original ATen: [aten.cat]
        triton_poi_fused_cat_71.run(buf326, buf486, buf511, buf537, 25088, grid=grid(25088), stream=stream0)
        buf490 = reinterpret_tensor(buf494, (4, 32, 14, 14), (169344, 196, 14, 1), 144256)  # alias
        buf515 = reinterpret_tensor(buf520, (4, 32, 14, 14), (175616, 196, 14, 1), 144256)  # alias
        buf541 = reinterpret_tensor(buf547, (4, 32, 14, 14), (181888, 196, 14, 1), 144256)  # alias
        # Source Nodes: [cat_82, cat_83, cat_84], Original ATen: [aten.cat]
        triton_poi_fused_cat_71.run(buf404, buf490, buf515, buf541, 25088, grid=grid(25088), stream=stream0)
        buf493 = reinterpret_tensor(buf494, (4, 32, 14, 14), (169344, 196, 14, 1), 163072)  # alias
        buf518 = reinterpret_tensor(buf520, (4, 32, 14, 14), (175616, 196, 14, 1), 163072)  # alias
        buf544 = reinterpret_tensor(buf547, (4, 32, 14, 14), (181888, 196, 14, 1), 163072)  # alias
        # Source Nodes: [cat_82, cat_83, cat_84], Original ATen: [aten.cat]
        triton_poi_fused_cat_71.run(buf473, buf493, buf518, buf544, 25088, grid=grid(25088), stream=stream0)
        buf495 = empty((4, 864, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer20_norm1, l__mod___features_denseblock3_denselayer20_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_72.run(buf494, primals_596, primals_597, primals_232, primals_233, buf495, 677376, grid=grid(677376), stream=stream0)
        del primals_233
        # Source Nodes: [bottleneck_output_74], Original ATen: [aten.convolution]
        buf496 = extern_kernels.convolution(buf495, primals_234, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf496, (4, 128, 14, 14), (25088, 196, 14, 1))
        buf497 = empty((4, 128, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer20_norm2, l__mod___features_denseblock3_denselayer20_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf496, primals_599, primals_600, primals_235, primals_236, buf497, 100352, grid=grid(100352), stream=stream0)
        del primals_236
        # Source Nodes: [new_features_74], Original ATen: [aten.convolution]
        buf498 = extern_kernels.convolution(buf497, primals_237, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf498, (4, 32, 14, 14), (6272, 196, 14, 1))
        buf507 = reinterpret_tensor(buf520, (4, 32, 14, 14), (175616, 196, 14, 1), 94080)  # alias
        buf533 = reinterpret_tensor(buf547, (4, 32, 14, 14), (181888, 196, 14, 1), 94080)  # alias
        buf560 = reinterpret_tensor(buf575, (4, 32, 14, 14), (188160, 196, 14, 1), 94080)  # alias
        # Source Nodes: [cat_81, cat_82, cat_83], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf264, buf507, buf533, buf560, 25088, grid=grid(25088), stream=stream0)
        buf508 = reinterpret_tensor(buf520, (4, 32, 14, 14), (175616, 196, 14, 1), 100352)  # alias
        buf534 = reinterpret_tensor(buf547, (4, 32, 14, 14), (181888, 196, 14, 1), 100352)  # alias
        buf561 = reinterpret_tensor(buf575, (4, 32, 14, 14), (188160, 196, 14, 1), 100352)  # alias
        # Source Nodes: [cat_81, cat_82, cat_83], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf278, buf508, buf534, buf561, 25088, grid=grid(25088), stream=stream0)
        buf512 = reinterpret_tensor(buf520, (4, 32, 14, 14), (175616, 196, 14, 1), 125440)  # alias
        buf538 = reinterpret_tensor(buf547, (4, 32, 14, 14), (181888, 196, 14, 1), 125440)  # alias
        buf565 = reinterpret_tensor(buf575, (4, 32, 14, 14), (188160, 196, 14, 1), 125440)  # alias
        # Source Nodes: [cat_81, cat_82, cat_83], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf344, buf512, buf538, buf565, 25088, grid=grid(25088), stream=stream0)
        buf516 = reinterpret_tensor(buf520, (4, 32, 14, 14), (175616, 196, 14, 1), 150528)  # alias
        buf542 = reinterpret_tensor(buf547, (4, 32, 14, 14), (181888, 196, 14, 1), 150528)  # alias
        buf569 = reinterpret_tensor(buf575, (4, 32, 14, 14), (188160, 196, 14, 1), 150528)  # alias
        # Source Nodes: [cat_81, cat_82, cat_83], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf426, buf516, buf542, buf569, 25088, grid=grid(25088), stream=stream0)
        buf519 = reinterpret_tensor(buf520, (4, 32, 14, 14), (175616, 196, 14, 1), 169344)  # alias
        buf545 = reinterpret_tensor(buf547, (4, 32, 14, 14), (181888, 196, 14, 1), 169344)  # alias
        buf572 = reinterpret_tensor(buf575, (4, 32, 14, 14), (188160, 196, 14, 1), 169344)  # alias
        # Source Nodes: [cat_81, cat_82, cat_83], Original ATen: [aten.cat]
        triton_poi_fused_cat_73.run(buf498, buf519, buf545, buf572, 25088, grid=grid(25088), stream=stream0)
        buf521 = empty((4, 896, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer21_norm1, l__mod___features_denseblock3_denselayer21_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_74.run(buf520, primals_602, primals_603, primals_238, primals_239, buf521, 702464, grid=grid(702464), stream=stream0)
        del primals_239
        # Source Nodes: [bottleneck_output_76], Original ATen: [aten.convolution]
        buf522 = extern_kernels.convolution(buf521, primals_240, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf522, (4, 128, 14, 14), (25088, 196, 14, 1))
        buf523 = empty((4, 128, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer21_norm2, l__mod___features_denseblock3_denselayer21_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf522, primals_605, primals_606, primals_241, primals_242, buf523, 100352, grid=grid(100352), stream=stream0)
        del primals_242
        # Source Nodes: [new_features_76], Original ATen: [aten.convolution]
        buf524 = extern_kernels.convolution(buf523, primals_243, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf524, (4, 32, 14, 14), (6272, 196, 14, 1))
        buf526 = reinterpret_tensor(buf547, (4, 32, 14, 14), (181888, 196, 14, 1), 50176)  # alias
        buf553 = reinterpret_tensor(buf575, (4, 32, 14, 14), (188160, 196, 14, 1), 50176)  # alias
        buf581 = reinterpret_tensor(buf604, (4, 32, 14, 14), (194432, 196, 14, 1), 50176)  # alias
        # Source Nodes: [cat_80, cat_81, cat_82], Original ATen: [aten.cat]
        triton_poi_fused_cat_75.run(buf203, buf526, buf553, buf581, 25088, grid=grid(25088), stream=stream0)
        buf527 = reinterpret_tensor(buf547, (4, 32, 14, 14), (181888, 196, 14, 1), 56448)  # alias
        buf554 = reinterpret_tensor(buf575, (4, 32, 14, 14), (188160, 196, 14, 1), 56448)  # alias
        buf582 = reinterpret_tensor(buf604, (4, 32, 14, 14), (194432, 196, 14, 1), 56448)  # alias
        # Source Nodes: [cat_80, cat_81, cat_82], Original ATen: [aten.cat]
        triton_poi_fused_cat_75.run(buf208, buf527, buf554, buf582, 25088, grid=grid(25088), stream=stream0)
        buf528 = reinterpret_tensor(buf547, (4, 32, 14, 14), (181888, 196, 14, 1), 62720)  # alias
        buf555 = reinterpret_tensor(buf575, (4, 32, 14, 14), (188160, 196, 14, 1), 62720)  # alias
        buf583 = reinterpret_tensor(buf604, (4, 32, 14, 14), (194432, 196, 14, 1), 62720)  # alias
        # Source Nodes: [cat_80, cat_81, cat_82], Original ATen: [aten.cat]
        triton_poi_fused_cat_75.run(buf213, buf528, buf555, buf583, 25088, grid=grid(25088), stream=stream0)
        buf529 = reinterpret_tensor(buf547, (4, 32, 14, 14), (181888, 196, 14, 1), 68992)  # alias
        buf556 = reinterpret_tensor(buf575, (4, 32, 14, 14), (188160, 196, 14, 1), 68992)  # alias
        buf584 = reinterpret_tensor(buf604, (4, 32, 14, 14), (194432, 196, 14, 1), 68992)  # alias
        # Source Nodes: [cat_80, cat_81, cat_82], Original ATen: [aten.cat]
        triton_poi_fused_cat_75.run(buf218, buf529, buf556, buf584, 25088, grid=grid(25088), stream=stream0)
        buf530 = reinterpret_tensor(buf547, (4, 32, 14, 14), (181888, 196, 14, 1), 75264)  # alias
        buf557 = reinterpret_tensor(buf575, (4, 32, 14, 14), (188160, 196, 14, 1), 75264)  # alias
        buf585 = reinterpret_tensor(buf604, (4, 32, 14, 14), (194432, 196, 14, 1), 75264)  # alias
        # Source Nodes: [cat_80, cat_81, cat_82], Original ATen: [aten.cat]
        triton_poi_fused_cat_75.run(buf228, buf530, buf557, buf585, 25088, grid=grid(25088), stream=stream0)
        buf531 = reinterpret_tensor(buf547, (4, 32, 14, 14), (181888, 196, 14, 1), 81536)  # alias
        buf558 = reinterpret_tensor(buf575, (4, 32, 14, 14), (188160, 196, 14, 1), 81536)  # alias
        buf586 = reinterpret_tensor(buf604, (4, 32, 14, 14), (194432, 196, 14, 1), 81536)  # alias
        # Source Nodes: [cat_80, cat_81, cat_82], Original ATen: [aten.cat]
        triton_poi_fused_cat_75.run(buf239, buf531, buf558, buf586, 25088, grid=grid(25088), stream=stream0)
        buf535 = reinterpret_tensor(buf547, (4, 32, 14, 14), (181888, 196, 14, 1), 106624)  # alias
        buf562 = reinterpret_tensor(buf575, (4, 32, 14, 14), (188160, 196, 14, 1), 106624)  # alias
        buf590 = reinterpret_tensor(buf604, (4, 32, 14, 14), (194432, 196, 14, 1), 106624)  # alias
        # Source Nodes: [cat_80, cat_81, cat_82], Original ATen: [aten.cat]
        triton_poi_fused_cat_75.run(buf293, buf535, buf562, buf590, 25088, grid=grid(25088), stream=stream0)
        buf536 = reinterpret_tensor(buf547, (4, 32, 14, 14), (181888, 196, 14, 1), 112896)  # alias
        buf563 = reinterpret_tensor(buf575, (4, 32, 14, 14), (188160, 196, 14, 1), 112896)  # alias
        buf591 = reinterpret_tensor(buf604, (4, 32, 14, 14), (194432, 196, 14, 1), 112896)  # alias
        # Source Nodes: [cat_80, cat_81, cat_82], Original ATen: [aten.cat]
        triton_poi_fused_cat_75.run(buf309, buf536, buf563, buf591, 25088, grid=grid(25088), stream=stream0)
        buf539 = reinterpret_tensor(buf547, (4, 32, 14, 14), (181888, 196, 14, 1), 131712)  # alias
        buf566 = reinterpret_tensor(buf575, (4, 32, 14, 14), (188160, 196, 14, 1), 131712)  # alias
        buf594 = reinterpret_tensor(buf604, (4, 32, 14, 14), (194432, 196, 14, 1), 131712)  # alias
        # Source Nodes: [cat_80, cat_81, cat_82], Original ATen: [aten.cat]
        triton_poi_fused_cat_75.run(buf363, buf539, buf566, buf594, 25088, grid=grid(25088), stream=stream0)
        buf540 = reinterpret_tensor(buf547, (4, 32, 14, 14), (181888, 196, 14, 1), 137984)  # alias
        buf567 = reinterpret_tensor(buf575, (4, 32, 14, 14), (188160, 196, 14, 1), 137984)  # alias
        buf595 = reinterpret_tensor(buf604, (4, 32, 14, 14), (194432, 196, 14, 1), 137984)  # alias
        # Source Nodes: [cat_80, cat_81, cat_82], Original ATen: [aten.cat]
        triton_poi_fused_cat_75.run(buf383, buf540, buf567, buf595, 25088, grid=grid(25088), stream=stream0)
        buf543 = reinterpret_tensor(buf547, (4, 32, 14, 14), (181888, 196, 14, 1), 156800)  # alias
        buf570 = reinterpret_tensor(buf575, (4, 32, 14, 14), (188160, 196, 14, 1), 156800)  # alias
        buf598 = reinterpret_tensor(buf604, (4, 32, 14, 14), (194432, 196, 14, 1), 156800)  # alias
        # Source Nodes: [cat_80, cat_81, cat_82], Original ATen: [aten.cat]
        triton_poi_fused_cat_75.run(buf449, buf543, buf570, buf598, 25088, grid=grid(25088), stream=stream0)
        buf546 = reinterpret_tensor(buf547, (4, 32, 14, 14), (181888, 196, 14, 1), 175616)  # alias
        buf573 = reinterpret_tensor(buf575, (4, 32, 14, 14), (188160, 196, 14, 1), 175616)  # alias
        buf601 = reinterpret_tensor(buf604, (4, 32, 14, 14), (194432, 196, 14, 1), 175616)  # alias
        # Source Nodes: [cat_80, cat_81, cat_82], Original ATen: [aten.cat]
        triton_poi_fused_cat_75.run(buf524, buf546, buf573, buf601, 25088, grid=grid(25088), stream=stream0)
        buf548 = empty((4, 928, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer22_norm1, l__mod___features_denseblock3_denselayer22_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_76.run(buf547, primals_608, primals_609, primals_244, primals_245, buf548, 727552, grid=grid(727552), stream=stream0)
        del primals_245
        # Source Nodes: [bottleneck_output_78], Original ATen: [aten.convolution]
        buf549 = extern_kernels.convolution(buf548, primals_246, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf549, (4, 128, 14, 14), (25088, 196, 14, 1))
        buf550 = empty((4, 128, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer22_norm2, l__mod___features_denseblock3_denselayer22_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf549, primals_611, primals_612, primals_247, primals_248, buf550, 100352, grid=grid(100352), stream=stream0)
        del primals_248
        # Source Nodes: [new_features_78], Original ATen: [aten.convolution]
        buf551 = extern_kernels.convolution(buf550, primals_249, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf551, (4, 32, 14, 14), (6272, 196, 14, 1))
        buf559 = reinterpret_tensor(buf575, (4, 32, 14, 14), (188160, 196, 14, 1), 87808)  # alias
        buf587 = reinterpret_tensor(buf604, (4, 32, 14, 14), (194432, 196, 14, 1), 87808)  # alias
        buf616 = reinterpret_tensor(buf634, (4, 32, 14, 14), (200704, 196, 14, 1), 87808)  # alias
        # Source Nodes: [cat_79, cat_80, cat_81], Original ATen: [aten.cat]
        triton_poi_fused_cat_77.run(buf251, buf559, buf587, buf616, 25088, grid=grid(25088), stream=stream0)
        del buf251
        buf564 = reinterpret_tensor(buf575, (4, 32, 14, 14), (188160, 196, 14, 1), 119168)  # alias
        buf592 = reinterpret_tensor(buf604, (4, 32, 14, 14), (194432, 196, 14, 1), 119168)  # alias
        buf621 = reinterpret_tensor(buf634, (4, 32, 14, 14), (200704, 196, 14, 1), 119168)  # alias
        # Source Nodes: [cat_79, cat_80, cat_81], Original ATen: [aten.cat]
        triton_poi_fused_cat_77.run(buf326, buf564, buf592, buf621, 25088, grid=grid(25088), stream=stream0)
        del buf326
        buf568 = reinterpret_tensor(buf575, (4, 32, 14, 14), (188160, 196, 14, 1), 144256)  # alias
        buf596 = reinterpret_tensor(buf604, (4, 32, 14, 14), (194432, 196, 14, 1), 144256)  # alias
        buf625 = reinterpret_tensor(buf634, (4, 32, 14, 14), (200704, 196, 14, 1), 144256)  # alias
        # Source Nodes: [cat_79, cat_80, cat_81], Original ATen: [aten.cat]
        triton_poi_fused_cat_77.run(buf404, buf568, buf596, buf625, 25088, grid=grid(25088), stream=stream0)
        del buf404
        buf571 = reinterpret_tensor(buf575, (4, 32, 14, 14), (188160, 196, 14, 1), 163072)  # alias
        buf599 = reinterpret_tensor(buf604, (4, 32, 14, 14), (194432, 196, 14, 1), 163072)  # alias
        buf628 = reinterpret_tensor(buf634, (4, 32, 14, 14), (200704, 196, 14, 1), 163072)  # alias
        # Source Nodes: [cat_79, cat_80, cat_81], Original ATen: [aten.cat]
        triton_poi_fused_cat_77.run(buf473, buf571, buf599, buf628, 25088, grid=grid(25088), stream=stream0)
        del buf473
        buf574 = reinterpret_tensor(buf575, (4, 32, 14, 14), (188160, 196, 14, 1), 181888)  # alias
        buf602 = reinterpret_tensor(buf604, (4, 32, 14, 14), (194432, 196, 14, 1), 181888)  # alias
        buf631 = reinterpret_tensor(buf634, (4, 32, 14, 14), (200704, 196, 14, 1), 181888)  # alias
        # Source Nodes: [cat_79, cat_80, cat_81], Original ATen: [aten.cat]
        triton_poi_fused_cat_77.run(buf551, buf574, buf602, buf631, 25088, grid=grid(25088), stream=stream0)
        del buf551
        buf576 = empty((4, 960, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer23_norm1, l__mod___features_denseblock3_denselayer23_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_78.run(buf575, primals_614, primals_615, primals_250, primals_251, buf576, 752640, grid=grid(752640), stream=stream0)
        del primals_251
        # Source Nodes: [bottleneck_output_80], Original ATen: [aten.convolution]
        buf577 = extern_kernels.convolution(buf576, primals_252, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf577, (4, 128, 14, 14), (25088, 196, 14, 1))
        buf578 = empty((4, 128, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer23_norm2, l__mod___features_denseblock3_denselayer23_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf577, primals_617, primals_618, primals_253, primals_254, buf578, 100352, grid=grid(100352), stream=stream0)
        del primals_254
        # Source Nodes: [new_features_80], Original ATen: [aten.convolution]
        buf579 = extern_kernels.convolution(buf578, primals_255, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf579, (4, 32, 14, 14), (6272, 196, 14, 1))
        buf588 = reinterpret_tensor(buf604, (4, 32, 14, 14), (194432, 196, 14, 1), 94080)  # alias
        buf617 = reinterpret_tensor(buf634, (4, 32, 14, 14), (200704, 196, 14, 1), 94080)  # alias
        # Source Nodes: [cat_79, cat_80], Original ATen: [aten.cat]
        triton_poi_fused_cat_79.run(buf264, buf588, buf617, 25088, grid=grid(25088), stream=stream0)
        del buf264
        buf589 = reinterpret_tensor(buf604, (4, 32, 14, 14), (194432, 196, 14, 1), 100352)  # alias
        buf618 = reinterpret_tensor(buf634, (4, 32, 14, 14), (200704, 196, 14, 1), 100352)  # alias
        # Source Nodes: [cat_79, cat_80], Original ATen: [aten.cat]
        triton_poi_fused_cat_79.run(buf278, buf589, buf618, 25088, grid=grid(25088), stream=stream0)
        del buf278
        buf593 = reinterpret_tensor(buf604, (4, 32, 14, 14), (194432, 196, 14, 1), 125440)  # alias
        buf622 = reinterpret_tensor(buf634, (4, 32, 14, 14), (200704, 196, 14, 1), 125440)  # alias
        # Source Nodes: [cat_79, cat_80], Original ATen: [aten.cat]
        triton_poi_fused_cat_79.run(buf344, buf593, buf622, 25088, grid=grid(25088), stream=stream0)
        del buf344
        buf597 = reinterpret_tensor(buf604, (4, 32, 14, 14), (194432, 196, 14, 1), 150528)  # alias
        buf626 = reinterpret_tensor(buf634, (4, 32, 14, 14), (200704, 196, 14, 1), 150528)  # alias
        # Source Nodes: [cat_79, cat_80], Original ATen: [aten.cat]
        triton_poi_fused_cat_79.run(buf426, buf597, buf626, 25088, grid=grid(25088), stream=stream0)
        buf600 = reinterpret_tensor(buf604, (4, 32, 14, 14), (194432, 196, 14, 1), 169344)  # alias
        buf629 = reinterpret_tensor(buf634, (4, 32, 14, 14), (200704, 196, 14, 1), 169344)  # alias
        # Source Nodes: [cat_79, cat_80], Original ATen: [aten.cat]
        triton_poi_fused_cat_79.run(buf498, buf600, buf629, 25088, grid=grid(25088), stream=stream0)
        buf603 = reinterpret_tensor(buf604, (4, 32, 14, 14), (194432, 196, 14, 1), 188160)  # alias
        buf632 = reinterpret_tensor(buf634, (4, 32, 14, 14), (200704, 196, 14, 1), 188160)  # alias
        # Source Nodes: [cat_79, cat_80], Original ATen: [aten.cat]
        triton_poi_fused_cat_79.run(buf579, buf603, buf632, 25088, grid=grid(25088), stream=stream0)
        buf605 = empty((4, 992, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer24_norm1, l__mod___features_denseblock3_denselayer24_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_80.run(buf604, primals_620, primals_621, primals_256, primals_257, buf605, 777728, grid=grid(777728), stream=stream0)
        del primals_257
        # Source Nodes: [bottleneck_output_82], Original ATen: [aten.convolution]
        buf606 = extern_kernels.convolution(buf605, primals_258, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf606, (4, 128, 14, 14), (25088, 196, 14, 1))
        buf607 = empty((4, 128, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock3_denselayer24_norm2, l__mod___features_denseblock3_denselayer24_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_37.run(buf606, primals_623, primals_624, primals_259, primals_260, buf607, 100352, grid=grid(100352), stream=stream0)
        del primals_260
        # Source Nodes: [new_features_82], Original ATen: [aten.convolution]
        buf608 = extern_kernels.convolution(buf607, primals_261, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf608, (4, 32, 14, 14), (6272, 196, 14, 1))
        buf610 = reinterpret_tensor(buf634, (4, 32, 14, 14), (200704, 196, 14, 1), 50176)  # alias
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_81.run(buf203, buf610, 25088, grid=grid(25088), stream=stream0)
        buf611 = reinterpret_tensor(buf634, (4, 32, 14, 14), (200704, 196, 14, 1), 56448)  # alias
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_81.run(buf208, buf611, 25088, grid=grid(25088), stream=stream0)
        buf612 = reinterpret_tensor(buf634, (4, 32, 14, 14), (200704, 196, 14, 1), 62720)  # alias
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_81.run(buf213, buf612, 25088, grid=grid(25088), stream=stream0)
        buf613 = reinterpret_tensor(buf634, (4, 32, 14, 14), (200704, 196, 14, 1), 68992)  # alias
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_81.run(buf218, buf613, 25088, grid=grid(25088), stream=stream0)
        buf614 = reinterpret_tensor(buf634, (4, 32, 14, 14), (200704, 196, 14, 1), 75264)  # alias
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_81.run(buf228, buf614, 25088, grid=grid(25088), stream=stream0)
        buf615 = reinterpret_tensor(buf634, (4, 32, 14, 14), (200704, 196, 14, 1), 81536)  # alias
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_81.run(buf239, buf615, 25088, grid=grid(25088), stream=stream0)
        buf619 = reinterpret_tensor(buf634, (4, 32, 14, 14), (200704, 196, 14, 1), 106624)  # alias
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_81.run(buf293, buf619, 25088, grid=grid(25088), stream=stream0)
        buf620 = reinterpret_tensor(buf634, (4, 32, 14, 14), (200704, 196, 14, 1), 112896)  # alias
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_81.run(buf309, buf620, 25088, grid=grid(25088), stream=stream0)
        buf623 = reinterpret_tensor(buf634, (4, 32, 14, 14), (200704, 196, 14, 1), 131712)  # alias
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_81.run(buf363, buf623, 25088, grid=grid(25088), stream=stream0)
        buf624 = reinterpret_tensor(buf634, (4, 32, 14, 14), (200704, 196, 14, 1), 137984)  # alias
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_81.run(buf383, buf624, 25088, grid=grid(25088), stream=stream0)
        buf627 = reinterpret_tensor(buf634, (4, 32, 14, 14), (200704, 196, 14, 1), 156800)  # alias
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_81.run(buf449, buf627, 25088, grid=grid(25088), stream=stream0)
        buf630 = reinterpret_tensor(buf634, (4, 32, 14, 14), (200704, 196, 14, 1), 175616)  # alias
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_81.run(buf524, buf630, 25088, grid=grid(25088), stream=stream0)
        buf633 = reinterpret_tensor(buf634, (4, 32, 14, 14), (200704, 196, 14, 1), 194432)  # alias
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_81.run(buf608, buf633, 25088, grid=grid(25088), stream=stream0)
        buf635 = empty((4, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_transition3_norm, l__mod___features_transition3_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_82.run(buf634, primals_626, primals_627, primals_262, primals_263, buf635, 802816, grid=grid(802816), stream=stream0)
        del primals_263
        # Source Nodes: [l__mod___features_transition3_conv], Original ATen: [aten.convolution]
        buf636 = extern_kernels.convolution(buf635, primals_264, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf636, (4, 512, 14, 14), (100352, 196, 14, 1))
        buf637 = empty((4, 512, 7, 7), device='cuda', dtype=torch.float32)
        buf638 = empty((4, 512, 7, 7), device='cuda', dtype=torch.float32)
        buf662 = empty((4, 640, 7, 7), device='cuda', dtype=torch.float32)
        buf657 = reinterpret_tensor(buf662, (4, 512, 7, 7), (31360, 49, 7, 1), 0)  # alias
        buf673 = empty((4, 672, 7, 7), device='cuda', dtype=torch.float32)
        buf667 = reinterpret_tensor(buf673, (4, 512, 7, 7), (32928, 49, 7, 1), 0)  # alias
        buf685 = empty((4, 704, 7, 7), device='cuda', dtype=torch.float32)
        buf678 = reinterpret_tensor(buf685, (4, 512, 7, 7), (34496, 49, 7, 1), 0)  # alias
        buf698 = empty((4, 736, 7, 7), device='cuda', dtype=torch.float32)
        buf690 = reinterpret_tensor(buf698, (4, 512, 7, 7), (36064, 49, 7, 1), 0)  # alias
        buf712 = empty((4, 768, 7, 7), device='cuda', dtype=torch.float32)
        buf703 = reinterpret_tensor(buf712, (4, 512, 7, 7), (37632, 49, 7, 1), 0)  # alias
        buf727 = empty((4, 800, 7, 7), device='cuda', dtype=torch.float32)
        buf717 = reinterpret_tensor(buf727, (4, 512, 7, 7), (39200, 49, 7, 1), 0)  # alias
        buf743 = empty((4, 832, 7, 7), device='cuda', dtype=torch.float32)
        buf732 = reinterpret_tensor(buf743, (4, 512, 7, 7), (40768, 49, 7, 1), 0)  # alias
        buf760 = empty((4, 864, 7, 7), device='cuda', dtype=torch.float32)
        buf748 = reinterpret_tensor(buf760, (4, 512, 7, 7), (42336, 49, 7, 1), 0)  # alias
        buf778 = empty((4, 896, 7, 7), device='cuda', dtype=torch.float32)
        buf765 = reinterpret_tensor(buf778, (4, 512, 7, 7), (43904, 49, 7, 1), 0)  # alias
        buf797 = empty((4, 928, 7, 7), device='cuda', dtype=torch.float32)
        buf783 = reinterpret_tensor(buf797, (4, 512, 7, 7), (45472, 49, 7, 1), 0)  # alias
        buf817 = empty((4, 960, 7, 7), device='cuda', dtype=torch.float32)
        buf802 = reinterpret_tensor(buf817, (4, 512, 7, 7), (47040, 49, 7, 1), 0)  # alias
        buf838 = empty((4, 992, 7, 7), device='cuda', dtype=torch.float32)
        buf822 = reinterpret_tensor(buf838, (4, 512, 7, 7), (48608, 49, 7, 1), 0)  # alias
        buf860 = empty((4, 1024, 7, 7), device='cuda', dtype=torch.float32)
        buf843 = reinterpret_tensor(buf860, (4, 512, 7, 7), (50176, 49, 7, 1), 0)  # alias
        # Source Nodes: [cat_62, cat_63, cat_64, cat_65, cat_66, cat_67, cat_68, cat_69, cat_70, cat_71, cat_72, cat_73, cat_74, l__mod___features_denseblock4_denselayer1_norm1, l__mod___features_denseblock4_denselayer1_relu1, l__mod___features_transition3_pool], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_relu_83.run(buf636, primals_629, primals_630, primals_265, primals_266, buf637, buf638, buf657, buf667, buf678, buf690, buf703, buf717, buf732, buf748, buf765, buf783, buf802, buf822, buf843, 100352, grid=grid(100352), stream=stream0)
        del primals_266
        # Source Nodes: [bottleneck_output_84], Original ATen: [aten.convolution]
        buf639 = extern_kernels.convolution(buf638, primals_267, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf639, (4, 128, 7, 7), (6272, 49, 7, 1))
        buf640 = reinterpret_tensor(buf608, (4, 128, 7, 7), (6272, 49, 7, 1), 0); del buf608  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer1_norm2, l__mod___features_denseblock4_denselayer1_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_84.run(buf639, primals_632, primals_633, primals_268, primals_269, buf640, 25088, grid=grid(25088), stream=stream0)
        del primals_269
        # Source Nodes: [new_features_84], Original ATen: [aten.convolution]
        buf641 = extern_kernels.convolution(buf640, primals_270, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf641, (4, 32, 7, 7), (1568, 49, 7, 1))
        buf642 = empty((4, 544, 7, 7), device='cuda', dtype=torch.float32)
        buf643 = empty((4, 544, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_77, l__mod___features_denseblock4_denselayer2_norm1, l__mod___features_denseblock4_denselayer2_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_85.run(buf637, buf641, primals_635, primals_636, primals_271, primals_272, buf642, buf643, 106624, grid=grid(106624), stream=stream0)
        del primals_272
        # Source Nodes: [bottleneck_output_86], Original ATen: [aten.convolution]
        buf644 = extern_kernels.convolution(buf643, primals_273, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf644, (4, 128, 7, 7), (6272, 49, 7, 1))
        buf645 = reinterpret_tensor(buf524, (4, 128, 7, 7), (6272, 49, 7, 1), 0); del buf524  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer2_norm2, l__mod___features_denseblock4_denselayer2_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_84.run(buf644, primals_638, primals_639, primals_274, primals_275, buf645, 25088, grid=grid(25088), stream=stream0)
        del primals_275
        # Source Nodes: [new_features_86], Original ATen: [aten.convolution]
        buf646 = extern_kernels.convolution(buf645, primals_276, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf646, (4, 32, 7, 7), (1568, 49, 7, 1))
        buf647 = empty((4, 576, 7, 7), device='cuda', dtype=torch.float32)
        buf648 = empty((4, 576, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_76, l__mod___features_denseblock4_denselayer3_norm1, l__mod___features_denseblock4_denselayer3_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_86.run(buf637, buf641, buf646, primals_641, primals_642, primals_277, primals_278, buf647, buf648, 112896, grid=grid(112896), stream=stream0)
        del primals_278
        # Source Nodes: [bottleneck_output_88], Original ATen: [aten.convolution]
        buf649 = extern_kernels.convolution(buf648, primals_279, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf649, (4, 128, 7, 7), (6272, 49, 7, 1))
        buf650 = reinterpret_tensor(buf449, (4, 128, 7, 7), (6272, 49, 7, 1), 0); del buf449  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer3_norm2, l__mod___features_denseblock4_denselayer3_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_84.run(buf649, primals_644, primals_645, primals_280, primals_281, buf650, 25088, grid=grid(25088), stream=stream0)
        del primals_281
        # Source Nodes: [new_features_88], Original ATen: [aten.convolution]
        buf651 = extern_kernels.convolution(buf650, primals_282, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf651, (4, 32, 7, 7), (1568, 49, 7, 1))
        buf652 = empty((4, 608, 7, 7), device='cuda', dtype=torch.float32)
        buf653 = empty((4, 608, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_75, l__mod___features_denseblock4_denselayer4_norm1, l__mod___features_denseblock4_denselayer4_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_87.run(buf637, buf641, buf646, buf651, primals_647, primals_648, primals_283, primals_284, buf652, buf653, 119168, grid=grid(119168), stream=stream0)
        del primals_284
        # Source Nodes: [bottleneck_output_90], Original ATen: [aten.convolution]
        buf654 = extern_kernels.convolution(buf653, primals_285, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf654, (4, 128, 7, 7), (6272, 49, 7, 1))
        buf655 = reinterpret_tensor(buf383, (4, 128, 7, 7), (6272, 49, 7, 1), 0); del buf383  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer4_norm2, l__mod___features_denseblock4_denselayer4_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_84.run(buf654, primals_650, primals_651, primals_286, primals_287, buf655, 25088, grid=grid(25088), stream=stream0)
        del primals_287
        # Source Nodes: [new_features_90], Original ATen: [aten.convolution]
        buf656 = extern_kernels.convolution(buf655, primals_288, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf656, (4, 32, 7, 7), (1568, 49, 7, 1))
        buf658 = reinterpret_tensor(buf662, (4, 32, 7, 7), (31360, 49, 7, 1), 25088)  # alias
        buf668 = reinterpret_tensor(buf673, (4, 32, 7, 7), (32928, 49, 7, 1), 25088)  # alias
        buf679 = reinterpret_tensor(buf685, (4, 32, 7, 7), (34496, 49, 7, 1), 25088)  # alias
        buf691 = reinterpret_tensor(buf698, (4, 32, 7, 7), (36064, 49, 7, 1), 25088)  # alias
        buf704 = reinterpret_tensor(buf712, (4, 32, 7, 7), (37632, 49, 7, 1), 25088)  # alias
        buf718 = reinterpret_tensor(buf727, (4, 32, 7, 7), (39200, 49, 7, 1), 25088)  # alias
        # Source Nodes: [cat_69, cat_70, cat_71, cat_72, cat_73, cat_74], Original ATen: [aten.cat]
        triton_poi_fused_cat_88.run(buf641, buf658, buf668, buf679, buf691, buf704, buf718, 6272, grid=grid(6272), stream=stream0)
        buf659 = reinterpret_tensor(buf662, (4, 32, 7, 7), (31360, 49, 7, 1), 26656)  # alias
        buf669 = reinterpret_tensor(buf673, (4, 32, 7, 7), (32928, 49, 7, 1), 26656)  # alias
        buf680 = reinterpret_tensor(buf685, (4, 32, 7, 7), (34496, 49, 7, 1), 26656)  # alias
        buf692 = reinterpret_tensor(buf698, (4, 32, 7, 7), (36064, 49, 7, 1), 26656)  # alias
        buf705 = reinterpret_tensor(buf712, (4, 32, 7, 7), (37632, 49, 7, 1), 26656)  # alias
        buf719 = reinterpret_tensor(buf727, (4, 32, 7, 7), (39200, 49, 7, 1), 26656)  # alias
        # Source Nodes: [cat_69, cat_70, cat_71, cat_72, cat_73, cat_74], Original ATen: [aten.cat]
        triton_poi_fused_cat_88.run(buf646, buf659, buf669, buf680, buf692, buf705, buf719, 6272, grid=grid(6272), stream=stream0)
        buf660 = reinterpret_tensor(buf662, (4, 32, 7, 7), (31360, 49, 7, 1), 28224)  # alias
        buf670 = reinterpret_tensor(buf673, (4, 32, 7, 7), (32928, 49, 7, 1), 28224)  # alias
        buf681 = reinterpret_tensor(buf685, (4, 32, 7, 7), (34496, 49, 7, 1), 28224)  # alias
        buf693 = reinterpret_tensor(buf698, (4, 32, 7, 7), (36064, 49, 7, 1), 28224)  # alias
        buf706 = reinterpret_tensor(buf712, (4, 32, 7, 7), (37632, 49, 7, 1), 28224)  # alias
        buf720 = reinterpret_tensor(buf727, (4, 32, 7, 7), (39200, 49, 7, 1), 28224)  # alias
        # Source Nodes: [cat_69, cat_70, cat_71, cat_72, cat_73, cat_74], Original ATen: [aten.cat]
        triton_poi_fused_cat_88.run(buf651, buf660, buf670, buf681, buf693, buf706, buf720, 6272, grid=grid(6272), stream=stream0)
        buf661 = reinterpret_tensor(buf662, (4, 32, 7, 7), (31360, 49, 7, 1), 29792)  # alias
        buf671 = reinterpret_tensor(buf673, (4, 32, 7, 7), (32928, 49, 7, 1), 29792)  # alias
        buf682 = reinterpret_tensor(buf685, (4, 32, 7, 7), (34496, 49, 7, 1), 29792)  # alias
        buf694 = reinterpret_tensor(buf698, (4, 32, 7, 7), (36064, 49, 7, 1), 29792)  # alias
        buf707 = reinterpret_tensor(buf712, (4, 32, 7, 7), (37632, 49, 7, 1), 29792)  # alias
        buf721 = reinterpret_tensor(buf727, (4, 32, 7, 7), (39200, 49, 7, 1), 29792)  # alias
        # Source Nodes: [cat_69, cat_70, cat_71, cat_72, cat_73, cat_74], Original ATen: [aten.cat]
        triton_poi_fused_cat_88.run(buf656, buf661, buf671, buf682, buf694, buf707, buf721, 6272, grid=grid(6272), stream=stream0)
        buf663 = empty((4, 640, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock4_denselayer5_norm1, l__mod___features_denseblock4_denselayer5_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_89.run(buf662, primals_653, primals_654, primals_289, primals_290, buf663, 125440, grid=grid(125440), stream=stream0)
        del primals_290
        # Source Nodes: [bottleneck_output_92], Original ATen: [aten.convolution]
        buf664 = extern_kernels.convolution(buf663, primals_291, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf664, (4, 128, 7, 7), (6272, 49, 7, 1))
        buf665 = reinterpret_tensor(buf363, (4, 128, 7, 7), (6272, 49, 7, 1), 0); del buf363  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer5_norm2, l__mod___features_denseblock4_denselayer5_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_84.run(buf664, primals_656, primals_657, primals_292, primals_293, buf665, 25088, grid=grid(25088), stream=stream0)
        del primals_293
        # Source Nodes: [new_features_92], Original ATen: [aten.convolution]
        buf666 = extern_kernels.convolution(buf665, primals_294, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf666, (4, 32, 7, 7), (1568, 49, 7, 1))
        buf672 = reinterpret_tensor(buf673, (4, 32, 7, 7), (32928, 49, 7, 1), 31360)  # alias
        buf683 = reinterpret_tensor(buf685, (4, 32, 7, 7), (34496, 49, 7, 1), 31360)  # alias
        buf695 = reinterpret_tensor(buf698, (4, 32, 7, 7), (36064, 49, 7, 1), 31360)  # alias
        buf708 = reinterpret_tensor(buf712, (4, 32, 7, 7), (37632, 49, 7, 1), 31360)  # alias
        buf722 = reinterpret_tensor(buf727, (4, 32, 7, 7), (39200, 49, 7, 1), 31360)  # alias
        # Source Nodes: [cat_69, cat_70, cat_71, cat_72, cat_73], Original ATen: [aten.cat]
        triton_poi_fused_cat_90.run(buf666, buf672, buf683, buf695, buf708, buf722, 6272, grid=grid(6272), stream=stream0)
        buf674 = empty((4, 672, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock4_denselayer6_norm1, l__mod___features_denseblock4_denselayer6_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_91.run(buf673, primals_659, primals_660, primals_295, primals_296, buf674, 131712, grid=grid(131712), stream=stream0)
        del primals_296
        # Source Nodes: [bottleneck_output_94], Original ATen: [aten.convolution]
        buf675 = extern_kernels.convolution(buf674, primals_297, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf675, (4, 128, 7, 7), (6272, 49, 7, 1))
        buf676 = reinterpret_tensor(buf309, (4, 128, 7, 7), (6272, 49, 7, 1), 0); del buf309  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer6_norm2, l__mod___features_denseblock4_denselayer6_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_84.run(buf675, primals_662, primals_663, primals_298, primals_299, buf676, 25088, grid=grid(25088), stream=stream0)
        del primals_299
        # Source Nodes: [new_features_94], Original ATen: [aten.convolution]
        buf677 = extern_kernels.convolution(buf676, primals_300, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf677, (4, 32, 7, 7), (1568, 49, 7, 1))
        buf684 = reinterpret_tensor(buf685, (4, 32, 7, 7), (34496, 49, 7, 1), 32928)  # alias
        buf696 = reinterpret_tensor(buf698, (4, 32, 7, 7), (36064, 49, 7, 1), 32928)  # alias
        buf709 = reinterpret_tensor(buf712, (4, 32, 7, 7), (37632, 49, 7, 1), 32928)  # alias
        buf723 = reinterpret_tensor(buf727, (4, 32, 7, 7), (39200, 49, 7, 1), 32928)  # alias
        buf738 = reinterpret_tensor(buf743, (4, 32, 7, 7), (40768, 49, 7, 1), 32928)  # alias
        # Source Nodes: [cat_68, cat_69, cat_70, cat_71, cat_72], Original ATen: [aten.cat]
        triton_poi_fused_cat_92.run(buf677, buf684, buf696, buf709, buf723, buf738, 6272, grid=grid(6272), stream=stream0)
        buf686 = empty((4, 704, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock4_denselayer7_norm1, l__mod___features_denseblock4_denselayer7_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_93.run(buf685, primals_665, primals_666, primals_301, primals_302, buf686, 137984, grid=grid(137984), stream=stream0)
        del primals_302
        # Source Nodes: [bottleneck_output_96], Original ATen: [aten.convolution]
        buf687 = extern_kernels.convolution(buf686, primals_303, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf687, (4, 128, 7, 7), (6272, 49, 7, 1))
        buf688 = reinterpret_tensor(buf293, (4, 128, 7, 7), (6272, 49, 7, 1), 0); del buf293  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer7_norm2, l__mod___features_denseblock4_denselayer7_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_84.run(buf687, primals_668, primals_669, primals_304, primals_305, buf688, 25088, grid=grid(25088), stream=stream0)
        del primals_305
        # Source Nodes: [new_features_96], Original ATen: [aten.convolution]
        buf689 = extern_kernels.convolution(buf688, primals_306, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf689, (4, 32, 7, 7), (1568, 49, 7, 1))
        buf697 = reinterpret_tensor(buf698, (4, 32, 7, 7), (36064, 49, 7, 1), 34496)  # alias
        buf710 = reinterpret_tensor(buf712, (4, 32, 7, 7), (37632, 49, 7, 1), 34496)  # alias
        buf724 = reinterpret_tensor(buf727, (4, 32, 7, 7), (39200, 49, 7, 1), 34496)  # alias
        buf739 = reinterpret_tensor(buf743, (4, 32, 7, 7), (40768, 49, 7, 1), 34496)  # alias
        buf755 = reinterpret_tensor(buf760, (4, 32, 7, 7), (42336, 49, 7, 1), 34496)  # alias
        # Source Nodes: [cat_67, cat_68, cat_69, cat_70, cat_71], Original ATen: [aten.cat]
        triton_poi_fused_cat_94.run(buf689, buf697, buf710, buf724, buf739, buf755, 6272, grid=grid(6272), stream=stream0)
        buf699 = empty((4, 736, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock4_denselayer8_norm1, l__mod___features_denseblock4_denselayer8_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_95.run(buf698, primals_671, primals_672, primals_307, primals_308, buf699, 144256, grid=grid(144256), stream=stream0)
        del primals_308
        # Source Nodes: [bottleneck_output_98], Original ATen: [aten.convolution]
        buf700 = extern_kernels.convolution(buf699, primals_309, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf700, (4, 128, 7, 7), (6272, 49, 7, 1))
        buf701 = reinterpret_tensor(buf239, (4, 128, 7, 7), (6272, 49, 7, 1), 0); del buf239  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer8_norm2, l__mod___features_denseblock4_denselayer8_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_84.run(buf700, primals_674, primals_675, primals_310, primals_311, buf701, 25088, grid=grid(25088), stream=stream0)
        del primals_311
        # Source Nodes: [new_features_98], Original ATen: [aten.convolution]
        buf702 = extern_kernels.convolution(buf701, primals_312, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf702, (4, 32, 7, 7), (1568, 49, 7, 1))
        buf711 = reinterpret_tensor(buf712, (4, 32, 7, 7), (37632, 49, 7, 1), 36064)  # alias
        buf725 = reinterpret_tensor(buf727, (4, 32, 7, 7), (39200, 49, 7, 1), 36064)  # alias
        buf740 = reinterpret_tensor(buf743, (4, 32, 7, 7), (40768, 49, 7, 1), 36064)  # alias
        buf756 = reinterpret_tensor(buf760, (4, 32, 7, 7), (42336, 49, 7, 1), 36064)  # alias
        buf773 = reinterpret_tensor(buf778, (4, 32, 7, 7), (43904, 49, 7, 1), 36064)  # alias
        # Source Nodes: [cat_66, cat_67, cat_68, cat_69, cat_70], Original ATen: [aten.cat]
        triton_poi_fused_cat_96.run(buf702, buf711, buf725, buf740, buf756, buf773, 6272, grid=grid(6272), stream=stream0)
        buf713 = empty((4, 768, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock4_denselayer9_norm1, l__mod___features_denseblock4_denselayer9_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_97.run(buf712, primals_677, primals_678, primals_313, primals_314, buf713, 150528, grid=grid(150528), stream=stream0)
        del primals_314
        # Source Nodes: [bottleneck_output_100], Original ATen: [aten.convolution]
        buf714 = extern_kernels.convolution(buf713, primals_315, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf714, (4, 128, 7, 7), (6272, 49, 7, 1))
        buf715 = reinterpret_tensor(buf228, (4, 128, 7, 7), (6272, 49, 7, 1), 0); del buf228  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer9_norm2, l__mod___features_denseblock4_denselayer9_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_84.run(buf714, primals_680, primals_681, primals_316, primals_317, buf715, 25088, grid=grid(25088), stream=stream0)
        del primals_317
        # Source Nodes: [new_features_100], Original ATen: [aten.convolution]
        buf716 = extern_kernels.convolution(buf715, primals_318, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf716, (4, 32, 7, 7), (1568, 49, 7, 1))
        buf726 = reinterpret_tensor(buf727, (4, 32, 7, 7), (39200, 49, 7, 1), 37632)  # alias
        buf741 = reinterpret_tensor(buf743, (4, 32, 7, 7), (40768, 49, 7, 1), 37632)  # alias
        buf757 = reinterpret_tensor(buf760, (4, 32, 7, 7), (42336, 49, 7, 1), 37632)  # alias
        buf774 = reinterpret_tensor(buf778, (4, 32, 7, 7), (43904, 49, 7, 1), 37632)  # alias
        # Source Nodes: [cat_66, cat_67, cat_68, cat_69], Original ATen: [aten.cat]
        triton_poi_fused_cat_98.run(buf716, buf726, buf741, buf757, buf774, 6272, grid=grid(6272), stream=stream0)
        buf728 = empty((4, 800, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock4_denselayer10_norm1, l__mod___features_denseblock4_denselayer10_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_99.run(buf727, primals_683, primals_684, primals_319, primals_320, buf728, 156800, grid=grid(156800), stream=stream0)
        del primals_320
        # Source Nodes: [bottleneck_output_102], Original ATen: [aten.convolution]
        buf729 = extern_kernels.convolution(buf728, primals_321, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf729, (4, 128, 7, 7), (6272, 49, 7, 1))
        buf730 = reinterpret_tensor(buf218, (4, 128, 7, 7), (6272, 49, 7, 1), 0); del buf218  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer10_norm2, l__mod___features_denseblock4_denselayer10_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_84.run(buf729, primals_686, primals_687, primals_322, primals_323, buf730, 25088, grid=grid(25088), stream=stream0)
        del primals_323
        # Source Nodes: [new_features_102], Original ATen: [aten.convolution]
        buf731 = extern_kernels.convolution(buf730, primals_324, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf731, (4, 32, 7, 7), (1568, 49, 7, 1))
        buf733 = reinterpret_tensor(buf743, (4, 32, 7, 7), (40768, 49, 7, 1), 25088)  # alias
        buf749 = reinterpret_tensor(buf760, (4, 32, 7, 7), (42336, 49, 7, 1), 25088)  # alias
        buf766 = reinterpret_tensor(buf778, (4, 32, 7, 7), (43904, 49, 7, 1), 25088)  # alias
        buf784 = reinterpret_tensor(buf797, (4, 32, 7, 7), (45472, 49, 7, 1), 25088)  # alias
        # Source Nodes: [cat_65, cat_66, cat_67, cat_68], Original ATen: [aten.cat]
        triton_poi_fused_cat_100.run(buf641, buf733, buf749, buf766, buf784, 6272, grid=grid(6272), stream=stream0)
        buf734 = reinterpret_tensor(buf743, (4, 32, 7, 7), (40768, 49, 7, 1), 26656)  # alias
        buf750 = reinterpret_tensor(buf760, (4, 32, 7, 7), (42336, 49, 7, 1), 26656)  # alias
        buf767 = reinterpret_tensor(buf778, (4, 32, 7, 7), (43904, 49, 7, 1), 26656)  # alias
        buf785 = reinterpret_tensor(buf797, (4, 32, 7, 7), (45472, 49, 7, 1), 26656)  # alias
        # Source Nodes: [cat_65, cat_66, cat_67, cat_68], Original ATen: [aten.cat]
        triton_poi_fused_cat_100.run(buf646, buf734, buf750, buf767, buf785, 6272, grid=grid(6272), stream=stream0)
        buf735 = reinterpret_tensor(buf743, (4, 32, 7, 7), (40768, 49, 7, 1), 28224)  # alias
        buf751 = reinterpret_tensor(buf760, (4, 32, 7, 7), (42336, 49, 7, 1), 28224)  # alias
        buf768 = reinterpret_tensor(buf778, (4, 32, 7, 7), (43904, 49, 7, 1), 28224)  # alias
        buf786 = reinterpret_tensor(buf797, (4, 32, 7, 7), (45472, 49, 7, 1), 28224)  # alias
        # Source Nodes: [cat_65, cat_66, cat_67, cat_68], Original ATen: [aten.cat]
        triton_poi_fused_cat_100.run(buf651, buf735, buf751, buf768, buf786, 6272, grid=grid(6272), stream=stream0)
        buf736 = reinterpret_tensor(buf743, (4, 32, 7, 7), (40768, 49, 7, 1), 29792)  # alias
        buf752 = reinterpret_tensor(buf760, (4, 32, 7, 7), (42336, 49, 7, 1), 29792)  # alias
        buf769 = reinterpret_tensor(buf778, (4, 32, 7, 7), (43904, 49, 7, 1), 29792)  # alias
        buf787 = reinterpret_tensor(buf797, (4, 32, 7, 7), (45472, 49, 7, 1), 29792)  # alias
        # Source Nodes: [cat_65, cat_66, cat_67, cat_68], Original ATen: [aten.cat]
        triton_poi_fused_cat_100.run(buf656, buf736, buf752, buf769, buf787, 6272, grid=grid(6272), stream=stream0)
        buf737 = reinterpret_tensor(buf743, (4, 32, 7, 7), (40768, 49, 7, 1), 31360)  # alias
        buf753 = reinterpret_tensor(buf760, (4, 32, 7, 7), (42336, 49, 7, 1), 31360)  # alias
        buf770 = reinterpret_tensor(buf778, (4, 32, 7, 7), (43904, 49, 7, 1), 31360)  # alias
        buf788 = reinterpret_tensor(buf797, (4, 32, 7, 7), (45472, 49, 7, 1), 31360)  # alias
        # Source Nodes: [cat_65, cat_66, cat_67, cat_68], Original ATen: [aten.cat]
        triton_poi_fused_cat_100.run(buf666, buf737, buf753, buf770, buf788, 6272, grid=grid(6272), stream=stream0)
        buf742 = reinterpret_tensor(buf743, (4, 32, 7, 7), (40768, 49, 7, 1), 39200)  # alias
        buf758 = reinterpret_tensor(buf760, (4, 32, 7, 7), (42336, 49, 7, 1), 39200)  # alias
        buf775 = reinterpret_tensor(buf778, (4, 32, 7, 7), (43904, 49, 7, 1), 39200)  # alias
        buf793 = reinterpret_tensor(buf797, (4, 32, 7, 7), (45472, 49, 7, 1), 39200)  # alias
        # Source Nodes: [cat_65, cat_66, cat_67, cat_68], Original ATen: [aten.cat]
        triton_poi_fused_cat_100.run(buf731, buf742, buf758, buf775, buf793, 6272, grid=grid(6272), stream=stream0)
        buf744 = empty((4, 832, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock4_denselayer11_norm1, l__mod___features_denseblock4_denselayer11_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_101.run(buf743, primals_689, primals_690, primals_325, primals_326, buf744, 163072, grid=grid(163072), stream=stream0)
        del primals_326
        # Source Nodes: [bottleneck_output_104], Original ATen: [aten.convolution]
        buf745 = extern_kernels.convolution(buf744, primals_327, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf745, (4, 128, 7, 7), (6272, 49, 7, 1))
        buf746 = reinterpret_tensor(buf213, (4, 128, 7, 7), (6272, 49, 7, 1), 0); del buf213  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer11_norm2, l__mod___features_denseblock4_denselayer11_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_84.run(buf745, primals_692, primals_693, primals_328, primals_329, buf746, 25088, grid=grid(25088), stream=stream0)
        del primals_329
        # Source Nodes: [new_features_104], Original ATen: [aten.convolution]
        buf747 = extern_kernels.convolution(buf746, primals_330, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf747, (4, 32, 7, 7), (1568, 49, 7, 1))
        buf754 = reinterpret_tensor(buf760, (4, 32, 7, 7), (42336, 49, 7, 1), 32928)  # alias
        buf771 = reinterpret_tensor(buf778, (4, 32, 7, 7), (43904, 49, 7, 1), 32928)  # alias
        buf789 = reinterpret_tensor(buf797, (4, 32, 7, 7), (45472, 49, 7, 1), 32928)  # alias
        buf808 = reinterpret_tensor(buf817, (4, 32, 7, 7), (47040, 49, 7, 1), 32928)  # alias
        # Source Nodes: [cat_64, cat_65, cat_66, cat_67], Original ATen: [aten.cat]
        triton_poi_fused_cat_102.run(buf677, buf754, buf771, buf789, buf808, 6272, grid=grid(6272), stream=stream0)
        buf759 = reinterpret_tensor(buf760, (4, 32, 7, 7), (42336, 49, 7, 1), 40768)  # alias
        buf776 = reinterpret_tensor(buf778, (4, 32, 7, 7), (43904, 49, 7, 1), 40768)  # alias
        buf794 = reinterpret_tensor(buf797, (4, 32, 7, 7), (45472, 49, 7, 1), 40768)  # alias
        buf813 = reinterpret_tensor(buf817, (4, 32, 7, 7), (47040, 49, 7, 1), 40768)  # alias
        # Source Nodes: [cat_64, cat_65, cat_66, cat_67], Original ATen: [aten.cat]
        triton_poi_fused_cat_102.run(buf747, buf759, buf776, buf794, buf813, 6272, grid=grid(6272), stream=stream0)
        buf761 = empty((4, 864, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock4_denselayer12_norm1, l__mod___features_denseblock4_denselayer12_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_103.run(buf760, primals_695, primals_696, primals_331, primals_332, buf761, 169344, grid=grid(169344), stream=stream0)
        del primals_332
        # Source Nodes: [bottleneck_output_106], Original ATen: [aten.convolution]
        buf762 = extern_kernels.convolution(buf761, primals_333, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf762, (4, 128, 7, 7), (6272, 49, 7, 1))
        buf763 = reinterpret_tensor(buf208, (4, 128, 7, 7), (6272, 49, 7, 1), 0); del buf208  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer12_norm2, l__mod___features_denseblock4_denselayer12_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_84.run(buf762, primals_698, primals_699, primals_334, primals_335, buf763, 25088, grid=grid(25088), stream=stream0)
        del primals_335
        # Source Nodes: [new_features_106], Original ATen: [aten.convolution]
        buf764 = extern_kernels.convolution(buf763, primals_336, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf764, (4, 32, 7, 7), (1568, 49, 7, 1))
        buf772 = reinterpret_tensor(buf778, (4, 32, 7, 7), (43904, 49, 7, 1), 34496)  # alias
        buf790 = reinterpret_tensor(buf797, (4, 32, 7, 7), (45472, 49, 7, 1), 34496)  # alias
        buf809 = reinterpret_tensor(buf817, (4, 32, 7, 7), (47040, 49, 7, 1), 34496)  # alias
        buf829 = reinterpret_tensor(buf838, (4, 32, 7, 7), (48608, 49, 7, 1), 34496)  # alias
        # Source Nodes: [cat_63, cat_64, cat_65, cat_66], Original ATen: [aten.cat]
        triton_poi_fused_cat_104.run(buf689, buf772, buf790, buf809, buf829, 6272, grid=grid(6272), stream=stream0)
        buf777 = reinterpret_tensor(buf778, (4, 32, 7, 7), (43904, 49, 7, 1), 42336)  # alias
        buf795 = reinterpret_tensor(buf797, (4, 32, 7, 7), (45472, 49, 7, 1), 42336)  # alias
        buf814 = reinterpret_tensor(buf817, (4, 32, 7, 7), (47040, 49, 7, 1), 42336)  # alias
        buf834 = reinterpret_tensor(buf838, (4, 32, 7, 7), (48608, 49, 7, 1), 42336)  # alias
        # Source Nodes: [cat_63, cat_64, cat_65, cat_66], Original ATen: [aten.cat]
        triton_poi_fused_cat_104.run(buf764, buf777, buf795, buf814, buf834, 6272, grid=grid(6272), stream=stream0)
        buf779 = empty((4, 896, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock4_denselayer13_norm1, l__mod___features_denseblock4_denselayer13_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_105.run(buf778, primals_701, primals_702, primals_337, primals_338, buf779, 175616, grid=grid(175616), stream=stream0)
        del primals_338
        # Source Nodes: [bottleneck_output_108], Original ATen: [aten.convolution]
        buf780 = extern_kernels.convolution(buf779, primals_339, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf780, (4, 128, 7, 7), (6272, 49, 7, 1))
        buf781 = reinterpret_tensor(buf203, (4, 128, 7, 7), (6272, 49, 7, 1), 0); del buf203  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer13_norm2, l__mod___features_denseblock4_denselayer13_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_84.run(buf780, primals_704, primals_705, primals_340, primals_341, buf781, 25088, grid=grid(25088), stream=stream0)
        del primals_341
        # Source Nodes: [new_features_108], Original ATen: [aten.convolution]
        buf782 = extern_kernels.convolution(buf781, primals_342, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf782, (4, 32, 7, 7), (1568, 49, 7, 1))
        buf791 = reinterpret_tensor(buf797, (4, 32, 7, 7), (45472, 49, 7, 1), 36064)  # alias
        buf810 = reinterpret_tensor(buf817, (4, 32, 7, 7), (47040, 49, 7, 1), 36064)  # alias
        buf830 = reinterpret_tensor(buf838, (4, 32, 7, 7), (48608, 49, 7, 1), 36064)  # alias
        buf851 = reinterpret_tensor(buf860, (4, 32, 7, 7), (50176, 49, 7, 1), 36064)  # alias
        # Source Nodes: [cat_62, cat_63, cat_64, cat_65], Original ATen: [aten.cat]
        triton_poi_fused_cat_106.run(buf702, buf791, buf810, buf830, buf851, 6272, grid=grid(6272), stream=stream0)
        del buf702
        buf792 = reinterpret_tensor(buf797, (4, 32, 7, 7), (45472, 49, 7, 1), 37632)  # alias
        buf811 = reinterpret_tensor(buf817, (4, 32, 7, 7), (47040, 49, 7, 1), 37632)  # alias
        buf831 = reinterpret_tensor(buf838, (4, 32, 7, 7), (48608, 49, 7, 1), 37632)  # alias
        buf852 = reinterpret_tensor(buf860, (4, 32, 7, 7), (50176, 49, 7, 1), 37632)  # alias
        # Source Nodes: [cat_62, cat_63, cat_64, cat_65], Original ATen: [aten.cat]
        triton_poi_fused_cat_106.run(buf716, buf792, buf811, buf831, buf852, 6272, grid=grid(6272), stream=stream0)
        del buf716
        buf796 = reinterpret_tensor(buf797, (4, 32, 7, 7), (45472, 49, 7, 1), 43904)  # alias
        buf815 = reinterpret_tensor(buf817, (4, 32, 7, 7), (47040, 49, 7, 1), 43904)  # alias
        buf835 = reinterpret_tensor(buf838, (4, 32, 7, 7), (48608, 49, 7, 1), 43904)  # alias
        buf856 = reinterpret_tensor(buf860, (4, 32, 7, 7), (50176, 49, 7, 1), 43904)  # alias
        # Source Nodes: [cat_62, cat_63, cat_64, cat_65], Original ATen: [aten.cat]
        triton_poi_fused_cat_106.run(buf782, buf796, buf815, buf835, buf856, 6272, grid=grid(6272), stream=stream0)
        del buf782
        buf798 = empty((4, 928, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock4_denselayer14_norm1, l__mod___features_denseblock4_denselayer14_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_107.run(buf797, primals_707, primals_708, primals_343, primals_344, buf798, 181888, grid=grid(181888), stream=stream0)
        del primals_344
        # Source Nodes: [bottleneck_output_110], Original ATen: [aten.convolution]
        buf799 = extern_kernels.convolution(buf798, primals_345, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf799, (4, 128, 7, 7), (6272, 49, 7, 1))
        buf800 = reinterpret_tensor(buf579, (4, 128, 7, 7), (6272, 49, 7, 1), 0); del buf579  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer14_norm2, l__mod___features_denseblock4_denselayer14_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_84.run(buf799, primals_710, primals_711, primals_346, primals_347, buf800, 25088, grid=grid(25088), stream=stream0)
        del primals_347
        # Source Nodes: [new_features_110], Original ATen: [aten.convolution]
        buf801 = extern_kernels.convolution(buf800, primals_348, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf801, (4, 32, 7, 7), (1568, 49, 7, 1))
        buf803 = reinterpret_tensor(buf817, (4, 32, 7, 7), (47040, 49, 7, 1), 25088)  # alias
        buf823 = reinterpret_tensor(buf838, (4, 32, 7, 7), (48608, 49, 7, 1), 25088)  # alias
        buf844 = reinterpret_tensor(buf860, (4, 32, 7, 7), (50176, 49, 7, 1), 25088)  # alias
        # Source Nodes: [cat_62, cat_63, cat_64], Original ATen: [aten.cat]
        triton_poi_fused_cat_108.run(buf641, buf803, buf823, buf844, 6272, grid=grid(6272), stream=stream0)
        del buf641
        buf804 = reinterpret_tensor(buf817, (4, 32, 7, 7), (47040, 49, 7, 1), 26656)  # alias
        buf824 = reinterpret_tensor(buf838, (4, 32, 7, 7), (48608, 49, 7, 1), 26656)  # alias
        buf845 = reinterpret_tensor(buf860, (4, 32, 7, 7), (50176, 49, 7, 1), 26656)  # alias
        # Source Nodes: [cat_62, cat_63, cat_64], Original ATen: [aten.cat]
        triton_poi_fused_cat_108.run(buf646, buf804, buf824, buf845, 6272, grid=grid(6272), stream=stream0)
        del buf646
        buf805 = reinterpret_tensor(buf817, (4, 32, 7, 7), (47040, 49, 7, 1), 28224)  # alias
        buf825 = reinterpret_tensor(buf838, (4, 32, 7, 7), (48608, 49, 7, 1), 28224)  # alias
        buf846 = reinterpret_tensor(buf860, (4, 32, 7, 7), (50176, 49, 7, 1), 28224)  # alias
        # Source Nodes: [cat_62, cat_63, cat_64], Original ATen: [aten.cat]
        triton_poi_fused_cat_108.run(buf651, buf805, buf825, buf846, 6272, grid=grid(6272), stream=stream0)
        del buf651
        buf806 = reinterpret_tensor(buf817, (4, 32, 7, 7), (47040, 49, 7, 1), 29792)  # alias
        buf826 = reinterpret_tensor(buf838, (4, 32, 7, 7), (48608, 49, 7, 1), 29792)  # alias
        buf847 = reinterpret_tensor(buf860, (4, 32, 7, 7), (50176, 49, 7, 1), 29792)  # alias
        # Source Nodes: [cat_62, cat_63, cat_64], Original ATen: [aten.cat]
        triton_poi_fused_cat_108.run(buf656, buf806, buf826, buf847, 6272, grid=grid(6272), stream=stream0)
        del buf656
        buf807 = reinterpret_tensor(buf817, (4, 32, 7, 7), (47040, 49, 7, 1), 31360)  # alias
        buf827 = reinterpret_tensor(buf838, (4, 32, 7, 7), (48608, 49, 7, 1), 31360)  # alias
        buf848 = reinterpret_tensor(buf860, (4, 32, 7, 7), (50176, 49, 7, 1), 31360)  # alias
        # Source Nodes: [cat_62, cat_63, cat_64], Original ATen: [aten.cat]
        triton_poi_fused_cat_108.run(buf666, buf807, buf827, buf848, 6272, grid=grid(6272), stream=stream0)
        del buf666
        buf812 = reinterpret_tensor(buf817, (4, 32, 7, 7), (47040, 49, 7, 1), 39200)  # alias
        buf832 = reinterpret_tensor(buf838, (4, 32, 7, 7), (48608, 49, 7, 1), 39200)  # alias
        buf853 = reinterpret_tensor(buf860, (4, 32, 7, 7), (50176, 49, 7, 1), 39200)  # alias
        # Source Nodes: [cat_62, cat_63, cat_64], Original ATen: [aten.cat]
        triton_poi_fused_cat_108.run(buf731, buf812, buf832, buf853, 6272, grid=grid(6272), stream=stream0)
        del buf731
        buf816 = reinterpret_tensor(buf817, (4, 32, 7, 7), (47040, 49, 7, 1), 45472)  # alias
        buf836 = reinterpret_tensor(buf838, (4, 32, 7, 7), (48608, 49, 7, 1), 45472)  # alias
        buf857 = reinterpret_tensor(buf860, (4, 32, 7, 7), (50176, 49, 7, 1), 45472)  # alias
        # Source Nodes: [cat_62, cat_63, cat_64], Original ATen: [aten.cat]
        triton_poi_fused_cat_108.run(buf801, buf816, buf836, buf857, 6272, grid=grid(6272), stream=stream0)
        del buf801
        buf818 = empty((4, 960, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock4_denselayer15_norm1, l__mod___features_denseblock4_denselayer15_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_109.run(buf817, primals_713, primals_714, primals_349, primals_350, buf818, 188160, grid=grid(188160), stream=stream0)
        del primals_350
        # Source Nodes: [bottleneck_output_112], Original ATen: [aten.convolution]
        buf819 = extern_kernels.convolution(buf818, primals_351, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf819, (4, 128, 7, 7), (6272, 49, 7, 1))
        buf820 = reinterpret_tensor(buf498, (4, 128, 7, 7), (6272, 49, 7, 1), 0); del buf498  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer15_norm2, l__mod___features_denseblock4_denselayer15_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_84.run(buf819, primals_716, primals_717, primals_352, primals_353, buf820, 25088, grid=grid(25088), stream=stream0)
        del primals_353
        # Source Nodes: [new_features_112], Original ATen: [aten.convolution]
        buf821 = extern_kernels.convolution(buf820, primals_354, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf821, (4, 32, 7, 7), (1568, 49, 7, 1))
        buf828 = reinterpret_tensor(buf838, (4, 32, 7, 7), (48608, 49, 7, 1), 32928)  # alias
        buf849 = reinterpret_tensor(buf860, (4, 32, 7, 7), (50176, 49, 7, 1), 32928)  # alias
        # Source Nodes: [cat_62, cat_63], Original ATen: [aten.cat]
        triton_poi_fused_cat_110.run(buf677, buf828, buf849, 6272, grid=grid(6272), stream=stream0)
        del buf677
        buf833 = reinterpret_tensor(buf838, (4, 32, 7, 7), (48608, 49, 7, 1), 40768)  # alias
        buf854 = reinterpret_tensor(buf860, (4, 32, 7, 7), (50176, 49, 7, 1), 40768)  # alias
        # Source Nodes: [cat_62, cat_63], Original ATen: [aten.cat]
        triton_poi_fused_cat_110.run(buf747, buf833, buf854, 6272, grid=grid(6272), stream=stream0)
        del buf747
        buf837 = reinterpret_tensor(buf838, (4, 32, 7, 7), (48608, 49, 7, 1), 47040)  # alias
        buf858 = reinterpret_tensor(buf860, (4, 32, 7, 7), (50176, 49, 7, 1), 47040)  # alias
        # Source Nodes: [cat_62, cat_63], Original ATen: [aten.cat]
        triton_poi_fused_cat_110.run(buf821, buf837, buf858, 6272, grid=grid(6272), stream=stream0)
        del buf821
        buf839 = empty((4, 992, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_denseblock4_denselayer16_norm1, l__mod___features_denseblock4_denselayer16_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_111.run(buf838, primals_719, primals_720, primals_355, primals_356, buf839, 194432, grid=grid(194432), stream=stream0)
        del primals_356
        # Source Nodes: [bottleneck_output_114], Original ATen: [aten.convolution]
        buf840 = extern_kernels.convolution(buf839, primals_357, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf840, (4, 128, 7, 7), (6272, 49, 7, 1))
        buf841 = reinterpret_tensor(buf426, (4, 128, 7, 7), (6272, 49, 7, 1), 0); del buf426  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer16_norm2, l__mod___features_denseblock4_denselayer16_relu2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_84.run(buf840, primals_722, primals_723, primals_358, primals_359, buf841, 25088, grid=grid(25088), stream=stream0)
        del primals_359
        # Source Nodes: [new_features_114], Original ATen: [aten.convolution]
        buf842 = extern_kernels.convolution(buf841, primals_360, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf842, (4, 32, 7, 7), (1568, 49, 7, 1))
        buf850 = reinterpret_tensor(buf860, (4, 32, 7, 7), (50176, 49, 7, 1), 34496)  # alias
        # Source Nodes: [cat_62], Original ATen: [aten.cat]
        triton_poi_fused_cat_112.run(buf689, buf850, 6272, grid=grid(6272), stream=stream0)
        del buf689
        buf855 = reinterpret_tensor(buf860, (4, 32, 7, 7), (50176, 49, 7, 1), 42336)  # alias
        # Source Nodes: [cat_62], Original ATen: [aten.cat]
        triton_poi_fused_cat_112.run(buf764, buf855, 6272, grid=grid(6272), stream=stream0)
        del buf764
        buf859 = reinterpret_tensor(buf860, (4, 32, 7, 7), (50176, 49, 7, 1), 48608)  # alias
        # Source Nodes: [cat_62], Original ATen: [aten.cat]
        triton_poi_fused_cat_112.run(buf842, buf859, 6272, grid=grid(6272), stream=stream0)
        del buf842
        buf865 = empty((4, 1024, 7, 7), device='cuda', dtype=torch.bool)
        buf862 = empty_strided((4, 1024, 1, 1), (1024, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf863 = reinterpret_tensor(buf862, (4, 1024), (1024, 1), 0); del buf862  # reuse
        # Source Nodes: [features, out, out_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu, aten.threshold_backward, aten.view]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_threshold_backward_view_113.run(buf863, buf860, primals_725, primals_726, primals_361, primals_362, buf865, 4096, 49, grid=grid(4096), stream=stream0)
        del primals_362
        buf864 = empty((4, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_364, buf863, reinterpret_tensor(primals_363, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf864)
        del primals_364
        return (buf864, primals_1, primals_2, primals_4, primals_6, primals_7, primals_9, primals_10, primals_12, primals_13, primals_15, primals_16, primals_18, primals_19, primals_21, primals_22, primals_24, primals_25, primals_27, primals_28, primals_30, primals_31, primals_33, primals_34, primals_36, primals_37, primals_39, primals_40, primals_42, primals_43, primals_45, primals_46, primals_48, primals_49, primals_51, primals_52, primals_54, primals_55, primals_57, primals_58, primals_60, primals_61, primals_63, primals_64, primals_66, primals_67, primals_69, primals_70, primals_72, primals_73, primals_75, primals_76, primals_78, primals_79, primals_81, primals_82, primals_84, primals_85, primals_87, primals_88, primals_90, primals_91, primals_93, primals_94, primals_96, primals_97, primals_99, primals_100, primals_102, primals_103, primals_105, primals_106, primals_108, primals_109, primals_111, primals_112, primals_114, primals_115, primals_117, primals_118, primals_120, primals_121, primals_123, primals_124, primals_126, primals_127, primals_129, primals_130, primals_132, primals_133, primals_135, primals_136, primals_138, primals_139, primals_141, primals_142, primals_144, primals_145, primals_147, primals_148, primals_150, primals_151, primals_153, primals_154, primals_156, primals_157, primals_159, primals_160, primals_162, primals_163, primals_165, primals_166, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, primals_291, primals_292, primals_294, primals_295, primals_297, primals_298, primals_300, primals_301, primals_303, primals_304, primals_306, primals_307, primals_309, primals_310, primals_312, primals_313, primals_315, primals_316, primals_318, primals_319, primals_321, primals_322, primals_324, primals_325, primals_327, primals_328, primals_330, primals_331, primals_333, primals_334, primals_336, primals_337, primals_339, primals_340, primals_342, primals_343, primals_345, primals_346, primals_348, primals_349, primals_351, primals_352, primals_354, primals_355, primals_357, primals_358, primals_360, primals_361, primals_365, primals_366, primals_369, primals_371, primals_372, primals_374, primals_375, primals_377, primals_378, primals_380, primals_381, primals_383, primals_384, primals_386, primals_387, primals_389, primals_390, primals_392, primals_393, primals_395, primals_396, primals_398, primals_399, primals_401, primals_402, primals_404, primals_405, primals_407, primals_408, primals_410, primals_411, primals_413, primals_414, primals_416, primals_417, primals_419, primals_420, primals_422, primals_423, primals_425, primals_426, primals_428, primals_429, primals_431, primals_432, primals_434, primals_435, primals_437, primals_438, primals_440, primals_441, primals_443, primals_444, primals_446, primals_447, primals_449, primals_450, primals_452, primals_453, primals_455, primals_456, primals_458, primals_459, primals_461, primals_462, primals_464, primals_465, primals_467, primals_468, primals_470, primals_471, primals_473, primals_474, primals_476, primals_477, primals_479, primals_480, primals_482, primals_483, primals_485, primals_486, primals_488, primals_489, primals_491, primals_492, primals_494, primals_495, primals_497, primals_498, primals_500, primals_501, primals_503, primals_504, primals_506, primals_507, primals_509, primals_510, primals_512, primals_513, primals_515, primals_516, primals_518, primals_519, primals_521, primals_522, primals_524, primals_525, primals_527, primals_528, primals_530, primals_531, primals_533, primals_534, primals_536, primals_537, primals_539, primals_540, primals_542, primals_543, primals_545, primals_546, primals_548, primals_549, primals_551, primals_552, primals_554, primals_555, primals_557, primals_558, primals_560, primals_561, primals_563, primals_564, primals_566, primals_567, primals_569, primals_570, primals_572, primals_573, primals_575, primals_576, primals_578, primals_579, primals_581, primals_582, primals_584, primals_585, primals_587, primals_588, primals_590, primals_591, primals_593, primals_594, primals_596, primals_597, primals_599, primals_600, primals_602, primals_603, primals_605, primals_606, primals_608, primals_609, primals_611, primals_612, primals_614, primals_615, primals_617, primals_618, primals_620, primals_621, primals_623, primals_624, primals_626, primals_627, primals_629, primals_630, primals_632, primals_633, primals_635, primals_636, primals_638, primals_639, primals_641, primals_642, primals_644, primals_645, primals_647, primals_648, primals_650, primals_651, primals_653, primals_654, primals_656, primals_657, primals_659, primals_660, primals_662, primals_663, primals_665, primals_666, primals_668, primals_669, primals_671, primals_672, primals_674, primals_675, primals_677, primals_678, primals_680, primals_681, primals_683, primals_684, primals_686, primals_687, primals_689, primals_690, primals_692, primals_693, primals_695, primals_696, primals_698, primals_699, primals_701, primals_702, primals_704, primals_705, primals_707, primals_708, primals_710, primals_711, primals_713, primals_714, primals_716, primals_717, primals_719, primals_720, primals_722, primals_723, primals_725, primals_726, primals_728, buf0, buf1, buf3, buf4, buf5, buf6, buf9, buf10, buf11, buf12, buf14, buf15, buf16, buf17, buf19, buf20, buf21, buf22, buf29, buf30, buf31, buf32, buf40, buf41, buf42, buf43, buf52, buf53, buf54, buf55, buf56, buf57, buf58, buf60, buf61, buf62, buf63, buf65, buf66, buf67, buf68, buf70, buf71, buf72, buf73, buf80, buf81, buf82, buf83, buf91, buf92, buf93, buf94, buf103, buf104, buf105, buf106, buf116, buf117, buf118, buf119, buf130, buf131, buf132, buf133, buf145, buf146, buf147, buf148, buf161, buf162, buf163, buf164, buf178, buf179, buf180, buf181, buf196, buf197, buf198, buf199, buf200, buf201, buf202, buf204, buf205, buf206, buf207, buf209, buf210, buf211, buf212, buf214, buf215, buf216, buf217, buf224, buf225, buf226, buf227, buf235, buf236, buf237, buf238, buf247, buf248, buf249, buf250, buf260, buf261, buf262, buf263, buf274, buf275, buf276, buf277, buf289, buf290, buf291, buf292, buf305, buf306, buf307, buf308, buf322, buf323, buf324, buf325, buf340, buf341, buf342, buf343, buf359, buf360, buf361, buf362, buf379, buf380, buf381, buf382, buf400, buf401, buf402, buf403, buf422, buf423, buf424, buf425, buf445, buf446, buf447, buf448, buf469, buf470, buf471, buf472, buf494, buf495, buf496, buf497, buf520, buf521, buf522, buf523, buf547, buf548, buf549, buf550, buf575, buf576, buf577, buf578, buf604, buf605, buf606, buf607, buf634, buf635, buf636, buf637, buf638, buf639, buf640, buf642, buf643, buf644, buf645, buf647, buf648, buf649, buf650, buf652, buf653, buf654, buf655, buf662, buf663, buf664, buf665, buf673, buf674, buf675, buf676, buf685, buf686, buf687, buf688, buf698, buf699, buf700, buf701, buf712, buf713, buf714, buf715, buf727, buf728, buf729, buf730, buf743, buf744, buf745, buf746, buf760, buf761, buf762, buf763, buf778, buf779, buf780, buf781, buf797, buf798, buf799, buf800, buf817, buf818, buf819, buf820, buf838, buf839, buf840, buf841, buf860, buf863, reinterpret_tensor(primals_363, (1000, 1024), (1024, 1), 0), buf865, buf866, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((128, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((128, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((128, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((128, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((128, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((128, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((128, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((128, 352, 1, 1), (352, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((128, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((128, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((128, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((128, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((128, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((128, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((128, 352, 1, 1), (352, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((128, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((128, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((128, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((128, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((544, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((544, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((128, 544, 1, 1), (544, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((128, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((128, 608, 1, 1), (608, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((128, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((128, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_201 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_204 = rand_strided((128, 704, 1, 1), (704, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_207 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_210 = rand_strided((128, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_213 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_216 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_219 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_222 = rand_strided((128, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_225 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_228 = rand_strided((128, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_231 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_234 = rand_strided((128, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_237 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_240 = rand_strided((128, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_243 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((928, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((928, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_246 = rand_strided((128, 928, 1, 1), (928, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_249 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_252 = rand_strided((128, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_255 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((992, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((992, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_258 = rand_strided((128, 992, 1, 1), (992, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_261 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_264 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_267 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_270 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((544, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((544, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_273 = rand_strided((128, 544, 1, 1), (544, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_276 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_279 = rand_strided((128, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_282 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_285 = rand_strided((128, 608, 1, 1), (608, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_288 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_291 = rand_strided((128, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_294 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_297 = rand_strided((128, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_300 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_303 = rand_strided((128, 704, 1, 1), (704, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_306 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_309 = rand_strided((128, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_312 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_315 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_318 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_321 = rand_strided((128, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_322 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_323 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_324 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_325 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_326 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_327 = rand_strided((128, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_328 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_329 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_330 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_331 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_332 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_333 = rand_strided((128, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_334 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_335 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_336 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_337 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_338 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_339 = rand_strided((128, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_340 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_341 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_342 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_343 = rand_strided((928, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_344 = rand_strided((928, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_345 = rand_strided((128, 928, 1, 1), (928, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_346 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_347 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_348 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_349 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_350 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_351 = rand_strided((128, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_352 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_353 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_354 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_355 = rand_strided((992, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_356 = rand_strided((992, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_357 = rand_strided((128, 992, 1, 1), (992, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_358 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_359 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_360 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_361 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_362 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_363 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_364 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_365 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_366 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_367 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_368 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_369 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_370 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_371 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_372 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_373 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_374 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_375 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_376 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_377 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_378 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_379 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_380 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_381 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_382 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_383 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_384 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_385 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_386 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_387 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_388 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_389 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_390 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_391 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_392 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_393 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_394 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_395 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_396 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_397 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_398 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_399 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_400 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_401 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_402 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_403 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_404 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_405 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_406 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_407 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_408 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_409 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_410 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_411 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_412 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_413 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_414 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_415 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_416 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_417 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_418 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_419 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_420 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_421 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_422 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_423 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_424 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_425 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_426 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_427 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_428 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_429 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_430 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_431 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_432 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_433 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_434 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_435 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_436 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_437 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_438 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_439 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_440 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_441 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_442 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_443 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_444 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_445 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_446 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_447 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_448 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_449 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_450 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_451 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_452 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_453 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_454 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_455 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_456 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_457 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_458 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_459 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_460 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_461 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_462 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_463 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_464 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_465 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_466 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_467 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_468 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_469 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_470 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_471 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_472 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_473 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_474 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_475 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_476 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_477 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_478 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_479 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_480 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_481 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_482 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_483 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_484 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_485 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_486 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_487 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_488 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_489 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_490 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_491 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_492 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_493 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_494 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_495 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_496 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_497 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_498 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_499 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_500 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_501 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_502 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_503 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_504 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_505 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_506 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_507 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_508 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_509 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_510 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_511 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_512 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_513 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_514 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_515 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_516 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_517 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_518 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_519 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_520 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_521 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_522 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_523 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_524 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_525 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_526 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_527 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_528 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_529 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_530 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_531 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_532 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_533 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_534 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_535 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_536 = rand_strided((544, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_537 = rand_strided((544, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_538 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_539 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_540 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_541 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_542 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_543 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_544 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_545 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_546 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_547 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_548 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_549 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_550 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_551 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_552 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_553 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_554 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_555 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_556 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_557 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_558 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_559 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_560 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_561 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_562 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_563 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_564 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_565 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_566 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_567 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_568 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_569 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_570 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_571 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_572 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_573 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_574 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_575 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_576 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_577 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_578 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_579 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_580 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_581 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_582 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_583 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_584 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_585 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_586 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_587 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_588 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_589 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_590 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_591 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_592 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_593 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_594 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_595 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_596 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_597 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_598 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_599 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_600 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_601 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_602 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_603 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_604 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_605 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_606 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_607 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_608 = rand_strided((928, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_609 = rand_strided((928, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_610 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_611 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_612 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_613 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_614 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_615 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_616 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_617 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_618 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_619 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_620 = rand_strided((992, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_621 = rand_strided((992, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_622 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_623 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_624 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_625 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_626 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_627 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_628 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_629 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_630 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_631 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_632 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_633 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_634 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_635 = rand_strided((544, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_636 = rand_strided((544, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_637 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_638 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_639 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_640 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_641 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_642 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_643 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_644 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_645 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_646 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_647 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_648 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_649 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_650 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_651 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_652 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_653 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_654 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_655 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_656 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_657 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_658 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_659 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_660 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_661 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_662 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_663 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_664 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_665 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_666 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_667 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_668 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_669 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_670 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_671 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_672 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_673 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_674 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_675 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_676 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_677 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_678 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_679 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_680 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_681 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_682 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_683 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_684 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_685 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_686 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_687 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_688 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_689 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_690 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_691 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_692 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_693 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_694 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_695 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_696 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_697 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_698 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_699 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_700 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_701 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_702 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_703 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_704 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_705 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_706 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_707 = rand_strided((928, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_708 = rand_strided((928, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_709 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_710 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_711 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_712 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_713 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_714 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_715 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_716 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_717 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_718 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_719 = rand_strided((992, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_720 = rand_strided((992, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_721 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_722 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_723 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_724 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_725 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_726 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_727 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_728 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321, primals_322, primals_323, primals_324, primals_325, primals_326, primals_327, primals_328, primals_329, primals_330, primals_331, primals_332, primals_333, primals_334, primals_335, primals_336, primals_337, primals_338, primals_339, primals_340, primals_341, primals_342, primals_343, primals_344, primals_345, primals_346, primals_347, primals_348, primals_349, primals_350, primals_351, primals_352, primals_353, primals_354, primals_355, primals_356, primals_357, primals_358, primals_359, primals_360, primals_361, primals_362, primals_363, primals_364, primals_365, primals_366, primals_367, primals_368, primals_369, primals_370, primals_371, primals_372, primals_373, primals_374, primals_375, primals_376, primals_377, primals_378, primals_379, primals_380, primals_381, primals_382, primals_383, primals_384, primals_385, primals_386, primals_387, primals_388, primals_389, primals_390, primals_391, primals_392, primals_393, primals_394, primals_395, primals_396, primals_397, primals_398, primals_399, primals_400, primals_401, primals_402, primals_403, primals_404, primals_405, primals_406, primals_407, primals_408, primals_409, primals_410, primals_411, primals_412, primals_413, primals_414, primals_415, primals_416, primals_417, primals_418, primals_419, primals_420, primals_421, primals_422, primals_423, primals_424, primals_425, primals_426, primals_427, primals_428, primals_429, primals_430, primals_431, primals_432, primals_433, primals_434, primals_435, primals_436, primals_437, primals_438, primals_439, primals_440, primals_441, primals_442, primals_443, primals_444, primals_445, primals_446, primals_447, primals_448, primals_449, primals_450, primals_451, primals_452, primals_453, primals_454, primals_455, primals_456, primals_457, primals_458, primals_459, primals_460, primals_461, primals_462, primals_463, primals_464, primals_465, primals_466, primals_467, primals_468, primals_469, primals_470, primals_471, primals_472, primals_473, primals_474, primals_475, primals_476, primals_477, primals_478, primals_479, primals_480, primals_481, primals_482, primals_483, primals_484, primals_485, primals_486, primals_487, primals_488, primals_489, primals_490, primals_491, primals_492, primals_493, primals_494, primals_495, primals_496, primals_497, primals_498, primals_499, primals_500, primals_501, primals_502, primals_503, primals_504, primals_505, primals_506, primals_507, primals_508, primals_509, primals_510, primals_511, primals_512, primals_513, primals_514, primals_515, primals_516, primals_517, primals_518, primals_519, primals_520, primals_521, primals_522, primals_523, primals_524, primals_525, primals_526, primals_527, primals_528, primals_529, primals_530, primals_531, primals_532, primals_533, primals_534, primals_535, primals_536, primals_537, primals_538, primals_539, primals_540, primals_541, primals_542, primals_543, primals_544, primals_545, primals_546, primals_547, primals_548, primals_549, primals_550, primals_551, primals_552, primals_553, primals_554, primals_555, primals_556, primals_557, primals_558, primals_559, primals_560, primals_561, primals_562, primals_563, primals_564, primals_565, primals_566, primals_567, primals_568, primals_569, primals_570, primals_571, primals_572, primals_573, primals_574, primals_575, primals_576, primals_577, primals_578, primals_579, primals_580, primals_581, primals_582, primals_583, primals_584, primals_585, primals_586, primals_587, primals_588, primals_589, primals_590, primals_591, primals_592, primals_593, primals_594, primals_595, primals_596, primals_597, primals_598, primals_599, primals_600, primals_601, primals_602, primals_603, primals_604, primals_605, primals_606, primals_607, primals_608, primals_609, primals_610, primals_611, primals_612, primals_613, primals_614, primals_615, primals_616, primals_617, primals_618, primals_619, primals_620, primals_621, primals_622, primals_623, primals_624, primals_625, primals_626, primals_627, primals_628, primals_629, primals_630, primals_631, primals_632, primals_633, primals_634, primals_635, primals_636, primals_637, primals_638, primals_639, primals_640, primals_641, primals_642, primals_643, primals_644, primals_645, primals_646, primals_647, primals_648, primals_649, primals_650, primals_651, primals_652, primals_653, primals_654, primals_655, primals_656, primals_657, primals_658, primals_659, primals_660, primals_661, primals_662, primals_663, primals_664, primals_665, primals_666, primals_667, primals_668, primals_669, primals_670, primals_671, primals_672, primals_673, primals_674, primals_675, primals_676, primals_677, primals_678, primals_679, primals_680, primals_681, primals_682, primals_683, primals_684, primals_685, primals_686, primals_687, primals_688, primals_689, primals_690, primals_691, primals_692, primals_693, primals_694, primals_695, primals_696, primals_697, primals_698, primals_699, primals_700, primals_701, primals_702, primals_703, primals_704, primals_705, primals_706, primals_707, primals_708, primals_709, primals_710, primals_711, primals_712, primals_713, primals_714, primals_715, primals_716, primals_717, primals_718, primals_719, primals_720, primals_721, primals_722, primals_723, primals_724, primals_725, primals_726, primals_727, primals_728]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('densenet121', benchmark_compiled_module)
