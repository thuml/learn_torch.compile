
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


# kernel path: /tmp/torchinductor_youkaichao/o2/co2mllfqlh6uwi7acgxcuwk6q2ywutrc3bqf5rc64pddsttrh7z3.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_0', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/ta/ctax5tfeo5dgv54q55eczmmmbbxakrx7zf45afbrg3og7jjochpj.py
# Source Nodes: [bottleneck_output, cat_117, cat_118, l__mod___features_denseblock1_denselayer1_norm1, l__mod___features_denseblock1_denselayer1_relu1, l__mod___features_norm0, l__mod___features_pool0, l__mod___features_relu0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# bottleneck_output => convolution_1
# cat_117 => cat_5
# cat_118 => cat_4
# l__mod___features_denseblock1_denselayer1_norm1 => add_3, mul_4, mul_5, sub_1
# l__mod___features_denseblock1_denselayer1_relu1 => relu_1
# l__mod___features_norm0 => add_1, mul_1, mul_2, sub
# l__mod___features_pool0 => max_pool2d_with_indices
# l__mod___features_relu0 => relu
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_max_pool2d_with_indices_relu_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_max_pool2d_with_indices_relu_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 56) % 56
    x0 = xindex % 56
    x5 = (xindex // 56)
    x3 = (xindex // 200704)
    x6 = xindex % 200704
    x2 = (xindex // 3136) % 64
    x8 = xindex
    tmp70 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp80 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp82 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
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
    tmp71 = tmp69 - tmp70
    tmp73 = 1e-05
    tmp74 = tmp72 + tmp73
    tmp75 = tl.sqrt(tmp74)
    tmp76 = 1 / tmp75
    tmp77 = 1.0
    tmp78 = tmp76 * tmp77
    tmp79 = tmp71 * tmp78
    tmp81 = tmp79 * tmp80
    tmp83 = tmp81 + tmp82
    tmp84 = triton_helpers.maximum(0, tmp83)
    tl.store(out_ptr0 + (x6 + (602112*x3)), tmp69, None)
    tl.store(out_ptr1 + (x8), tmp84, None)
    tl.store(out_ptr2 + (x6 + (702464*x3)), tmp69, None)
    tl.store(out_ptr3 + (x6 + (802816*x3)), tmp69, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ic/cicf6gwynmnv4fnuwiqtvnu7pz4cnhkx7w7v4tp4il7hyxxi33mr.py
# Source Nodes: [l__mod___features_denseblock1_denselayer1_norm2, l__mod___features_denseblock1_denselayer1_relu2, new_features], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___features_denseblock1_denselayer1_norm2 => add_5, mul_7, mul_8, sub_2
# l__mod___features_denseblock1_denselayer1_relu2 => relu_2
# new_features => convolution_2
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xq/cxqvtz7pfujydbhahttc345quoryhyjtk257tdoiquvkllondskt.py
# Source Nodes: [bottleneck_output_2, cat_122, l__mod___features_denseblock1_denselayer2_norm1, l__mod___features_denseblock1_denselayer2_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# bottleneck_output_2 => convolution_3
# cat_122 => cat
# l__mod___features_denseblock1_denselayer2_norm1 => add_7, mul_10, mul_11, sub_3
# l__mod___features_denseblock1_denselayer2_relu1 => relu_3
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3136) % 96
    x2 = (xindex // 301056)
    x3 = xindex % 301056
    x4 = xindex
    tmp15 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp17 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (602112*x2)), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 96, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-200704) + x3 + (100352*x2)), tmp8, other=0.0)
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
    tl.store(out_ptr0 + (x4), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fu/cfuwiogh5ifb3mzukfry6w2n2j4tbvbtm3sqmxp2x5xhmxdubnlz.py
# Source Nodes: [bottleneck_output_4, cat_121, l__mod___features_denseblock1_denselayer3_norm1, l__mod___features_denseblock1_denselayer3_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# bottleneck_output_4 => convolution_5
# cat_121 => cat_1
# l__mod___features_denseblock1_denselayer3_norm1 => add_11, mul_16, mul_17, sub_5
# l__mod___features_denseblock1_denselayer3_relu1 => relu_5
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr0 + (x3 + (602112*x2)), tmp4, other=0.0)
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
    tl.store(out_ptr0 + (x4), tmp37, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ls/clsdsplthc5f34vy43tsygsdi3vopu43wcrkecq6noyftabybbri.py
# Source Nodes: [bottleneck_output_6, cat_120, l__mod___features_denseblock1_denselayer4_norm1, l__mod___features_denseblock1_denselayer4_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# bottleneck_output_6 => convolution_7
# cat_120 => cat_2
# l__mod___features_denseblock1_denselayer4_norm1 => add_15, mul_22, mul_23, sub_7
# l__mod___features_denseblock1_denselayer4_relu1 => relu_7
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp5 = tl.load(in_ptr0 + (x3 + (602112*x2)), tmp4, other=0.0)
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
    tl.store(out_ptr0 + (x4), tmp45, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/eq/ceq7rqspvpo27svnoe2con3s5qc7vgnjgrfr3joffzgwucmpslxl.py
# Source Nodes: [cat_117, cat_118, cat_119], Original ATen: [aten.cat]
# cat_117 => cat_5
# cat_118 => cat_4
# cat_119 => cat_3
triton_poi_fused_cat_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_6', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/mi/cmijngckzb7yhatdevpif2tmnhbva5hfvgfudtruw7fhqo7wblad.py
# Source Nodes: [bottleneck_output_8, l__mod___features_denseblock1_denselayer5_norm1, l__mod___features_denseblock1_denselayer5_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_8 => convolution_9
# l__mod___features_denseblock1_denselayer5_norm1 => add_19, mul_28, mul_29, sub_9
# l__mod___features_denseblock1_denselayer5_relu1 => relu_9
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 192
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ei/ceisvteslo22xypuk4bg2jacll2vzfeuxe2bzfj36meqgfyl7dig.py
# Source Nodes: [cat_117, cat_118], Original ATen: [aten.cat]
# cat_117 => cat_5
# cat_118 => cat_4
triton_poi_fused_cat_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_8', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/b7/cb7xhgv3xfed4tnpieqse7vv6fw6cg67eupvcaaa72r3wgnmkl4k.py
# Source Nodes: [bottleneck_output_10, l__mod___features_denseblock1_denselayer6_norm1, l__mod___features_denseblock1_denselayer6_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_10 => convolution_11
# l__mod___features_denseblock1_denselayer6_norm1 => add_23, mul_34, mul_35, sub_11
# l__mod___features_denseblock1_denselayer6_relu1 => relu_11
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2809856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 224
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kj/ckjaro4hhohoboersbawvpxumsgbpvp5cufjlxnganxnornrkfjj.py
# Source Nodes: [cat_117], Original ATen: [aten.cat]
# cat_117 => cat_5
triton_poi_fused_cat_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_10', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/yj/cyj6hohlwht7mbs3wpaj623cuyfhh76bdg7x7l6oxjjkeitmu523.py
# Source Nodes: [l__mod___features_transition1_conv, l__mod___features_transition1_norm, l__mod___features_transition1_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___features_transition1_conv => convolution_13
# l__mod___features_transition1_norm => add_27, mul_40, mul_41, sub_13
# l__mod___features_transition1_relu => relu_13
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qs/cqs7bewktnkomsuw5ynivi6xwjiea7ww3ia7d4joirkcvmpzgbmx.py
# Source Nodes: [bottleneck_output_12, cat_104, cat_105, cat_106, cat_107, cat_108, cat_109, cat_110, cat_111, l__mod___features_denseblock2_denselayer1_norm1, l__mod___features_denseblock2_denselayer1_relu1, l__mod___features_transition1_pool], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d, aten.cat, aten.convolution, aten.relu]
# bottleneck_output_12 => convolution_14
# cat_104 => cat_17
# cat_105 => cat_16
# cat_106 => cat_15
# cat_107 => cat_14
# cat_108 => cat_13
# cat_109 => cat_12
# cat_110 => cat_11
# cat_111 => cat_10
# l__mod___features_denseblock2_denselayer1_norm1 => add_29, mul_43, mul_44, sub_14
# l__mod___features_denseblock2_denselayer1_relu1 => relu_14
# l__mod___features_transition1_pool => avg_pool2d
triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(15,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 28
    x4 = (xindex // 28)
    x2 = (xindex // 784) % 128
    x5 = xindex
    x3 = (xindex // 100352)
    x7 = xindex % 100352
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (112*x4)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (112*x4)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (56 + (2*x0) + (112*x4)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (57 + (2*x0) + (112*x4)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x5), tmp23, None)
    tl.store(out_ptr1 + (x7 + (200704*x3)), tmp8, None)
    tl.store(out_ptr2 + (x7 + (225792*x3)), tmp8, None)
    tl.store(out_ptr3 + (x7 + (250880*x3)), tmp8, None)
    tl.store(out_ptr4 + (x7 + (275968*x3)), tmp8, None)
    tl.store(out_ptr5 + (x7 + (301056*x3)), tmp8, None)
    tl.store(out_ptr6 + (x7 + (326144*x3)), tmp8, None)
    tl.store(out_ptr7 + (x7 + (351232*x3)), tmp8, None)
    tl.store(out_ptr8 + (x7 + (376320*x3)), tmp8, None)
    tl.store(out_ptr9 + (x7 + (401408*x3)), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/i2/ci2vxtjcukfr7sih5kniq3esnzkyhrnvwf7j5jyqw3wwvanrvgps.py
# Source Nodes: [l__mod___features_denseblock2_denselayer1_norm2, l__mod___features_denseblock2_denselayer1_relu2, new_features_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___features_denseblock2_denselayer1_norm2 => add_31, mul_46, mul_47, sub_15
# l__mod___features_denseblock2_denselayer1_relu2 => relu_15
# new_features_12 => convolution_15
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zf/czfmu5zl3ol5ha5arkhazw4lbn5f4ubb2ed3wt5xcjr6doqvsfe3.py
# Source Nodes: [bottleneck_output_14, cat_115, l__mod___features_denseblock2_denselayer2_norm1, l__mod___features_denseblock2_denselayer2_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# bottleneck_output_14 => convolution_16
# cat_115 => cat_6
# l__mod___features_denseblock2_denselayer2_norm1 => add_33, mul_49, mul_50, sub_16
# l__mod___features_denseblock2_denselayer2_relu1 => relu_16
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 784) % 160
    x0 = xindex % 28
    x3 = (xindex // 125440)
    x4 = (xindex // 28) % 4480
    x5 = xindex % 125440
    x6 = xindex
    tmp23 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((2*x0) + (112*x4) + (401408*x3)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (1 + (2*x0) + (112*x4) + (401408*x3)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp6 + tmp5
    tmp8 = tl.load(in_ptr0 + (56 + (2*x0) + (112*x4) + (401408*x3)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp8 + tmp7
    tmp10 = tl.load(in_ptr0 + (57 + (2*x0) + (112*x4) + (401408*x3)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp10 + tmp9
    tmp12 = 0.25
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1], 160, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr1 + ((-100352) + x5 + (25088*x3)), tmp16, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp16, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp15, tmp21)
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
    tl.store(in_out_ptr0 + (x6), tmp37, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/h2/ch2v7ppvhgeg7xatmitidoaf6avxaj52ykhg2sdedlkycxqg3ttr.py
# Source Nodes: [bottleneck_output_16, cat_114, l__mod___features_denseblock2_denselayer3_norm1, l__mod___features_denseblock2_denselayer3_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# bottleneck_output_16 => convolution_18
# cat_114 => cat_7
# l__mod___features_denseblock2_denselayer3_norm1 => add_37, mul_55, mul_56, sub_18
# l__mod___features_denseblock2_denselayer3_relu1 => relu_18
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 784) % 192
    x0 = xindex % 28
    x3 = (xindex // 150528)
    x4 = (xindex // 28) % 5376
    x5 = xindex % 150528
    x6 = xindex
    tmp31 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((2*x0) + (112*x4) + (401408*x3)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (1 + (2*x0) + (112*x4) + (401408*x3)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp6 + tmp5
    tmp8 = tl.load(in_ptr0 + (56 + (2*x0) + (112*x4) + (401408*x3)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp8 + tmp7
    tmp10 = tl.load(in_ptr0 + (57 + (2*x0) + (112*x4) + (401408*x3)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp10 + tmp9
    tmp12 = 0.25
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1], 160, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr1 + ((-100352) + x5 + (25088*x3)), tmp19, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp0 >= tmp17
    tmp24 = tl.full([1], 192, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tl.load(in_ptr2 + ((-125440) + x5 + (25088*x3)), tmp23, other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = tl.where(tmp19, tmp22, tmp28)
    tmp30 = tl.where(tmp4, tmp15, tmp29)
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
    tl.store(in_out_ptr0 + (x6), tmp45, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/2f/c2fa2gocg6cnxgzqbjzszzwbm7dpnlzjmp57oapgzffinuwklfh6.py
# Source Nodes: [bottleneck_output_18, cat_113, l__mod___features_denseblock2_denselayer4_norm1, l__mod___features_denseblock2_denselayer4_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# bottleneck_output_18 => convolution_20
# cat_113 => cat_8
# l__mod___features_denseblock2_denselayer4_norm1 => add_41, mul_61, mul_62, sub_20
# l__mod___features_denseblock2_denselayer4_relu1 => relu_20
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 702464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 784) % 224
    x0 = xindex % 28
    x3 = (xindex // 175616)
    x4 = (xindex // 28) % 6272
    x5 = xindex % 175616
    x6 = xindex
    tmp39 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr5 + (x2), None, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr6 + (x2), None, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr7 + (x2), None, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((2*x0) + (112*x4) + (401408*x3)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (1 + (2*x0) + (112*x4) + (401408*x3)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp6 + tmp5
    tmp8 = tl.load(in_ptr0 + (56 + (2*x0) + (112*x4) + (401408*x3)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp8 + tmp7
    tmp10 = tl.load(in_ptr0 + (57 + (2*x0) + (112*x4) + (401408*x3)), tmp4, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp10 + tmp9
    tmp12 = 0.25
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1], 160, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr1 + ((-100352) + x5 + (25088*x3)), tmp19, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp0 >= tmp17
    tmp24 = tl.full([1], 192, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tmp23 & tmp25
    tmp27 = tl.load(in_ptr2 + ((-125440) + x5 + (25088*x3)), tmp26, other=0.0)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp26, tmp27, tmp28)
    tmp30 = tmp0 >= tmp24
    tmp31 = tl.full([1], 224, tl.int64)
    tmp32 = tmp0 < tmp31
    tmp33 = tl.load(in_ptr3 + ((-150528) + x5 + (25088*x3)), tmp30, other=0.0)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp30, tmp33, tmp34)
    tmp36 = tl.where(tmp26, tmp29, tmp35)
    tmp37 = tl.where(tmp19, tmp22, tmp36)
    tmp38 = tl.where(tmp4, tmp15, tmp37)
    tmp40 = tmp38 - tmp39
    tmp42 = 1e-05
    tmp43 = tmp41 + tmp42
    tmp44 = tl.sqrt(tmp43)
    tmp45 = 1 / tmp44
    tmp46 = 1.0
    tmp47 = tmp45 * tmp46
    tmp48 = tmp40 * tmp47
    tmp50 = tmp48 * tmp49
    tmp52 = tmp50 + tmp51
    tmp53 = triton_helpers.maximum(0, tmp52)
    tl.store(in_out_ptr0 + (x6), tmp53, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5y/c5ycwas6ibm33n35gyeeqdalhcwbsr4wxvoijt2jzqqiqh5hs6gp.py
# Source Nodes: [cat_107, cat_108, cat_109, cat_110, cat_111, cat_112], Original ATen: [aten.cat]
# cat_107 => cat_14
# cat_108 => cat_13
# cat_109 => cat_12
# cat_110 => cat_11
# cat_111 => cat_10
# cat_112 => cat_9
triton_poi_fused_cat_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_17', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/dx/cdxhqn63u4xka7emewmz5p2mey2z2gehmbyxpwsegern6iyededl.py
# Source Nodes: [bottleneck_output_20, l__mod___features_denseblock2_denselayer5_norm1, l__mod___features_denseblock2_denselayer5_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_20 => convolution_22
# l__mod___features_denseblock2_denselayer5_norm1 => add_45, mul_67, mul_68, sub_22
# l__mod___features_denseblock2_denselayer5_relu1 => relu_22
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ar/carxsijq6oeqocfospxzm6lwcuz4lbqyuypgzw2izbvt4y4r7lha.py
# Source Nodes: [cat_107, cat_108, cat_109, cat_110, cat_111], Original ATen: [aten.cat]
# cat_107 => cat_14
# cat_108 => cat_13
# cat_109 => cat_12
# cat_110 => cat_11
# cat_111 => cat_10
triton_poi_fused_cat_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_19', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ko/cko4u4wkpxhbvwruluvj3an6we4shhowu4ajtv2cgy35jk2o4pth.py
# Source Nodes: [bottleneck_output_22, l__mod___features_denseblock2_denselayer6_norm1, l__mod___features_denseblock2_denselayer6_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_22 => convolution_24
# l__mod___features_denseblock2_denselayer6_norm1 => add_49, mul_73, mul_74, sub_24
# l__mod___features_denseblock2_denselayer6_relu1 => relu_24
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 903168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 288
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/u6/cu6p4upi54gk5x5rljvi3czquvm4rqhj6zt2g7nyqkrk5esw23qa.py
# Source Nodes: [cat_106, cat_107, cat_108, cat_109, cat_110], Original ATen: [aten.cat]
# cat_106 => cat_15
# cat_107 => cat_14
# cat_108 => cat_13
# cat_109 => cat_12
# cat_110 => cat_11
triton_poi_fused_cat_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_21', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/5k/c5kvjn6bdndwnbguma4iovsav5fsol3ktlwx3n5maqcdq5mxfovs.py
# Source Nodes: [bottleneck_output_24, l__mod___features_denseblock2_denselayer7_norm1, l__mod___features_denseblock2_denselayer7_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_24 => convolution_26
# l__mod___features_denseblock2_denselayer7_norm1 => add_53, mul_79, mul_80, sub_26
# l__mod___features_denseblock2_denselayer7_relu1 => relu_26
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 320
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jr/cjrffd2em4csqnwafnp4jmklgojrqstf3ya4jflmv33jyl2hbw3h.py
# Source Nodes: [cat_105, cat_106, cat_107, cat_108, cat_109], Original ATen: [aten.cat]
# cat_105 => cat_16
# cat_106 => cat_15
# cat_107 => cat_14
# cat_108 => cat_13
# cat_109 => cat_12
triton_poi_fused_cat_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_23', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/74/c74johvymw3q7kdd65smejruop5xfccmvc5pr4miwgrrau7akelz.py
# Source Nodes: [bottleneck_output_26, l__mod___features_denseblock2_denselayer8_norm1, l__mod___features_denseblock2_denselayer8_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_26 => convolution_28
# l__mod___features_denseblock2_denselayer8_norm1 => add_57, mul_85, mul_86, sub_28
# l__mod___features_denseblock2_denselayer8_relu1 => relu_28
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1103872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 352
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/pa/cpadcf2rg3lyqr5wfaua2r27accesjj3tooovkopw7q6fazpbfkw.py
# Source Nodes: [cat_104, cat_105, cat_106, cat_107, cat_108], Original ATen: [aten.cat]
# cat_104 => cat_17
# cat_105 => cat_16
# cat_106 => cat_15
# cat_107 => cat_14
# cat_108 => cat_13
triton_poi_fused_cat_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_25', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/5d/c5dbxfwb5oxnmvw3rwpu54yltgpnekcu3d2mncxo4bgigaqxxdmw.py
# Source Nodes: [bottleneck_output_28, l__mod___features_denseblock2_denselayer9_norm1, l__mod___features_denseblock2_denselayer9_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_28 => convolution_30
# l__mod___features_denseblock2_denselayer9_norm1 => add_61, mul_91, mul_92, sub_30
# l__mod___features_denseblock2_denselayer9_relu1 => relu_30
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 384
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qo/cqo6to7ncyw6i2o2zsezkegwdaaiiqt526om3he33urak6zt4zen.py
# Source Nodes: [cat_104, cat_105, cat_106, cat_107], Original ATen: [aten.cat]
# cat_104 => cat_17
# cat_105 => cat_16
# cat_106 => cat_15
# cat_107 => cat_14
triton_poi_fused_cat_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_27', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/pt/cptji4hkd72icwf36eq3ghyd7jjt7prpgzcyyzkzqm4ctwcm2sl4.py
# Source Nodes: [bottleneck_output_30, l__mod___features_denseblock2_denselayer10_norm1, l__mod___features_denseblock2_denselayer10_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_30 => convolution_32
# l__mod___features_denseblock2_denselayer10_norm1 => add_65, mul_97, mul_98, sub_32
# l__mod___features_denseblock2_denselayer10_relu1 => relu_32
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1304576
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 416
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vf/cvfxjxquvdcz2cttuloqbw4n3bwlfcgafdzgj4sxvitzrrhfihen.py
# Source Nodes: [cat_104, cat_105, cat_106], Original ATen: [aten.cat]
# cat_104 => cat_17
# cat_105 => cat_16
# cat_106 => cat_15
triton_poi_fused_cat_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_29', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ds/cds4ybjtxsane4l5yhx4bxg6z7otpz2s6p3cv67zunodvwiyf3os.py
# Source Nodes: [bottleneck_output_32, l__mod___features_denseblock2_denselayer11_norm1, l__mod___features_denseblock2_denselayer11_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_32 => convolution_34
# l__mod___features_denseblock2_denselayer11_norm1 => add_69, mul_103, mul_104, sub_34
# l__mod___features_denseblock2_denselayer11_relu1 => relu_34
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1404928
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 448
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/33/c33d7vgwupw5yygbxuogw765oa6oidxo7dcismqmog5ryvh2j7ps.py
# Source Nodes: [cat_104, cat_105], Original ATen: [aten.cat]
# cat_104 => cat_17
# cat_105 => cat_16
triton_poi_fused_cat_31 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_31', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/qn/cqndy6e6oppq3sxuser4fhpel7od53gfhibahnpkwx4n3hufpm7t.py
# Source Nodes: [bottleneck_output_34, l__mod___features_denseblock2_denselayer12_norm1, l__mod___features_denseblock2_denselayer12_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_34 => convolution_36
# l__mod___features_denseblock2_denselayer12_norm1 => add_73, mul_109, mul_110, sub_36
# l__mod___features_denseblock2_denselayer12_relu1 => relu_36
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1505280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 480
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/es/cesadklhdtp6ln7ylvsjat7br5h2xfmhtotl3ro2cokkzk2fe3r4.py
# Source Nodes: [cat_104], Original ATen: [aten.cat]
# cat_104 => cat_17
triton_poi_fused_cat_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_33', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/3f/c3fhng35jxsg6rvqp3aqz6e76ltnsupdsev7gxvukvdzv62hqbes.py
# Source Nodes: [l__mod___features_transition2_conv, l__mod___features_transition2_norm, l__mod___features_transition2_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___features_transition2_conv => convolution_38
# l__mod___features_transition2_norm => add_77, mul_115, mul_116, sub_38
# l__mod___features_transition2_relu => relu_38
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 512
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5q/c5q6z63po6xjk4n6cvd7kyoixc3x7ttixds3tq4zoldbhbfyvjuc.py
# Source Nodes: [bottleneck_output_36, cat_79, cat_80, cat_81, cat_82, cat_83, cat_84, cat_85, cat_86, cat_87, cat_88, cat_89, cat_90, cat_91, cat_92, cat_93, cat_94, cat_95, cat_96, cat_97, cat_98, l__mod___features_denseblock3_denselayer1_norm1, l__mod___features_denseblock3_denselayer1_relu1, l__mod___features_transition2_pool], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d, aten.cat, aten.convolution, aten.relu]
# bottleneck_output_36 => convolution_39
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
# l__mod___features_denseblock3_denselayer1_norm1 => add_79, mul_118, mul_119, sub_39
# l__mod___features_denseblock3_denselayer1_relu1 => relu_39
# l__mod___features_transition2_pool => avg_pool2d_1
triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: '*fp32', 25: '*fp32', 26: '*fp32', 27: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(27,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 14
    x4 = (xindex // 14)
    x2 = (xindex // 196) % 256
    x5 = xindex
    x3 = (xindex // 50176)
    x7 = xindex % 50176
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (56*x4)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (56*x4)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (28 + (2*x0) + (56*x4)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (29 + (2*x0) + (56*x4)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x5), tmp23, None)
    tl.store(out_ptr1 + (x7 + (75264*x3)), tmp8, None)
    tl.store(out_ptr2 + (x7 + (81536*x3)), tmp8, None)
    tl.store(out_ptr3 + (x7 + (87808*x3)), tmp8, None)
    tl.store(out_ptr4 + (x7 + (94080*x3)), tmp8, None)
    tl.store(out_ptr5 + (x7 + (100352*x3)), tmp8, None)
    tl.store(out_ptr6 + (x7 + (106624*x3)), tmp8, None)
    tl.store(out_ptr7 + (x7 + (112896*x3)), tmp8, None)
    tl.store(out_ptr8 + (x7 + (119168*x3)), tmp8, None)
    tl.store(out_ptr9 + (x7 + (125440*x3)), tmp8, None)
    tl.store(out_ptr10 + (x7 + (131712*x3)), tmp8, None)
    tl.store(out_ptr11 + (x7 + (137984*x3)), tmp8, None)
    tl.store(out_ptr12 + (x7 + (144256*x3)), tmp8, None)
    tl.store(out_ptr13 + (x7 + (150528*x3)), tmp8, None)
    tl.store(out_ptr14 + (x7 + (156800*x3)), tmp8, None)
    tl.store(out_ptr15 + (x7 + (163072*x3)), tmp8, None)
    tl.store(out_ptr16 + (x7 + (169344*x3)), tmp8, None)
    tl.store(out_ptr17 + (x7 + (175616*x3)), tmp8, None)
    tl.store(out_ptr18 + (x7 + (181888*x3)), tmp8, None)
    tl.store(out_ptr19 + (x7 + (188160*x3)), tmp8, None)
    tl.store(out_ptr20 + (x7 + (194432*x3)), tmp8, None)
    tl.store(out_ptr21 + (x7 + (200704*x3)), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zx/czxfm3qk5phfvrrnj4ovyssvcq5zzvpnmau5bkekcc2yuuaxqral.py
# Source Nodes: [l__mod___features_denseblock3_denselayer1_norm2, l__mod___features_denseblock3_denselayer1_relu2, new_features_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___features_denseblock3_denselayer1_norm2 => add_81, mul_121, mul_122, sub_40
# l__mod___features_denseblock3_denselayer1_relu2 => relu_40
# new_features_36 => convolution_40
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tq/ctqs7exgiulppy764cikrusco75ohkdh55nncqgakhrgljlkxurj.py
# Source Nodes: [bottleneck_output_38, cat_102, l__mod___features_denseblock3_denselayer2_norm1, l__mod___features_denseblock3_denselayer2_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# bottleneck_output_38 => convolution_41
# cat_102 => cat_18
# l__mod___features_denseblock3_denselayer2_norm1 => add_83, mul_124, mul_125, sub_41
# l__mod___features_denseblock3_denselayer2_relu1 => relu_41
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_37', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 225792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 196) % 288
    x0 = xindex % 14
    x3 = (xindex // 56448)
    x4 = (xindex // 14) % 4032
    x5 = xindex % 56448
    x6 = xindex
    tmp23 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((2*x0) + (56*x4) + (200704*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (1 + (2*x0) + (56*x4) + (200704*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp6 + tmp5
    tmp8 = tl.load(in_ptr0 + (28 + (2*x0) + (56*x4) + (200704*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp8 + tmp7
    tmp10 = tl.load(in_ptr0 + (29 + (2*x0) + (56*x4) + (200704*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp10 + tmp9
    tmp12 = 0.25
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1], 288, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr1 + ((-50176) + x5 + (6272*x3)), tmp16 & xmask, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp16, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp15, tmp21)
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
    tl.store(in_out_ptr0 + (x6), tmp37, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ok/cok72se4ulgxdporacu3yz3p5orehtvqhn2xvkjyjjhmneqmlg6w.py
# Source Nodes: [bottleneck_output_40, cat_101, l__mod___features_denseblock3_denselayer3_norm1, l__mod___features_denseblock3_denselayer3_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# bottleneck_output_40 => convolution_43
# cat_101 => cat_19
# l__mod___features_denseblock3_denselayer3_norm1 => add_87, mul_130, mul_131, sub_43
# l__mod___features_denseblock3_denselayer3_relu1 => relu_43
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 250880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 196) % 320
    x0 = xindex % 14
    x3 = (xindex // 62720)
    x4 = (xindex // 14) % 4480
    x5 = xindex % 62720
    x6 = xindex
    tmp31 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((2*x0) + (56*x4) + (200704*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (1 + (2*x0) + (56*x4) + (200704*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp6 + tmp5
    tmp8 = tl.load(in_ptr0 + (28 + (2*x0) + (56*x4) + (200704*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp8 + tmp7
    tmp10 = tl.load(in_ptr0 + (29 + (2*x0) + (56*x4) + (200704*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp10 + tmp9
    tmp12 = 0.25
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1], 288, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr1 + ((-50176) + x5 + (6272*x3)), tmp19 & xmask, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp0 >= tmp17
    tmp24 = tl.full([1], 320, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tl.load(in_ptr2 + ((-56448) + x5 + (6272*x3)), tmp23 & xmask, other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = tl.where(tmp19, tmp22, tmp28)
    tmp30 = tl.where(tmp4, tmp15, tmp29)
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
    tl.store(in_out_ptr0 + (x6), tmp45, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qh/cqhu3m527uczolcgaffu4djh66ljuvj3cnmnfjsbopbsj7q3cwcr.py
# Source Nodes: [bottleneck_output_42, cat_100, l__mod___features_denseblock3_denselayer4_norm1, l__mod___features_denseblock3_denselayer4_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# bottleneck_output_42 => convolution_45
# cat_100 => cat_20
# l__mod___features_denseblock3_denselayer4_norm1 => add_91, mul_136, mul_137, sub_45
# l__mod___features_denseblock3_denselayer4_relu1 => relu_45
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_39 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_39', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 275968
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 196) % 352
    x0 = xindex % 14
    x3 = (xindex // 68992)
    x4 = (xindex // 14) % 4928
    x5 = xindex % 68992
    x6 = xindex
    tmp39 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((2*x0) + (56*x4) + (200704*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (1 + (2*x0) + (56*x4) + (200704*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp6 + tmp5
    tmp8 = tl.load(in_ptr0 + (28 + (2*x0) + (56*x4) + (200704*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp8 + tmp7
    tmp10 = tl.load(in_ptr0 + (29 + (2*x0) + (56*x4) + (200704*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp10 + tmp9
    tmp12 = 0.25
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1], 288, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr1 + ((-50176) + x5 + (6272*x3)), tmp19 & xmask, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp0 >= tmp17
    tmp24 = tl.full([1], 320, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tmp23 & tmp25
    tmp27 = tl.load(in_ptr2 + ((-56448) + x5 + (6272*x3)), tmp26 & xmask, other=0.0)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp26, tmp27, tmp28)
    tmp30 = tmp0 >= tmp24
    tmp31 = tl.full([1], 352, tl.int64)
    tmp32 = tmp0 < tmp31
    tmp33 = tl.load(in_ptr3 + ((-62720) + x5 + (6272*x3)), tmp30 & xmask, other=0.0)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp30, tmp33, tmp34)
    tmp36 = tl.where(tmp26, tmp29, tmp35)
    tmp37 = tl.where(tmp19, tmp22, tmp36)
    tmp38 = tl.where(tmp4, tmp15, tmp37)
    tmp40 = tmp38 - tmp39
    tmp42 = 1e-05
    tmp43 = tmp41 + tmp42
    tmp44 = tl.sqrt(tmp43)
    tmp45 = 1 / tmp44
    tmp46 = 1.0
    tmp47 = tmp45 * tmp46
    tmp48 = tmp40 * tmp47
    tmp50 = tmp48 * tmp49
    tmp52 = tmp50 + tmp51
    tmp53 = triton_helpers.maximum(0, tmp52)
    tl.store(in_out_ptr0 + (x6), tmp53, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/my/cmy6ngjpj54zqm3cs5nuzxn4frmsdaj27wwqqyqzus7vy2j3gde2.py
# Source Nodes: [cat_94, cat_95, cat_96, cat_97, cat_98, cat_99], Original ATen: [aten.cat]
# cat_94 => cat_26
# cat_95 => cat_25
# cat_96 => cat_24
# cat_97 => cat_23
# cat_98 => cat_22
# cat_99 => cat_21
triton_poi_fused_cat_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_40', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/tw/ctwylr5a6tmlt3cdclspohrn6e5cdkrlrizc5fzl4ankhqjqrvqm.py
# Source Nodes: [bottleneck_output_44, l__mod___features_denseblock3_denselayer5_norm1, l__mod___features_denseblock3_denselayer5_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_44 => convolution_47
# l__mod___features_denseblock3_denselayer5_norm1 => add_95, mul_142, mul_143, sub_47
# l__mod___features_denseblock3_denselayer5_relu1 => relu_47
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_41', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 301056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 384
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nv/cnvkbqwogyupqyfvocwmi4gecpua7vkjcvaaxnxsotzqq6fv3tld.py
# Source Nodes: [cat_94, cat_95, cat_96, cat_97, cat_98], Original ATen: [aten.cat]
# cat_94 => cat_26
# cat_95 => cat_25
# cat_96 => cat_24
# cat_97 => cat_23
# cat_98 => cat_22
triton_poi_fused_cat_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_42', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/fs/cfsqsd6e5ott2bero3annva2j5flrgyqeuj3twrah7jdhmbro3cq.py
# Source Nodes: [bottleneck_output_46, l__mod___features_denseblock3_denselayer6_norm1, l__mod___features_denseblock3_denselayer6_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_46 => convolution_49
# l__mod___features_denseblock3_denselayer6_norm1 => add_99, mul_148, mul_149, sub_49
# l__mod___features_denseblock3_denselayer6_relu1 => relu_49
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_43', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 326144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 416
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vr/cvrrkfjxwoxbfpr2ju6f62w7ekuhuy5jwip2yv3pti4xf4d3vb5m.py
# Source Nodes: [cat_93, cat_94, cat_95, cat_96, cat_97], Original ATen: [aten.cat]
# cat_93 => cat_27
# cat_94 => cat_26
# cat_95 => cat_25
# cat_96 => cat_24
# cat_97 => cat_23
triton_poi_fused_cat_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_44', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/kr/ckr6tfawerao3oq3dx6gqmt4z5kpd3ayn5nunfns64vp6d6jjok7.py
# Source Nodes: [bottleneck_output_48, l__mod___features_denseblock3_denselayer7_norm1, l__mod___features_denseblock3_denselayer7_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_48 => convolution_51
# l__mod___features_denseblock3_denselayer7_norm1 => add_103, mul_154, mul_155, sub_51
# l__mod___features_denseblock3_denselayer7_relu1 => relu_51
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_45', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 351232
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 448
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5s/c5sarvizwudwc7vhakuuld2uxkc2r3tmkqwqamnoefkyrwknvag6.py
# Source Nodes: [cat_92, cat_93, cat_94, cat_95, cat_96], Original ATen: [aten.cat]
# cat_92 => cat_28
# cat_93 => cat_27
# cat_94 => cat_26
# cat_95 => cat_25
# cat_96 => cat_24
triton_poi_fused_cat_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_46', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/of/cofsibhgwnv6bozlrirxjsxcpspo5o5yjxy2tdcglkl7hz4ackkb.py
# Source Nodes: [bottleneck_output_50, l__mod___features_denseblock3_denselayer8_norm1, l__mod___features_denseblock3_denselayer8_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_50 => convolution_53
# l__mod___features_denseblock3_denselayer8_norm1 => add_107, mul_160, mul_161, sub_53
# l__mod___features_denseblock3_denselayer8_relu1 => relu_53
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_47', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 480
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dl/cdl4bceczs7ehy3r2g37ivlz46yjzsyf4je2tq5bp2qg776ffkzz.py
# Source Nodes: [cat_91, cat_92, cat_93, cat_94, cat_95], Original ATen: [aten.cat]
# cat_91 => cat_29
# cat_92 => cat_28
# cat_93 => cat_27
# cat_94 => cat_26
# cat_95 => cat_25
triton_poi_fused_cat_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_48', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/un/cunfqx6laroiaqtzr2zzqdoms2ujdp2izwrpt5p46e52ova5epu4.py
# Source Nodes: [bottleneck_output_52, l__mod___features_denseblock3_denselayer9_norm1, l__mod___features_denseblock3_denselayer9_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_52 => convolution_55
# l__mod___features_denseblock3_denselayer9_norm1 => add_111, mul_166, mul_167, sub_55
# l__mod___features_denseblock3_denselayer9_relu1 => relu_55
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_49', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 512
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xb/cxboiixyhk6yppfrsqfqs4o4pxv7mykck7qwtj35gvegadqsyzbt.py
# Source Nodes: [cat_91, cat_92, cat_93, cat_94], Original ATen: [aten.cat]
# cat_91 => cat_29
# cat_92 => cat_28
# cat_93 => cat_27
# cat_94 => cat_26
triton_poi_fused_cat_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_50', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/43/c43q2rz742u5usiaornthhbb4ntjdbh5xnyzrhjlfqgws2bmkyb3.py
# Source Nodes: [bottleneck_output_54, l__mod___features_denseblock3_denselayer10_norm1, l__mod___features_denseblock3_denselayer10_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_54 => convolution_57
# l__mod___features_denseblock3_denselayer10_norm1 => add_115, mul_172, mul_173, sub_57
# l__mod___features_denseblock3_denselayer10_relu1 => relu_57
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_51 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_51', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 426496
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 544
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qe/cqeeizbi6q5b5xrfzwxgu22loqj4dznunxnkqb7dwha4ojsnuq7k.py
# Source Nodes: [cat_90, cat_91, cat_92, cat_93], Original ATen: [aten.cat]
# cat_90 => cat_30
# cat_91 => cat_29
# cat_92 => cat_28
# cat_93 => cat_27
triton_poi_fused_cat_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_52', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/hg/chg7hmjvwnndcst5dcxttfdohbv344hp5qp2negmuknuepo3kmx7.py
# Source Nodes: [bottleneck_output_56, l__mod___features_denseblock3_denselayer11_norm1, l__mod___features_denseblock3_denselayer11_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_56 => convolution_59
# l__mod___features_denseblock3_denselayer11_norm1 => add_119, mul_178, mul_179, sub_59
# l__mod___features_denseblock3_denselayer11_relu1 => relu_59
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_53', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 576
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/33/c33nm4qgmynau5pedpo7ngyqe5g2l7aiadl5m47vkuucefky2nla.py
# Source Nodes: [cat_89, cat_90, cat_91, cat_92], Original ATen: [aten.cat]
# cat_89 => cat_31
# cat_90 => cat_30
# cat_91 => cat_29
# cat_92 => cat_28
triton_poi_fused_cat_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_54', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/um/cumndwxdxbkm2tzpr2wq2256c2rbayekshqw6jtm2wwgdxocbn74.py
# Source Nodes: [bottleneck_output_58, l__mod___features_denseblock3_denselayer12_norm1, l__mod___features_denseblock3_denselayer12_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_58 => convolution_61
# l__mod___features_denseblock3_denselayer12_norm1 => add_123, mul_184, mul_185, sub_61
# l__mod___features_denseblock3_denselayer12_relu1 => relu_61
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_55', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 476672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 608
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hp/chpwejeoq25mjy3v2kd6nwl23rzrvdljpvhyqu27rdjprg5ejfd6.py
# Source Nodes: [cat_88, cat_89, cat_90, cat_91], Original ATen: [aten.cat]
# cat_88 => cat_32
# cat_89 => cat_31
# cat_90 => cat_30
# cat_91 => cat_29
triton_poi_fused_cat_56 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_56', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/vf/cvf7bl2eear5r2fevbr6bxmtijrkhtgvpkt62znhvxg7lm2rqqmk.py
# Source Nodes: [bottleneck_output_60, l__mod___features_denseblock3_denselayer13_norm1, l__mod___features_denseblock3_denselayer13_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_60 => convolution_63
# l__mod___features_denseblock3_denselayer13_norm1 => add_127, mul_190, mul_191, sub_63
# l__mod___features_denseblock3_denselayer13_relu1 => relu_63
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_57', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 640
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/x7/cx7mgby5hhfexq3tbxqbgvdwwbdzgvo5pkymxi76ykznhknx24o7.py
# Source Nodes: [cat_87, cat_88, cat_89, cat_90], Original ATen: [aten.cat]
# cat_87 => cat_33
# cat_88 => cat_32
# cat_89 => cat_31
# cat_90 => cat_30
triton_poi_fused_cat_58 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_58', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/jx/cjxpgj6tz77vvq4vgmti4ylnxxwmdvnqp4645sa2a4ndtls6vbdh.py
# Source Nodes: [bottleneck_output_62, l__mod___features_denseblock3_denselayer14_norm1, l__mod___features_denseblock3_denselayer14_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_62 => convolution_65
# l__mod___features_denseblock3_denselayer14_norm1 => add_131, mul_196, mul_197, sub_65
# l__mod___features_denseblock3_denselayer14_relu1 => relu_65
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_59 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_59', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 526848
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 672
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5q/c5qnd4cvhfdzefk5rxwwtfphn2odadzcbqgifxji22zifn4lo7vs.py
# Source Nodes: [cat_86, cat_87, cat_88, cat_89], Original ATen: [aten.cat]
# cat_86 => cat_34
# cat_87 => cat_33
# cat_88 => cat_32
# cat_89 => cat_31
triton_poi_fused_cat_60 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_60', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ol/colvbkuwzitxunhi44iqcmehe3njhttnemz7i2e2psguvskr4hhf.py
# Source Nodes: [bottleneck_output_64, l__mod___features_denseblock3_denselayer15_norm1, l__mod___features_denseblock3_denselayer15_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_64 => convolution_67
# l__mod___features_denseblock3_denselayer15_norm1 => add_135, mul_202, mul_203, sub_67
# l__mod___features_denseblock3_denselayer15_relu1 => relu_67
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_61 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_61', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 551936
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 704
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cq/ccqgy23pgikcz3wsof5urjn3tmvplv35tmsl5o6kpkauptqyz2le.py
# Source Nodes: [cat_86, cat_87, cat_88], Original ATen: [aten.cat]
# cat_86 => cat_34
# cat_87 => cat_33
# cat_88 => cat_32
triton_poi_fused_cat_62 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_62', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/x2/cx2xxzdgykn2vsjfuz5gwiittjmduotsrqqmu6hm2ifdgl6ywr5r.py
# Source Nodes: [bottleneck_output_66, l__mod___features_denseblock3_denselayer16_norm1, l__mod___features_denseblock3_denselayer16_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_66 => convolution_69
# l__mod___features_denseblock3_denselayer16_norm1 => add_139, mul_208, mul_209, sub_69
# l__mod___features_denseblock3_denselayer16_relu1 => relu_69
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_63 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_63', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 577024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 736
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qf/cqf7yzkfshabgid6h5xxtv4b4wpuxqc6btecmo6fsncehri4einh.py
# Source Nodes: [cat_85, cat_86, cat_87], Original ATen: [aten.cat]
# cat_85 => cat_35
# cat_86 => cat_34
# cat_87 => cat_33
triton_poi_fused_cat_64 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_64', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/e3/ce3i53axoynguj3cusecddvdabuwhddtzfn5qjrvr3zsmvkqnkpe.py
# Source Nodes: [bottleneck_output_68, l__mod___features_denseblock3_denselayer17_norm1, l__mod___features_denseblock3_denselayer17_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_68 => convolution_71
# l__mod___features_denseblock3_denselayer17_norm1 => add_143, mul_214, mul_215, sub_71
# l__mod___features_denseblock3_denselayer17_relu1 => relu_71
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_65 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_65', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 768
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xi/cxigibn6glxwm3m736atjunijumgtjqyw2zkkhqujllnuxjzallj.py
# Source Nodes: [cat_84, cat_85, cat_86], Original ATen: [aten.cat]
# cat_84 => cat_36
# cat_85 => cat_35
# cat_86 => cat_34
triton_poi_fused_cat_66 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_66', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/d3/cd3uljdueyonmtzseqw2syeu5swxxzrbxkgx5gavlpzotwok2qx6.py
# Source Nodes: [bottleneck_output_70, l__mod___features_denseblock3_denselayer18_norm1, l__mod___features_denseblock3_denselayer18_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_70 => convolution_73
# l__mod___features_denseblock3_denselayer18_norm1 => add_147, mul_220, mul_221, sub_73
# l__mod___features_denseblock3_denselayer18_relu1 => relu_73
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_67 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_67', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 627200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 800
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3f/c3fbzexjo4gt64tk3a4vfesstudnxiemyiyc4mfawni3pocc6rjv.py
# Source Nodes: [cat_83, cat_84, cat_85], Original ATen: [aten.cat]
# cat_83 => cat_37
# cat_84 => cat_36
# cat_85 => cat_35
triton_poi_fused_cat_68 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_68', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/fw/cfwncoqsj3vi7msm7vqgbh625omtwgaxxrcw5mtklzg4cu6vo46b.py
# Source Nodes: [bottleneck_output_72, l__mod___features_denseblock3_denselayer19_norm1, l__mod___features_denseblock3_denselayer19_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_72 => convolution_75
# l__mod___features_denseblock3_denselayer19_norm1 => add_151, mul_226, mul_227, sub_75
# l__mod___features_denseblock3_denselayer19_relu1 => relu_75
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_69 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_69', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 652288
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 832
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ui/cuizn6cxs4lpyzd276ae7ova47splgkz7optaf3gxqg2x4qythta.py
# Source Nodes: [cat_82, cat_83, cat_84], Original ATen: [aten.cat]
# cat_82 => cat_38
# cat_83 => cat_37
# cat_84 => cat_36
triton_poi_fused_cat_70 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_70', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/a6/ca6axsgdqfmbhj2vmubedw2gotqvagj6ltgqpsw7672swwyypoc3.py
# Source Nodes: [bottleneck_output_74, l__mod___features_denseblock3_denselayer20_norm1, l__mod___features_denseblock3_denselayer20_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_74 => convolution_77
# l__mod___features_denseblock3_denselayer20_norm1 => add_155, mul_232, mul_233, sub_77
# l__mod___features_denseblock3_denselayer20_relu1 => relu_77
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_71 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_71', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 677376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 864
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ww/cwwv2cxnftyen2pwlyay5ucgvvq5srh76lpv6kwrmnv3ezxb4mwu.py
# Source Nodes: [cat_81, cat_82, cat_83], Original ATen: [aten.cat]
# cat_81 => cat_39
# cat_82 => cat_38
# cat_83 => cat_37
triton_poi_fused_cat_72 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_72', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ks/cks62bllpym6xzoj4f4uytdkfxdlikferehiu7bymm4r2dqrjf3b.py
# Source Nodes: [bottleneck_output_76, l__mod___features_denseblock3_denselayer21_norm1, l__mod___features_denseblock3_denselayer21_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_76 => convolution_79
# l__mod___features_denseblock3_denselayer21_norm1 => add_159, mul_238, mul_239, sub_79
# l__mod___features_denseblock3_denselayer21_relu1 => relu_79
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_73 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_73', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 702464
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 896
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/27/c277ogvpb366g7vzejp2ke3awul3tw4ojc5jxyovmgus3vp4l7jm.py
# Source Nodes: [cat_80, cat_81, cat_82], Original ATen: [aten.cat]
# cat_80 => cat_40
# cat_81 => cat_39
# cat_82 => cat_38
triton_poi_fused_cat_74 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_74', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/fm/cfmkdsb6imspop3eyoiegwrmugfxshamoqbp4i7p6nac6ufdd4dd.py
# Source Nodes: [bottleneck_output_78, l__mod___features_denseblock3_denselayer22_norm1, l__mod___features_denseblock3_denselayer22_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_78 => convolution_81
# l__mod___features_denseblock3_denselayer22_norm1 => add_163, mul_244, mul_245, sub_81
# l__mod___features_denseblock3_denselayer22_relu1 => relu_81
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_75 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_75', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 727552
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 928
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ed/ced2bnkpskucwzmjx4xfmm2ci7bv7esvnaqxdqatveelaqtyx3nu.py
# Source Nodes: [cat_79, cat_80, cat_81], Original ATen: [aten.cat]
# cat_79 => cat_41
# cat_80 => cat_40
# cat_81 => cat_39
triton_poi_fused_cat_76 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_76', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/cc/cccbqqs2oiynofjy6qh5ei22e6wfakje5eb6qdjyaqouxjjt5dbl.py
# Source Nodes: [bottleneck_output_80, l__mod___features_denseblock3_denselayer23_norm1, l__mod___features_denseblock3_denselayer23_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_80 => convolution_83
# l__mod___features_denseblock3_denselayer23_norm1 => add_167, mul_250, mul_251, sub_83
# l__mod___features_denseblock3_denselayer23_relu1 => relu_83
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_77 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_77', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 960
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/au/caurigzyyn56mvwlz36bdpgsdydmur4n4hehllfkreeo65dlgadq.py
# Source Nodes: [cat_79, cat_80], Original ATen: [aten.cat]
# cat_79 => cat_41
# cat_80 => cat_40
triton_poi_fused_cat_78 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_78', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/t2/ct26z5mxulsgtjta44rvpe5d5cbdmzbit2ex4ph7isln7r4dakqx.py
# Source Nodes: [bottleneck_output_82, l__mod___features_denseblock3_denselayer24_norm1, l__mod___features_denseblock3_denselayer24_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_82 => convolution_85
# l__mod___features_denseblock3_denselayer24_norm1 => add_171, mul_256, mul_257, sub_85
# l__mod___features_denseblock3_denselayer24_relu1 => relu_85
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_79 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_79', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 777728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 992
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cv/ccv7cfv63puv7lpykrj3mdrqmoih645fxwf2fcm5wrefndz33g4m.py
# Source Nodes: [cat_79], Original ATen: [aten.cat]
# cat_79 => cat_41
triton_poi_fused_cat_80 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_80', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/3w/c3w5nhjuqqe2j5i4yhb6waeovsyzxq3qfc6pa6n3q75k2iqiu5zy.py
# Source Nodes: [l__mod___features_transition3_conv, l__mod___features_transition3_norm, l__mod___features_transition3_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___features_transition3_conv => convolution_87
# l__mod___features_transition3_norm => add_175, mul_262, mul_263, sub_87
# l__mod___features_transition3_relu => relu_87
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_81 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_81', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1024
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rn/crnidulq6qnlvdbjslpdhe6pintlk3l5z37vhlb5lvelxxo3gc6u.py
# Source Nodes: [bottleneck_output_84, cat_62, cat_63, cat_64, cat_65, cat_66, cat_67, cat_68, cat_69, cat_70, cat_71, cat_72, cat_73, l__mod___features_denseblock4_denselayer1_norm1, l__mod___features_denseblock4_denselayer1_relu1, l__mod___features_transition3_pool], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d, aten.cat, aten.convolution, aten.relu]
# bottleneck_output_84 => convolution_88
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
# l__mod___features_denseblock4_denselayer1_norm1 => add_177, mul_265, mul_266, sub_88
# l__mod___features_denseblock4_denselayer1_relu1 => relu_88
# l__mod___features_transition3_pool => avg_pool2d_2
triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_82 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(19,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_82', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 7
    x4 = (xindex // 7)
    x2 = (xindex // 49) % 512
    x5 = xindex
    x3 = (xindex // 25088)
    x7 = xindex % 25088
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (28*x4)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (28*x4)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (14 + (2*x0) + (28*x4)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (15 + (2*x0) + (28*x4)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last')
    tmp19 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last')
    tmp21 = tl.load(in_ptr4 + (x2), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x5), tmp23, None)
    tl.store(out_ptr1 + (x7 + (31360*x3)), tmp8, None)
    tl.store(out_ptr2 + (x7 + (32928*x3)), tmp8, None)
    tl.store(out_ptr3 + (x7 + (34496*x3)), tmp8, None)
    tl.store(out_ptr4 + (x7 + (36064*x3)), tmp8, None)
    tl.store(out_ptr5 + (x7 + (37632*x3)), tmp8, None)
    tl.store(out_ptr6 + (x7 + (39200*x3)), tmp8, None)
    tl.store(out_ptr7 + (x7 + (40768*x3)), tmp8, None)
    tl.store(out_ptr8 + (x7 + (42336*x3)), tmp8, None)
    tl.store(out_ptr9 + (x7 + (43904*x3)), tmp8, None)
    tl.store(out_ptr10 + (x7 + (45472*x3)), tmp8, None)
    tl.store(out_ptr11 + (x7 + (47040*x3)), tmp8, None)
    tl.store(out_ptr12 + (x7 + (48608*x3)), tmp8, None)
    tl.store(out_ptr13 + (x7 + (50176*x3)), tmp8, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zc/czcshpqkxbcmmk6srxkgtzr5ppexp2qr4kplj5ywh2242jjer5wb.py
# Source Nodes: [l__mod___features_denseblock4_denselayer1_norm2, l__mod___features_denseblock4_denselayer1_relu2, new_features_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___features_denseblock4_denselayer1_norm2 => add_179, mul_268, mul_269, sub_89
# l__mod___features_denseblock4_denselayer1_relu2 => relu_89
# new_features_84 => convolution_89
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_83 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_83', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ej/cejby5vslwlbjwkrpubypalwgtip6y2ltv7g2dj5ppcnepxpmkx4.py
# Source Nodes: [bottleneck_output_86, cat_77, l__mod___features_denseblock4_denselayer2_norm1, l__mod___features_denseblock4_denselayer2_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# bottleneck_output_86 => convolution_90
# cat_77 => cat_42
# l__mod___features_denseblock4_denselayer2_norm1 => add_181, mul_271, mul_272, sub_90
# l__mod___features_denseblock4_denselayer2_relu1 => relu_90
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_84 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_84', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 106624
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 49) % 544
    x0 = xindex % 7
    x3 = (xindex // 26656)
    x4 = (xindex // 7) % 3808
    x5 = xindex % 26656
    x6 = xindex
    tmp23 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((2*x0) + (28*x4) + (100352*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (1 + (2*x0) + (28*x4) + (100352*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp6 + tmp5
    tmp8 = tl.load(in_ptr0 + (14 + (2*x0) + (28*x4) + (100352*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp8 + tmp7
    tmp10 = tl.load(in_ptr0 + (15 + (2*x0) + (28*x4) + (100352*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp10 + tmp9
    tmp12 = 0.25
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1], 544, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr1 + ((-25088) + x5 + (1568*x3)), tmp16 & xmask, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp16, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp15, tmp21)
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
    tl.store(in_out_ptr0 + (x6), tmp37, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xl/cxlcx2yec3rai7qkqdutz7ndujmatp5abq77utjqwyssz5wqts56.py
# Source Nodes: [bottleneck_output_88, cat_76, l__mod___features_denseblock4_denselayer3_norm1, l__mod___features_denseblock4_denselayer3_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# bottleneck_output_88 => convolution_92
# cat_76 => cat_43
# l__mod___features_denseblock4_denselayer3_norm1 => add_185, mul_277, mul_278, sub_92
# l__mod___features_denseblock4_denselayer3_relu1 => relu_92
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_85 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_85', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 112896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 49) % 576
    x0 = xindex % 7
    x3 = (xindex // 28224)
    x4 = (xindex // 7) % 4032
    x5 = xindex % 28224
    x6 = xindex
    tmp31 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp43 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((2*x0) + (28*x4) + (100352*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (1 + (2*x0) + (28*x4) + (100352*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp6 + tmp5
    tmp8 = tl.load(in_ptr0 + (14 + (2*x0) + (28*x4) + (100352*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp8 + tmp7
    tmp10 = tl.load(in_ptr0 + (15 + (2*x0) + (28*x4) + (100352*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp10 + tmp9
    tmp12 = 0.25
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1], 544, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr1 + ((-25088) + x5 + (1568*x3)), tmp19 & xmask, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp0 >= tmp17
    tmp24 = tl.full([1], 576, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tl.load(in_ptr2 + ((-26656) + x5 + (1568*x3)), tmp23 & xmask, other=0.0)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp23, tmp26, tmp27)
    tmp29 = tl.where(tmp19, tmp22, tmp28)
    tmp30 = tl.where(tmp4, tmp15, tmp29)
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
    tl.store(in_out_ptr0 + (x6), tmp45, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jv/cjv37w3o2feb7ecflgmhzhexhxobmm2dqgfeapsjanwmak4wpfhe.py
# Source Nodes: [bottleneck_output_90, cat_75, l__mod___features_denseblock4_denselayer4_norm1, l__mod___features_denseblock4_denselayer4_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# bottleneck_output_90 => convolution_94
# cat_75 => cat_44
# l__mod___features_denseblock4_denselayer4_norm1 => add_189, mul_283, mul_284, sub_94
# l__mod___features_denseblock4_denselayer4_relu1 => relu_94
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_86 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_86', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 119168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 49) % 608
    x0 = xindex % 7
    x3 = (xindex // 29792)
    x4 = (xindex // 7) % 4256
    x5 = xindex % 29792
    x6 = xindex
    tmp39 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp41 = tl.load(in_ptr5 + (x2), xmask, eviction_policy='evict_last')
    tmp49 = tl.load(in_ptr6 + (x2), xmask, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr7 + (x2), xmask, eviction_policy='evict_last')
    tmp0 = x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 512, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + ((2*x0) + (28*x4) + (100352*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr0 + (1 + (2*x0) + (28*x4) + (100352*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp6 + tmp5
    tmp8 = tl.load(in_ptr0 + (14 + (2*x0) + (28*x4) + (100352*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = tmp8 + tmp7
    tmp10 = tl.load(in_ptr0 + (15 + (2*x0) + (28*x4) + (100352*x3)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp11 = tmp10 + tmp9
    tmp12 = 0.25
    tmp13 = tmp11 * tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1], 544, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tmp16 & tmp18
    tmp20 = tl.load(in_ptr1 + ((-25088) + x5 + (1568*x3)), tmp19 & xmask, other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp0 >= tmp17
    tmp24 = tl.full([1], 576, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tmp23 & tmp25
    tmp27 = tl.load(in_ptr2 + ((-26656) + x5 + (1568*x3)), tmp26 & xmask, other=0.0)
    tmp28 = tl.full(tmp27.shape, 0.0, tmp27.dtype)
    tmp29 = tl.where(tmp26, tmp27, tmp28)
    tmp30 = tmp0 >= tmp24
    tmp31 = tl.full([1], 608, tl.int64)
    tmp32 = tmp0 < tmp31
    tmp33 = tl.load(in_ptr3 + ((-28224) + x5 + (1568*x3)), tmp30 & xmask, other=0.0)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp30, tmp33, tmp34)
    tmp36 = tl.where(tmp26, tmp29, tmp35)
    tmp37 = tl.where(tmp19, tmp22, tmp36)
    tmp38 = tl.where(tmp4, tmp15, tmp37)
    tmp40 = tmp38 - tmp39
    tmp42 = 1e-05
    tmp43 = tmp41 + tmp42
    tmp44 = tl.sqrt(tmp43)
    tmp45 = 1 / tmp44
    tmp46 = 1.0
    tmp47 = tmp45 * tmp46
    tmp48 = tmp40 * tmp47
    tmp50 = tmp48 * tmp49
    tmp52 = tmp50 + tmp51
    tmp53 = triton_helpers.maximum(0, tmp52)
    tl.store(in_out_ptr0 + (x6), tmp53, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/46/c46ivol5fwvu5k5ud73tmdpetwifwy3atvznepo3eossrtu5ljbe.py
# Source Nodes: [cat_69, cat_70, cat_71, cat_72, cat_73, cat_74], Original ATen: [aten.cat]
# cat_69 => cat_50
# cat_70 => cat_49
# cat_71 => cat_48
# cat_72 => cat_47
# cat_73 => cat_46
# cat_74 => cat_45
triton_poi_fused_cat_87 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_87', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/xa/cxa37dwxkhkjnjik3tsvdopl5qs4zoohjjdoyw7o6jimuo56qtpn.py
# Source Nodes: [bottleneck_output_92, l__mod___features_denseblock4_denselayer5_norm1, l__mod___features_denseblock4_denselayer5_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_92 => convolution_96
# l__mod___features_denseblock4_denselayer5_norm1 => add_193, mul_289, mul_290, sub_96
# l__mod___features_denseblock4_denselayer5_relu1 => relu_96
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_88 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_88', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 125440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 640
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/am/camugebiqsh57pgxkzqskgbp4yay27vapxj4xdxtwee7vpf2cse6.py
# Source Nodes: [cat_69, cat_70, cat_71, cat_72, cat_73], Original ATen: [aten.cat]
# cat_69 => cat_50
# cat_70 => cat_49
# cat_71 => cat_48
# cat_72 => cat_47
# cat_73 => cat_46
triton_poi_fused_cat_89 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_89', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/px/cpx3gupayltyqvur4i4l4xmcnm2wvhr635rntlt62ie6il5t6xuf.py
# Source Nodes: [bottleneck_output_94, l__mod___features_denseblock4_denselayer6_norm1, l__mod___features_denseblock4_denselayer6_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_94 => convolution_98
# l__mod___features_denseblock4_denselayer6_norm1 => add_197, mul_295, mul_296, sub_98
# l__mod___features_denseblock4_denselayer6_relu1 => relu_98
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_90 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_90', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131712
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 672
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tz/ctzje6ygwrsyre7yjeegpj7qh26kqnrur62vbw5o6g4bpzl4bsbv.py
# Source Nodes: [cat_68, cat_69, cat_70, cat_71, cat_72], Original ATen: [aten.cat]
# cat_68 => cat_51
# cat_69 => cat_50
# cat_70 => cat_49
# cat_71 => cat_48
# cat_72 => cat_47
triton_poi_fused_cat_91 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_91', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ct/cct3ienfz5xojuqzfv7e2acvjfug4kw2tga3k2iew7rgpg73uf25.py
# Source Nodes: [bottleneck_output_96, l__mod___features_denseblock4_denselayer7_norm1, l__mod___features_denseblock4_denselayer7_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_96 => convolution_100
# l__mod___features_denseblock4_denselayer7_norm1 => add_201, mul_301, mul_302, sub_100
# l__mod___features_denseblock4_denselayer7_relu1 => relu_100
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_92 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_92', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 137984
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 704
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m6/cm6f25n3q6tpbdpck2spfbemwvqxzzgruayqbvxa3bbj4cy7qjms.py
# Source Nodes: [cat_67, cat_68, cat_69, cat_70, cat_71], Original ATen: [aten.cat]
# cat_67 => cat_52
# cat_68 => cat_51
# cat_69 => cat_50
# cat_70 => cat_49
# cat_71 => cat_48
triton_poi_fused_cat_93 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_93', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/mh/cmhv6fhiceom255crldkbrrhazlbb7ijmliyah7ofcehzc6mcafw.py
# Source Nodes: [bottleneck_output_98, l__mod___features_denseblock4_denselayer8_norm1, l__mod___features_denseblock4_denselayer8_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_98 => convolution_102
# l__mod___features_denseblock4_denselayer8_norm1 => add_205, mul_307, mul_308, sub_102
# l__mod___features_denseblock4_denselayer8_relu1 => relu_102
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_94 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_94', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 144256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 736
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ph/cphzpxcjdvasy6asp3ljidttdwsb4qkyvl7iuz4v4jnlctu3myoy.py
# Source Nodes: [cat_66, cat_67, cat_68, cat_69, cat_70], Original ATen: [aten.cat]
# cat_66 => cat_53
# cat_67 => cat_52
# cat_68 => cat_51
# cat_69 => cat_50
# cat_70 => cat_49
triton_poi_fused_cat_95 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_95', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/7c/c7ciy6xa7xqcllxonr6u7zlmkvls7jq2flklotlfqlwskafn7psu.py
# Source Nodes: [bottleneck_output_100, l__mod___features_denseblock4_denselayer9_norm1, l__mod___features_denseblock4_denselayer9_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_100 => convolution_104
# l__mod___features_denseblock4_denselayer9_norm1 => add_209, mul_313, mul_314, sub_104
# l__mod___features_denseblock4_denselayer9_relu1 => relu_104
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_96 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_96', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 768
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hi/chiu6drpc2ukxd5vyqjldmrfidkwirdvbrygkvjwze24njve544t.py
# Source Nodes: [cat_66, cat_67, cat_68, cat_69], Original ATen: [aten.cat]
# cat_66 => cat_53
# cat_67 => cat_52
# cat_68 => cat_51
# cat_69 => cat_50
triton_poi_fused_cat_97 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_97', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/jy/cjytby3g5t22qjcnljgqfsaaan7p4sh2okvoayhpmfhw6r2j63zm.py
# Source Nodes: [bottleneck_output_102, l__mod___features_denseblock4_denselayer10_norm1, l__mod___features_denseblock4_denselayer10_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_102 => convolution_106
# l__mod___features_denseblock4_denselayer10_norm1 => add_213, mul_319, mul_320, sub_106
# l__mod___features_denseblock4_denselayer10_relu1 => relu_106
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_98 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_98', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 156800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 800
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/22/c222opfloyetut3udth47mbhkucx2zztuzprdq4xl6m772xtrwky.py
# Source Nodes: [cat_65, cat_66, cat_67, cat_68], Original ATen: [aten.cat]
# cat_65 => cat_54
# cat_66 => cat_53
# cat_67 => cat_52
# cat_68 => cat_51
triton_poi_fused_cat_99 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_99', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/4s/c4sztyh6r22t5eudec2m2wd5ruhbdzx7j2nbbc77a5r3innb5iwj.py
# Source Nodes: [bottleneck_output_104, l__mod___features_denseblock4_denselayer11_norm1, l__mod___features_denseblock4_denselayer11_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_104 => convolution_108
# l__mod___features_denseblock4_denselayer11_norm1 => add_217, mul_325, mul_326, sub_108
# l__mod___features_denseblock4_denselayer11_relu1 => relu_108
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_100 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_100', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 832
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qz/cqzkn63e5mmuejwbspnog7omxwvsp4pdpqiuvkjghtbbqpmpgqxh.py
# Source Nodes: [cat_64, cat_65, cat_66, cat_67], Original ATen: [aten.cat]
# cat_64 => cat_55
# cat_65 => cat_54
# cat_66 => cat_53
# cat_67 => cat_52
triton_poi_fused_cat_101 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_101', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/vz/cvzdh74ax5uwtc4oydqopu4nx5ar5hfikqkbu4nbljzs4jma2rjq.py
# Source Nodes: [bottleneck_output_106, l__mod___features_denseblock4_denselayer12_norm1, l__mod___features_denseblock4_denselayer12_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_106 => convolution_110
# l__mod___features_denseblock4_denselayer12_norm1 => add_221, mul_331, mul_332, sub_110
# l__mod___features_denseblock4_denselayer12_relu1 => relu_110
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_102 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_102', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 169344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 864
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/py/cpy4sx77gcprnveqcc2ca5l2l3bucuelgsvclahllorw3cg2gt6v.py
# Source Nodes: [cat_63, cat_64, cat_65, cat_66], Original ATen: [aten.cat]
# cat_63 => cat_56
# cat_64 => cat_55
# cat_65 => cat_54
# cat_66 => cat_53
triton_poi_fused_cat_103 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_103', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/2g/c2gmuez6tlpvlxcvek7iqnjqbaxofstigxar5axglg5sb3jtjzu5.py
# Source Nodes: [bottleneck_output_108, l__mod___features_denseblock4_denselayer13_norm1, l__mod___features_denseblock4_denselayer13_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_108 => convolution_112
# l__mod___features_denseblock4_denselayer13_norm1 => add_225, mul_337, mul_338, sub_112
# l__mod___features_denseblock4_denselayer13_relu1 => relu_112
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_104 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_104', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 175616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 896
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ol/col36zmyn5dbgytj2fx2e645fo7ilnkps36tcm4nnoruktrmtgvj.py
# Source Nodes: [cat_62, cat_63, cat_64, cat_65], Original ATen: [aten.cat]
# cat_62 => cat_57
# cat_63 => cat_56
# cat_64 => cat_55
# cat_65 => cat_54
triton_poi_fused_cat_105 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_105', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/lt/cltfxiaj3orssn4mpj3vkopiept7jjvo7kslcl4afctxfqldo4p2.py
# Source Nodes: [bottleneck_output_110, l__mod___features_denseblock4_denselayer14_norm1, l__mod___features_denseblock4_denselayer14_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_110 => convolution_114
# l__mod___features_denseblock4_denselayer14_norm1 => add_229, mul_343, mul_344, sub_114
# l__mod___features_denseblock4_denselayer14_relu1 => relu_114
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_106 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_106', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 181888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 928
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oz/cozhhbl2gfuic65nfqo3nqvipw4qcgpk4tr7jmujbrb6wxdofnvf.py
# Source Nodes: [cat_62, cat_63, cat_64], Original ATen: [aten.cat]
# cat_62 => cat_57
# cat_63 => cat_56
# cat_64 => cat_55
triton_poi_fused_cat_107 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_107', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/qo/cqo2dpytbgr6vfjiciwipby7jo3hwrsrdh6l6jw7xok64bi7efwt.py
# Source Nodes: [bottleneck_output_112, l__mod___features_denseblock4_denselayer15_norm1, l__mod___features_denseblock4_denselayer15_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_112 => convolution_116
# l__mod___features_denseblock4_denselayer15_norm1 => add_233, mul_349, mul_350, sub_116
# l__mod___features_denseblock4_denselayer15_relu1 => relu_116
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_108 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_108', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 188160
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 960
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k7/ck75sr2amajtzcs3dxkms55imqd33nf6c7qmqexlcgpbxwwkbkm6.py
# Source Nodes: [cat_62, cat_63], Original ATen: [aten.cat]
# cat_62 => cat_57
# cat_63 => cat_56
triton_poi_fused_cat_109 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_109', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/z3/cz3gf42fdzre2qqc6pf6hu7agggmc5hfwsr2nepuo5pduchm7vec.py
# Source Nodes: [bottleneck_output_114, l__mod___features_denseblock4_denselayer16_norm1, l__mod___features_denseblock4_denselayer16_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# bottleneck_output_114 => convolution_118
# l__mod___features_denseblock4_denselayer16_norm1 => add_237, mul_355, mul_356, sub_118
# l__mod___features_denseblock4_denselayer16_relu1 => relu_118
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_110 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_110', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 194432
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 992
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xu/cxuqwynbk4y3baflom7piuyagpw6bo45nnyyfdfvzfiaafrjlyqu.py
# Source Nodes: [cat_62], Original ATen: [aten.cat]
# cat_62 => cat_57
triton_poi_fused_cat_111 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_111', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/bm/cbmsmp7leag6j4wlikt6q2bzxe6qirsogp55yi45g3ko3r3ywnyg.py
# Source Nodes: [features, out, out_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
# features => add_241, mul_361, mul_362, sub_120
# out => relu_120
# out_1 => mean
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_112 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_112', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    tmp18 = tl.where(rmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = 49.0
    tmp21 = tmp19 / tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg6_1, (128, ), (1, ))
    assert_size_stride(arg7_1, (128, ), (1, ))
    assert_size_stride(arg8_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg9_1, (96, ), (1, ))
    assert_size_stride(arg10_1, (96, ), (1, ))
    assert_size_stride(arg11_1, (128, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg12_1, (128, ), (1, ))
    assert_size_stride(arg13_1, (128, ), (1, ))
    assert_size_stride(arg14_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg15_1, (128, ), (1, ))
    assert_size_stride(arg16_1, (128, ), (1, ))
    assert_size_stride(arg17_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg18_1, (128, ), (1, ))
    assert_size_stride(arg19_1, (128, ), (1, ))
    assert_size_stride(arg20_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg21_1, (160, ), (1, ))
    assert_size_stride(arg22_1, (160, ), (1, ))
    assert_size_stride(arg23_1, (128, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg24_1, (128, ), (1, ))
    assert_size_stride(arg25_1, (128, ), (1, ))
    assert_size_stride(arg26_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg27_1, (192, ), (1, ))
    assert_size_stride(arg28_1, (192, ), (1, ))
    assert_size_stride(arg29_1, (128, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg30_1, (128, ), (1, ))
    assert_size_stride(arg31_1, (128, ), (1, ))
    assert_size_stride(arg32_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg33_1, (224, ), (1, ))
    assert_size_stride(arg34_1, (224, ), (1, ))
    assert_size_stride(arg35_1, (128, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg36_1, (128, ), (1, ))
    assert_size_stride(arg37_1, (128, ), (1, ))
    assert_size_stride(arg38_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg39_1, (256, ), (1, ))
    assert_size_stride(arg40_1, (256, ), (1, ))
    assert_size_stride(arg41_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg42_1, (128, ), (1, ))
    assert_size_stride(arg43_1, (128, ), (1, ))
    assert_size_stride(arg44_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg45_1, (128, ), (1, ))
    assert_size_stride(arg46_1, (128, ), (1, ))
    assert_size_stride(arg47_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg48_1, (160, ), (1, ))
    assert_size_stride(arg49_1, (160, ), (1, ))
    assert_size_stride(arg50_1, (128, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg51_1, (128, ), (1, ))
    assert_size_stride(arg52_1, (128, ), (1, ))
    assert_size_stride(arg53_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg54_1, (192, ), (1, ))
    assert_size_stride(arg55_1, (192, ), (1, ))
    assert_size_stride(arg56_1, (128, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg57_1, (128, ), (1, ))
    assert_size_stride(arg58_1, (128, ), (1, ))
    assert_size_stride(arg59_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg60_1, (224, ), (1, ))
    assert_size_stride(arg61_1, (224, ), (1, ))
    assert_size_stride(arg62_1, (128, 224, 1, 1), (224, 1, 1, 1))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (128, ), (1, ))
    assert_size_stride(arg65_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg66_1, (256, ), (1, ))
    assert_size_stride(arg67_1, (256, ), (1, ))
    assert_size_stride(arg68_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg69_1, (128, ), (1, ))
    assert_size_stride(arg70_1, (128, ), (1, ))
    assert_size_stride(arg71_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg72_1, (288, ), (1, ))
    assert_size_stride(arg73_1, (288, ), (1, ))
    assert_size_stride(arg74_1, (128, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg75_1, (128, ), (1, ))
    assert_size_stride(arg76_1, (128, ), (1, ))
    assert_size_stride(arg77_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg78_1, (320, ), (1, ))
    assert_size_stride(arg79_1, (320, ), (1, ))
    assert_size_stride(arg80_1, (128, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(arg81_1, (128, ), (1, ))
    assert_size_stride(arg82_1, (128, ), (1, ))
    assert_size_stride(arg83_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg84_1, (352, ), (1, ))
    assert_size_stride(arg85_1, (352, ), (1, ))
    assert_size_stride(arg86_1, (128, 352, 1, 1), (352, 1, 1, 1))
    assert_size_stride(arg87_1, (128, ), (1, ))
    assert_size_stride(arg88_1, (128, ), (1, ))
    assert_size_stride(arg89_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg90_1, (384, ), (1, ))
    assert_size_stride(arg91_1, (384, ), (1, ))
    assert_size_stride(arg92_1, (128, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg93_1, (128, ), (1, ))
    assert_size_stride(arg94_1, (128, ), (1, ))
    assert_size_stride(arg95_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg96_1, (416, ), (1, ))
    assert_size_stride(arg97_1, (416, ), (1, ))
    assert_size_stride(arg98_1, (128, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg99_1, (128, ), (1, ))
    assert_size_stride(arg100_1, (128, ), (1, ))
    assert_size_stride(arg101_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg102_1, (448, ), (1, ))
    assert_size_stride(arg103_1, (448, ), (1, ))
    assert_size_stride(arg104_1, (128, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg105_1, (128, ), (1, ))
    assert_size_stride(arg106_1, (128, ), (1, ))
    assert_size_stride(arg107_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg108_1, (480, ), (1, ))
    assert_size_stride(arg109_1, (480, ), (1, ))
    assert_size_stride(arg110_1, (128, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg111_1, (128, ), (1, ))
    assert_size_stride(arg112_1, (128, ), (1, ))
    assert_size_stride(arg113_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg114_1, (512, ), (1, ))
    assert_size_stride(arg115_1, (512, ), (1, ))
    assert_size_stride(arg116_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg117_1, (256, ), (1, ))
    assert_size_stride(arg118_1, (256, ), (1, ))
    assert_size_stride(arg119_1, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg120_1, (128, ), (1, ))
    assert_size_stride(arg121_1, (128, ), (1, ))
    assert_size_stride(arg122_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg123_1, (288, ), (1, ))
    assert_size_stride(arg124_1, (288, ), (1, ))
    assert_size_stride(arg125_1, (128, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg126_1, (128, ), (1, ))
    assert_size_stride(arg127_1, (128, ), (1, ))
    assert_size_stride(arg128_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg129_1, (320, ), (1, ))
    assert_size_stride(arg130_1, (320, ), (1, ))
    assert_size_stride(arg131_1, (128, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(arg132_1, (128, ), (1, ))
    assert_size_stride(arg133_1, (128, ), (1, ))
    assert_size_stride(arg134_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg135_1, (352, ), (1, ))
    assert_size_stride(arg136_1, (352, ), (1, ))
    assert_size_stride(arg137_1, (128, 352, 1, 1), (352, 1, 1, 1))
    assert_size_stride(arg138_1, (128, ), (1, ))
    assert_size_stride(arg139_1, (128, ), (1, ))
    assert_size_stride(arg140_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg141_1, (384, ), (1, ))
    assert_size_stride(arg142_1, (384, ), (1, ))
    assert_size_stride(arg143_1, (128, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg144_1, (128, ), (1, ))
    assert_size_stride(arg145_1, (128, ), (1, ))
    assert_size_stride(arg146_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg147_1, (416, ), (1, ))
    assert_size_stride(arg148_1, (416, ), (1, ))
    assert_size_stride(arg149_1, (128, 416, 1, 1), (416, 1, 1, 1))
    assert_size_stride(arg150_1, (128, ), (1, ))
    assert_size_stride(arg151_1, (128, ), (1, ))
    assert_size_stride(arg152_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg153_1, (448, ), (1, ))
    assert_size_stride(arg154_1, (448, ), (1, ))
    assert_size_stride(arg155_1, (128, 448, 1, 1), (448, 1, 1, 1))
    assert_size_stride(arg156_1, (128, ), (1, ))
    assert_size_stride(arg157_1, (128, ), (1, ))
    assert_size_stride(arg158_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg159_1, (480, ), (1, ))
    assert_size_stride(arg160_1, (480, ), (1, ))
    assert_size_stride(arg161_1, (128, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg162_1, (128, ), (1, ))
    assert_size_stride(arg163_1, (128, ), (1, ))
    assert_size_stride(arg164_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg165_1, (512, ), (1, ))
    assert_size_stride(arg166_1, (512, ), (1, ))
    assert_size_stride(arg167_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg168_1, (128, ), (1, ))
    assert_size_stride(arg169_1, (128, ), (1, ))
    assert_size_stride(arg170_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg171_1, (544, ), (1, ))
    assert_size_stride(arg172_1, (544, ), (1, ))
    assert_size_stride(arg173_1, (128, 544, 1, 1), (544, 1, 1, 1))
    assert_size_stride(arg174_1, (128, ), (1, ))
    assert_size_stride(arg175_1, (128, ), (1, ))
    assert_size_stride(arg176_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg177_1, (576, ), (1, ))
    assert_size_stride(arg178_1, (576, ), (1, ))
    assert_size_stride(arg179_1, (128, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(arg180_1, (128, ), (1, ))
    assert_size_stride(arg181_1, (128, ), (1, ))
    assert_size_stride(arg182_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg183_1, (608, ), (1, ))
    assert_size_stride(arg184_1, (608, ), (1, ))
    assert_size_stride(arg185_1, (128, 608, 1, 1), (608, 1, 1, 1))
    assert_size_stride(arg186_1, (128, ), (1, ))
    assert_size_stride(arg187_1, (128, ), (1, ))
    assert_size_stride(arg188_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg189_1, (640, ), (1, ))
    assert_size_stride(arg190_1, (640, ), (1, ))
    assert_size_stride(arg191_1, (128, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg192_1, (128, ), (1, ))
    assert_size_stride(arg193_1, (128, ), (1, ))
    assert_size_stride(arg194_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg195_1, (672, ), (1, ))
    assert_size_stride(arg196_1, (672, ), (1, ))
    assert_size_stride(arg197_1, (128, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg198_1, (128, ), (1, ))
    assert_size_stride(arg199_1, (128, ), (1, ))
    assert_size_stride(arg200_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg201_1, (704, ), (1, ))
    assert_size_stride(arg202_1, (704, ), (1, ))
    assert_size_stride(arg203_1, (128, 704, 1, 1), (704, 1, 1, 1))
    assert_size_stride(arg204_1, (128, ), (1, ))
    assert_size_stride(arg205_1, (128, ), (1, ))
    assert_size_stride(arg206_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg207_1, (736, ), (1, ))
    assert_size_stride(arg208_1, (736, ), (1, ))
    assert_size_stride(arg209_1, (128, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg210_1, (128, ), (1, ))
    assert_size_stride(arg211_1, (128, ), (1, ))
    assert_size_stride(arg212_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg213_1, (768, ), (1, ))
    assert_size_stride(arg214_1, (768, ), (1, ))
    assert_size_stride(arg215_1, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg216_1, (128, ), (1, ))
    assert_size_stride(arg217_1, (128, ), (1, ))
    assert_size_stride(arg218_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg219_1, (800, ), (1, ))
    assert_size_stride(arg220_1, (800, ), (1, ))
    assert_size_stride(arg221_1, (128, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg222_1, (128, ), (1, ))
    assert_size_stride(arg223_1, (128, ), (1, ))
    assert_size_stride(arg224_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg225_1, (832, ), (1, ))
    assert_size_stride(arg226_1, (832, ), (1, ))
    assert_size_stride(arg227_1, (128, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(arg228_1, (128, ), (1, ))
    assert_size_stride(arg229_1, (128, ), (1, ))
    assert_size_stride(arg230_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg231_1, (864, ), (1, ))
    assert_size_stride(arg232_1, (864, ), (1, ))
    assert_size_stride(arg233_1, (128, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg234_1, (128, ), (1, ))
    assert_size_stride(arg235_1, (128, ), (1, ))
    assert_size_stride(arg236_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg237_1, (896, ), (1, ))
    assert_size_stride(arg238_1, (896, ), (1, ))
    assert_size_stride(arg239_1, (128, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg240_1, (128, ), (1, ))
    assert_size_stride(arg241_1, (128, ), (1, ))
    assert_size_stride(arg242_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg243_1, (928, ), (1, ))
    assert_size_stride(arg244_1, (928, ), (1, ))
    assert_size_stride(arg245_1, (128, 928, 1, 1), (928, 1, 1, 1))
    assert_size_stride(arg246_1, (128, ), (1, ))
    assert_size_stride(arg247_1, (128, ), (1, ))
    assert_size_stride(arg248_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg249_1, (960, ), (1, ))
    assert_size_stride(arg250_1, (960, ), (1, ))
    assert_size_stride(arg251_1, (128, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg252_1, (128, ), (1, ))
    assert_size_stride(arg253_1, (128, ), (1, ))
    assert_size_stride(arg254_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg255_1, (992, ), (1, ))
    assert_size_stride(arg256_1, (992, ), (1, ))
    assert_size_stride(arg257_1, (128, 992, 1, 1), (992, 1, 1, 1))
    assert_size_stride(arg258_1, (128, ), (1, ))
    assert_size_stride(arg259_1, (128, ), (1, ))
    assert_size_stride(arg260_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg261_1, (1024, ), (1, ))
    assert_size_stride(arg262_1, (1024, ), (1, ))
    assert_size_stride(arg263_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg264_1, (512, ), (1, ))
    assert_size_stride(arg265_1, (512, ), (1, ))
    assert_size_stride(arg266_1, (128, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg267_1, (128, ), (1, ))
    assert_size_stride(arg268_1, (128, ), (1, ))
    assert_size_stride(arg269_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg270_1, (544, ), (1, ))
    assert_size_stride(arg271_1, (544, ), (1, ))
    assert_size_stride(arg272_1, (128, 544, 1, 1), (544, 1, 1, 1))
    assert_size_stride(arg273_1, (128, ), (1, ))
    assert_size_stride(arg274_1, (128, ), (1, ))
    assert_size_stride(arg275_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg276_1, (576, ), (1, ))
    assert_size_stride(arg277_1, (576, ), (1, ))
    assert_size_stride(arg278_1, (128, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(arg279_1, (128, ), (1, ))
    assert_size_stride(arg280_1, (128, ), (1, ))
    assert_size_stride(arg281_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg282_1, (608, ), (1, ))
    assert_size_stride(arg283_1, (608, ), (1, ))
    assert_size_stride(arg284_1, (128, 608, 1, 1), (608, 1, 1, 1))
    assert_size_stride(arg285_1, (128, ), (1, ))
    assert_size_stride(arg286_1, (128, ), (1, ))
    assert_size_stride(arg287_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg288_1, (640, ), (1, ))
    assert_size_stride(arg289_1, (640, ), (1, ))
    assert_size_stride(arg290_1, (128, 640, 1, 1), (640, 1, 1, 1))
    assert_size_stride(arg291_1, (128, ), (1, ))
    assert_size_stride(arg292_1, (128, ), (1, ))
    assert_size_stride(arg293_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg294_1, (672, ), (1, ))
    assert_size_stride(arg295_1, (672, ), (1, ))
    assert_size_stride(arg296_1, (128, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg297_1, (128, ), (1, ))
    assert_size_stride(arg298_1, (128, ), (1, ))
    assert_size_stride(arg299_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg300_1, (704, ), (1, ))
    assert_size_stride(arg301_1, (704, ), (1, ))
    assert_size_stride(arg302_1, (128, 704, 1, 1), (704, 1, 1, 1))
    assert_size_stride(arg303_1, (128, ), (1, ))
    assert_size_stride(arg304_1, (128, ), (1, ))
    assert_size_stride(arg305_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg306_1, (736, ), (1, ))
    assert_size_stride(arg307_1, (736, ), (1, ))
    assert_size_stride(arg308_1, (128, 736, 1, 1), (736, 1, 1, 1))
    assert_size_stride(arg309_1, (128, ), (1, ))
    assert_size_stride(arg310_1, (128, ), (1, ))
    assert_size_stride(arg311_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg312_1, (768, ), (1, ))
    assert_size_stride(arg313_1, (768, ), (1, ))
    assert_size_stride(arg314_1, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg315_1, (128, ), (1, ))
    assert_size_stride(arg316_1, (128, ), (1, ))
    assert_size_stride(arg317_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg318_1, (800, ), (1, ))
    assert_size_stride(arg319_1, (800, ), (1, ))
    assert_size_stride(arg320_1, (128, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg321_1, (128, ), (1, ))
    assert_size_stride(arg322_1, (128, ), (1, ))
    assert_size_stride(arg323_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg324_1, (832, ), (1, ))
    assert_size_stride(arg325_1, (832, ), (1, ))
    assert_size_stride(arg326_1, (128, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(arg327_1, (128, ), (1, ))
    assert_size_stride(arg328_1, (128, ), (1, ))
    assert_size_stride(arg329_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg330_1, (864, ), (1, ))
    assert_size_stride(arg331_1, (864, ), (1, ))
    assert_size_stride(arg332_1, (128, 864, 1, 1), (864, 1, 1, 1))
    assert_size_stride(arg333_1, (128, ), (1, ))
    assert_size_stride(arg334_1, (128, ), (1, ))
    assert_size_stride(arg335_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg336_1, (896, ), (1, ))
    assert_size_stride(arg337_1, (896, ), (1, ))
    assert_size_stride(arg338_1, (128, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg339_1, (128, ), (1, ))
    assert_size_stride(arg340_1, (128, ), (1, ))
    assert_size_stride(arg341_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg342_1, (928, ), (1, ))
    assert_size_stride(arg343_1, (928, ), (1, ))
    assert_size_stride(arg344_1, (128, 928, 1, 1), (928, 1, 1, 1))
    assert_size_stride(arg345_1, (128, ), (1, ))
    assert_size_stride(arg346_1, (128, ), (1, ))
    assert_size_stride(arg347_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg348_1, (960, ), (1, ))
    assert_size_stride(arg349_1, (960, ), (1, ))
    assert_size_stride(arg350_1, (128, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg351_1, (128, ), (1, ))
    assert_size_stride(arg352_1, (128, ), (1, ))
    assert_size_stride(arg353_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg354_1, (992, ), (1, ))
    assert_size_stride(arg355_1, (992, ), (1, ))
    assert_size_stride(arg356_1, (128, 992, 1, 1), (992, 1, 1, 1))
    assert_size_stride(arg357_1, (128, ), (1, ))
    assert_size_stride(arg358_1, (128, ), (1, ))
    assert_size_stride(arg359_1, (32, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg360_1, (1024, ), (1, ))
    assert_size_stride(arg361_1, (1024, ), (1, ))
    assert_size_stride(arg362_1, (1000, 1024), (1024, 1))
    assert_size_stride(arg363_1, (1000, ), (1, ))
    assert_size_stride(arg364_1, (64, ), (1, ))
    assert_size_stride(arg365_1, (64, ), (1, ))
    assert_size_stride(arg366_1, (), ())
    assert_size_stride(arg367_1, (64, ), (1, ))
    assert_size_stride(arg368_1, (64, ), (1, ))
    assert_size_stride(arg369_1, (), ())
    assert_size_stride(arg370_1, (128, ), (1, ))
    assert_size_stride(arg371_1, (128, ), (1, ))
    assert_size_stride(arg372_1, (), ())
    assert_size_stride(arg373_1, (96, ), (1, ))
    assert_size_stride(arg374_1, (96, ), (1, ))
    assert_size_stride(arg375_1, (), ())
    assert_size_stride(arg376_1, (128, ), (1, ))
    assert_size_stride(arg377_1, (128, ), (1, ))
    assert_size_stride(arg378_1, (), ())
    assert_size_stride(arg379_1, (128, ), (1, ))
    assert_size_stride(arg380_1, (128, ), (1, ))
    assert_size_stride(arg381_1, (), ())
    assert_size_stride(arg382_1, (128, ), (1, ))
    assert_size_stride(arg383_1, (128, ), (1, ))
    assert_size_stride(arg384_1, (), ())
    assert_size_stride(arg385_1, (160, ), (1, ))
    assert_size_stride(arg386_1, (160, ), (1, ))
    assert_size_stride(arg387_1, (), ())
    assert_size_stride(arg388_1, (128, ), (1, ))
    assert_size_stride(arg389_1, (128, ), (1, ))
    assert_size_stride(arg390_1, (), ())
    assert_size_stride(arg391_1, (192, ), (1, ))
    assert_size_stride(arg392_1, (192, ), (1, ))
    assert_size_stride(arg393_1, (), ())
    assert_size_stride(arg394_1, (128, ), (1, ))
    assert_size_stride(arg395_1, (128, ), (1, ))
    assert_size_stride(arg396_1, (), ())
    assert_size_stride(arg397_1, (224, ), (1, ))
    assert_size_stride(arg398_1, (224, ), (1, ))
    assert_size_stride(arg399_1, (), ())
    assert_size_stride(arg400_1, (128, ), (1, ))
    assert_size_stride(arg401_1, (128, ), (1, ))
    assert_size_stride(arg402_1, (), ())
    assert_size_stride(arg403_1, (256, ), (1, ))
    assert_size_stride(arg404_1, (256, ), (1, ))
    assert_size_stride(arg405_1, (), ())
    assert_size_stride(arg406_1, (128, ), (1, ))
    assert_size_stride(arg407_1, (128, ), (1, ))
    assert_size_stride(arg408_1, (), ())
    assert_size_stride(arg409_1, (128, ), (1, ))
    assert_size_stride(arg410_1, (128, ), (1, ))
    assert_size_stride(arg411_1, (), ())
    assert_size_stride(arg412_1, (160, ), (1, ))
    assert_size_stride(arg413_1, (160, ), (1, ))
    assert_size_stride(arg414_1, (), ())
    assert_size_stride(arg415_1, (128, ), (1, ))
    assert_size_stride(arg416_1, (128, ), (1, ))
    assert_size_stride(arg417_1, (), ())
    assert_size_stride(arg418_1, (192, ), (1, ))
    assert_size_stride(arg419_1, (192, ), (1, ))
    assert_size_stride(arg420_1, (), ())
    assert_size_stride(arg421_1, (128, ), (1, ))
    assert_size_stride(arg422_1, (128, ), (1, ))
    assert_size_stride(arg423_1, (), ())
    assert_size_stride(arg424_1, (224, ), (1, ))
    assert_size_stride(arg425_1, (224, ), (1, ))
    assert_size_stride(arg426_1, (), ())
    assert_size_stride(arg427_1, (128, ), (1, ))
    assert_size_stride(arg428_1, (128, ), (1, ))
    assert_size_stride(arg429_1, (), ())
    assert_size_stride(arg430_1, (256, ), (1, ))
    assert_size_stride(arg431_1, (256, ), (1, ))
    assert_size_stride(arg432_1, (), ())
    assert_size_stride(arg433_1, (128, ), (1, ))
    assert_size_stride(arg434_1, (128, ), (1, ))
    assert_size_stride(arg435_1, (), ())
    assert_size_stride(arg436_1, (288, ), (1, ))
    assert_size_stride(arg437_1, (288, ), (1, ))
    assert_size_stride(arg438_1, (), ())
    assert_size_stride(arg439_1, (128, ), (1, ))
    assert_size_stride(arg440_1, (128, ), (1, ))
    assert_size_stride(arg441_1, (), ())
    assert_size_stride(arg442_1, (320, ), (1, ))
    assert_size_stride(arg443_1, (320, ), (1, ))
    assert_size_stride(arg444_1, (), ())
    assert_size_stride(arg445_1, (128, ), (1, ))
    assert_size_stride(arg446_1, (128, ), (1, ))
    assert_size_stride(arg447_1, (), ())
    assert_size_stride(arg448_1, (352, ), (1, ))
    assert_size_stride(arg449_1, (352, ), (1, ))
    assert_size_stride(arg450_1, (), ())
    assert_size_stride(arg451_1, (128, ), (1, ))
    assert_size_stride(arg452_1, (128, ), (1, ))
    assert_size_stride(arg453_1, (), ())
    assert_size_stride(arg454_1, (384, ), (1, ))
    assert_size_stride(arg455_1, (384, ), (1, ))
    assert_size_stride(arg456_1, (), ())
    assert_size_stride(arg457_1, (128, ), (1, ))
    assert_size_stride(arg458_1, (128, ), (1, ))
    assert_size_stride(arg459_1, (), ())
    assert_size_stride(arg460_1, (416, ), (1, ))
    assert_size_stride(arg461_1, (416, ), (1, ))
    assert_size_stride(arg462_1, (), ())
    assert_size_stride(arg463_1, (128, ), (1, ))
    assert_size_stride(arg464_1, (128, ), (1, ))
    assert_size_stride(arg465_1, (), ())
    assert_size_stride(arg466_1, (448, ), (1, ))
    assert_size_stride(arg467_1, (448, ), (1, ))
    assert_size_stride(arg468_1, (), ())
    assert_size_stride(arg469_1, (128, ), (1, ))
    assert_size_stride(arg470_1, (128, ), (1, ))
    assert_size_stride(arg471_1, (), ())
    assert_size_stride(arg472_1, (480, ), (1, ))
    assert_size_stride(arg473_1, (480, ), (1, ))
    assert_size_stride(arg474_1, (), ())
    assert_size_stride(arg475_1, (128, ), (1, ))
    assert_size_stride(arg476_1, (128, ), (1, ))
    assert_size_stride(arg477_1, (), ())
    assert_size_stride(arg478_1, (512, ), (1, ))
    assert_size_stride(arg479_1, (512, ), (1, ))
    assert_size_stride(arg480_1, (), ())
    assert_size_stride(arg481_1, (256, ), (1, ))
    assert_size_stride(arg482_1, (256, ), (1, ))
    assert_size_stride(arg483_1, (), ())
    assert_size_stride(arg484_1, (128, ), (1, ))
    assert_size_stride(arg485_1, (128, ), (1, ))
    assert_size_stride(arg486_1, (), ())
    assert_size_stride(arg487_1, (288, ), (1, ))
    assert_size_stride(arg488_1, (288, ), (1, ))
    assert_size_stride(arg489_1, (), ())
    assert_size_stride(arg490_1, (128, ), (1, ))
    assert_size_stride(arg491_1, (128, ), (1, ))
    assert_size_stride(arg492_1, (), ())
    assert_size_stride(arg493_1, (320, ), (1, ))
    assert_size_stride(arg494_1, (320, ), (1, ))
    assert_size_stride(arg495_1, (), ())
    assert_size_stride(arg496_1, (128, ), (1, ))
    assert_size_stride(arg497_1, (128, ), (1, ))
    assert_size_stride(arg498_1, (), ())
    assert_size_stride(arg499_1, (352, ), (1, ))
    assert_size_stride(arg500_1, (352, ), (1, ))
    assert_size_stride(arg501_1, (), ())
    assert_size_stride(arg502_1, (128, ), (1, ))
    assert_size_stride(arg503_1, (128, ), (1, ))
    assert_size_stride(arg504_1, (), ())
    assert_size_stride(arg505_1, (384, ), (1, ))
    assert_size_stride(arg506_1, (384, ), (1, ))
    assert_size_stride(arg507_1, (), ())
    assert_size_stride(arg508_1, (128, ), (1, ))
    assert_size_stride(arg509_1, (128, ), (1, ))
    assert_size_stride(arg510_1, (), ())
    assert_size_stride(arg511_1, (416, ), (1, ))
    assert_size_stride(arg512_1, (416, ), (1, ))
    assert_size_stride(arg513_1, (), ())
    assert_size_stride(arg514_1, (128, ), (1, ))
    assert_size_stride(arg515_1, (128, ), (1, ))
    assert_size_stride(arg516_1, (), ())
    assert_size_stride(arg517_1, (448, ), (1, ))
    assert_size_stride(arg518_1, (448, ), (1, ))
    assert_size_stride(arg519_1, (), ())
    assert_size_stride(arg520_1, (128, ), (1, ))
    assert_size_stride(arg521_1, (128, ), (1, ))
    assert_size_stride(arg522_1, (), ())
    assert_size_stride(arg523_1, (480, ), (1, ))
    assert_size_stride(arg524_1, (480, ), (1, ))
    assert_size_stride(arg525_1, (), ())
    assert_size_stride(arg526_1, (128, ), (1, ))
    assert_size_stride(arg527_1, (128, ), (1, ))
    assert_size_stride(arg528_1, (), ())
    assert_size_stride(arg529_1, (512, ), (1, ))
    assert_size_stride(arg530_1, (512, ), (1, ))
    assert_size_stride(arg531_1, (), ())
    assert_size_stride(arg532_1, (128, ), (1, ))
    assert_size_stride(arg533_1, (128, ), (1, ))
    assert_size_stride(arg534_1, (), ())
    assert_size_stride(arg535_1, (544, ), (1, ))
    assert_size_stride(arg536_1, (544, ), (1, ))
    assert_size_stride(arg537_1, (), ())
    assert_size_stride(arg538_1, (128, ), (1, ))
    assert_size_stride(arg539_1, (128, ), (1, ))
    assert_size_stride(arg540_1, (), ())
    assert_size_stride(arg541_1, (576, ), (1, ))
    assert_size_stride(arg542_1, (576, ), (1, ))
    assert_size_stride(arg543_1, (), ())
    assert_size_stride(arg544_1, (128, ), (1, ))
    assert_size_stride(arg545_1, (128, ), (1, ))
    assert_size_stride(arg546_1, (), ())
    assert_size_stride(arg547_1, (608, ), (1, ))
    assert_size_stride(arg548_1, (608, ), (1, ))
    assert_size_stride(arg549_1, (), ())
    assert_size_stride(arg550_1, (128, ), (1, ))
    assert_size_stride(arg551_1, (128, ), (1, ))
    assert_size_stride(arg552_1, (), ())
    assert_size_stride(arg553_1, (640, ), (1, ))
    assert_size_stride(arg554_1, (640, ), (1, ))
    assert_size_stride(arg555_1, (), ())
    assert_size_stride(arg556_1, (128, ), (1, ))
    assert_size_stride(arg557_1, (128, ), (1, ))
    assert_size_stride(arg558_1, (), ())
    assert_size_stride(arg559_1, (672, ), (1, ))
    assert_size_stride(arg560_1, (672, ), (1, ))
    assert_size_stride(arg561_1, (), ())
    assert_size_stride(arg562_1, (128, ), (1, ))
    assert_size_stride(arg563_1, (128, ), (1, ))
    assert_size_stride(arg564_1, (), ())
    assert_size_stride(arg565_1, (704, ), (1, ))
    assert_size_stride(arg566_1, (704, ), (1, ))
    assert_size_stride(arg567_1, (), ())
    assert_size_stride(arg568_1, (128, ), (1, ))
    assert_size_stride(arg569_1, (128, ), (1, ))
    assert_size_stride(arg570_1, (), ())
    assert_size_stride(arg571_1, (736, ), (1, ))
    assert_size_stride(arg572_1, (736, ), (1, ))
    assert_size_stride(arg573_1, (), ())
    assert_size_stride(arg574_1, (128, ), (1, ))
    assert_size_stride(arg575_1, (128, ), (1, ))
    assert_size_stride(arg576_1, (), ())
    assert_size_stride(arg577_1, (768, ), (1, ))
    assert_size_stride(arg578_1, (768, ), (1, ))
    assert_size_stride(arg579_1, (), ())
    assert_size_stride(arg580_1, (128, ), (1, ))
    assert_size_stride(arg581_1, (128, ), (1, ))
    assert_size_stride(arg582_1, (), ())
    assert_size_stride(arg583_1, (800, ), (1, ))
    assert_size_stride(arg584_1, (800, ), (1, ))
    assert_size_stride(arg585_1, (), ())
    assert_size_stride(arg586_1, (128, ), (1, ))
    assert_size_stride(arg587_1, (128, ), (1, ))
    assert_size_stride(arg588_1, (), ())
    assert_size_stride(arg589_1, (832, ), (1, ))
    assert_size_stride(arg590_1, (832, ), (1, ))
    assert_size_stride(arg591_1, (), ())
    assert_size_stride(arg592_1, (128, ), (1, ))
    assert_size_stride(arg593_1, (128, ), (1, ))
    assert_size_stride(arg594_1, (), ())
    assert_size_stride(arg595_1, (864, ), (1, ))
    assert_size_stride(arg596_1, (864, ), (1, ))
    assert_size_stride(arg597_1, (), ())
    assert_size_stride(arg598_1, (128, ), (1, ))
    assert_size_stride(arg599_1, (128, ), (1, ))
    assert_size_stride(arg600_1, (), ())
    assert_size_stride(arg601_1, (896, ), (1, ))
    assert_size_stride(arg602_1, (896, ), (1, ))
    assert_size_stride(arg603_1, (), ())
    assert_size_stride(arg604_1, (128, ), (1, ))
    assert_size_stride(arg605_1, (128, ), (1, ))
    assert_size_stride(arg606_1, (), ())
    assert_size_stride(arg607_1, (928, ), (1, ))
    assert_size_stride(arg608_1, (928, ), (1, ))
    assert_size_stride(arg609_1, (), ())
    assert_size_stride(arg610_1, (128, ), (1, ))
    assert_size_stride(arg611_1, (128, ), (1, ))
    assert_size_stride(arg612_1, (), ())
    assert_size_stride(arg613_1, (960, ), (1, ))
    assert_size_stride(arg614_1, (960, ), (1, ))
    assert_size_stride(arg615_1, (), ())
    assert_size_stride(arg616_1, (128, ), (1, ))
    assert_size_stride(arg617_1, (128, ), (1, ))
    assert_size_stride(arg618_1, (), ())
    assert_size_stride(arg619_1, (992, ), (1, ))
    assert_size_stride(arg620_1, (992, ), (1, ))
    assert_size_stride(arg621_1, (), ())
    assert_size_stride(arg622_1, (128, ), (1, ))
    assert_size_stride(arg623_1, (128, ), (1, ))
    assert_size_stride(arg624_1, (), ())
    assert_size_stride(arg625_1, (1024, ), (1, ))
    assert_size_stride(arg626_1, (1024, ), (1, ))
    assert_size_stride(arg627_1, (), ())
    assert_size_stride(arg628_1, (512, ), (1, ))
    assert_size_stride(arg629_1, (512, ), (1, ))
    assert_size_stride(arg630_1, (), ())
    assert_size_stride(arg631_1, (128, ), (1, ))
    assert_size_stride(arg632_1, (128, ), (1, ))
    assert_size_stride(arg633_1, (), ())
    assert_size_stride(arg634_1, (544, ), (1, ))
    assert_size_stride(arg635_1, (544, ), (1, ))
    assert_size_stride(arg636_1, (), ())
    assert_size_stride(arg637_1, (128, ), (1, ))
    assert_size_stride(arg638_1, (128, ), (1, ))
    assert_size_stride(arg639_1, (), ())
    assert_size_stride(arg640_1, (576, ), (1, ))
    assert_size_stride(arg641_1, (576, ), (1, ))
    assert_size_stride(arg642_1, (), ())
    assert_size_stride(arg643_1, (128, ), (1, ))
    assert_size_stride(arg644_1, (128, ), (1, ))
    assert_size_stride(arg645_1, (), ())
    assert_size_stride(arg646_1, (608, ), (1, ))
    assert_size_stride(arg647_1, (608, ), (1, ))
    assert_size_stride(arg648_1, (), ())
    assert_size_stride(arg649_1, (128, ), (1, ))
    assert_size_stride(arg650_1, (128, ), (1, ))
    assert_size_stride(arg651_1, (), ())
    assert_size_stride(arg652_1, (640, ), (1, ))
    assert_size_stride(arg653_1, (640, ), (1, ))
    assert_size_stride(arg654_1, (), ())
    assert_size_stride(arg655_1, (128, ), (1, ))
    assert_size_stride(arg656_1, (128, ), (1, ))
    assert_size_stride(arg657_1, (), ())
    assert_size_stride(arg658_1, (672, ), (1, ))
    assert_size_stride(arg659_1, (672, ), (1, ))
    assert_size_stride(arg660_1, (), ())
    assert_size_stride(arg661_1, (128, ), (1, ))
    assert_size_stride(arg662_1, (128, ), (1, ))
    assert_size_stride(arg663_1, (), ())
    assert_size_stride(arg664_1, (704, ), (1, ))
    assert_size_stride(arg665_1, (704, ), (1, ))
    assert_size_stride(arg666_1, (), ())
    assert_size_stride(arg667_1, (128, ), (1, ))
    assert_size_stride(arg668_1, (128, ), (1, ))
    assert_size_stride(arg669_1, (), ())
    assert_size_stride(arg670_1, (736, ), (1, ))
    assert_size_stride(arg671_1, (736, ), (1, ))
    assert_size_stride(arg672_1, (), ())
    assert_size_stride(arg673_1, (128, ), (1, ))
    assert_size_stride(arg674_1, (128, ), (1, ))
    assert_size_stride(arg675_1, (), ())
    assert_size_stride(arg676_1, (768, ), (1, ))
    assert_size_stride(arg677_1, (768, ), (1, ))
    assert_size_stride(arg678_1, (), ())
    assert_size_stride(arg679_1, (128, ), (1, ))
    assert_size_stride(arg680_1, (128, ), (1, ))
    assert_size_stride(arg681_1, (), ())
    assert_size_stride(arg682_1, (800, ), (1, ))
    assert_size_stride(arg683_1, (800, ), (1, ))
    assert_size_stride(arg684_1, (), ())
    assert_size_stride(arg685_1, (128, ), (1, ))
    assert_size_stride(arg686_1, (128, ), (1, ))
    assert_size_stride(arg687_1, (), ())
    assert_size_stride(arg688_1, (832, ), (1, ))
    assert_size_stride(arg689_1, (832, ), (1, ))
    assert_size_stride(arg690_1, (), ())
    assert_size_stride(arg691_1, (128, ), (1, ))
    assert_size_stride(arg692_1, (128, ), (1, ))
    assert_size_stride(arg693_1, (), ())
    assert_size_stride(arg694_1, (864, ), (1, ))
    assert_size_stride(arg695_1, (864, ), (1, ))
    assert_size_stride(arg696_1, (), ())
    assert_size_stride(arg697_1, (128, ), (1, ))
    assert_size_stride(arg698_1, (128, ), (1, ))
    assert_size_stride(arg699_1, (), ())
    assert_size_stride(arg700_1, (896, ), (1, ))
    assert_size_stride(arg701_1, (896, ), (1, ))
    assert_size_stride(arg702_1, (), ())
    assert_size_stride(arg703_1, (128, ), (1, ))
    assert_size_stride(arg704_1, (128, ), (1, ))
    assert_size_stride(arg705_1, (), ())
    assert_size_stride(arg706_1, (928, ), (1, ))
    assert_size_stride(arg707_1, (928, ), (1, ))
    assert_size_stride(arg708_1, (), ())
    assert_size_stride(arg709_1, (128, ), (1, ))
    assert_size_stride(arg710_1, (128, ), (1, ))
    assert_size_stride(arg711_1, (), ())
    assert_size_stride(arg712_1, (960, ), (1, ))
    assert_size_stride(arg713_1, (960, ), (1, ))
    assert_size_stride(arg714_1, (), ())
    assert_size_stride(arg715_1, (128, ), (1, ))
    assert_size_stride(arg716_1, (128, ), (1, ))
    assert_size_stride(arg717_1, (), ())
    assert_size_stride(arg718_1, (992, ), (1, ))
    assert_size_stride(arg719_1, (992, ), (1, ))
    assert_size_stride(arg720_1, (), ())
    assert_size_stride(arg721_1, (128, ), (1, ))
    assert_size_stride(arg722_1, (128, ), (1, ))
    assert_size_stride(arg723_1, (), ())
    assert_size_stride(arg724_1, (1024, ), (1, ))
    assert_size_stride(arg725_1, (1024, ), (1, ))
    assert_size_stride(arg726_1, (), ())
    assert_size_stride(arg727_1, (4, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [l__mod___features_conv0], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg727_1, arg0_1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 112, 112), (802816, 12544, 112, 1))
        del arg0_1
        del arg727_1
        buf1 = buf0; del buf0  # reuse
        # Source Nodes: [l__mod___features_norm0, l__mod___features_relu0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf1, arg364_1, arg365_1, arg1_1, arg2_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg1_1
        del arg2_1
        del arg364_1
        del arg365_1
        buf23 = empty((4, 192, 56, 56), device='cuda', dtype=torch.float32)
        buf2 = reinterpret_tensor(buf23, (4, 64, 56, 56), (602112, 3136, 56, 1), 0)  # alias
        buf3 = empty((4, 64, 56, 56), device='cuda', dtype=torch.float32)
        buf34 = empty((4, 224, 56, 56), device='cuda', dtype=torch.float32)
        buf28 = reinterpret_tensor(buf34, (4, 64, 56, 56), (702464, 3136, 56, 1), 0)  # alias
        buf46 = empty((4, 256, 56, 56), device='cuda', dtype=torch.float32)
        buf39 = reinterpret_tensor(buf46, (4, 64, 56, 56), (802816, 3136, 56, 1), 0)  # alias
        # Source Nodes: [bottleneck_output, cat_117, cat_118, l__mod___features_denseblock1_denselayer1_norm1, l__mod___features_denseblock1_denselayer1_relu1, l__mod___features_norm0, l__mod___features_pool0, l__mod___features_relu0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_max_pool2d_with_indices_relu_1.run(buf1, arg367_1, arg368_1, arg3_1, arg4_1, buf2, buf3, buf28, buf39, 802816, grid=grid(802816), stream=stream0)
        del arg367_1
        del arg368_1
        del arg3_1
        del arg4_1
        del buf1
        # Source Nodes: [bottleneck_output, l__mod___features_denseblock1_denselayer1_norm1, l__mod___features_denseblock1_denselayer1_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf4 = extern_kernels.convolution(buf3, arg5_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 128, 56, 56), (401408, 3136, 56, 1))
        del arg5_1
        buf5 = buf4; del buf4  # reuse
        # Source Nodes: [l__mod___features_denseblock1_denselayer1_norm2, l__mod___features_denseblock1_denselayer1_relu2, new_features], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf5, arg370_1, arg371_1, arg6_1, arg7_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg370_1
        del arg371_1
        del arg6_1
        del arg7_1
        # Source Nodes: [l__mod___features_denseblock1_denselayer1_norm2, l__mod___features_denseblock1_denselayer1_relu2, new_features], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf6 = extern_kernels.convolution(buf5, arg8_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 32, 56, 56), (100352, 3136, 56, 1))
        del arg8_1
        del buf5
        buf7 = empty((4, 96, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [bottleneck_output_2, cat_122, l__mod___features_denseblock1_denselayer2_norm1, l__mod___features_denseblock1_denselayer2_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_3.run(buf2, buf6, arg373_1, arg374_1, arg9_1, arg10_1, buf7, 1204224, grid=grid(1204224), stream=stream0)
        del arg10_1
        del arg373_1
        del arg374_1
        del arg9_1
        # Source Nodes: [bottleneck_output_2, cat_122, l__mod___features_denseblock1_denselayer2_norm1, l__mod___features_denseblock1_denselayer2_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        buf8 = extern_kernels.convolution(buf7, arg11_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 128, 56, 56), (401408, 3136, 56, 1))
        del arg11_1
        buf9 = buf8; del buf8  # reuse
        # Source Nodes: [l__mod___features_denseblock1_denselayer2_norm2, l__mod___features_denseblock1_denselayer2_relu2, new_features_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf9, arg376_1, arg377_1, arg12_1, arg13_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg12_1
        del arg13_1
        del arg376_1
        del arg377_1
        # Source Nodes: [l__mod___features_denseblock1_denselayer2_norm2, l__mod___features_denseblock1_denselayer2_relu2, new_features_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf10 = extern_kernels.convolution(buf9, arg14_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 32, 56, 56), (100352, 3136, 56, 1))
        del arg14_1
        buf11 = buf9; del buf9  # reuse
        # Source Nodes: [bottleneck_output_4, cat_121, l__mod___features_denseblock1_denselayer3_norm1, l__mod___features_denseblock1_denselayer3_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_4.run(buf2, buf6, buf10, arg379_1, arg380_1, arg15_1, arg16_1, buf11, 1605632, grid=grid(1605632), stream=stream0)
        del arg15_1
        del arg16_1
        del arg379_1
        del arg380_1
        # Source Nodes: [bottleneck_output_4, cat_121, l__mod___features_denseblock1_denselayer3_norm1, l__mod___features_denseblock1_denselayer3_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        buf12 = extern_kernels.convolution(buf11, arg17_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 128, 56, 56), (401408, 3136, 56, 1))
        del arg17_1
        del buf11
        buf13 = buf12; del buf12  # reuse
        # Source Nodes: [l__mod___features_denseblock1_denselayer3_norm2, l__mod___features_denseblock1_denselayer3_relu2, new_features_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf13, arg382_1, arg383_1, arg18_1, arg19_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg18_1
        del arg19_1
        del arg382_1
        del arg383_1
        # Source Nodes: [l__mod___features_denseblock1_denselayer3_norm2, l__mod___features_denseblock1_denselayer3_relu2, new_features_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf14 = extern_kernels.convolution(buf13, arg20_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 32, 56, 56), (100352, 3136, 56, 1))
        del arg20_1
        del buf13
        buf15 = empty((4, 160, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [bottleneck_output_6, cat_120, l__mod___features_denseblock1_denselayer4_norm1, l__mod___features_denseblock1_denselayer4_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_5.run(buf2, buf6, buf10, buf14, arg385_1, arg386_1, arg21_1, arg22_1, buf15, 2007040, grid=grid(2007040), stream=stream0)
        del arg21_1
        del arg22_1
        del arg385_1
        del arg386_1
        # Source Nodes: [bottleneck_output_6, cat_120, l__mod___features_denseblock1_denselayer4_norm1, l__mod___features_denseblock1_denselayer4_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        buf16 = extern_kernels.convolution(buf15, arg23_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 128, 56, 56), (401408, 3136, 56, 1))
        del arg23_1
        del buf15
        buf17 = buf16; del buf16  # reuse
        # Source Nodes: [l__mod___features_denseblock1_denselayer4_norm2, l__mod___features_denseblock1_denselayer4_relu2, new_features_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf17, arg388_1, arg389_1, arg24_1, arg25_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg24_1
        del arg25_1
        del arg388_1
        del arg389_1
        # Source Nodes: [l__mod___features_denseblock1_denselayer4_norm2, l__mod___features_denseblock1_denselayer4_relu2, new_features_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf18 = extern_kernels.convolution(buf17, arg26_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 32, 56, 56), (100352, 3136, 56, 1))
        del arg26_1
        del buf17
        buf19 = reinterpret_tensor(buf23, (4, 32, 56, 56), (602112, 3136, 56, 1), 200704)  # alias
        buf29 = reinterpret_tensor(buf34, (4, 32, 56, 56), (702464, 3136, 56, 1), 200704)  # alias
        buf40 = reinterpret_tensor(buf46, (4, 32, 56, 56), (802816, 3136, 56, 1), 200704)  # alias
        # Source Nodes: [cat_117, cat_118, cat_119], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf6, buf19, buf29, buf40, 401408, grid=grid(401408), stream=stream0)
        del buf6
        buf20 = reinterpret_tensor(buf23, (4, 32, 56, 56), (602112, 3136, 56, 1), 301056)  # alias
        buf30 = reinterpret_tensor(buf34, (4, 32, 56, 56), (702464, 3136, 56, 1), 301056)  # alias
        buf41 = reinterpret_tensor(buf46, (4, 32, 56, 56), (802816, 3136, 56, 1), 301056)  # alias
        # Source Nodes: [cat_117, cat_118, cat_119], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf10, buf20, buf30, buf41, 401408, grid=grid(401408), stream=stream0)
        del buf10
        buf21 = reinterpret_tensor(buf23, (4, 32, 56, 56), (602112, 3136, 56, 1), 401408)  # alias
        buf31 = reinterpret_tensor(buf34, (4, 32, 56, 56), (702464, 3136, 56, 1), 401408)  # alias
        buf42 = reinterpret_tensor(buf46, (4, 32, 56, 56), (802816, 3136, 56, 1), 401408)  # alias
        # Source Nodes: [cat_117, cat_118, cat_119], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf14, buf21, buf31, buf42, 401408, grid=grid(401408), stream=stream0)
        del buf14
        buf22 = reinterpret_tensor(buf23, (4, 32, 56, 56), (602112, 3136, 56, 1), 501760)  # alias
        buf32 = reinterpret_tensor(buf34, (4, 32, 56, 56), (702464, 3136, 56, 1), 501760)  # alias
        buf43 = reinterpret_tensor(buf46, (4, 32, 56, 56), (802816, 3136, 56, 1), 501760)  # alias
        # Source Nodes: [cat_117, cat_118, cat_119], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf18, buf22, buf32, buf43, 401408, grid=grid(401408), stream=stream0)
        del buf18
        buf24 = buf23; del buf23  # reuse
        # Source Nodes: [bottleneck_output_8, l__mod___features_denseblock1_denselayer5_norm1, l__mod___features_denseblock1_denselayer5_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_7.run(buf24, arg391_1, arg392_1, arg27_1, arg28_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg27_1
        del arg28_1
        del arg391_1
        del arg392_1
        del buf19
        del buf2
        del buf20
        del buf21
        del buf22
        # Source Nodes: [bottleneck_output_8, l__mod___features_denseblock1_denselayer5_norm1, l__mod___features_denseblock1_denselayer5_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf25 = extern_kernels.convolution(buf24, arg29_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 128, 56, 56), (401408, 3136, 56, 1))
        del arg29_1
        del buf24
        buf26 = buf25; del buf25  # reuse
        # Source Nodes: [l__mod___features_denseblock1_denselayer5_norm2, l__mod___features_denseblock1_denselayer5_relu2, new_features_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf26, arg394_1, arg395_1, arg30_1, arg31_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg30_1
        del arg31_1
        del arg394_1
        del arg395_1
        # Source Nodes: [l__mod___features_denseblock1_denselayer5_norm2, l__mod___features_denseblock1_denselayer5_relu2, new_features_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf27 = extern_kernels.convolution(buf26, arg32_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 32, 56, 56), (100352, 3136, 56, 1))
        del arg32_1
        del buf26
        buf33 = reinterpret_tensor(buf34, (4, 32, 56, 56), (702464, 3136, 56, 1), 602112)  # alias
        buf44 = reinterpret_tensor(buf46, (4, 32, 56, 56), (802816, 3136, 56, 1), 602112)  # alias
        # Source Nodes: [cat_117, cat_118], Original ATen: [aten.cat]
        triton_poi_fused_cat_8.run(buf27, buf33, buf44, 401408, grid=grid(401408), stream=stream0)
        del buf27
        buf35 = buf34; del buf34  # reuse
        # Source Nodes: [bottleneck_output_10, l__mod___features_denseblock1_denselayer6_norm1, l__mod___features_denseblock1_denselayer6_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf35, arg397_1, arg398_1, arg33_1, arg34_1, 2809856, grid=grid(2809856), stream=stream0)
        del arg33_1
        del arg34_1
        del arg397_1
        del arg398_1
        del buf28
        del buf29
        del buf30
        del buf31
        del buf32
        del buf33
        # Source Nodes: [bottleneck_output_10, l__mod___features_denseblock1_denselayer6_norm1, l__mod___features_denseblock1_denselayer6_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf36 = extern_kernels.convolution(buf35, arg35_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 128, 56, 56), (401408, 3136, 56, 1))
        del arg35_1
        del buf35
        buf37 = buf36; del buf36  # reuse
        # Source Nodes: [l__mod___features_denseblock1_denselayer6_norm2, l__mod___features_denseblock1_denselayer6_relu2, new_features_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf37, arg400_1, arg401_1, arg36_1, arg37_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg36_1
        del arg37_1
        del arg400_1
        del arg401_1
        # Source Nodes: [l__mod___features_denseblock1_denselayer6_norm2, l__mod___features_denseblock1_denselayer6_relu2, new_features_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf38 = extern_kernels.convolution(buf37, arg38_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 32, 56, 56), (100352, 3136, 56, 1))
        del arg38_1
        buf45 = reinterpret_tensor(buf46, (4, 32, 56, 56), (802816, 3136, 56, 1), 702464)  # alias
        # Source Nodes: [cat_117], Original ATen: [aten.cat]
        triton_poi_fused_cat_10.run(buf38, buf45, 401408, grid=grid(401408), stream=stream0)
        buf47 = buf46; del buf46  # reuse
        # Source Nodes: [l__mod___features_transition1_conv, l__mod___features_transition1_norm, l__mod___features_transition1_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_11.run(buf47, arg403_1, arg404_1, arg39_1, arg40_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg39_1
        del arg403_1
        del arg404_1
        del arg40_1
        del buf39
        del buf40
        del buf41
        del buf42
        del buf43
        del buf44
        del buf45
        # Source Nodes: [l__mod___features_transition1_conv, l__mod___features_transition1_norm, l__mod___features_transition1_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf48 = extern_kernels.convolution(buf47, arg41_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 128, 56, 56), (401408, 3136, 56, 1))
        del arg41_1
        del buf47
        buf49 = reinterpret_tensor(buf38, (4, 128, 28, 28), (100352, 784, 28, 1), 0); del buf38  # reuse
        buf73 = reinterpret_tensor(buf3, (4, 256, 28, 28), (200704, 784, 28, 1), 0); del buf3  # reuse
        buf68 = reinterpret_tensor(buf73, (4, 128, 28, 28), (200704, 784, 28, 1), 0)  # alias
        buf84 = empty((4, 288, 28, 28), device='cuda', dtype=torch.float32)
        buf78 = reinterpret_tensor(buf84, (4, 128, 28, 28), (225792, 784, 28, 1), 0)  # alias
        buf96 = empty((4, 320, 28, 28), device='cuda', dtype=torch.float32)
        buf89 = reinterpret_tensor(buf96, (4, 128, 28, 28), (250880, 784, 28, 1), 0)  # alias
        buf109 = empty((4, 352, 28, 28), device='cuda', dtype=torch.float32)
        buf101 = reinterpret_tensor(buf109, (4, 128, 28, 28), (275968, 784, 28, 1), 0)  # alias
        buf123 = reinterpret_tensor(buf7, (4, 384, 28, 28), (301056, 784, 28, 1), 0); del buf7  # reuse
        buf114 = reinterpret_tensor(buf123, (4, 128, 28, 28), (301056, 784, 28, 1), 0)  # alias
        buf138 = empty((4, 416, 28, 28), device='cuda', dtype=torch.float32)
        buf128 = reinterpret_tensor(buf138, (4, 128, 28, 28), (326144, 784, 28, 1), 0)  # alias
        buf154 = empty((4, 448, 28, 28), device='cuda', dtype=torch.float32)
        buf143 = reinterpret_tensor(buf154, (4, 128, 28, 28), (351232, 784, 28, 1), 0)  # alias
        buf171 = empty((4, 480, 28, 28), device='cuda', dtype=torch.float32)
        buf159 = reinterpret_tensor(buf171, (4, 128, 28, 28), (376320, 784, 28, 1), 0)  # alias
        buf189 = reinterpret_tensor(buf37, (4, 512, 28, 28), (401408, 784, 28, 1), 0); del buf37  # reuse
        buf176 = reinterpret_tensor(buf189, (4, 128, 28, 28), (401408, 784, 28, 1), 0)  # alias
        # Source Nodes: [bottleneck_output_12, cat_104, cat_105, cat_106, cat_107, cat_108, cat_109, cat_110, cat_111, l__mod___features_denseblock2_denselayer1_norm1, l__mod___features_denseblock2_denselayer1_relu1, l__mod___features_transition1_pool], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_12.run(buf48, arg406_1, arg407_1, arg42_1, arg43_1, buf49, buf68, buf78, buf89, buf101, buf114, buf128, buf143, buf159, buf176, 401408, grid=grid(401408), stream=stream0)
        del arg406_1
        del arg407_1
        del arg42_1
        del arg43_1
        # Source Nodes: [bottleneck_output_12, l__mod___features_denseblock2_denselayer1_norm1, l__mod___features_denseblock2_denselayer1_relu1, l__mod___features_transition1_pool], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d, aten.convolution, aten.relu]
        buf50 = extern_kernels.convolution(buf49, arg44_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 128, 28, 28), (100352, 784, 28, 1))
        del arg44_1
        del buf49
        buf51 = buf50; del buf50  # reuse
        # Source Nodes: [l__mod___features_denseblock2_denselayer1_norm2, l__mod___features_denseblock2_denselayer1_relu2, new_features_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf51, arg409_1, arg410_1, arg45_1, arg46_1, 401408, grid=grid(401408), stream=stream0)
        del arg409_1
        del arg410_1
        del arg45_1
        del arg46_1
        # Source Nodes: [l__mod___features_denseblock2_denselayer1_norm2, l__mod___features_denseblock2_denselayer1_relu2, new_features_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf52 = extern_kernels.convolution(buf51, arg47_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 32, 28, 28), (25088, 784, 28, 1))
        del arg47_1
        del buf51
        buf53 = empty((4, 160, 28, 28), device='cuda', dtype=torch.float32)
        buf54 = buf53; del buf53  # reuse
        # Source Nodes: [bottleneck_output_14, cat_115, l__mod___features_denseblock2_denselayer2_norm1, l__mod___features_denseblock2_denselayer2_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_14.run(buf54, buf48, buf52, arg412_1, arg413_1, arg48_1, arg49_1, 501760, grid=grid(501760), stream=stream0)
        del arg412_1
        del arg413_1
        del arg48_1
        del arg49_1
        # Source Nodes: [bottleneck_output_14, l__mod___features_denseblock2_denselayer2_relu1], Original ATen: [aten.convolution, aten.relu]
        buf55 = extern_kernels.convolution(buf54, arg50_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 128, 28, 28), (100352, 784, 28, 1))
        del arg50_1
        buf56 = buf55; del buf55  # reuse
        # Source Nodes: [l__mod___features_denseblock2_denselayer2_norm2, l__mod___features_denseblock2_denselayer2_relu2, new_features_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf56, arg415_1, arg416_1, arg51_1, arg52_1, 401408, grid=grid(401408), stream=stream0)
        del arg415_1
        del arg416_1
        del arg51_1
        del arg52_1
        # Source Nodes: [l__mod___features_denseblock2_denselayer2_norm2, l__mod___features_denseblock2_denselayer2_relu2, new_features_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf57 = extern_kernels.convolution(buf56, arg53_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (4, 32, 28, 28), (25088, 784, 28, 1))
        del arg53_1
        del buf56
        buf58 = empty((4, 192, 28, 28), device='cuda', dtype=torch.float32)
        buf59 = buf58; del buf58  # reuse
        # Source Nodes: [bottleneck_output_16, cat_114, l__mod___features_denseblock2_denselayer3_norm1, l__mod___features_denseblock2_denselayer3_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_15.run(buf59, buf48, buf52, buf57, arg418_1, arg419_1, arg54_1, arg55_1, 602112, grid=grid(602112), stream=stream0)
        del arg418_1
        del arg419_1
        del arg54_1
        del arg55_1
        # Source Nodes: [bottleneck_output_16, l__mod___features_denseblock2_denselayer3_norm1, l__mod___features_denseblock2_denselayer3_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf60 = extern_kernels.convolution(buf59, arg56_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 128, 28, 28), (100352, 784, 28, 1))
        del arg56_1
        buf61 = buf60; del buf60  # reuse
        # Source Nodes: [l__mod___features_denseblock2_denselayer3_norm2, l__mod___features_denseblock2_denselayer3_relu2, new_features_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf61, arg421_1, arg422_1, arg57_1, arg58_1, 401408, grid=grid(401408), stream=stream0)
        del arg421_1
        del arg422_1
        del arg57_1
        del arg58_1
        # Source Nodes: [l__mod___features_denseblock2_denselayer3_norm2, l__mod___features_denseblock2_denselayer3_relu2, new_features_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf62 = extern_kernels.convolution(buf61, arg59_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 32, 28, 28), (25088, 784, 28, 1))
        del arg59_1
        del buf61
        buf63 = empty((4, 224, 28, 28), device='cuda', dtype=torch.float32)
        buf64 = buf63; del buf63  # reuse
        # Source Nodes: [bottleneck_output_18, cat_113, l__mod___features_denseblock2_denselayer4_norm1, l__mod___features_denseblock2_denselayer4_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_16.run(buf64, buf48, buf52, buf57, buf62, arg424_1, arg425_1, arg60_1, arg61_1, 702464, grid=grid(702464), stream=stream0)
        del arg424_1
        del arg425_1
        del arg60_1
        del arg61_1
        del buf48
        # Source Nodes: [bottleneck_output_18, l__mod___features_denseblock2_denselayer4_norm1, l__mod___features_denseblock2_denselayer4_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf65 = extern_kernels.convolution(buf64, arg62_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (4, 128, 28, 28), (100352, 784, 28, 1))
        del arg62_1
        buf66 = buf65; del buf65  # reuse
        # Source Nodes: [l__mod___features_denseblock2_denselayer4_norm2, l__mod___features_denseblock2_denselayer4_relu2, new_features_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf66, arg427_1, arg428_1, arg63_1, arg64_1, 401408, grid=grid(401408), stream=stream0)
        del arg427_1
        del arg428_1
        del arg63_1
        del arg64_1
        # Source Nodes: [l__mod___features_denseblock2_denselayer4_norm2, l__mod___features_denseblock2_denselayer4_relu2, new_features_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf67 = extern_kernels.convolution(buf66, arg65_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (4, 32, 28, 28), (25088, 784, 28, 1))
        del arg65_1
        del buf66
        buf69 = reinterpret_tensor(buf73, (4, 32, 28, 28), (200704, 784, 28, 1), 100352)  # alias
        buf79 = reinterpret_tensor(buf84, (4, 32, 28, 28), (225792, 784, 28, 1), 100352)  # alias
        buf90 = reinterpret_tensor(buf96, (4, 32, 28, 28), (250880, 784, 28, 1), 100352)  # alias
        buf102 = reinterpret_tensor(buf109, (4, 32, 28, 28), (275968, 784, 28, 1), 100352)  # alias
        buf115 = reinterpret_tensor(buf123, (4, 32, 28, 28), (301056, 784, 28, 1), 100352)  # alias
        buf129 = reinterpret_tensor(buf138, (4, 32, 28, 28), (326144, 784, 28, 1), 100352)  # alias
        # Source Nodes: [cat_107, cat_108, cat_109, cat_110, cat_111, cat_112], Original ATen: [aten.cat]
        triton_poi_fused_cat_17.run(buf52, buf69, buf79, buf90, buf102, buf115, buf129, 100352, grid=grid(100352), stream=stream0)
        buf70 = reinterpret_tensor(buf73, (4, 32, 28, 28), (200704, 784, 28, 1), 125440)  # alias
        buf80 = reinterpret_tensor(buf84, (4, 32, 28, 28), (225792, 784, 28, 1), 125440)  # alias
        buf91 = reinterpret_tensor(buf96, (4, 32, 28, 28), (250880, 784, 28, 1), 125440)  # alias
        buf103 = reinterpret_tensor(buf109, (4, 32, 28, 28), (275968, 784, 28, 1), 125440)  # alias
        buf116 = reinterpret_tensor(buf123, (4, 32, 28, 28), (301056, 784, 28, 1), 125440)  # alias
        buf130 = reinterpret_tensor(buf138, (4, 32, 28, 28), (326144, 784, 28, 1), 125440)  # alias
        # Source Nodes: [cat_107, cat_108, cat_109, cat_110, cat_111, cat_112], Original ATen: [aten.cat]
        triton_poi_fused_cat_17.run(buf57, buf70, buf80, buf91, buf103, buf116, buf130, 100352, grid=grid(100352), stream=stream0)
        buf71 = reinterpret_tensor(buf73, (4, 32, 28, 28), (200704, 784, 28, 1), 150528)  # alias
        buf81 = reinterpret_tensor(buf84, (4, 32, 28, 28), (225792, 784, 28, 1), 150528)  # alias
        buf92 = reinterpret_tensor(buf96, (4, 32, 28, 28), (250880, 784, 28, 1), 150528)  # alias
        buf104 = reinterpret_tensor(buf109, (4, 32, 28, 28), (275968, 784, 28, 1), 150528)  # alias
        buf117 = reinterpret_tensor(buf123, (4, 32, 28, 28), (301056, 784, 28, 1), 150528)  # alias
        buf131 = reinterpret_tensor(buf138, (4, 32, 28, 28), (326144, 784, 28, 1), 150528)  # alias
        # Source Nodes: [cat_107, cat_108, cat_109, cat_110, cat_111, cat_112], Original ATen: [aten.cat]
        triton_poi_fused_cat_17.run(buf62, buf71, buf81, buf92, buf104, buf117, buf131, 100352, grid=grid(100352), stream=stream0)
        buf72 = reinterpret_tensor(buf73, (4, 32, 28, 28), (200704, 784, 28, 1), 175616)  # alias
        buf82 = reinterpret_tensor(buf84, (4, 32, 28, 28), (225792, 784, 28, 1), 175616)  # alias
        buf93 = reinterpret_tensor(buf96, (4, 32, 28, 28), (250880, 784, 28, 1), 175616)  # alias
        buf105 = reinterpret_tensor(buf109, (4, 32, 28, 28), (275968, 784, 28, 1), 175616)  # alias
        buf118 = reinterpret_tensor(buf123, (4, 32, 28, 28), (301056, 784, 28, 1), 175616)  # alias
        buf132 = reinterpret_tensor(buf138, (4, 32, 28, 28), (326144, 784, 28, 1), 175616)  # alias
        # Source Nodes: [cat_107, cat_108, cat_109, cat_110, cat_111, cat_112], Original ATen: [aten.cat]
        triton_poi_fused_cat_17.run(buf67, buf72, buf82, buf93, buf105, buf118, buf132, 100352, grid=grid(100352), stream=stream0)
        buf74 = buf73; del buf73  # reuse
        # Source Nodes: [bottleneck_output_20, l__mod___features_denseblock2_denselayer5_norm1, l__mod___features_denseblock2_denselayer5_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_18.run(buf74, arg430_1, arg431_1, arg66_1, arg67_1, 802816, grid=grid(802816), stream=stream0)
        del arg430_1
        del arg431_1
        del arg66_1
        del arg67_1
        del buf68
        del buf69
        del buf70
        del buf71
        del buf72
        # Source Nodes: [bottleneck_output_20, l__mod___features_denseblock2_denselayer5_norm1, l__mod___features_denseblock2_denselayer5_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf75 = extern_kernels.convolution(buf74, arg68_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (4, 128, 28, 28), (100352, 784, 28, 1))
        del arg68_1
        buf76 = buf75; del buf75  # reuse
        # Source Nodes: [l__mod___features_denseblock2_denselayer5_norm2, l__mod___features_denseblock2_denselayer5_relu2, new_features_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf76, arg433_1, arg434_1, arg69_1, arg70_1, 401408, grid=grid(401408), stream=stream0)
        del arg433_1
        del arg434_1
        del arg69_1
        del arg70_1
        # Source Nodes: [l__mod___features_denseblock2_denselayer5_norm2, l__mod___features_denseblock2_denselayer5_relu2, new_features_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf77 = extern_kernels.convolution(buf76, arg71_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (4, 32, 28, 28), (25088, 784, 28, 1))
        del arg71_1
        del buf76
        buf83 = reinterpret_tensor(buf84, (4, 32, 28, 28), (225792, 784, 28, 1), 200704)  # alias
        buf94 = reinterpret_tensor(buf96, (4, 32, 28, 28), (250880, 784, 28, 1), 200704)  # alias
        buf106 = reinterpret_tensor(buf109, (4, 32, 28, 28), (275968, 784, 28, 1), 200704)  # alias
        buf119 = reinterpret_tensor(buf123, (4, 32, 28, 28), (301056, 784, 28, 1), 200704)  # alias
        buf133 = reinterpret_tensor(buf138, (4, 32, 28, 28), (326144, 784, 28, 1), 200704)  # alias
        # Source Nodes: [cat_107, cat_108, cat_109, cat_110, cat_111], Original ATen: [aten.cat]
        triton_poi_fused_cat_19.run(buf77, buf83, buf94, buf106, buf119, buf133, 100352, grid=grid(100352), stream=stream0)
        buf85 = buf84; del buf84  # reuse
        # Source Nodes: [bottleneck_output_22, l__mod___features_denseblock2_denselayer6_norm1, l__mod___features_denseblock2_denselayer6_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_20.run(buf85, arg436_1, arg437_1, arg72_1, arg73_1, 903168, grid=grid(903168), stream=stream0)
        del arg436_1
        del arg437_1
        del arg72_1
        del arg73_1
        del buf78
        del buf79
        del buf80
        del buf81
        del buf82
        del buf83
        # Source Nodes: [bottleneck_output_22, l__mod___features_denseblock2_denselayer6_norm1, l__mod___features_denseblock2_denselayer6_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf86 = extern_kernels.convolution(buf85, arg74_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (4, 128, 28, 28), (100352, 784, 28, 1))
        del arg74_1
        del buf85
        buf87 = buf86; del buf86  # reuse
        # Source Nodes: [l__mod___features_denseblock2_denselayer6_norm2, l__mod___features_denseblock2_denselayer6_relu2, new_features_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf87, arg439_1, arg440_1, arg75_1, arg76_1, 401408, grid=grid(401408), stream=stream0)
        del arg439_1
        del arg440_1
        del arg75_1
        del arg76_1
        # Source Nodes: [l__mod___features_denseblock2_denselayer6_norm2, l__mod___features_denseblock2_denselayer6_relu2, new_features_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf88 = extern_kernels.convolution(buf87, arg77_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 32, 28, 28), (25088, 784, 28, 1))
        del arg77_1
        del buf87
        buf95 = reinterpret_tensor(buf96, (4, 32, 28, 28), (250880, 784, 28, 1), 225792)  # alias
        buf107 = reinterpret_tensor(buf109, (4, 32, 28, 28), (275968, 784, 28, 1), 225792)  # alias
        buf120 = reinterpret_tensor(buf123, (4, 32, 28, 28), (301056, 784, 28, 1), 225792)  # alias
        buf134 = reinterpret_tensor(buf138, (4, 32, 28, 28), (326144, 784, 28, 1), 225792)  # alias
        buf149 = reinterpret_tensor(buf154, (4, 32, 28, 28), (351232, 784, 28, 1), 225792)  # alias
        # Source Nodes: [cat_106, cat_107, cat_108, cat_109, cat_110], Original ATen: [aten.cat]
        triton_poi_fused_cat_21.run(buf88, buf95, buf107, buf120, buf134, buf149, 100352, grid=grid(100352), stream=stream0)
        buf97 = buf96; del buf96  # reuse
        # Source Nodes: [bottleneck_output_24, l__mod___features_denseblock2_denselayer7_norm1, l__mod___features_denseblock2_denselayer7_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf97, arg442_1, arg443_1, arg78_1, arg79_1, 1003520, grid=grid(1003520), stream=stream0)
        del arg442_1
        del arg443_1
        del arg78_1
        del arg79_1
        del buf89
        del buf90
        del buf91
        del buf92
        del buf93
        del buf94
        del buf95
        # Source Nodes: [bottleneck_output_24, l__mod___features_denseblock2_denselayer7_norm1, l__mod___features_denseblock2_denselayer7_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf98 = extern_kernels.convolution(buf97, arg80_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 128, 28, 28), (100352, 784, 28, 1))
        del arg80_1
        del buf97
        buf99 = buf98; del buf98  # reuse
        # Source Nodes: [l__mod___features_denseblock2_denselayer7_norm2, l__mod___features_denseblock2_denselayer7_relu2, new_features_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf99, arg445_1, arg446_1, arg81_1, arg82_1, 401408, grid=grid(401408), stream=stream0)
        del arg445_1
        del arg446_1
        del arg81_1
        del arg82_1
        # Source Nodes: [l__mod___features_denseblock2_denselayer7_norm2, l__mod___features_denseblock2_denselayer7_relu2, new_features_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf100 = extern_kernels.convolution(buf99, arg83_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (4, 32, 28, 28), (25088, 784, 28, 1))
        del arg83_1
        del buf99
        buf108 = reinterpret_tensor(buf109, (4, 32, 28, 28), (275968, 784, 28, 1), 250880)  # alias
        buf121 = reinterpret_tensor(buf123, (4, 32, 28, 28), (301056, 784, 28, 1), 250880)  # alias
        buf135 = reinterpret_tensor(buf138, (4, 32, 28, 28), (326144, 784, 28, 1), 250880)  # alias
        buf150 = reinterpret_tensor(buf154, (4, 32, 28, 28), (351232, 784, 28, 1), 250880)  # alias
        buf166 = reinterpret_tensor(buf171, (4, 32, 28, 28), (376320, 784, 28, 1), 250880)  # alias
        # Source Nodes: [cat_105, cat_106, cat_107, cat_108, cat_109], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf100, buf108, buf121, buf135, buf150, buf166, 100352, grid=grid(100352), stream=stream0)
        buf110 = buf109; del buf109  # reuse
        # Source Nodes: [bottleneck_output_26, l__mod___features_denseblock2_denselayer8_norm1, l__mod___features_denseblock2_denselayer8_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf110, arg448_1, arg449_1, arg84_1, arg85_1, 1103872, grid=grid(1103872), stream=stream0)
        del arg448_1
        del arg449_1
        del arg84_1
        del arg85_1
        del buf101
        del buf102
        del buf103
        del buf104
        del buf105
        del buf106
        del buf107
        del buf108
        # Source Nodes: [bottleneck_output_26, l__mod___features_denseblock2_denselayer8_norm1, l__mod___features_denseblock2_denselayer8_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf111 = extern_kernels.convolution(buf110, arg86_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (4, 128, 28, 28), (100352, 784, 28, 1))
        del arg86_1
        del buf110
        buf112 = buf111; del buf111  # reuse
        # Source Nodes: [l__mod___features_denseblock2_denselayer8_norm2, l__mod___features_denseblock2_denselayer8_relu2, new_features_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf112, arg451_1, arg452_1, arg87_1, arg88_1, 401408, grid=grid(401408), stream=stream0)
        del arg451_1
        del arg452_1
        del arg87_1
        del arg88_1
        # Source Nodes: [l__mod___features_denseblock2_denselayer8_norm2, l__mod___features_denseblock2_denselayer8_relu2, new_features_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf113 = extern_kernels.convolution(buf112, arg89_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (4, 32, 28, 28), (25088, 784, 28, 1))
        del arg89_1
        del buf112
        buf122 = reinterpret_tensor(buf123, (4, 32, 28, 28), (301056, 784, 28, 1), 275968)  # alias
        buf136 = reinterpret_tensor(buf138, (4, 32, 28, 28), (326144, 784, 28, 1), 275968)  # alias
        buf151 = reinterpret_tensor(buf154, (4, 32, 28, 28), (351232, 784, 28, 1), 275968)  # alias
        buf167 = reinterpret_tensor(buf171, (4, 32, 28, 28), (376320, 784, 28, 1), 275968)  # alias
        buf184 = reinterpret_tensor(buf189, (4, 32, 28, 28), (401408, 784, 28, 1), 275968)  # alias
        # Source Nodes: [cat_104, cat_105, cat_106, cat_107, cat_108], Original ATen: [aten.cat]
        triton_poi_fused_cat_25.run(buf113, buf122, buf136, buf151, buf167, buf184, 100352, grid=grid(100352), stream=stream0)
        del buf113
        buf124 = buf123; del buf123  # reuse
        # Source Nodes: [bottleneck_output_28, l__mod___features_denseblock2_denselayer9_norm1, l__mod___features_denseblock2_denselayer9_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_26.run(buf124, arg454_1, arg455_1, arg90_1, arg91_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg454_1
        del arg455_1
        del arg90_1
        del arg91_1
        del buf114
        del buf115
        del buf116
        del buf117
        del buf118
        del buf119
        del buf120
        del buf121
        del buf122
        # Source Nodes: [bottleneck_output_28, l__mod___features_denseblock2_denselayer9_norm1, l__mod___features_denseblock2_denselayer9_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf125 = extern_kernels.convolution(buf124, arg92_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf125, (4, 128, 28, 28), (100352, 784, 28, 1))
        del arg92_1
        del buf124
        buf126 = buf125; del buf125  # reuse
        # Source Nodes: [l__mod___features_denseblock2_denselayer9_norm2, l__mod___features_denseblock2_denselayer9_relu2, new_features_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf126, arg457_1, arg458_1, arg93_1, arg94_1, 401408, grid=grid(401408), stream=stream0)
        del arg457_1
        del arg458_1
        del arg93_1
        del arg94_1
        # Source Nodes: [l__mod___features_denseblock2_denselayer9_norm2, l__mod___features_denseblock2_denselayer9_relu2, new_features_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf127 = extern_kernels.convolution(buf126, arg95_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (4, 32, 28, 28), (25088, 784, 28, 1))
        del arg95_1
        del buf126
        buf137 = reinterpret_tensor(buf138, (4, 32, 28, 28), (326144, 784, 28, 1), 301056)  # alias
        buf152 = reinterpret_tensor(buf154, (4, 32, 28, 28), (351232, 784, 28, 1), 301056)  # alias
        buf168 = reinterpret_tensor(buf171, (4, 32, 28, 28), (376320, 784, 28, 1), 301056)  # alias
        buf185 = reinterpret_tensor(buf189, (4, 32, 28, 28), (401408, 784, 28, 1), 301056)  # alias
        # Source Nodes: [cat_104, cat_105, cat_106, cat_107], Original ATen: [aten.cat]
        triton_poi_fused_cat_27.run(buf127, buf137, buf152, buf168, buf185, 100352, grid=grid(100352), stream=stream0)
        del buf127
        buf139 = buf138; del buf138  # reuse
        # Source Nodes: [bottleneck_output_30, l__mod___features_denseblock2_denselayer10_norm1, l__mod___features_denseblock2_denselayer10_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(buf139, arg460_1, arg461_1, arg96_1, arg97_1, 1304576, grid=grid(1304576), stream=stream0)
        del arg460_1
        del arg461_1
        del arg96_1
        del arg97_1
        del buf128
        del buf129
        del buf130
        del buf131
        del buf132
        del buf133
        del buf134
        del buf135
        del buf136
        del buf137
        # Source Nodes: [bottleneck_output_30, l__mod___features_denseblock2_denselayer10_norm1, l__mod___features_denseblock2_denselayer10_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf140 = extern_kernels.convolution(buf139, arg98_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (4, 128, 28, 28), (100352, 784, 28, 1))
        del arg98_1
        del buf139
        buf141 = buf140; del buf140  # reuse
        # Source Nodes: [l__mod___features_denseblock2_denselayer10_norm2, l__mod___features_denseblock2_denselayer10_relu2, new_features_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf141, arg463_1, arg464_1, arg99_1, arg100_1, 401408, grid=grid(401408), stream=stream0)
        del arg100_1
        del arg463_1
        del arg464_1
        del arg99_1
        # Source Nodes: [l__mod___features_denseblock2_denselayer10_norm2, l__mod___features_denseblock2_denselayer10_relu2, new_features_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf142 = extern_kernels.convolution(buf141, arg101_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf142, (4, 32, 28, 28), (25088, 784, 28, 1))
        del arg101_1
        del buf141
        buf144 = reinterpret_tensor(buf154, (4, 32, 28, 28), (351232, 784, 28, 1), 100352)  # alias
        buf160 = reinterpret_tensor(buf171, (4, 32, 28, 28), (376320, 784, 28, 1), 100352)  # alias
        buf177 = reinterpret_tensor(buf189, (4, 32, 28, 28), (401408, 784, 28, 1), 100352)  # alias
        # Source Nodes: [cat_104, cat_105, cat_106], Original ATen: [aten.cat]
        triton_poi_fused_cat_29.run(buf52, buf144, buf160, buf177, 100352, grid=grid(100352), stream=stream0)
        del buf52
        buf145 = reinterpret_tensor(buf154, (4, 32, 28, 28), (351232, 784, 28, 1), 125440)  # alias
        buf161 = reinterpret_tensor(buf171, (4, 32, 28, 28), (376320, 784, 28, 1), 125440)  # alias
        buf178 = reinterpret_tensor(buf189, (4, 32, 28, 28), (401408, 784, 28, 1), 125440)  # alias
        # Source Nodes: [cat_104, cat_105, cat_106], Original ATen: [aten.cat]
        triton_poi_fused_cat_29.run(buf57, buf145, buf161, buf178, 100352, grid=grid(100352), stream=stream0)
        del buf57
        buf146 = reinterpret_tensor(buf154, (4, 32, 28, 28), (351232, 784, 28, 1), 150528)  # alias
        buf162 = reinterpret_tensor(buf171, (4, 32, 28, 28), (376320, 784, 28, 1), 150528)  # alias
        buf179 = reinterpret_tensor(buf189, (4, 32, 28, 28), (401408, 784, 28, 1), 150528)  # alias
        # Source Nodes: [cat_104, cat_105, cat_106], Original ATen: [aten.cat]
        triton_poi_fused_cat_29.run(buf62, buf146, buf162, buf179, 100352, grid=grid(100352), stream=stream0)
        del buf62
        buf147 = reinterpret_tensor(buf154, (4, 32, 28, 28), (351232, 784, 28, 1), 175616)  # alias
        buf163 = reinterpret_tensor(buf171, (4, 32, 28, 28), (376320, 784, 28, 1), 175616)  # alias
        buf180 = reinterpret_tensor(buf189, (4, 32, 28, 28), (401408, 784, 28, 1), 175616)  # alias
        # Source Nodes: [cat_104, cat_105, cat_106], Original ATen: [aten.cat]
        triton_poi_fused_cat_29.run(buf67, buf147, buf163, buf180, 100352, grid=grid(100352), stream=stream0)
        del buf67
        buf148 = reinterpret_tensor(buf154, (4, 32, 28, 28), (351232, 784, 28, 1), 200704)  # alias
        buf164 = reinterpret_tensor(buf171, (4, 32, 28, 28), (376320, 784, 28, 1), 200704)  # alias
        buf181 = reinterpret_tensor(buf189, (4, 32, 28, 28), (401408, 784, 28, 1), 200704)  # alias
        # Source Nodes: [cat_104, cat_105, cat_106], Original ATen: [aten.cat]
        triton_poi_fused_cat_29.run(buf77, buf148, buf164, buf181, 100352, grid=grid(100352), stream=stream0)
        del buf77
        buf153 = reinterpret_tensor(buf154, (4, 32, 28, 28), (351232, 784, 28, 1), 326144)  # alias
        buf169 = reinterpret_tensor(buf171, (4, 32, 28, 28), (376320, 784, 28, 1), 326144)  # alias
        buf186 = reinterpret_tensor(buf189, (4, 32, 28, 28), (401408, 784, 28, 1), 326144)  # alias
        # Source Nodes: [cat_104, cat_105, cat_106], Original ATen: [aten.cat]
        triton_poi_fused_cat_29.run(buf142, buf153, buf169, buf186, 100352, grid=grid(100352), stream=stream0)
        del buf142
        buf155 = buf154; del buf154  # reuse
        # Source Nodes: [bottleneck_output_32, l__mod___features_denseblock2_denselayer11_norm1, l__mod___features_denseblock2_denselayer11_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30.run(buf155, arg466_1, arg467_1, arg102_1, arg103_1, 1404928, grid=grid(1404928), stream=stream0)
        del arg102_1
        del arg103_1
        del arg466_1
        del arg467_1
        del buf143
        del buf144
        del buf145
        del buf146
        del buf147
        del buf148
        del buf149
        del buf150
        del buf151
        del buf152
        del buf153
        # Source Nodes: [bottleneck_output_32, l__mod___features_denseblock2_denselayer11_norm1, l__mod___features_denseblock2_denselayer11_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf156 = extern_kernels.convolution(buf155, arg104_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf156, (4, 128, 28, 28), (100352, 784, 28, 1))
        del arg104_1
        del buf155
        buf157 = buf156; del buf156  # reuse
        # Source Nodes: [l__mod___features_denseblock2_denselayer11_norm2, l__mod___features_denseblock2_denselayer11_relu2, new_features_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf157, arg469_1, arg470_1, arg105_1, arg106_1, 401408, grid=grid(401408), stream=stream0)
        del arg105_1
        del arg106_1
        del arg469_1
        del arg470_1
        # Source Nodes: [l__mod___features_denseblock2_denselayer11_norm2, l__mod___features_denseblock2_denselayer11_relu2, new_features_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf158 = extern_kernels.convolution(buf157, arg107_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (4, 32, 28, 28), (25088, 784, 28, 1))
        del arg107_1
        del buf157
        buf165 = reinterpret_tensor(buf171, (4, 32, 28, 28), (376320, 784, 28, 1), 225792)  # alias
        buf182 = reinterpret_tensor(buf189, (4, 32, 28, 28), (401408, 784, 28, 1), 225792)  # alias
        # Source Nodes: [cat_104, cat_105], Original ATen: [aten.cat]
        triton_poi_fused_cat_31.run(buf88, buf165, buf182, 100352, grid=grid(100352), stream=stream0)
        del buf88
        buf170 = reinterpret_tensor(buf171, (4, 32, 28, 28), (376320, 784, 28, 1), 351232)  # alias
        buf187 = reinterpret_tensor(buf189, (4, 32, 28, 28), (401408, 784, 28, 1), 351232)  # alias
        # Source Nodes: [cat_104, cat_105], Original ATen: [aten.cat]
        triton_poi_fused_cat_31.run(buf158, buf170, buf187, 100352, grid=grid(100352), stream=stream0)
        del buf158
        buf172 = buf171; del buf171  # reuse
        # Source Nodes: [bottleneck_output_34, l__mod___features_denseblock2_denselayer12_norm1, l__mod___features_denseblock2_denselayer12_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_32.run(buf172, arg472_1, arg473_1, arg108_1, arg109_1, 1505280, grid=grid(1505280), stream=stream0)
        del arg108_1
        del arg109_1
        del arg472_1
        del arg473_1
        del buf159
        del buf160
        del buf161
        del buf162
        del buf163
        del buf164
        del buf165
        del buf166
        del buf167
        del buf168
        del buf169
        del buf170
        # Source Nodes: [bottleneck_output_34, l__mod___features_denseblock2_denselayer12_norm1, l__mod___features_denseblock2_denselayer12_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf173 = extern_kernels.convolution(buf172, arg110_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (4, 128, 28, 28), (100352, 784, 28, 1))
        del arg110_1
        del buf172
        buf174 = buf173; del buf173  # reuse
        # Source Nodes: [l__mod___features_denseblock2_denselayer12_norm2, l__mod___features_denseblock2_denselayer12_relu2, new_features_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(buf174, arg475_1, arg476_1, arg111_1, arg112_1, 401408, grid=grid(401408), stream=stream0)
        del arg111_1
        del arg112_1
        del arg475_1
        del arg476_1
        # Source Nodes: [l__mod___features_denseblock2_denselayer12_norm2, l__mod___features_denseblock2_denselayer12_relu2, new_features_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf175 = extern_kernels.convolution(buf174, arg113_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (4, 32, 28, 28), (25088, 784, 28, 1))
        del arg113_1
        buf183 = reinterpret_tensor(buf189, (4, 32, 28, 28), (401408, 784, 28, 1), 250880)  # alias
        # Source Nodes: [cat_104], Original ATen: [aten.cat]
        triton_poi_fused_cat_33.run(buf100, buf183, 100352, grid=grid(100352), stream=stream0)
        del buf100
        buf188 = reinterpret_tensor(buf189, (4, 32, 28, 28), (401408, 784, 28, 1), 376320)  # alias
        # Source Nodes: [cat_104], Original ATen: [aten.cat]
        triton_poi_fused_cat_33.run(buf175, buf188, 100352, grid=grid(100352), stream=stream0)
        del buf175
        buf190 = buf189; del buf189  # reuse
        # Source Nodes: [l__mod___features_transition2_conv, l__mod___features_transition2_norm, l__mod___features_transition2_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34.run(buf190, arg478_1, arg479_1, arg114_1, arg115_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg114_1
        del arg115_1
        del arg478_1
        del arg479_1
        del buf176
        del buf177
        del buf178
        del buf179
        del buf180
        del buf181
        del buf182
        del buf183
        del buf184
        del buf185
        del buf186
        del buf187
        del buf188
        # Source Nodes: [l__mod___features_transition2_conv, l__mod___features_transition2_norm, l__mod___features_transition2_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf191 = extern_kernels.convolution(buf190, arg116_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf191, (4, 256, 28, 28), (200704, 784, 28, 1))
        del arg116_1
        del buf190
        buf192 = empty((4, 256, 14, 14), device='cuda', dtype=torch.float32)
        buf216 = empty((4, 384, 14, 14), device='cuda', dtype=torch.float32)
        buf211 = reinterpret_tensor(buf216, (4, 256, 14, 14), (75264, 196, 14, 1), 0)  # alias
        buf227 = empty((4, 416, 14, 14), device='cuda', dtype=torch.float32)
        buf221 = reinterpret_tensor(buf227, (4, 256, 14, 14), (81536, 196, 14, 1), 0)  # alias
        buf239 = empty((4, 448, 14, 14), device='cuda', dtype=torch.float32)
        buf232 = reinterpret_tensor(buf239, (4, 256, 14, 14), (87808, 196, 14, 1), 0)  # alias
        buf252 = empty((4, 480, 14, 14), device='cuda', dtype=torch.float32)
        buf244 = reinterpret_tensor(buf252, (4, 256, 14, 14), (94080, 196, 14, 1), 0)  # alias
        buf266 = reinterpret_tensor(buf174, (4, 512, 14, 14), (100352, 196, 14, 1), 0); del buf174  # reuse
        buf257 = reinterpret_tensor(buf266, (4, 256, 14, 14), (100352, 196, 14, 1), 0)  # alias
        buf281 = empty((4, 544, 14, 14), device='cuda', dtype=torch.float32)
        buf271 = reinterpret_tensor(buf281, (4, 256, 14, 14), (106624, 196, 14, 1), 0)  # alias
        buf297 = empty((4, 576, 14, 14), device='cuda', dtype=torch.float32)
        buf286 = reinterpret_tensor(buf297, (4, 256, 14, 14), (112896, 196, 14, 1), 0)  # alias
        buf314 = empty((4, 608, 14, 14), device='cuda', dtype=torch.float32)
        buf302 = reinterpret_tensor(buf314, (4, 256, 14, 14), (119168, 196, 14, 1), 0)  # alias
        buf332 = reinterpret_tensor(buf54, (4, 640, 14, 14), (125440, 196, 14, 1), 0); del buf54  # reuse
        buf319 = reinterpret_tensor(buf332, (4, 256, 14, 14), (125440, 196, 14, 1), 0)  # alias
        buf351 = empty((4, 672, 14, 14), device='cuda', dtype=torch.float32)
        buf337 = reinterpret_tensor(buf351, (4, 256, 14, 14), (131712, 196, 14, 1), 0)  # alias
        buf371 = empty((4, 704, 14, 14), device='cuda', dtype=torch.float32)
        buf356 = reinterpret_tensor(buf371, (4, 256, 14, 14), (137984, 196, 14, 1), 0)  # alias
        buf392 = empty((4, 736, 14, 14), device='cuda', dtype=torch.float32)
        buf376 = reinterpret_tensor(buf392, (4, 256, 14, 14), (144256, 196, 14, 1), 0)  # alias
        buf414 = reinterpret_tensor(buf59, (4, 768, 14, 14), (150528, 196, 14, 1), 0); del buf59  # reuse
        buf397 = reinterpret_tensor(buf414, (4, 256, 14, 14), (150528, 196, 14, 1), 0)  # alias
        buf437 = empty((4, 800, 14, 14), device='cuda', dtype=torch.float32)
        buf419 = reinterpret_tensor(buf437, (4, 256, 14, 14), (156800, 196, 14, 1), 0)  # alias
        buf461 = empty((4, 832, 14, 14), device='cuda', dtype=torch.float32)
        buf442 = reinterpret_tensor(buf461, (4, 256, 14, 14), (163072, 196, 14, 1), 0)  # alias
        buf486 = empty((4, 864, 14, 14), device='cuda', dtype=torch.float32)
        buf466 = reinterpret_tensor(buf486, (4, 256, 14, 14), (169344, 196, 14, 1), 0)  # alias
        buf512 = reinterpret_tensor(buf64, (4, 896, 14, 14), (175616, 196, 14, 1), 0); del buf64  # reuse
        buf491 = reinterpret_tensor(buf512, (4, 256, 14, 14), (175616, 196, 14, 1), 0)  # alias
        buf539 = empty((4, 928, 14, 14), device='cuda', dtype=torch.float32)
        buf517 = reinterpret_tensor(buf539, (4, 256, 14, 14), (181888, 196, 14, 1), 0)  # alias
        buf567 = empty((4, 960, 14, 14), device='cuda', dtype=torch.float32)
        buf544 = reinterpret_tensor(buf567, (4, 256, 14, 14), (188160, 196, 14, 1), 0)  # alias
        buf596 = empty((4, 992, 14, 14), device='cuda', dtype=torch.float32)
        buf572 = reinterpret_tensor(buf596, (4, 256, 14, 14), (194432, 196, 14, 1), 0)  # alias
        buf626 = reinterpret_tensor(buf74, (4, 1024, 14, 14), (200704, 196, 14, 1), 0); del buf74  # reuse
        buf601 = reinterpret_tensor(buf626, (4, 256, 14, 14), (200704, 196, 14, 1), 0)  # alias
        # Source Nodes: [bottleneck_output_36, cat_79, cat_80, cat_81, cat_82, cat_83, cat_84, cat_85, cat_86, cat_87, cat_88, cat_89, cat_90, cat_91, cat_92, cat_93, cat_94, cat_95, cat_96, cat_97, cat_98, l__mod___features_denseblock3_denselayer1_norm1, l__mod___features_denseblock3_denselayer1_relu1, l__mod___features_transition2_pool], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_35.run(buf191, arg481_1, arg482_1, arg117_1, arg118_1, buf192, buf211, buf221, buf232, buf244, buf257, buf271, buf286, buf302, buf319, buf337, buf356, buf376, buf397, buf419, buf442, buf466, buf491, buf517, buf544, buf572, buf601, 200704, grid=grid(200704), stream=stream0)
        del arg117_1
        del arg118_1
        del arg481_1
        del arg482_1
        # Source Nodes: [bottleneck_output_36, l__mod___features_denseblock3_denselayer1_norm1, l__mod___features_denseblock3_denselayer1_relu1, l__mod___features_transition2_pool], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d, aten.convolution, aten.relu]
        buf193 = extern_kernels.convolution(buf192, arg119_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (4, 128, 14, 14), (25088, 196, 14, 1))
        del arg119_1
        buf194 = buf193; del buf193  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer1_norm2, l__mod___features_denseblock3_denselayer1_relu2, new_features_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(buf194, arg484_1, arg485_1, arg120_1, arg121_1, 100352, grid=grid(100352), stream=stream0)
        del arg120_1
        del arg121_1
        del arg484_1
        del arg485_1
        # Source Nodes: [l__mod___features_denseblock3_denselayer1_norm2, l__mod___features_denseblock3_denselayer1_relu2, new_features_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf195 = extern_kernels.convolution(buf194, arg122_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf195, (4, 32, 14, 14), (6272, 196, 14, 1))
        del arg122_1
        del buf194
        buf196 = empty((4, 288, 14, 14), device='cuda', dtype=torch.float32)
        buf197 = buf196; del buf196  # reuse
        # Source Nodes: [bottleneck_output_38, cat_102, l__mod___features_denseblock3_denselayer2_norm1, l__mod___features_denseblock3_denselayer2_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_37.run(buf197, buf191, buf195, arg487_1, arg488_1, arg123_1, arg124_1, 225792, grid=grid(225792), stream=stream0)
        del arg123_1
        del arg124_1
        del arg487_1
        del arg488_1
        # Source Nodes: [bottleneck_output_38, l__mod___features_denseblock3_denselayer2_relu1], Original ATen: [aten.convolution, aten.relu]
        buf198 = extern_kernels.convolution(buf197, arg125_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (4, 128, 14, 14), (25088, 196, 14, 1))
        del arg125_1
        del buf197
        buf199 = buf198; del buf198  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer2_norm2, l__mod___features_denseblock3_denselayer2_relu2, new_features_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(buf199, arg490_1, arg491_1, arg126_1, arg127_1, 100352, grid=grid(100352), stream=stream0)
        del arg126_1
        del arg127_1
        del arg490_1
        del arg491_1
        # Source Nodes: [l__mod___features_denseblock3_denselayer2_norm2, l__mod___features_denseblock3_denselayer2_relu2, new_features_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf200 = extern_kernels.convolution(buf199, arg128_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (4, 32, 14, 14), (6272, 196, 14, 1))
        del arg128_1
        del buf199
        buf201 = empty((4, 320, 14, 14), device='cuda', dtype=torch.float32)
        buf202 = buf201; del buf201  # reuse
        # Source Nodes: [bottleneck_output_40, cat_101, l__mod___features_denseblock3_denselayer3_norm1, l__mod___features_denseblock3_denselayer3_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_38.run(buf202, buf191, buf195, buf200, arg493_1, arg494_1, arg129_1, arg130_1, 250880, grid=grid(250880), stream=stream0)
        del arg129_1
        del arg130_1
        del arg493_1
        del arg494_1
        # Source Nodes: [bottleneck_output_40, l__mod___features_denseblock3_denselayer3_norm1, l__mod___features_denseblock3_denselayer3_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf203 = extern_kernels.convolution(buf202, arg131_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (4, 128, 14, 14), (25088, 196, 14, 1))
        del arg131_1
        del buf202
        buf204 = buf203; del buf203  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer3_norm2, l__mod___features_denseblock3_denselayer3_relu2, new_features_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(buf204, arg496_1, arg497_1, arg132_1, arg133_1, 100352, grid=grid(100352), stream=stream0)
        del arg132_1
        del arg133_1
        del arg496_1
        del arg497_1
        # Source Nodes: [l__mod___features_denseblock3_denselayer3_norm2, l__mod___features_denseblock3_denselayer3_relu2, new_features_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf205 = extern_kernels.convolution(buf204, arg134_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf205, (4, 32, 14, 14), (6272, 196, 14, 1))
        del arg134_1
        del buf204
        buf206 = empty((4, 352, 14, 14), device='cuda', dtype=torch.float32)
        buf207 = buf206; del buf206  # reuse
        # Source Nodes: [bottleneck_output_42, cat_100, l__mod___features_denseblock3_denselayer4_norm1, l__mod___features_denseblock3_denselayer4_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_39.run(buf207, buf191, buf195, buf200, buf205, arg499_1, arg500_1, arg135_1, arg136_1, 275968, grid=grid(275968), stream=stream0)
        del arg135_1
        del arg136_1
        del arg499_1
        del arg500_1
        del buf191
        # Source Nodes: [bottleneck_output_42, l__mod___features_denseblock3_denselayer4_norm1, l__mod___features_denseblock3_denselayer4_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf208 = extern_kernels.convolution(buf207, arg137_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf208, (4, 128, 14, 14), (25088, 196, 14, 1))
        del arg137_1
        del buf207
        buf209 = buf208; del buf208  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer4_norm2, l__mod___features_denseblock3_denselayer4_relu2, new_features_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(buf209, arg502_1, arg503_1, arg138_1, arg139_1, 100352, grid=grid(100352), stream=stream0)
        del arg138_1
        del arg139_1
        del arg502_1
        del arg503_1
        # Source Nodes: [l__mod___features_denseblock3_denselayer4_norm2, l__mod___features_denseblock3_denselayer4_relu2, new_features_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf210 = extern_kernels.convolution(buf209, arg140_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf210, (4, 32, 14, 14), (6272, 196, 14, 1))
        del arg140_1
        del buf209
        buf212 = reinterpret_tensor(buf216, (4, 32, 14, 14), (75264, 196, 14, 1), 50176)  # alias
        buf222 = reinterpret_tensor(buf227, (4, 32, 14, 14), (81536, 196, 14, 1), 50176)  # alias
        buf233 = reinterpret_tensor(buf239, (4, 32, 14, 14), (87808, 196, 14, 1), 50176)  # alias
        buf245 = reinterpret_tensor(buf252, (4, 32, 14, 14), (94080, 196, 14, 1), 50176)  # alias
        buf258 = reinterpret_tensor(buf266, (4, 32, 14, 14), (100352, 196, 14, 1), 50176)  # alias
        buf272 = reinterpret_tensor(buf281, (4, 32, 14, 14), (106624, 196, 14, 1), 50176)  # alias
        # Source Nodes: [cat_94, cat_95, cat_96, cat_97, cat_98, cat_99], Original ATen: [aten.cat]
        triton_poi_fused_cat_40.run(buf195, buf212, buf222, buf233, buf245, buf258, buf272, 25088, grid=grid(25088), stream=stream0)
        buf213 = reinterpret_tensor(buf216, (4, 32, 14, 14), (75264, 196, 14, 1), 56448)  # alias
        buf223 = reinterpret_tensor(buf227, (4, 32, 14, 14), (81536, 196, 14, 1), 56448)  # alias
        buf234 = reinterpret_tensor(buf239, (4, 32, 14, 14), (87808, 196, 14, 1), 56448)  # alias
        buf246 = reinterpret_tensor(buf252, (4, 32, 14, 14), (94080, 196, 14, 1), 56448)  # alias
        buf259 = reinterpret_tensor(buf266, (4, 32, 14, 14), (100352, 196, 14, 1), 56448)  # alias
        buf273 = reinterpret_tensor(buf281, (4, 32, 14, 14), (106624, 196, 14, 1), 56448)  # alias
        # Source Nodes: [cat_94, cat_95, cat_96, cat_97, cat_98, cat_99], Original ATen: [aten.cat]
        triton_poi_fused_cat_40.run(buf200, buf213, buf223, buf234, buf246, buf259, buf273, 25088, grid=grid(25088), stream=stream0)
        buf214 = reinterpret_tensor(buf216, (4, 32, 14, 14), (75264, 196, 14, 1), 62720)  # alias
        buf224 = reinterpret_tensor(buf227, (4, 32, 14, 14), (81536, 196, 14, 1), 62720)  # alias
        buf235 = reinterpret_tensor(buf239, (4, 32, 14, 14), (87808, 196, 14, 1), 62720)  # alias
        buf247 = reinterpret_tensor(buf252, (4, 32, 14, 14), (94080, 196, 14, 1), 62720)  # alias
        buf260 = reinterpret_tensor(buf266, (4, 32, 14, 14), (100352, 196, 14, 1), 62720)  # alias
        buf274 = reinterpret_tensor(buf281, (4, 32, 14, 14), (106624, 196, 14, 1), 62720)  # alias
        # Source Nodes: [cat_94, cat_95, cat_96, cat_97, cat_98, cat_99], Original ATen: [aten.cat]
        triton_poi_fused_cat_40.run(buf205, buf214, buf224, buf235, buf247, buf260, buf274, 25088, grid=grid(25088), stream=stream0)
        buf215 = reinterpret_tensor(buf216, (4, 32, 14, 14), (75264, 196, 14, 1), 68992)  # alias
        buf225 = reinterpret_tensor(buf227, (4, 32, 14, 14), (81536, 196, 14, 1), 68992)  # alias
        buf236 = reinterpret_tensor(buf239, (4, 32, 14, 14), (87808, 196, 14, 1), 68992)  # alias
        buf248 = reinterpret_tensor(buf252, (4, 32, 14, 14), (94080, 196, 14, 1), 68992)  # alias
        buf261 = reinterpret_tensor(buf266, (4, 32, 14, 14), (100352, 196, 14, 1), 68992)  # alias
        buf275 = reinterpret_tensor(buf281, (4, 32, 14, 14), (106624, 196, 14, 1), 68992)  # alias
        # Source Nodes: [cat_94, cat_95, cat_96, cat_97, cat_98, cat_99], Original ATen: [aten.cat]
        triton_poi_fused_cat_40.run(buf210, buf215, buf225, buf236, buf248, buf261, buf275, 25088, grid=grid(25088), stream=stream0)
        buf217 = buf216; del buf216  # reuse
        # Source Nodes: [bottleneck_output_44, l__mod___features_denseblock3_denselayer5_norm1, l__mod___features_denseblock3_denselayer5_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_41.run(buf217, arg505_1, arg506_1, arg141_1, arg142_1, 301056, grid=grid(301056), stream=stream0)
        del arg141_1
        del arg142_1
        del arg505_1
        del arg506_1
        del buf211
        del buf212
        del buf213
        del buf214
        del buf215
        # Source Nodes: [bottleneck_output_44, l__mod___features_denseblock3_denselayer5_norm1, l__mod___features_denseblock3_denselayer5_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf218 = extern_kernels.convolution(buf217, arg143_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf218, (4, 128, 14, 14), (25088, 196, 14, 1))
        del arg143_1
        del buf217
        buf219 = buf218; del buf218  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer5_norm2, l__mod___features_denseblock3_denselayer5_relu2, new_features_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(buf219, arg508_1, arg509_1, arg144_1, arg145_1, 100352, grid=grid(100352), stream=stream0)
        del arg144_1
        del arg145_1
        del arg508_1
        del arg509_1
        # Source Nodes: [l__mod___features_denseblock3_denselayer5_norm2, l__mod___features_denseblock3_denselayer5_relu2, new_features_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf220 = extern_kernels.convolution(buf219, arg146_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf220, (4, 32, 14, 14), (6272, 196, 14, 1))
        del arg146_1
        del buf219
        buf226 = reinterpret_tensor(buf227, (4, 32, 14, 14), (81536, 196, 14, 1), 75264)  # alias
        buf237 = reinterpret_tensor(buf239, (4, 32, 14, 14), (87808, 196, 14, 1), 75264)  # alias
        buf249 = reinterpret_tensor(buf252, (4, 32, 14, 14), (94080, 196, 14, 1), 75264)  # alias
        buf262 = reinterpret_tensor(buf266, (4, 32, 14, 14), (100352, 196, 14, 1), 75264)  # alias
        buf276 = reinterpret_tensor(buf281, (4, 32, 14, 14), (106624, 196, 14, 1), 75264)  # alias
        # Source Nodes: [cat_94, cat_95, cat_96, cat_97, cat_98], Original ATen: [aten.cat]
        triton_poi_fused_cat_42.run(buf220, buf226, buf237, buf249, buf262, buf276, 25088, grid=grid(25088), stream=stream0)
        buf228 = buf227; del buf227  # reuse
        # Source Nodes: [bottleneck_output_46, l__mod___features_denseblock3_denselayer6_norm1, l__mod___features_denseblock3_denselayer6_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_43.run(buf228, arg511_1, arg512_1, arg147_1, arg148_1, 326144, grid=grid(326144), stream=stream0)
        del arg147_1
        del arg148_1
        del arg511_1
        del arg512_1
        del buf221
        del buf222
        del buf223
        del buf224
        del buf225
        del buf226
        # Source Nodes: [bottleneck_output_46, l__mod___features_denseblock3_denselayer6_norm1, l__mod___features_denseblock3_denselayer6_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf229 = extern_kernels.convolution(buf228, arg149_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf229, (4, 128, 14, 14), (25088, 196, 14, 1))
        del arg149_1
        del buf228
        buf230 = buf229; del buf229  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer6_norm2, l__mod___features_denseblock3_denselayer6_relu2, new_features_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(buf230, arg514_1, arg515_1, arg150_1, arg151_1, 100352, grid=grid(100352), stream=stream0)
        del arg150_1
        del arg151_1
        del arg514_1
        del arg515_1
        # Source Nodes: [l__mod___features_denseblock3_denselayer6_norm2, l__mod___features_denseblock3_denselayer6_relu2, new_features_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf231 = extern_kernels.convolution(buf230, arg152_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf231, (4, 32, 14, 14), (6272, 196, 14, 1))
        del arg152_1
        del buf230
        buf238 = reinterpret_tensor(buf239, (4, 32, 14, 14), (87808, 196, 14, 1), 81536)  # alias
        buf250 = reinterpret_tensor(buf252, (4, 32, 14, 14), (94080, 196, 14, 1), 81536)  # alias
        buf263 = reinterpret_tensor(buf266, (4, 32, 14, 14), (100352, 196, 14, 1), 81536)  # alias
        buf277 = reinterpret_tensor(buf281, (4, 32, 14, 14), (106624, 196, 14, 1), 81536)  # alias
        buf292 = reinterpret_tensor(buf297, (4, 32, 14, 14), (112896, 196, 14, 1), 81536)  # alias
        # Source Nodes: [cat_93, cat_94, cat_95, cat_96, cat_97], Original ATen: [aten.cat]
        triton_poi_fused_cat_44.run(buf231, buf238, buf250, buf263, buf277, buf292, 25088, grid=grid(25088), stream=stream0)
        buf240 = buf239; del buf239  # reuse
        # Source Nodes: [bottleneck_output_48, l__mod___features_denseblock3_denselayer7_norm1, l__mod___features_denseblock3_denselayer7_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_45.run(buf240, arg517_1, arg518_1, arg153_1, arg154_1, 351232, grid=grid(351232), stream=stream0)
        del arg153_1
        del arg154_1
        del arg517_1
        del arg518_1
        del buf232
        del buf233
        del buf234
        del buf235
        del buf236
        del buf237
        del buf238
        # Source Nodes: [bottleneck_output_48, l__mod___features_denseblock3_denselayer7_norm1, l__mod___features_denseblock3_denselayer7_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf241 = extern_kernels.convolution(buf240, arg155_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (4, 128, 14, 14), (25088, 196, 14, 1))
        del arg155_1
        del buf240
        buf242 = buf241; del buf241  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer7_norm2, l__mod___features_denseblock3_denselayer7_relu2, new_features_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(buf242, arg520_1, arg521_1, arg156_1, arg157_1, 100352, grid=grid(100352), stream=stream0)
        del arg156_1
        del arg157_1
        del arg520_1
        del arg521_1
        # Source Nodes: [l__mod___features_denseblock3_denselayer7_norm2, l__mod___features_denseblock3_denselayer7_relu2, new_features_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf243 = extern_kernels.convolution(buf242, arg158_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf243, (4, 32, 14, 14), (6272, 196, 14, 1))
        del arg158_1
        del buf242
        buf251 = reinterpret_tensor(buf252, (4, 32, 14, 14), (94080, 196, 14, 1), 87808)  # alias
        buf264 = reinterpret_tensor(buf266, (4, 32, 14, 14), (100352, 196, 14, 1), 87808)  # alias
        buf278 = reinterpret_tensor(buf281, (4, 32, 14, 14), (106624, 196, 14, 1), 87808)  # alias
        buf293 = reinterpret_tensor(buf297, (4, 32, 14, 14), (112896, 196, 14, 1), 87808)  # alias
        buf309 = reinterpret_tensor(buf314, (4, 32, 14, 14), (119168, 196, 14, 1), 87808)  # alias
        # Source Nodes: [cat_92, cat_93, cat_94, cat_95, cat_96], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf243, buf251, buf264, buf278, buf293, buf309, 25088, grid=grid(25088), stream=stream0)
        buf253 = buf252; del buf252  # reuse
        # Source Nodes: [bottleneck_output_50, l__mod___features_denseblock3_denselayer8_norm1, l__mod___features_denseblock3_denselayer8_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_47.run(buf253, arg523_1, arg524_1, arg159_1, arg160_1, 376320, grid=grid(376320), stream=stream0)
        del arg159_1
        del arg160_1
        del arg523_1
        del arg524_1
        del buf244
        del buf245
        del buf246
        del buf247
        del buf248
        del buf249
        del buf250
        del buf251
        # Source Nodes: [bottleneck_output_50, l__mod___features_denseblock3_denselayer8_norm1, l__mod___features_denseblock3_denselayer8_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf254 = extern_kernels.convolution(buf253, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (4, 128, 14, 14), (25088, 196, 14, 1))
        del arg161_1
        del buf253
        buf255 = buf254; del buf254  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer8_norm2, l__mod___features_denseblock3_denselayer8_relu2, new_features_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(buf255, arg526_1, arg527_1, arg162_1, arg163_1, 100352, grid=grid(100352), stream=stream0)
        del arg162_1
        del arg163_1
        del arg526_1
        del arg527_1
        # Source Nodes: [l__mod___features_denseblock3_denselayer8_norm2, l__mod___features_denseblock3_denselayer8_relu2, new_features_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf256 = extern_kernels.convolution(buf255, arg164_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf256, (4, 32, 14, 14), (6272, 196, 14, 1))
        del arg164_1
        del buf255
        buf265 = reinterpret_tensor(buf266, (4, 32, 14, 14), (100352, 196, 14, 1), 94080)  # alias
        buf279 = reinterpret_tensor(buf281, (4, 32, 14, 14), (106624, 196, 14, 1), 94080)  # alias
        buf294 = reinterpret_tensor(buf297, (4, 32, 14, 14), (112896, 196, 14, 1), 94080)  # alias
        buf310 = reinterpret_tensor(buf314, (4, 32, 14, 14), (119168, 196, 14, 1), 94080)  # alias
        buf327 = reinterpret_tensor(buf332, (4, 32, 14, 14), (125440, 196, 14, 1), 94080)  # alias
        # Source Nodes: [cat_91, cat_92, cat_93, cat_94, cat_95], Original ATen: [aten.cat]
        triton_poi_fused_cat_48.run(buf256, buf265, buf279, buf294, buf310, buf327, 25088, grid=grid(25088), stream=stream0)
        buf267 = buf266; del buf266  # reuse
        # Source Nodes: [bottleneck_output_52, l__mod___features_denseblock3_denselayer9_norm1, l__mod___features_denseblock3_denselayer9_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_49.run(buf267, arg529_1, arg530_1, arg165_1, arg166_1, 401408, grid=grid(401408), stream=stream0)
        del arg165_1
        del arg166_1
        del arg529_1
        del arg530_1
        del buf257
        del buf258
        del buf259
        del buf260
        del buf261
        del buf262
        del buf263
        del buf264
        del buf265
        # Source Nodes: [bottleneck_output_52, l__mod___features_denseblock3_denselayer9_norm1, l__mod___features_denseblock3_denselayer9_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf268 = extern_kernels.convolution(buf267, arg167_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf268, (4, 128, 14, 14), (25088, 196, 14, 1))
        del arg167_1
        del buf267
        buf269 = buf268; del buf268  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer9_norm2, l__mod___features_denseblock3_denselayer9_relu2, new_features_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(buf269, arg532_1, arg533_1, arg168_1, arg169_1, 100352, grid=grid(100352), stream=stream0)
        del arg168_1
        del arg169_1
        del arg532_1
        del arg533_1
        # Source Nodes: [l__mod___features_denseblock3_denselayer9_norm2, l__mod___features_denseblock3_denselayer9_relu2, new_features_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf270 = extern_kernels.convolution(buf269, arg170_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf270, (4, 32, 14, 14), (6272, 196, 14, 1))
        del arg170_1
        del buf269
        buf280 = reinterpret_tensor(buf281, (4, 32, 14, 14), (106624, 196, 14, 1), 100352)  # alias
        buf295 = reinterpret_tensor(buf297, (4, 32, 14, 14), (112896, 196, 14, 1), 100352)  # alias
        buf311 = reinterpret_tensor(buf314, (4, 32, 14, 14), (119168, 196, 14, 1), 100352)  # alias
        buf328 = reinterpret_tensor(buf332, (4, 32, 14, 14), (125440, 196, 14, 1), 100352)  # alias
        # Source Nodes: [cat_91, cat_92, cat_93, cat_94], Original ATen: [aten.cat]
        triton_poi_fused_cat_50.run(buf270, buf280, buf295, buf311, buf328, 25088, grid=grid(25088), stream=stream0)
        buf282 = buf281; del buf281  # reuse
        # Source Nodes: [bottleneck_output_54, l__mod___features_denseblock3_denselayer10_norm1, l__mod___features_denseblock3_denselayer10_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_51.run(buf282, arg535_1, arg536_1, arg171_1, arg172_1, 426496, grid=grid(426496), stream=stream0)
        del arg171_1
        del arg172_1
        del arg535_1
        del arg536_1
        del buf271
        del buf272
        del buf273
        del buf274
        del buf275
        del buf276
        del buf277
        del buf278
        del buf279
        del buf280
        # Source Nodes: [bottleneck_output_54, l__mod___features_denseblock3_denselayer10_norm1, l__mod___features_denseblock3_denselayer10_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf283 = extern_kernels.convolution(buf282, arg173_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf283, (4, 128, 14, 14), (25088, 196, 14, 1))
        del arg173_1
        del buf282
        buf284 = buf283; del buf283  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer10_norm2, l__mod___features_denseblock3_denselayer10_relu2, new_features_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(buf284, arg538_1, arg539_1, arg174_1, arg175_1, 100352, grid=grid(100352), stream=stream0)
        del arg174_1
        del arg175_1
        del arg538_1
        del arg539_1
        # Source Nodes: [l__mod___features_denseblock3_denselayer10_norm2, l__mod___features_denseblock3_denselayer10_relu2, new_features_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf285 = extern_kernels.convolution(buf284, arg176_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf285, (4, 32, 14, 14), (6272, 196, 14, 1))
        del arg176_1
        del buf284
        buf287 = reinterpret_tensor(buf297, (4, 32, 14, 14), (112896, 196, 14, 1), 50176)  # alias
        buf303 = reinterpret_tensor(buf314, (4, 32, 14, 14), (119168, 196, 14, 1), 50176)  # alias
        buf320 = reinterpret_tensor(buf332, (4, 32, 14, 14), (125440, 196, 14, 1), 50176)  # alias
        buf338 = reinterpret_tensor(buf351, (4, 32, 14, 14), (131712, 196, 14, 1), 50176)  # alias
        # Source Nodes: [cat_90, cat_91, cat_92, cat_93], Original ATen: [aten.cat]
        triton_poi_fused_cat_52.run(buf195, buf287, buf303, buf320, buf338, 25088, grid=grid(25088), stream=stream0)
        buf288 = reinterpret_tensor(buf297, (4, 32, 14, 14), (112896, 196, 14, 1), 56448)  # alias
        buf304 = reinterpret_tensor(buf314, (4, 32, 14, 14), (119168, 196, 14, 1), 56448)  # alias
        buf321 = reinterpret_tensor(buf332, (4, 32, 14, 14), (125440, 196, 14, 1), 56448)  # alias
        buf339 = reinterpret_tensor(buf351, (4, 32, 14, 14), (131712, 196, 14, 1), 56448)  # alias
        # Source Nodes: [cat_90, cat_91, cat_92, cat_93], Original ATen: [aten.cat]
        triton_poi_fused_cat_52.run(buf200, buf288, buf304, buf321, buf339, 25088, grid=grid(25088), stream=stream0)
        buf289 = reinterpret_tensor(buf297, (4, 32, 14, 14), (112896, 196, 14, 1), 62720)  # alias
        buf305 = reinterpret_tensor(buf314, (4, 32, 14, 14), (119168, 196, 14, 1), 62720)  # alias
        buf322 = reinterpret_tensor(buf332, (4, 32, 14, 14), (125440, 196, 14, 1), 62720)  # alias
        buf340 = reinterpret_tensor(buf351, (4, 32, 14, 14), (131712, 196, 14, 1), 62720)  # alias
        # Source Nodes: [cat_90, cat_91, cat_92, cat_93], Original ATen: [aten.cat]
        triton_poi_fused_cat_52.run(buf205, buf289, buf305, buf322, buf340, 25088, grid=grid(25088), stream=stream0)
        buf290 = reinterpret_tensor(buf297, (4, 32, 14, 14), (112896, 196, 14, 1), 68992)  # alias
        buf306 = reinterpret_tensor(buf314, (4, 32, 14, 14), (119168, 196, 14, 1), 68992)  # alias
        buf323 = reinterpret_tensor(buf332, (4, 32, 14, 14), (125440, 196, 14, 1), 68992)  # alias
        buf341 = reinterpret_tensor(buf351, (4, 32, 14, 14), (131712, 196, 14, 1), 68992)  # alias
        # Source Nodes: [cat_90, cat_91, cat_92, cat_93], Original ATen: [aten.cat]
        triton_poi_fused_cat_52.run(buf210, buf290, buf306, buf323, buf341, 25088, grid=grid(25088), stream=stream0)
        buf291 = reinterpret_tensor(buf297, (4, 32, 14, 14), (112896, 196, 14, 1), 75264)  # alias
        buf307 = reinterpret_tensor(buf314, (4, 32, 14, 14), (119168, 196, 14, 1), 75264)  # alias
        buf324 = reinterpret_tensor(buf332, (4, 32, 14, 14), (125440, 196, 14, 1), 75264)  # alias
        buf342 = reinterpret_tensor(buf351, (4, 32, 14, 14), (131712, 196, 14, 1), 75264)  # alias
        # Source Nodes: [cat_90, cat_91, cat_92, cat_93], Original ATen: [aten.cat]
        triton_poi_fused_cat_52.run(buf220, buf291, buf307, buf324, buf342, 25088, grid=grid(25088), stream=stream0)
        buf296 = reinterpret_tensor(buf297, (4, 32, 14, 14), (112896, 196, 14, 1), 106624)  # alias
        buf312 = reinterpret_tensor(buf314, (4, 32, 14, 14), (119168, 196, 14, 1), 106624)  # alias
        buf329 = reinterpret_tensor(buf332, (4, 32, 14, 14), (125440, 196, 14, 1), 106624)  # alias
        buf347 = reinterpret_tensor(buf351, (4, 32, 14, 14), (131712, 196, 14, 1), 106624)  # alias
        # Source Nodes: [cat_90, cat_91, cat_92, cat_93], Original ATen: [aten.cat]
        triton_poi_fused_cat_52.run(buf285, buf296, buf312, buf329, buf347, 25088, grid=grid(25088), stream=stream0)
        buf298 = buf297; del buf297  # reuse
        # Source Nodes: [bottleneck_output_56, l__mod___features_denseblock3_denselayer11_norm1, l__mod___features_denseblock3_denselayer11_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_53.run(buf298, arg541_1, arg542_1, arg177_1, arg178_1, 451584, grid=grid(451584), stream=stream0)
        del arg177_1
        del arg178_1
        del arg541_1
        del arg542_1
        del buf286
        del buf287
        del buf288
        del buf289
        del buf290
        del buf291
        del buf292
        del buf293
        del buf294
        del buf295
        del buf296
        # Source Nodes: [bottleneck_output_56, l__mod___features_denseblock3_denselayer11_norm1, l__mod___features_denseblock3_denselayer11_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf299 = extern_kernels.convolution(buf298, arg179_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf299, (4, 128, 14, 14), (25088, 196, 14, 1))
        del arg179_1
        del buf298
        buf300 = buf299; del buf299  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer11_norm2, l__mod___features_denseblock3_denselayer11_relu2, new_features_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(buf300, arg544_1, arg545_1, arg180_1, arg181_1, 100352, grid=grid(100352), stream=stream0)
        del arg180_1
        del arg181_1
        del arg544_1
        del arg545_1
        # Source Nodes: [l__mod___features_denseblock3_denselayer11_norm2, l__mod___features_denseblock3_denselayer11_relu2, new_features_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf301 = extern_kernels.convolution(buf300, arg182_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf301, (4, 32, 14, 14), (6272, 196, 14, 1))
        del arg182_1
        del buf300
        buf308 = reinterpret_tensor(buf314, (4, 32, 14, 14), (119168, 196, 14, 1), 81536)  # alias
        buf325 = reinterpret_tensor(buf332, (4, 32, 14, 14), (125440, 196, 14, 1), 81536)  # alias
        buf343 = reinterpret_tensor(buf351, (4, 32, 14, 14), (131712, 196, 14, 1), 81536)  # alias
        buf362 = reinterpret_tensor(buf371, (4, 32, 14, 14), (137984, 196, 14, 1), 81536)  # alias
        # Source Nodes: [cat_89, cat_90, cat_91, cat_92], Original ATen: [aten.cat]
        triton_poi_fused_cat_54.run(buf231, buf308, buf325, buf343, buf362, 25088, grid=grid(25088), stream=stream0)
        buf313 = reinterpret_tensor(buf314, (4, 32, 14, 14), (119168, 196, 14, 1), 112896)  # alias
        buf330 = reinterpret_tensor(buf332, (4, 32, 14, 14), (125440, 196, 14, 1), 112896)  # alias
        buf348 = reinterpret_tensor(buf351, (4, 32, 14, 14), (131712, 196, 14, 1), 112896)  # alias
        buf367 = reinterpret_tensor(buf371, (4, 32, 14, 14), (137984, 196, 14, 1), 112896)  # alias
        # Source Nodes: [cat_89, cat_90, cat_91, cat_92], Original ATen: [aten.cat]
        triton_poi_fused_cat_54.run(buf301, buf313, buf330, buf348, buf367, 25088, grid=grid(25088), stream=stream0)
        buf315 = buf314; del buf314  # reuse
        # Source Nodes: [bottleneck_output_58, l__mod___features_denseblock3_denselayer12_norm1, l__mod___features_denseblock3_denselayer12_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_55.run(buf315, arg547_1, arg548_1, arg183_1, arg184_1, 476672, grid=grid(476672), stream=stream0)
        del arg183_1
        del arg184_1
        del arg547_1
        del arg548_1
        del buf302
        del buf303
        del buf304
        del buf305
        del buf306
        del buf307
        del buf308
        del buf309
        del buf310
        del buf311
        del buf312
        del buf313
        # Source Nodes: [bottleneck_output_58, l__mod___features_denseblock3_denselayer12_norm1, l__mod___features_denseblock3_denselayer12_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf316 = extern_kernels.convolution(buf315, arg185_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf316, (4, 128, 14, 14), (25088, 196, 14, 1))
        del arg185_1
        del buf315
        buf317 = buf316; del buf316  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer12_norm2, l__mod___features_denseblock3_denselayer12_relu2, new_features_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(buf317, arg550_1, arg551_1, arg186_1, arg187_1, 100352, grid=grid(100352), stream=stream0)
        del arg186_1
        del arg187_1
        del arg550_1
        del arg551_1
        # Source Nodes: [l__mod___features_denseblock3_denselayer12_norm2, l__mod___features_denseblock3_denselayer12_relu2, new_features_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf318 = extern_kernels.convolution(buf317, arg188_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf318, (4, 32, 14, 14), (6272, 196, 14, 1))
        del arg188_1
        del buf317
        buf326 = reinterpret_tensor(buf332, (4, 32, 14, 14), (125440, 196, 14, 1), 87808)  # alias
        buf344 = reinterpret_tensor(buf351, (4, 32, 14, 14), (131712, 196, 14, 1), 87808)  # alias
        buf363 = reinterpret_tensor(buf371, (4, 32, 14, 14), (137984, 196, 14, 1), 87808)  # alias
        buf383 = reinterpret_tensor(buf392, (4, 32, 14, 14), (144256, 196, 14, 1), 87808)  # alias
        # Source Nodes: [cat_88, cat_89, cat_90, cat_91], Original ATen: [aten.cat]
        triton_poi_fused_cat_56.run(buf243, buf326, buf344, buf363, buf383, 25088, grid=grid(25088), stream=stream0)
        buf331 = reinterpret_tensor(buf332, (4, 32, 14, 14), (125440, 196, 14, 1), 119168)  # alias
        buf349 = reinterpret_tensor(buf351, (4, 32, 14, 14), (131712, 196, 14, 1), 119168)  # alias
        buf368 = reinterpret_tensor(buf371, (4, 32, 14, 14), (137984, 196, 14, 1), 119168)  # alias
        buf388 = reinterpret_tensor(buf392, (4, 32, 14, 14), (144256, 196, 14, 1), 119168)  # alias
        # Source Nodes: [cat_88, cat_89, cat_90, cat_91], Original ATen: [aten.cat]
        triton_poi_fused_cat_56.run(buf318, buf331, buf349, buf368, buf388, 25088, grid=grid(25088), stream=stream0)
        buf333 = buf332; del buf332  # reuse
        # Source Nodes: [bottleneck_output_60, l__mod___features_denseblock3_denselayer13_norm1, l__mod___features_denseblock3_denselayer13_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_57.run(buf333, arg553_1, arg554_1, arg189_1, arg190_1, 501760, grid=grid(501760), stream=stream0)
        del arg189_1
        del arg190_1
        del arg553_1
        del arg554_1
        del buf319
        del buf320
        del buf321
        del buf322
        del buf323
        del buf324
        del buf325
        del buf326
        del buf327
        del buf328
        del buf329
        del buf330
        del buf331
        # Source Nodes: [bottleneck_output_60, l__mod___features_denseblock3_denselayer13_norm1, l__mod___features_denseblock3_denselayer13_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf334 = extern_kernels.convolution(buf333, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf334, (4, 128, 14, 14), (25088, 196, 14, 1))
        del arg191_1
        del buf333
        buf335 = buf334; del buf334  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer13_norm2, l__mod___features_denseblock3_denselayer13_relu2, new_features_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(buf335, arg556_1, arg557_1, arg192_1, arg193_1, 100352, grid=grid(100352), stream=stream0)
        del arg192_1
        del arg193_1
        del arg556_1
        del arg557_1
        # Source Nodes: [l__mod___features_denseblock3_denselayer13_norm2, l__mod___features_denseblock3_denselayer13_relu2, new_features_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf336 = extern_kernels.convolution(buf335, arg194_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf336, (4, 32, 14, 14), (6272, 196, 14, 1))
        del arg194_1
        del buf335
        buf345 = reinterpret_tensor(buf351, (4, 32, 14, 14), (131712, 196, 14, 1), 94080)  # alias
        buf364 = reinterpret_tensor(buf371, (4, 32, 14, 14), (137984, 196, 14, 1), 94080)  # alias
        buf384 = reinterpret_tensor(buf392, (4, 32, 14, 14), (144256, 196, 14, 1), 94080)  # alias
        buf405 = reinterpret_tensor(buf414, (4, 32, 14, 14), (150528, 196, 14, 1), 94080)  # alias
        # Source Nodes: [cat_87, cat_88, cat_89, cat_90], Original ATen: [aten.cat]
        triton_poi_fused_cat_58.run(buf256, buf345, buf364, buf384, buf405, 25088, grid=grid(25088), stream=stream0)
        buf346 = reinterpret_tensor(buf351, (4, 32, 14, 14), (131712, 196, 14, 1), 100352)  # alias
        buf365 = reinterpret_tensor(buf371, (4, 32, 14, 14), (137984, 196, 14, 1), 100352)  # alias
        buf385 = reinterpret_tensor(buf392, (4, 32, 14, 14), (144256, 196, 14, 1), 100352)  # alias
        buf406 = reinterpret_tensor(buf414, (4, 32, 14, 14), (150528, 196, 14, 1), 100352)  # alias
        # Source Nodes: [cat_87, cat_88, cat_89, cat_90], Original ATen: [aten.cat]
        triton_poi_fused_cat_58.run(buf270, buf346, buf365, buf385, buf406, 25088, grid=grid(25088), stream=stream0)
        buf350 = reinterpret_tensor(buf351, (4, 32, 14, 14), (131712, 196, 14, 1), 125440)  # alias
        buf369 = reinterpret_tensor(buf371, (4, 32, 14, 14), (137984, 196, 14, 1), 125440)  # alias
        buf389 = reinterpret_tensor(buf392, (4, 32, 14, 14), (144256, 196, 14, 1), 125440)  # alias
        buf410 = reinterpret_tensor(buf414, (4, 32, 14, 14), (150528, 196, 14, 1), 125440)  # alias
        # Source Nodes: [cat_87, cat_88, cat_89, cat_90], Original ATen: [aten.cat]
        triton_poi_fused_cat_58.run(buf336, buf350, buf369, buf389, buf410, 25088, grid=grid(25088), stream=stream0)
        buf352 = buf351; del buf351  # reuse
        # Source Nodes: [bottleneck_output_62, l__mod___features_denseblock3_denselayer14_norm1, l__mod___features_denseblock3_denselayer14_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_59.run(buf352, arg559_1, arg560_1, arg195_1, arg196_1, 526848, grid=grid(526848), stream=stream0)
        del arg195_1
        del arg196_1
        del arg559_1
        del arg560_1
        del buf337
        del buf338
        del buf339
        del buf340
        del buf341
        del buf342
        del buf343
        del buf344
        del buf345
        del buf346
        del buf347
        del buf348
        del buf349
        del buf350
        # Source Nodes: [bottleneck_output_62, l__mod___features_denseblock3_denselayer14_norm1, l__mod___features_denseblock3_denselayer14_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf353 = extern_kernels.convolution(buf352, arg197_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf353, (4, 128, 14, 14), (25088, 196, 14, 1))
        del arg197_1
        del buf352
        buf354 = buf353; del buf353  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer14_norm2, l__mod___features_denseblock3_denselayer14_relu2, new_features_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(buf354, arg562_1, arg563_1, arg198_1, arg199_1, 100352, grid=grid(100352), stream=stream0)
        del arg198_1
        del arg199_1
        del arg562_1
        del arg563_1
        # Source Nodes: [l__mod___features_denseblock3_denselayer14_norm2, l__mod___features_denseblock3_denselayer14_relu2, new_features_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf355 = extern_kernels.convolution(buf354, arg200_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf355, (4, 32, 14, 14), (6272, 196, 14, 1))
        del arg200_1
        del buf354
        buf357 = reinterpret_tensor(buf371, (4, 32, 14, 14), (137984, 196, 14, 1), 50176)  # alias
        buf377 = reinterpret_tensor(buf392, (4, 32, 14, 14), (144256, 196, 14, 1), 50176)  # alias
        buf398 = reinterpret_tensor(buf414, (4, 32, 14, 14), (150528, 196, 14, 1), 50176)  # alias
        buf420 = reinterpret_tensor(buf437, (4, 32, 14, 14), (156800, 196, 14, 1), 50176)  # alias
        # Source Nodes: [cat_86, cat_87, cat_88, cat_89], Original ATen: [aten.cat]
        triton_poi_fused_cat_60.run(buf195, buf357, buf377, buf398, buf420, 25088, grid=grid(25088), stream=stream0)
        buf358 = reinterpret_tensor(buf371, (4, 32, 14, 14), (137984, 196, 14, 1), 56448)  # alias
        buf378 = reinterpret_tensor(buf392, (4, 32, 14, 14), (144256, 196, 14, 1), 56448)  # alias
        buf399 = reinterpret_tensor(buf414, (4, 32, 14, 14), (150528, 196, 14, 1), 56448)  # alias
        buf421 = reinterpret_tensor(buf437, (4, 32, 14, 14), (156800, 196, 14, 1), 56448)  # alias
        # Source Nodes: [cat_86, cat_87, cat_88, cat_89], Original ATen: [aten.cat]
        triton_poi_fused_cat_60.run(buf200, buf358, buf378, buf399, buf421, 25088, grid=grid(25088), stream=stream0)
        buf359 = reinterpret_tensor(buf371, (4, 32, 14, 14), (137984, 196, 14, 1), 62720)  # alias
        buf379 = reinterpret_tensor(buf392, (4, 32, 14, 14), (144256, 196, 14, 1), 62720)  # alias
        buf400 = reinterpret_tensor(buf414, (4, 32, 14, 14), (150528, 196, 14, 1), 62720)  # alias
        buf422 = reinterpret_tensor(buf437, (4, 32, 14, 14), (156800, 196, 14, 1), 62720)  # alias
        # Source Nodes: [cat_86, cat_87, cat_88, cat_89], Original ATen: [aten.cat]
        triton_poi_fused_cat_60.run(buf205, buf359, buf379, buf400, buf422, 25088, grid=grid(25088), stream=stream0)
        buf360 = reinterpret_tensor(buf371, (4, 32, 14, 14), (137984, 196, 14, 1), 68992)  # alias
        buf380 = reinterpret_tensor(buf392, (4, 32, 14, 14), (144256, 196, 14, 1), 68992)  # alias
        buf401 = reinterpret_tensor(buf414, (4, 32, 14, 14), (150528, 196, 14, 1), 68992)  # alias
        buf423 = reinterpret_tensor(buf437, (4, 32, 14, 14), (156800, 196, 14, 1), 68992)  # alias
        # Source Nodes: [cat_86, cat_87, cat_88, cat_89], Original ATen: [aten.cat]
        triton_poi_fused_cat_60.run(buf210, buf360, buf380, buf401, buf423, 25088, grid=grid(25088), stream=stream0)
        buf361 = reinterpret_tensor(buf371, (4, 32, 14, 14), (137984, 196, 14, 1), 75264)  # alias
        buf381 = reinterpret_tensor(buf392, (4, 32, 14, 14), (144256, 196, 14, 1), 75264)  # alias
        buf402 = reinterpret_tensor(buf414, (4, 32, 14, 14), (150528, 196, 14, 1), 75264)  # alias
        buf424 = reinterpret_tensor(buf437, (4, 32, 14, 14), (156800, 196, 14, 1), 75264)  # alias
        # Source Nodes: [cat_86, cat_87, cat_88, cat_89], Original ATen: [aten.cat]
        triton_poi_fused_cat_60.run(buf220, buf361, buf381, buf402, buf424, 25088, grid=grid(25088), stream=stream0)
        buf366 = reinterpret_tensor(buf371, (4, 32, 14, 14), (137984, 196, 14, 1), 106624)  # alias
        buf386 = reinterpret_tensor(buf392, (4, 32, 14, 14), (144256, 196, 14, 1), 106624)  # alias
        buf407 = reinterpret_tensor(buf414, (4, 32, 14, 14), (150528, 196, 14, 1), 106624)  # alias
        buf429 = reinterpret_tensor(buf437, (4, 32, 14, 14), (156800, 196, 14, 1), 106624)  # alias
        # Source Nodes: [cat_86, cat_87, cat_88, cat_89], Original ATen: [aten.cat]
        triton_poi_fused_cat_60.run(buf285, buf366, buf386, buf407, buf429, 25088, grid=grid(25088), stream=stream0)
        buf370 = reinterpret_tensor(buf371, (4, 32, 14, 14), (137984, 196, 14, 1), 131712)  # alias
        buf390 = reinterpret_tensor(buf392, (4, 32, 14, 14), (144256, 196, 14, 1), 131712)  # alias
        buf411 = reinterpret_tensor(buf414, (4, 32, 14, 14), (150528, 196, 14, 1), 131712)  # alias
        buf433 = reinterpret_tensor(buf437, (4, 32, 14, 14), (156800, 196, 14, 1), 131712)  # alias
        # Source Nodes: [cat_86, cat_87, cat_88, cat_89], Original ATen: [aten.cat]
        triton_poi_fused_cat_60.run(buf355, buf370, buf390, buf411, buf433, 25088, grid=grid(25088), stream=stream0)
        buf372 = buf371; del buf371  # reuse
        # Source Nodes: [bottleneck_output_64, l__mod___features_denseblock3_denselayer15_norm1, l__mod___features_denseblock3_denselayer15_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_61.run(buf372, arg565_1, arg566_1, arg201_1, arg202_1, 551936, grid=grid(551936), stream=stream0)
        del arg201_1
        del arg202_1
        del arg565_1
        del arg566_1
        del buf356
        del buf357
        del buf358
        del buf359
        del buf360
        del buf361
        del buf362
        del buf363
        del buf364
        del buf365
        del buf366
        del buf367
        del buf368
        del buf369
        del buf370
        # Source Nodes: [bottleneck_output_64, l__mod___features_denseblock3_denselayer15_norm1, l__mod___features_denseblock3_denselayer15_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf373 = extern_kernels.convolution(buf372, arg203_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf373, (4, 128, 14, 14), (25088, 196, 14, 1))
        del arg203_1
        del buf372
        buf374 = buf373; del buf373  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer15_norm2, l__mod___features_denseblock3_denselayer15_relu2, new_features_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(buf374, arg568_1, arg569_1, arg204_1, arg205_1, 100352, grid=grid(100352), stream=stream0)
        del arg204_1
        del arg205_1
        del arg568_1
        del arg569_1
        # Source Nodes: [l__mod___features_denseblock3_denselayer15_norm2, l__mod___features_denseblock3_denselayer15_relu2, new_features_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf375 = extern_kernels.convolution(buf374, arg206_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf375, (4, 32, 14, 14), (6272, 196, 14, 1))
        del arg206_1
        del buf374
        buf382 = reinterpret_tensor(buf392, (4, 32, 14, 14), (144256, 196, 14, 1), 81536)  # alias
        buf403 = reinterpret_tensor(buf414, (4, 32, 14, 14), (150528, 196, 14, 1), 81536)  # alias
        buf425 = reinterpret_tensor(buf437, (4, 32, 14, 14), (156800, 196, 14, 1), 81536)  # alias
        # Source Nodes: [cat_86, cat_87, cat_88], Original ATen: [aten.cat]
        triton_poi_fused_cat_62.run(buf231, buf382, buf403, buf425, 25088, grid=grid(25088), stream=stream0)
        buf387 = reinterpret_tensor(buf392, (4, 32, 14, 14), (144256, 196, 14, 1), 112896)  # alias
        buf408 = reinterpret_tensor(buf414, (4, 32, 14, 14), (150528, 196, 14, 1), 112896)  # alias
        buf430 = reinterpret_tensor(buf437, (4, 32, 14, 14), (156800, 196, 14, 1), 112896)  # alias
        # Source Nodes: [cat_86, cat_87, cat_88], Original ATen: [aten.cat]
        triton_poi_fused_cat_62.run(buf301, buf387, buf408, buf430, 25088, grid=grid(25088), stream=stream0)
        buf391 = reinterpret_tensor(buf392, (4, 32, 14, 14), (144256, 196, 14, 1), 137984)  # alias
        buf412 = reinterpret_tensor(buf414, (4, 32, 14, 14), (150528, 196, 14, 1), 137984)  # alias
        buf434 = reinterpret_tensor(buf437, (4, 32, 14, 14), (156800, 196, 14, 1), 137984)  # alias
        # Source Nodes: [cat_86, cat_87, cat_88], Original ATen: [aten.cat]
        triton_poi_fused_cat_62.run(buf375, buf391, buf412, buf434, 25088, grid=grid(25088), stream=stream0)
        buf393 = buf392; del buf392  # reuse
        # Source Nodes: [bottleneck_output_66, l__mod___features_denseblock3_denselayer16_norm1, l__mod___features_denseblock3_denselayer16_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_63.run(buf393, arg571_1, arg572_1, arg207_1, arg208_1, 577024, grid=grid(577024), stream=stream0)
        del arg207_1
        del arg208_1
        del arg571_1
        del arg572_1
        del buf376
        del buf377
        del buf378
        del buf379
        del buf380
        del buf381
        del buf382
        del buf383
        del buf384
        del buf385
        del buf386
        del buf387
        del buf388
        del buf389
        del buf390
        del buf391
        # Source Nodes: [bottleneck_output_66, l__mod___features_denseblock3_denselayer16_norm1, l__mod___features_denseblock3_denselayer16_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf394 = extern_kernels.convolution(buf393, arg209_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf394, (4, 128, 14, 14), (25088, 196, 14, 1))
        del arg209_1
        del buf393
        buf395 = buf394; del buf394  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer16_norm2, l__mod___features_denseblock3_denselayer16_relu2, new_features_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(buf395, arg574_1, arg575_1, arg210_1, arg211_1, 100352, grid=grid(100352), stream=stream0)
        del arg210_1
        del arg211_1
        del arg574_1
        del arg575_1
        # Source Nodes: [l__mod___features_denseblock3_denselayer16_norm2, l__mod___features_denseblock3_denselayer16_relu2, new_features_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf396 = extern_kernels.convolution(buf395, arg212_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf396, (4, 32, 14, 14), (6272, 196, 14, 1))
        del arg212_1
        del buf395
        buf404 = reinterpret_tensor(buf414, (4, 32, 14, 14), (150528, 196, 14, 1), 87808)  # alias
        buf426 = reinterpret_tensor(buf437, (4, 32, 14, 14), (156800, 196, 14, 1), 87808)  # alias
        buf449 = reinterpret_tensor(buf461, (4, 32, 14, 14), (163072, 196, 14, 1), 87808)  # alias
        # Source Nodes: [cat_85, cat_86, cat_87], Original ATen: [aten.cat]
        triton_poi_fused_cat_64.run(buf243, buf404, buf426, buf449, 25088, grid=grid(25088), stream=stream0)
        buf409 = reinterpret_tensor(buf414, (4, 32, 14, 14), (150528, 196, 14, 1), 119168)  # alias
        buf431 = reinterpret_tensor(buf437, (4, 32, 14, 14), (156800, 196, 14, 1), 119168)  # alias
        buf454 = reinterpret_tensor(buf461, (4, 32, 14, 14), (163072, 196, 14, 1), 119168)  # alias
        # Source Nodes: [cat_85, cat_86, cat_87], Original ATen: [aten.cat]
        triton_poi_fused_cat_64.run(buf318, buf409, buf431, buf454, 25088, grid=grid(25088), stream=stream0)
        buf413 = reinterpret_tensor(buf414, (4, 32, 14, 14), (150528, 196, 14, 1), 144256)  # alias
        buf435 = reinterpret_tensor(buf437, (4, 32, 14, 14), (156800, 196, 14, 1), 144256)  # alias
        buf458 = reinterpret_tensor(buf461, (4, 32, 14, 14), (163072, 196, 14, 1), 144256)  # alias
        # Source Nodes: [cat_85, cat_86, cat_87], Original ATen: [aten.cat]
        triton_poi_fused_cat_64.run(buf396, buf413, buf435, buf458, 25088, grid=grid(25088), stream=stream0)
        buf415 = buf414; del buf414  # reuse
        # Source Nodes: [bottleneck_output_68, l__mod___features_denseblock3_denselayer17_norm1, l__mod___features_denseblock3_denselayer17_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_65.run(buf415, arg577_1, arg578_1, arg213_1, arg214_1, 602112, grid=grid(602112), stream=stream0)
        del arg213_1
        del arg214_1
        del arg577_1
        del arg578_1
        del buf397
        del buf398
        del buf399
        del buf400
        del buf401
        del buf402
        del buf403
        del buf404
        del buf405
        del buf406
        del buf407
        del buf408
        del buf409
        del buf410
        del buf411
        del buf412
        del buf413
        # Source Nodes: [bottleneck_output_68, l__mod___features_denseblock3_denselayer17_norm1, l__mod___features_denseblock3_denselayer17_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf416 = extern_kernels.convolution(buf415, arg215_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf416, (4, 128, 14, 14), (25088, 196, 14, 1))
        del arg215_1
        del buf415
        buf417 = buf416; del buf416  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer17_norm2, l__mod___features_denseblock3_denselayer17_relu2, new_features_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(buf417, arg580_1, arg581_1, arg216_1, arg217_1, 100352, grid=grid(100352), stream=stream0)
        del arg216_1
        del arg217_1
        del arg580_1
        del arg581_1
        # Source Nodes: [l__mod___features_denseblock3_denselayer17_norm2, l__mod___features_denseblock3_denselayer17_relu2, new_features_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf418 = extern_kernels.convolution(buf417, arg218_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf418, (4, 32, 14, 14), (6272, 196, 14, 1))
        del arg218_1
        del buf417
        buf427 = reinterpret_tensor(buf437, (4, 32, 14, 14), (156800, 196, 14, 1), 94080)  # alias
        buf450 = reinterpret_tensor(buf461, (4, 32, 14, 14), (163072, 196, 14, 1), 94080)  # alias
        buf474 = reinterpret_tensor(buf486, (4, 32, 14, 14), (169344, 196, 14, 1), 94080)  # alias
        # Source Nodes: [cat_84, cat_85, cat_86], Original ATen: [aten.cat]
        triton_poi_fused_cat_66.run(buf256, buf427, buf450, buf474, 25088, grid=grid(25088), stream=stream0)
        buf428 = reinterpret_tensor(buf437, (4, 32, 14, 14), (156800, 196, 14, 1), 100352)  # alias
        buf451 = reinterpret_tensor(buf461, (4, 32, 14, 14), (163072, 196, 14, 1), 100352)  # alias
        buf475 = reinterpret_tensor(buf486, (4, 32, 14, 14), (169344, 196, 14, 1), 100352)  # alias
        # Source Nodes: [cat_84, cat_85, cat_86], Original ATen: [aten.cat]
        triton_poi_fused_cat_66.run(buf270, buf428, buf451, buf475, 25088, grid=grid(25088), stream=stream0)
        buf432 = reinterpret_tensor(buf437, (4, 32, 14, 14), (156800, 196, 14, 1), 125440)  # alias
        buf455 = reinterpret_tensor(buf461, (4, 32, 14, 14), (163072, 196, 14, 1), 125440)  # alias
        buf479 = reinterpret_tensor(buf486, (4, 32, 14, 14), (169344, 196, 14, 1), 125440)  # alias
        # Source Nodes: [cat_84, cat_85, cat_86], Original ATen: [aten.cat]
        triton_poi_fused_cat_66.run(buf336, buf432, buf455, buf479, 25088, grid=grid(25088), stream=stream0)
        buf436 = reinterpret_tensor(buf437, (4, 32, 14, 14), (156800, 196, 14, 1), 150528)  # alias
        buf459 = reinterpret_tensor(buf461, (4, 32, 14, 14), (163072, 196, 14, 1), 150528)  # alias
        buf483 = reinterpret_tensor(buf486, (4, 32, 14, 14), (169344, 196, 14, 1), 150528)  # alias
        # Source Nodes: [cat_84, cat_85, cat_86], Original ATen: [aten.cat]
        triton_poi_fused_cat_66.run(buf418, buf436, buf459, buf483, 25088, grid=grid(25088), stream=stream0)
        buf438 = buf437; del buf437  # reuse
        # Source Nodes: [bottleneck_output_70, l__mod___features_denseblock3_denselayer18_norm1, l__mod___features_denseblock3_denselayer18_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_67.run(buf438, arg583_1, arg584_1, arg219_1, arg220_1, 627200, grid=grid(627200), stream=stream0)
        del arg219_1
        del arg220_1
        del arg583_1
        del arg584_1
        del buf419
        del buf420
        del buf421
        del buf422
        del buf423
        del buf424
        del buf425
        del buf426
        del buf427
        del buf428
        del buf429
        del buf430
        del buf431
        del buf432
        del buf433
        del buf434
        del buf435
        del buf436
        # Source Nodes: [bottleneck_output_70, l__mod___features_denseblock3_denselayer18_norm1, l__mod___features_denseblock3_denselayer18_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf439 = extern_kernels.convolution(buf438, arg221_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf439, (4, 128, 14, 14), (25088, 196, 14, 1))
        del arg221_1
        del buf438
        buf440 = buf439; del buf439  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer18_norm2, l__mod___features_denseblock3_denselayer18_relu2, new_features_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(buf440, arg586_1, arg587_1, arg222_1, arg223_1, 100352, grid=grid(100352), stream=stream0)
        del arg222_1
        del arg223_1
        del arg586_1
        del arg587_1
        # Source Nodes: [l__mod___features_denseblock3_denselayer18_norm2, l__mod___features_denseblock3_denselayer18_relu2, new_features_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf441 = extern_kernels.convolution(buf440, arg224_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf441, (4, 32, 14, 14), (6272, 196, 14, 1))
        del arg224_1
        del buf440
        buf443 = reinterpret_tensor(buf461, (4, 32, 14, 14), (163072, 196, 14, 1), 50176)  # alias
        buf467 = reinterpret_tensor(buf486, (4, 32, 14, 14), (169344, 196, 14, 1), 50176)  # alias
        buf492 = reinterpret_tensor(buf512, (4, 32, 14, 14), (175616, 196, 14, 1), 50176)  # alias
        # Source Nodes: [cat_83, cat_84, cat_85], Original ATen: [aten.cat]
        triton_poi_fused_cat_68.run(buf195, buf443, buf467, buf492, 25088, grid=grid(25088), stream=stream0)
        buf444 = reinterpret_tensor(buf461, (4, 32, 14, 14), (163072, 196, 14, 1), 56448)  # alias
        buf468 = reinterpret_tensor(buf486, (4, 32, 14, 14), (169344, 196, 14, 1), 56448)  # alias
        buf493 = reinterpret_tensor(buf512, (4, 32, 14, 14), (175616, 196, 14, 1), 56448)  # alias
        # Source Nodes: [cat_83, cat_84, cat_85], Original ATen: [aten.cat]
        triton_poi_fused_cat_68.run(buf200, buf444, buf468, buf493, 25088, grid=grid(25088), stream=stream0)
        buf445 = reinterpret_tensor(buf461, (4, 32, 14, 14), (163072, 196, 14, 1), 62720)  # alias
        buf469 = reinterpret_tensor(buf486, (4, 32, 14, 14), (169344, 196, 14, 1), 62720)  # alias
        buf494 = reinterpret_tensor(buf512, (4, 32, 14, 14), (175616, 196, 14, 1), 62720)  # alias
        # Source Nodes: [cat_83, cat_84, cat_85], Original ATen: [aten.cat]
        triton_poi_fused_cat_68.run(buf205, buf445, buf469, buf494, 25088, grid=grid(25088), stream=stream0)
        buf446 = reinterpret_tensor(buf461, (4, 32, 14, 14), (163072, 196, 14, 1), 68992)  # alias
        buf470 = reinterpret_tensor(buf486, (4, 32, 14, 14), (169344, 196, 14, 1), 68992)  # alias
        buf495 = reinterpret_tensor(buf512, (4, 32, 14, 14), (175616, 196, 14, 1), 68992)  # alias
        # Source Nodes: [cat_83, cat_84, cat_85], Original ATen: [aten.cat]
        triton_poi_fused_cat_68.run(buf210, buf446, buf470, buf495, 25088, grid=grid(25088), stream=stream0)
        buf447 = reinterpret_tensor(buf461, (4, 32, 14, 14), (163072, 196, 14, 1), 75264)  # alias
        buf471 = reinterpret_tensor(buf486, (4, 32, 14, 14), (169344, 196, 14, 1), 75264)  # alias
        buf496 = reinterpret_tensor(buf512, (4, 32, 14, 14), (175616, 196, 14, 1), 75264)  # alias
        # Source Nodes: [cat_83, cat_84, cat_85], Original ATen: [aten.cat]
        triton_poi_fused_cat_68.run(buf220, buf447, buf471, buf496, 25088, grid=grid(25088), stream=stream0)
        buf448 = reinterpret_tensor(buf461, (4, 32, 14, 14), (163072, 196, 14, 1), 81536)  # alias
        buf472 = reinterpret_tensor(buf486, (4, 32, 14, 14), (169344, 196, 14, 1), 81536)  # alias
        buf497 = reinterpret_tensor(buf512, (4, 32, 14, 14), (175616, 196, 14, 1), 81536)  # alias
        # Source Nodes: [cat_83, cat_84, cat_85], Original ATen: [aten.cat]
        triton_poi_fused_cat_68.run(buf231, buf448, buf472, buf497, 25088, grid=grid(25088), stream=stream0)
        buf452 = reinterpret_tensor(buf461, (4, 32, 14, 14), (163072, 196, 14, 1), 106624)  # alias
        buf476 = reinterpret_tensor(buf486, (4, 32, 14, 14), (169344, 196, 14, 1), 106624)  # alias
        buf501 = reinterpret_tensor(buf512, (4, 32, 14, 14), (175616, 196, 14, 1), 106624)  # alias
        # Source Nodes: [cat_83, cat_84, cat_85], Original ATen: [aten.cat]
        triton_poi_fused_cat_68.run(buf285, buf452, buf476, buf501, 25088, grid=grid(25088), stream=stream0)
        buf453 = reinterpret_tensor(buf461, (4, 32, 14, 14), (163072, 196, 14, 1), 112896)  # alias
        buf477 = reinterpret_tensor(buf486, (4, 32, 14, 14), (169344, 196, 14, 1), 112896)  # alias
        buf502 = reinterpret_tensor(buf512, (4, 32, 14, 14), (175616, 196, 14, 1), 112896)  # alias
        # Source Nodes: [cat_83, cat_84, cat_85], Original ATen: [aten.cat]
        triton_poi_fused_cat_68.run(buf301, buf453, buf477, buf502, 25088, grid=grid(25088), stream=stream0)
        buf456 = reinterpret_tensor(buf461, (4, 32, 14, 14), (163072, 196, 14, 1), 131712)  # alias
        buf480 = reinterpret_tensor(buf486, (4, 32, 14, 14), (169344, 196, 14, 1), 131712)  # alias
        buf505 = reinterpret_tensor(buf512, (4, 32, 14, 14), (175616, 196, 14, 1), 131712)  # alias
        # Source Nodes: [cat_83, cat_84, cat_85], Original ATen: [aten.cat]
        triton_poi_fused_cat_68.run(buf355, buf456, buf480, buf505, 25088, grid=grid(25088), stream=stream0)
        buf457 = reinterpret_tensor(buf461, (4, 32, 14, 14), (163072, 196, 14, 1), 137984)  # alias
        buf481 = reinterpret_tensor(buf486, (4, 32, 14, 14), (169344, 196, 14, 1), 137984)  # alias
        buf506 = reinterpret_tensor(buf512, (4, 32, 14, 14), (175616, 196, 14, 1), 137984)  # alias
        # Source Nodes: [cat_83, cat_84, cat_85], Original ATen: [aten.cat]
        triton_poi_fused_cat_68.run(buf375, buf457, buf481, buf506, 25088, grid=grid(25088), stream=stream0)
        buf460 = reinterpret_tensor(buf461, (4, 32, 14, 14), (163072, 196, 14, 1), 156800)  # alias
        buf484 = reinterpret_tensor(buf486, (4, 32, 14, 14), (169344, 196, 14, 1), 156800)  # alias
        buf509 = reinterpret_tensor(buf512, (4, 32, 14, 14), (175616, 196, 14, 1), 156800)  # alias
        # Source Nodes: [cat_83, cat_84, cat_85], Original ATen: [aten.cat]
        triton_poi_fused_cat_68.run(buf441, buf460, buf484, buf509, 25088, grid=grid(25088), stream=stream0)
        buf462 = buf461; del buf461  # reuse
        # Source Nodes: [bottleneck_output_72, l__mod___features_denseblock3_denselayer19_norm1, l__mod___features_denseblock3_denselayer19_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_69.run(buf462, arg589_1, arg590_1, arg225_1, arg226_1, 652288, grid=grid(652288), stream=stream0)
        del arg225_1
        del arg226_1
        del arg589_1
        del arg590_1
        del buf442
        del buf443
        del buf444
        del buf445
        del buf446
        del buf447
        del buf448
        del buf449
        del buf450
        del buf451
        del buf452
        del buf453
        del buf454
        del buf455
        del buf456
        del buf457
        del buf458
        del buf459
        del buf460
        # Source Nodes: [bottleneck_output_72, l__mod___features_denseblock3_denselayer19_norm1, l__mod___features_denseblock3_denselayer19_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf463 = extern_kernels.convolution(buf462, arg227_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf463, (4, 128, 14, 14), (25088, 196, 14, 1))
        del arg227_1
        del buf462
        buf464 = buf463; del buf463  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer19_norm2, l__mod___features_denseblock3_denselayer19_relu2, new_features_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(buf464, arg592_1, arg593_1, arg228_1, arg229_1, 100352, grid=grid(100352), stream=stream0)
        del arg228_1
        del arg229_1
        del arg592_1
        del arg593_1
        # Source Nodes: [l__mod___features_denseblock3_denselayer19_norm2, l__mod___features_denseblock3_denselayer19_relu2, new_features_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf465 = extern_kernels.convolution(buf464, arg230_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf465, (4, 32, 14, 14), (6272, 196, 14, 1))
        del arg230_1
        del buf464
        buf473 = reinterpret_tensor(buf486, (4, 32, 14, 14), (169344, 196, 14, 1), 87808)  # alias
        buf498 = reinterpret_tensor(buf512, (4, 32, 14, 14), (175616, 196, 14, 1), 87808)  # alias
        buf524 = reinterpret_tensor(buf539, (4, 32, 14, 14), (181888, 196, 14, 1), 87808)  # alias
        # Source Nodes: [cat_82, cat_83, cat_84], Original ATen: [aten.cat]
        triton_poi_fused_cat_70.run(buf243, buf473, buf498, buf524, 25088, grid=grid(25088), stream=stream0)
        buf478 = reinterpret_tensor(buf486, (4, 32, 14, 14), (169344, 196, 14, 1), 119168)  # alias
        buf503 = reinterpret_tensor(buf512, (4, 32, 14, 14), (175616, 196, 14, 1), 119168)  # alias
        buf529 = reinterpret_tensor(buf539, (4, 32, 14, 14), (181888, 196, 14, 1), 119168)  # alias
        # Source Nodes: [cat_82, cat_83, cat_84], Original ATen: [aten.cat]
        triton_poi_fused_cat_70.run(buf318, buf478, buf503, buf529, 25088, grid=grid(25088), stream=stream0)
        buf482 = reinterpret_tensor(buf486, (4, 32, 14, 14), (169344, 196, 14, 1), 144256)  # alias
        buf507 = reinterpret_tensor(buf512, (4, 32, 14, 14), (175616, 196, 14, 1), 144256)  # alias
        buf533 = reinterpret_tensor(buf539, (4, 32, 14, 14), (181888, 196, 14, 1), 144256)  # alias
        # Source Nodes: [cat_82, cat_83, cat_84], Original ATen: [aten.cat]
        triton_poi_fused_cat_70.run(buf396, buf482, buf507, buf533, 25088, grid=grid(25088), stream=stream0)
        buf485 = reinterpret_tensor(buf486, (4, 32, 14, 14), (169344, 196, 14, 1), 163072)  # alias
        buf510 = reinterpret_tensor(buf512, (4, 32, 14, 14), (175616, 196, 14, 1), 163072)  # alias
        buf536 = reinterpret_tensor(buf539, (4, 32, 14, 14), (181888, 196, 14, 1), 163072)  # alias
        # Source Nodes: [cat_82, cat_83, cat_84], Original ATen: [aten.cat]
        triton_poi_fused_cat_70.run(buf465, buf485, buf510, buf536, 25088, grid=grid(25088), stream=stream0)
        buf487 = buf486; del buf486  # reuse
        # Source Nodes: [bottleneck_output_74, l__mod___features_denseblock3_denselayer20_norm1, l__mod___features_denseblock3_denselayer20_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_71.run(buf487, arg595_1, arg596_1, arg231_1, arg232_1, 677376, grid=grid(677376), stream=stream0)
        del arg231_1
        del arg232_1
        del arg595_1
        del arg596_1
        del buf466
        del buf467
        del buf468
        del buf469
        del buf470
        del buf471
        del buf472
        del buf473
        del buf474
        del buf475
        del buf476
        del buf477
        del buf478
        del buf479
        del buf480
        del buf481
        del buf482
        del buf483
        del buf484
        del buf485
        # Source Nodes: [bottleneck_output_74, l__mod___features_denseblock3_denselayer20_norm1, l__mod___features_denseblock3_denselayer20_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf488 = extern_kernels.convolution(buf487, arg233_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf488, (4, 128, 14, 14), (25088, 196, 14, 1))
        del arg233_1
        del buf487
        buf489 = buf488; del buf488  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer20_norm2, l__mod___features_denseblock3_denselayer20_relu2, new_features_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(buf489, arg598_1, arg599_1, arg234_1, arg235_1, 100352, grid=grid(100352), stream=stream0)
        del arg234_1
        del arg235_1
        del arg598_1
        del arg599_1
        # Source Nodes: [l__mod___features_denseblock3_denselayer20_norm2, l__mod___features_denseblock3_denselayer20_relu2, new_features_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf490 = extern_kernels.convolution(buf489, arg236_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf490, (4, 32, 14, 14), (6272, 196, 14, 1))
        del arg236_1
        del buf489
        buf499 = reinterpret_tensor(buf512, (4, 32, 14, 14), (175616, 196, 14, 1), 94080)  # alias
        buf525 = reinterpret_tensor(buf539, (4, 32, 14, 14), (181888, 196, 14, 1), 94080)  # alias
        buf552 = reinterpret_tensor(buf567, (4, 32, 14, 14), (188160, 196, 14, 1), 94080)  # alias
        # Source Nodes: [cat_81, cat_82, cat_83], Original ATen: [aten.cat]
        triton_poi_fused_cat_72.run(buf256, buf499, buf525, buf552, 25088, grid=grid(25088), stream=stream0)
        buf500 = reinterpret_tensor(buf512, (4, 32, 14, 14), (175616, 196, 14, 1), 100352)  # alias
        buf526 = reinterpret_tensor(buf539, (4, 32, 14, 14), (181888, 196, 14, 1), 100352)  # alias
        buf553 = reinterpret_tensor(buf567, (4, 32, 14, 14), (188160, 196, 14, 1), 100352)  # alias
        # Source Nodes: [cat_81, cat_82, cat_83], Original ATen: [aten.cat]
        triton_poi_fused_cat_72.run(buf270, buf500, buf526, buf553, 25088, grid=grid(25088), stream=stream0)
        buf504 = reinterpret_tensor(buf512, (4, 32, 14, 14), (175616, 196, 14, 1), 125440)  # alias
        buf530 = reinterpret_tensor(buf539, (4, 32, 14, 14), (181888, 196, 14, 1), 125440)  # alias
        buf557 = reinterpret_tensor(buf567, (4, 32, 14, 14), (188160, 196, 14, 1), 125440)  # alias
        # Source Nodes: [cat_81, cat_82, cat_83], Original ATen: [aten.cat]
        triton_poi_fused_cat_72.run(buf336, buf504, buf530, buf557, 25088, grid=grid(25088), stream=stream0)
        buf508 = reinterpret_tensor(buf512, (4, 32, 14, 14), (175616, 196, 14, 1), 150528)  # alias
        buf534 = reinterpret_tensor(buf539, (4, 32, 14, 14), (181888, 196, 14, 1), 150528)  # alias
        buf561 = reinterpret_tensor(buf567, (4, 32, 14, 14), (188160, 196, 14, 1), 150528)  # alias
        # Source Nodes: [cat_81, cat_82, cat_83], Original ATen: [aten.cat]
        triton_poi_fused_cat_72.run(buf418, buf508, buf534, buf561, 25088, grid=grid(25088), stream=stream0)
        buf511 = reinterpret_tensor(buf512, (4, 32, 14, 14), (175616, 196, 14, 1), 169344)  # alias
        buf537 = reinterpret_tensor(buf539, (4, 32, 14, 14), (181888, 196, 14, 1), 169344)  # alias
        buf564 = reinterpret_tensor(buf567, (4, 32, 14, 14), (188160, 196, 14, 1), 169344)  # alias
        # Source Nodes: [cat_81, cat_82, cat_83], Original ATen: [aten.cat]
        triton_poi_fused_cat_72.run(buf490, buf511, buf537, buf564, 25088, grid=grid(25088), stream=stream0)
        buf513 = buf512; del buf512  # reuse
        # Source Nodes: [bottleneck_output_76, l__mod___features_denseblock3_denselayer21_norm1, l__mod___features_denseblock3_denselayer21_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_73.run(buf513, arg601_1, arg602_1, arg237_1, arg238_1, 702464, grid=grid(702464), stream=stream0)
        del arg237_1
        del arg238_1
        del arg601_1
        del arg602_1
        del buf491
        del buf492
        del buf493
        del buf494
        del buf495
        del buf496
        del buf497
        del buf498
        del buf499
        del buf500
        del buf501
        del buf502
        del buf503
        del buf504
        del buf505
        del buf506
        del buf507
        del buf508
        del buf509
        del buf510
        del buf511
        # Source Nodes: [bottleneck_output_76, l__mod___features_denseblock3_denselayer21_norm1, l__mod___features_denseblock3_denselayer21_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf514 = extern_kernels.convolution(buf513, arg239_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf514, (4, 128, 14, 14), (25088, 196, 14, 1))
        del arg239_1
        del buf513
        buf515 = buf514; del buf514  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer21_norm2, l__mod___features_denseblock3_denselayer21_relu2, new_features_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(buf515, arg604_1, arg605_1, arg240_1, arg241_1, 100352, grid=grid(100352), stream=stream0)
        del arg240_1
        del arg241_1
        del arg604_1
        del arg605_1
        # Source Nodes: [l__mod___features_denseblock3_denselayer21_norm2, l__mod___features_denseblock3_denselayer21_relu2, new_features_76], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf516 = extern_kernels.convolution(buf515, arg242_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf516, (4, 32, 14, 14), (6272, 196, 14, 1))
        del arg242_1
        del buf515
        buf518 = reinterpret_tensor(buf539, (4, 32, 14, 14), (181888, 196, 14, 1), 50176)  # alias
        buf545 = reinterpret_tensor(buf567, (4, 32, 14, 14), (188160, 196, 14, 1), 50176)  # alias
        buf573 = reinterpret_tensor(buf596, (4, 32, 14, 14), (194432, 196, 14, 1), 50176)  # alias
        # Source Nodes: [cat_80, cat_81, cat_82], Original ATen: [aten.cat]
        triton_poi_fused_cat_74.run(buf195, buf518, buf545, buf573, 25088, grid=grid(25088), stream=stream0)
        buf519 = reinterpret_tensor(buf539, (4, 32, 14, 14), (181888, 196, 14, 1), 56448)  # alias
        buf546 = reinterpret_tensor(buf567, (4, 32, 14, 14), (188160, 196, 14, 1), 56448)  # alias
        buf574 = reinterpret_tensor(buf596, (4, 32, 14, 14), (194432, 196, 14, 1), 56448)  # alias
        # Source Nodes: [cat_80, cat_81, cat_82], Original ATen: [aten.cat]
        triton_poi_fused_cat_74.run(buf200, buf519, buf546, buf574, 25088, grid=grid(25088), stream=stream0)
        buf520 = reinterpret_tensor(buf539, (4, 32, 14, 14), (181888, 196, 14, 1), 62720)  # alias
        buf547 = reinterpret_tensor(buf567, (4, 32, 14, 14), (188160, 196, 14, 1), 62720)  # alias
        buf575 = reinterpret_tensor(buf596, (4, 32, 14, 14), (194432, 196, 14, 1), 62720)  # alias
        # Source Nodes: [cat_80, cat_81, cat_82], Original ATen: [aten.cat]
        triton_poi_fused_cat_74.run(buf205, buf520, buf547, buf575, 25088, grid=grid(25088), stream=stream0)
        buf521 = reinterpret_tensor(buf539, (4, 32, 14, 14), (181888, 196, 14, 1), 68992)  # alias
        buf548 = reinterpret_tensor(buf567, (4, 32, 14, 14), (188160, 196, 14, 1), 68992)  # alias
        buf576 = reinterpret_tensor(buf596, (4, 32, 14, 14), (194432, 196, 14, 1), 68992)  # alias
        # Source Nodes: [cat_80, cat_81, cat_82], Original ATen: [aten.cat]
        triton_poi_fused_cat_74.run(buf210, buf521, buf548, buf576, 25088, grid=grid(25088), stream=stream0)
        buf522 = reinterpret_tensor(buf539, (4, 32, 14, 14), (181888, 196, 14, 1), 75264)  # alias
        buf549 = reinterpret_tensor(buf567, (4, 32, 14, 14), (188160, 196, 14, 1), 75264)  # alias
        buf577 = reinterpret_tensor(buf596, (4, 32, 14, 14), (194432, 196, 14, 1), 75264)  # alias
        # Source Nodes: [cat_80, cat_81, cat_82], Original ATen: [aten.cat]
        triton_poi_fused_cat_74.run(buf220, buf522, buf549, buf577, 25088, grid=grid(25088), stream=stream0)
        buf523 = reinterpret_tensor(buf539, (4, 32, 14, 14), (181888, 196, 14, 1), 81536)  # alias
        buf550 = reinterpret_tensor(buf567, (4, 32, 14, 14), (188160, 196, 14, 1), 81536)  # alias
        buf578 = reinterpret_tensor(buf596, (4, 32, 14, 14), (194432, 196, 14, 1), 81536)  # alias
        # Source Nodes: [cat_80, cat_81, cat_82], Original ATen: [aten.cat]
        triton_poi_fused_cat_74.run(buf231, buf523, buf550, buf578, 25088, grid=grid(25088), stream=stream0)
        buf527 = reinterpret_tensor(buf539, (4, 32, 14, 14), (181888, 196, 14, 1), 106624)  # alias
        buf554 = reinterpret_tensor(buf567, (4, 32, 14, 14), (188160, 196, 14, 1), 106624)  # alias
        buf582 = reinterpret_tensor(buf596, (4, 32, 14, 14), (194432, 196, 14, 1), 106624)  # alias
        # Source Nodes: [cat_80, cat_81, cat_82], Original ATen: [aten.cat]
        triton_poi_fused_cat_74.run(buf285, buf527, buf554, buf582, 25088, grid=grid(25088), stream=stream0)
        buf528 = reinterpret_tensor(buf539, (4, 32, 14, 14), (181888, 196, 14, 1), 112896)  # alias
        buf555 = reinterpret_tensor(buf567, (4, 32, 14, 14), (188160, 196, 14, 1), 112896)  # alias
        buf583 = reinterpret_tensor(buf596, (4, 32, 14, 14), (194432, 196, 14, 1), 112896)  # alias
        # Source Nodes: [cat_80, cat_81, cat_82], Original ATen: [aten.cat]
        triton_poi_fused_cat_74.run(buf301, buf528, buf555, buf583, 25088, grid=grid(25088), stream=stream0)
        buf531 = reinterpret_tensor(buf539, (4, 32, 14, 14), (181888, 196, 14, 1), 131712)  # alias
        buf558 = reinterpret_tensor(buf567, (4, 32, 14, 14), (188160, 196, 14, 1), 131712)  # alias
        buf586 = reinterpret_tensor(buf596, (4, 32, 14, 14), (194432, 196, 14, 1), 131712)  # alias
        # Source Nodes: [cat_80, cat_81, cat_82], Original ATen: [aten.cat]
        triton_poi_fused_cat_74.run(buf355, buf531, buf558, buf586, 25088, grid=grid(25088), stream=stream0)
        buf532 = reinterpret_tensor(buf539, (4, 32, 14, 14), (181888, 196, 14, 1), 137984)  # alias
        buf559 = reinterpret_tensor(buf567, (4, 32, 14, 14), (188160, 196, 14, 1), 137984)  # alias
        buf587 = reinterpret_tensor(buf596, (4, 32, 14, 14), (194432, 196, 14, 1), 137984)  # alias
        # Source Nodes: [cat_80, cat_81, cat_82], Original ATen: [aten.cat]
        triton_poi_fused_cat_74.run(buf375, buf532, buf559, buf587, 25088, grid=grid(25088), stream=stream0)
        buf535 = reinterpret_tensor(buf539, (4, 32, 14, 14), (181888, 196, 14, 1), 156800)  # alias
        buf562 = reinterpret_tensor(buf567, (4, 32, 14, 14), (188160, 196, 14, 1), 156800)  # alias
        buf590 = reinterpret_tensor(buf596, (4, 32, 14, 14), (194432, 196, 14, 1), 156800)  # alias
        # Source Nodes: [cat_80, cat_81, cat_82], Original ATen: [aten.cat]
        triton_poi_fused_cat_74.run(buf441, buf535, buf562, buf590, 25088, grid=grid(25088), stream=stream0)
        buf538 = reinterpret_tensor(buf539, (4, 32, 14, 14), (181888, 196, 14, 1), 175616)  # alias
        buf565 = reinterpret_tensor(buf567, (4, 32, 14, 14), (188160, 196, 14, 1), 175616)  # alias
        buf593 = reinterpret_tensor(buf596, (4, 32, 14, 14), (194432, 196, 14, 1), 175616)  # alias
        # Source Nodes: [cat_80, cat_81, cat_82], Original ATen: [aten.cat]
        triton_poi_fused_cat_74.run(buf516, buf538, buf565, buf593, 25088, grid=grid(25088), stream=stream0)
        buf540 = buf539; del buf539  # reuse
        # Source Nodes: [bottleneck_output_78, l__mod___features_denseblock3_denselayer22_norm1, l__mod___features_denseblock3_denselayer22_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_75.run(buf540, arg607_1, arg608_1, arg243_1, arg244_1, 727552, grid=grid(727552), stream=stream0)
        del arg243_1
        del arg244_1
        del arg607_1
        del arg608_1
        del buf517
        del buf518
        del buf519
        del buf520
        del buf521
        del buf522
        del buf523
        del buf524
        del buf525
        del buf526
        del buf527
        del buf528
        del buf529
        del buf530
        del buf531
        del buf532
        del buf533
        del buf534
        del buf535
        del buf536
        del buf537
        del buf538
        # Source Nodes: [bottleneck_output_78, l__mod___features_denseblock3_denselayer22_norm1, l__mod___features_denseblock3_denselayer22_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf541 = extern_kernels.convolution(buf540, arg245_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf541, (4, 128, 14, 14), (25088, 196, 14, 1))
        del arg245_1
        del buf540
        buf542 = buf541; del buf541  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer22_norm2, l__mod___features_denseblock3_denselayer22_relu2, new_features_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(buf542, arg610_1, arg611_1, arg246_1, arg247_1, 100352, grid=grid(100352), stream=stream0)
        del arg246_1
        del arg247_1
        del arg610_1
        del arg611_1
        # Source Nodes: [l__mod___features_denseblock3_denselayer22_norm2, l__mod___features_denseblock3_denselayer22_relu2, new_features_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf543 = extern_kernels.convolution(buf542, arg248_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf543, (4, 32, 14, 14), (6272, 196, 14, 1))
        del arg248_1
        del buf542
        buf551 = reinterpret_tensor(buf567, (4, 32, 14, 14), (188160, 196, 14, 1), 87808)  # alias
        buf579 = reinterpret_tensor(buf596, (4, 32, 14, 14), (194432, 196, 14, 1), 87808)  # alias
        buf608 = reinterpret_tensor(buf626, (4, 32, 14, 14), (200704, 196, 14, 1), 87808)  # alias
        # Source Nodes: [cat_79, cat_80, cat_81], Original ATen: [aten.cat]
        triton_poi_fused_cat_76.run(buf243, buf551, buf579, buf608, 25088, grid=grid(25088), stream=stream0)
        del buf243
        buf556 = reinterpret_tensor(buf567, (4, 32, 14, 14), (188160, 196, 14, 1), 119168)  # alias
        buf584 = reinterpret_tensor(buf596, (4, 32, 14, 14), (194432, 196, 14, 1), 119168)  # alias
        buf613 = reinterpret_tensor(buf626, (4, 32, 14, 14), (200704, 196, 14, 1), 119168)  # alias
        # Source Nodes: [cat_79, cat_80, cat_81], Original ATen: [aten.cat]
        triton_poi_fused_cat_76.run(buf318, buf556, buf584, buf613, 25088, grid=grid(25088), stream=stream0)
        del buf318
        buf560 = reinterpret_tensor(buf567, (4, 32, 14, 14), (188160, 196, 14, 1), 144256)  # alias
        buf588 = reinterpret_tensor(buf596, (4, 32, 14, 14), (194432, 196, 14, 1), 144256)  # alias
        buf617 = reinterpret_tensor(buf626, (4, 32, 14, 14), (200704, 196, 14, 1), 144256)  # alias
        # Source Nodes: [cat_79, cat_80, cat_81], Original ATen: [aten.cat]
        triton_poi_fused_cat_76.run(buf396, buf560, buf588, buf617, 25088, grid=grid(25088), stream=stream0)
        del buf396
        buf563 = reinterpret_tensor(buf567, (4, 32, 14, 14), (188160, 196, 14, 1), 163072)  # alias
        buf591 = reinterpret_tensor(buf596, (4, 32, 14, 14), (194432, 196, 14, 1), 163072)  # alias
        buf620 = reinterpret_tensor(buf626, (4, 32, 14, 14), (200704, 196, 14, 1), 163072)  # alias
        # Source Nodes: [cat_79, cat_80, cat_81], Original ATen: [aten.cat]
        triton_poi_fused_cat_76.run(buf465, buf563, buf591, buf620, 25088, grid=grid(25088), stream=stream0)
        del buf465
        buf566 = reinterpret_tensor(buf567, (4, 32, 14, 14), (188160, 196, 14, 1), 181888)  # alias
        buf594 = reinterpret_tensor(buf596, (4, 32, 14, 14), (194432, 196, 14, 1), 181888)  # alias
        buf623 = reinterpret_tensor(buf626, (4, 32, 14, 14), (200704, 196, 14, 1), 181888)  # alias
        # Source Nodes: [cat_79, cat_80, cat_81], Original ATen: [aten.cat]
        triton_poi_fused_cat_76.run(buf543, buf566, buf594, buf623, 25088, grid=grid(25088), stream=stream0)
        del buf543
        buf568 = buf567; del buf567  # reuse
        # Source Nodes: [bottleneck_output_80, l__mod___features_denseblock3_denselayer23_norm1, l__mod___features_denseblock3_denselayer23_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_77.run(buf568, arg613_1, arg614_1, arg249_1, arg250_1, 752640, grid=grid(752640), stream=stream0)
        del arg249_1
        del arg250_1
        del arg613_1
        del arg614_1
        del buf544
        del buf545
        del buf546
        del buf547
        del buf548
        del buf549
        del buf550
        del buf551
        del buf552
        del buf553
        del buf554
        del buf555
        del buf556
        del buf557
        del buf558
        del buf559
        del buf560
        del buf561
        del buf562
        del buf563
        del buf564
        del buf565
        del buf566
        # Source Nodes: [bottleneck_output_80, l__mod___features_denseblock3_denselayer23_norm1, l__mod___features_denseblock3_denselayer23_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf569 = extern_kernels.convolution(buf568, arg251_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf569, (4, 128, 14, 14), (25088, 196, 14, 1))
        del arg251_1
        del buf568
        buf570 = buf569; del buf569  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer23_norm2, l__mod___features_denseblock3_denselayer23_relu2, new_features_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(buf570, arg616_1, arg617_1, arg252_1, arg253_1, 100352, grid=grid(100352), stream=stream0)
        del arg252_1
        del arg253_1
        del arg616_1
        del arg617_1
        # Source Nodes: [l__mod___features_denseblock3_denselayer23_norm2, l__mod___features_denseblock3_denselayer23_relu2, new_features_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf571 = extern_kernels.convolution(buf570, arg254_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf571, (4, 32, 14, 14), (6272, 196, 14, 1))
        del arg254_1
        del buf570
        buf580 = reinterpret_tensor(buf596, (4, 32, 14, 14), (194432, 196, 14, 1), 94080)  # alias
        buf609 = reinterpret_tensor(buf626, (4, 32, 14, 14), (200704, 196, 14, 1), 94080)  # alias
        # Source Nodes: [cat_79, cat_80], Original ATen: [aten.cat]
        triton_poi_fused_cat_78.run(buf256, buf580, buf609, 25088, grid=grid(25088), stream=stream0)
        del buf256
        buf581 = reinterpret_tensor(buf596, (4, 32, 14, 14), (194432, 196, 14, 1), 100352)  # alias
        buf610 = reinterpret_tensor(buf626, (4, 32, 14, 14), (200704, 196, 14, 1), 100352)  # alias
        # Source Nodes: [cat_79, cat_80], Original ATen: [aten.cat]
        triton_poi_fused_cat_78.run(buf270, buf581, buf610, 25088, grid=grid(25088), stream=stream0)
        del buf270
        buf585 = reinterpret_tensor(buf596, (4, 32, 14, 14), (194432, 196, 14, 1), 125440)  # alias
        buf614 = reinterpret_tensor(buf626, (4, 32, 14, 14), (200704, 196, 14, 1), 125440)  # alias
        # Source Nodes: [cat_79, cat_80], Original ATen: [aten.cat]
        triton_poi_fused_cat_78.run(buf336, buf585, buf614, 25088, grid=grid(25088), stream=stream0)
        del buf336
        buf589 = reinterpret_tensor(buf596, (4, 32, 14, 14), (194432, 196, 14, 1), 150528)  # alias
        buf618 = reinterpret_tensor(buf626, (4, 32, 14, 14), (200704, 196, 14, 1), 150528)  # alias
        # Source Nodes: [cat_79, cat_80], Original ATen: [aten.cat]
        triton_poi_fused_cat_78.run(buf418, buf589, buf618, 25088, grid=grid(25088), stream=stream0)
        del buf418
        buf592 = reinterpret_tensor(buf596, (4, 32, 14, 14), (194432, 196, 14, 1), 169344)  # alias
        buf621 = reinterpret_tensor(buf626, (4, 32, 14, 14), (200704, 196, 14, 1), 169344)  # alias
        # Source Nodes: [cat_79, cat_80], Original ATen: [aten.cat]
        triton_poi_fused_cat_78.run(buf490, buf592, buf621, 25088, grid=grid(25088), stream=stream0)
        del buf490
        buf595 = reinterpret_tensor(buf596, (4, 32, 14, 14), (194432, 196, 14, 1), 188160)  # alias
        buf624 = reinterpret_tensor(buf626, (4, 32, 14, 14), (200704, 196, 14, 1), 188160)  # alias
        # Source Nodes: [cat_79, cat_80], Original ATen: [aten.cat]
        triton_poi_fused_cat_78.run(buf571, buf595, buf624, 25088, grid=grid(25088), stream=stream0)
        del buf571
        buf597 = buf596; del buf596  # reuse
        # Source Nodes: [bottleneck_output_82, l__mod___features_denseblock3_denselayer24_norm1, l__mod___features_denseblock3_denselayer24_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_79.run(buf597, arg619_1, arg620_1, arg255_1, arg256_1, 777728, grid=grid(777728), stream=stream0)
        del arg255_1
        del arg256_1
        del arg619_1
        del arg620_1
        del buf572
        del buf573
        del buf574
        del buf575
        del buf576
        del buf577
        del buf578
        del buf579
        del buf580
        del buf581
        del buf582
        del buf583
        del buf584
        del buf585
        del buf586
        del buf587
        del buf588
        del buf589
        del buf590
        del buf591
        del buf592
        del buf593
        del buf594
        del buf595
        # Source Nodes: [bottleneck_output_82, l__mod___features_denseblock3_denselayer24_norm1, l__mod___features_denseblock3_denselayer24_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf598 = extern_kernels.convolution(buf597, arg257_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf598, (4, 128, 14, 14), (25088, 196, 14, 1))
        del arg257_1
        del buf597
        buf599 = buf598; del buf598  # reuse
        # Source Nodes: [l__mod___features_denseblock3_denselayer24_norm2, l__mod___features_denseblock3_denselayer24_relu2, new_features_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_36.run(buf599, arg622_1, arg623_1, arg258_1, arg259_1, 100352, grid=grid(100352), stream=stream0)
        del arg258_1
        del arg259_1
        del arg622_1
        del arg623_1
        # Source Nodes: [l__mod___features_denseblock3_denselayer24_norm2, l__mod___features_denseblock3_denselayer24_relu2, new_features_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf600 = extern_kernels.convolution(buf599, arg260_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf600, (4, 32, 14, 14), (6272, 196, 14, 1))
        del arg260_1
        buf602 = reinterpret_tensor(buf626, (4, 32, 14, 14), (200704, 196, 14, 1), 50176)  # alias
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_80.run(buf195, buf602, 25088, grid=grid(25088), stream=stream0)
        del buf195
        buf603 = reinterpret_tensor(buf626, (4, 32, 14, 14), (200704, 196, 14, 1), 56448)  # alias
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_80.run(buf200, buf603, 25088, grid=grid(25088), stream=stream0)
        del buf200
        buf604 = reinterpret_tensor(buf626, (4, 32, 14, 14), (200704, 196, 14, 1), 62720)  # alias
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_80.run(buf205, buf604, 25088, grid=grid(25088), stream=stream0)
        del buf205
        buf605 = reinterpret_tensor(buf626, (4, 32, 14, 14), (200704, 196, 14, 1), 68992)  # alias
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_80.run(buf210, buf605, 25088, grid=grid(25088), stream=stream0)
        del buf210
        buf606 = reinterpret_tensor(buf626, (4, 32, 14, 14), (200704, 196, 14, 1), 75264)  # alias
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_80.run(buf220, buf606, 25088, grid=grid(25088), stream=stream0)
        del buf220
        buf607 = reinterpret_tensor(buf626, (4, 32, 14, 14), (200704, 196, 14, 1), 81536)  # alias
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_80.run(buf231, buf607, 25088, grid=grid(25088), stream=stream0)
        del buf231
        buf611 = reinterpret_tensor(buf626, (4, 32, 14, 14), (200704, 196, 14, 1), 106624)  # alias
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_80.run(buf285, buf611, 25088, grid=grid(25088), stream=stream0)
        del buf285
        buf612 = reinterpret_tensor(buf626, (4, 32, 14, 14), (200704, 196, 14, 1), 112896)  # alias
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_80.run(buf301, buf612, 25088, grid=grid(25088), stream=stream0)
        del buf301
        buf615 = reinterpret_tensor(buf626, (4, 32, 14, 14), (200704, 196, 14, 1), 131712)  # alias
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_80.run(buf355, buf615, 25088, grid=grid(25088), stream=stream0)
        del buf355
        buf616 = reinterpret_tensor(buf626, (4, 32, 14, 14), (200704, 196, 14, 1), 137984)  # alias
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_80.run(buf375, buf616, 25088, grid=grid(25088), stream=stream0)
        del buf375
        buf619 = reinterpret_tensor(buf626, (4, 32, 14, 14), (200704, 196, 14, 1), 156800)  # alias
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_80.run(buf441, buf619, 25088, grid=grid(25088), stream=stream0)
        del buf441
        buf622 = reinterpret_tensor(buf626, (4, 32, 14, 14), (200704, 196, 14, 1), 175616)  # alias
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_80.run(buf516, buf622, 25088, grid=grid(25088), stream=stream0)
        del buf516
        buf625 = reinterpret_tensor(buf626, (4, 32, 14, 14), (200704, 196, 14, 1), 194432)  # alias
        # Source Nodes: [cat_79], Original ATen: [aten.cat]
        triton_poi_fused_cat_80.run(buf600, buf625, 25088, grid=grid(25088), stream=stream0)
        del buf600
        buf627 = buf626; del buf626  # reuse
        # Source Nodes: [l__mod___features_transition3_conv, l__mod___features_transition3_norm, l__mod___features_transition3_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_81.run(buf627, arg625_1, arg626_1, arg261_1, arg262_1, 802816, grid=grid(802816), stream=stream0)
        del arg261_1
        del arg262_1
        del arg625_1
        del arg626_1
        del buf601
        del buf602
        del buf603
        del buf604
        del buf605
        del buf606
        del buf607
        del buf608
        del buf609
        del buf610
        del buf611
        del buf612
        del buf613
        del buf614
        del buf615
        del buf616
        del buf617
        del buf618
        del buf619
        del buf620
        del buf621
        del buf622
        del buf623
        del buf624
        del buf625
        # Source Nodes: [l__mod___features_transition3_conv, l__mod___features_transition3_norm, l__mod___features_transition3_relu], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf628 = extern_kernels.convolution(buf627, arg263_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf628, (4, 512, 14, 14), (100352, 196, 14, 1))
        del arg263_1
        del buf627
        buf629 = reinterpret_tensor(buf599, (4, 512, 7, 7), (25088, 49, 7, 1), 0); del buf599  # reuse
        buf653 = empty((4, 640, 7, 7), device='cuda', dtype=torch.float32)
        buf648 = reinterpret_tensor(buf653, (4, 512, 7, 7), (31360, 49, 7, 1), 0)  # alias
        buf664 = empty((4, 672, 7, 7), device='cuda', dtype=torch.float32)
        buf658 = reinterpret_tensor(buf664, (4, 512, 7, 7), (32928, 49, 7, 1), 0)  # alias
        buf676 = empty((4, 704, 7, 7), device='cuda', dtype=torch.float32)
        buf669 = reinterpret_tensor(buf676, (4, 512, 7, 7), (34496, 49, 7, 1), 0)  # alias
        buf689 = empty((4, 736, 7, 7), device='cuda', dtype=torch.float32)
        buf681 = reinterpret_tensor(buf689, (4, 512, 7, 7), (36064, 49, 7, 1), 0)  # alias
        buf703 = empty((4, 768, 7, 7), device='cuda', dtype=torch.float32)
        buf694 = reinterpret_tensor(buf703, (4, 512, 7, 7), (37632, 49, 7, 1), 0)  # alias
        buf718 = empty((4, 800, 7, 7), device='cuda', dtype=torch.float32)
        buf708 = reinterpret_tensor(buf718, (4, 512, 7, 7), (39200, 49, 7, 1), 0)  # alias
        buf734 = empty((4, 832, 7, 7), device='cuda', dtype=torch.float32)
        buf723 = reinterpret_tensor(buf734, (4, 512, 7, 7), (40768, 49, 7, 1), 0)  # alias
        buf751 = empty((4, 864, 7, 7), device='cuda', dtype=torch.float32)
        buf739 = reinterpret_tensor(buf751, (4, 512, 7, 7), (42336, 49, 7, 1), 0)  # alias
        buf769 = empty((4, 896, 7, 7), device='cuda', dtype=torch.float32)
        buf756 = reinterpret_tensor(buf769, (4, 512, 7, 7), (43904, 49, 7, 1), 0)  # alias
        buf788 = empty((4, 928, 7, 7), device='cuda', dtype=torch.float32)
        buf774 = reinterpret_tensor(buf788, (4, 512, 7, 7), (45472, 49, 7, 1), 0)  # alias
        buf808 = empty((4, 960, 7, 7), device='cuda', dtype=torch.float32)
        buf793 = reinterpret_tensor(buf808, (4, 512, 7, 7), (47040, 49, 7, 1), 0)  # alias
        buf829 = empty((4, 992, 7, 7), device='cuda', dtype=torch.float32)
        buf813 = reinterpret_tensor(buf829, (4, 512, 7, 7), (48608, 49, 7, 1), 0)  # alias
        buf851 = reinterpret_tensor(buf192, (4, 1024, 7, 7), (50176, 49, 7, 1), 0); del buf192  # reuse
        buf834 = reinterpret_tensor(buf851, (4, 512, 7, 7), (50176, 49, 7, 1), 0)  # alias
        # Source Nodes: [bottleneck_output_84, cat_62, cat_63, cat_64, cat_65, cat_66, cat_67, cat_68, cat_69, cat_70, cat_71, cat_72, cat_73, l__mod___features_denseblock4_denselayer1_norm1, l__mod___features_denseblock4_denselayer1_relu1, l__mod___features_transition3_pool], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_avg_pool2d_cat_convolution_relu_82.run(buf628, arg628_1, arg629_1, arg264_1, arg265_1, buf629, buf648, buf658, buf669, buf681, buf694, buf708, buf723, buf739, buf756, buf774, buf793, buf813, buf834, 100352, grid=grid(100352), stream=stream0)
        del arg264_1
        del arg265_1
        del arg628_1
        del arg629_1
        # Source Nodes: [bottleneck_output_84, l__mod___features_denseblock4_denselayer1_norm1, l__mod___features_denseblock4_denselayer1_relu1, l__mod___features_transition3_pool], Original ATen: [aten._native_batch_norm_legit_no_training, aten.avg_pool2d, aten.convolution, aten.relu]
        buf630 = extern_kernels.convolution(buf629, arg266_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf630, (4, 128, 7, 7), (6272, 49, 7, 1))
        del arg266_1
        del buf629
        buf631 = buf630; del buf630  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer1_norm2, l__mod___features_denseblock4_denselayer1_relu2, new_features_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_83.run(buf631, arg631_1, arg632_1, arg267_1, arg268_1, 25088, grid=grid(25088), stream=stream0)
        del arg267_1
        del arg268_1
        del arg631_1
        del arg632_1
        # Source Nodes: [l__mod___features_denseblock4_denselayer1_norm2, l__mod___features_denseblock4_denselayer1_relu2, new_features_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf632 = extern_kernels.convolution(buf631, arg269_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf632, (4, 32, 7, 7), (1568, 49, 7, 1))
        del arg269_1
        del buf631
        buf633 = empty((4, 544, 7, 7), device='cuda', dtype=torch.float32)
        buf634 = buf633; del buf633  # reuse
        # Source Nodes: [bottleneck_output_86, cat_77, l__mod___features_denseblock4_denselayer2_norm1, l__mod___features_denseblock4_denselayer2_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_84.run(buf634, buf628, buf632, arg634_1, arg635_1, arg270_1, arg271_1, 106624, grid=grid(106624), stream=stream0)
        del arg270_1
        del arg271_1
        del arg634_1
        del arg635_1
        # Source Nodes: [bottleneck_output_86, l__mod___features_denseblock4_denselayer2_relu1], Original ATen: [aten.convolution, aten.relu]
        buf635 = extern_kernels.convolution(buf634, arg272_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf635, (4, 128, 7, 7), (6272, 49, 7, 1))
        del arg272_1
        del buf634
        buf636 = buf635; del buf635  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer2_norm2, l__mod___features_denseblock4_denselayer2_relu2, new_features_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_83.run(buf636, arg637_1, arg638_1, arg273_1, arg274_1, 25088, grid=grid(25088), stream=stream0)
        del arg273_1
        del arg274_1
        del arg637_1
        del arg638_1
        # Source Nodes: [l__mod___features_denseblock4_denselayer2_norm2, l__mod___features_denseblock4_denselayer2_relu2, new_features_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf637 = extern_kernels.convolution(buf636, arg275_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf637, (4, 32, 7, 7), (1568, 49, 7, 1))
        del arg275_1
        del buf636
        buf638 = empty((4, 576, 7, 7), device='cuda', dtype=torch.float32)
        buf639 = buf638; del buf638  # reuse
        # Source Nodes: [bottleneck_output_88, cat_76, l__mod___features_denseblock4_denselayer3_norm1, l__mod___features_denseblock4_denselayer3_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_85.run(buf639, buf628, buf632, buf637, arg640_1, arg641_1, arg276_1, arg277_1, 112896, grid=grid(112896), stream=stream0)
        del arg276_1
        del arg277_1
        del arg640_1
        del arg641_1
        # Source Nodes: [bottleneck_output_88, l__mod___features_denseblock4_denselayer3_norm1, l__mod___features_denseblock4_denselayer3_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf640 = extern_kernels.convolution(buf639, arg278_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf640, (4, 128, 7, 7), (6272, 49, 7, 1))
        del arg278_1
        del buf639
        buf641 = buf640; del buf640  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer3_norm2, l__mod___features_denseblock4_denselayer3_relu2, new_features_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_83.run(buf641, arg643_1, arg644_1, arg279_1, arg280_1, 25088, grid=grid(25088), stream=stream0)
        del arg279_1
        del arg280_1
        del arg643_1
        del arg644_1
        # Source Nodes: [l__mod___features_denseblock4_denselayer3_norm2, l__mod___features_denseblock4_denselayer3_relu2, new_features_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf642 = extern_kernels.convolution(buf641, arg281_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf642, (4, 32, 7, 7), (1568, 49, 7, 1))
        del arg281_1
        del buf641
        buf643 = empty((4, 608, 7, 7), device='cuda', dtype=torch.float32)
        buf644 = buf643; del buf643  # reuse
        # Source Nodes: [bottleneck_output_90, cat_75, l__mod___features_denseblock4_denselayer4_norm1, l__mod___features_denseblock4_denselayer4_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_86.run(buf644, buf628, buf632, buf637, buf642, arg646_1, arg647_1, arg282_1, arg283_1, 119168, grid=grid(119168), stream=stream0)
        del arg282_1
        del arg283_1
        del arg646_1
        del arg647_1
        del buf628
        # Source Nodes: [bottleneck_output_90, l__mod___features_denseblock4_denselayer4_norm1, l__mod___features_denseblock4_denselayer4_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf645 = extern_kernels.convolution(buf644, arg284_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf645, (4, 128, 7, 7), (6272, 49, 7, 1))
        del arg284_1
        del buf644
        buf646 = buf645; del buf645  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer4_norm2, l__mod___features_denseblock4_denselayer4_relu2, new_features_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_83.run(buf646, arg649_1, arg650_1, arg285_1, arg286_1, 25088, grid=grid(25088), stream=stream0)
        del arg285_1
        del arg286_1
        del arg649_1
        del arg650_1
        # Source Nodes: [l__mod___features_denseblock4_denselayer4_norm2, l__mod___features_denseblock4_denselayer4_relu2, new_features_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf647 = extern_kernels.convolution(buf646, arg287_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf647, (4, 32, 7, 7), (1568, 49, 7, 1))
        del arg287_1
        del buf646
        buf649 = reinterpret_tensor(buf653, (4, 32, 7, 7), (31360, 49, 7, 1), 25088)  # alias
        buf659 = reinterpret_tensor(buf664, (4, 32, 7, 7), (32928, 49, 7, 1), 25088)  # alias
        buf670 = reinterpret_tensor(buf676, (4, 32, 7, 7), (34496, 49, 7, 1), 25088)  # alias
        buf682 = reinterpret_tensor(buf689, (4, 32, 7, 7), (36064, 49, 7, 1), 25088)  # alias
        buf695 = reinterpret_tensor(buf703, (4, 32, 7, 7), (37632, 49, 7, 1), 25088)  # alias
        buf709 = reinterpret_tensor(buf718, (4, 32, 7, 7), (39200, 49, 7, 1), 25088)  # alias
        # Source Nodes: [cat_69, cat_70, cat_71, cat_72, cat_73, cat_74], Original ATen: [aten.cat]
        triton_poi_fused_cat_87.run(buf632, buf649, buf659, buf670, buf682, buf695, buf709, 6272, grid=grid(6272), stream=stream0)
        buf650 = reinterpret_tensor(buf653, (4, 32, 7, 7), (31360, 49, 7, 1), 26656)  # alias
        buf660 = reinterpret_tensor(buf664, (4, 32, 7, 7), (32928, 49, 7, 1), 26656)  # alias
        buf671 = reinterpret_tensor(buf676, (4, 32, 7, 7), (34496, 49, 7, 1), 26656)  # alias
        buf683 = reinterpret_tensor(buf689, (4, 32, 7, 7), (36064, 49, 7, 1), 26656)  # alias
        buf696 = reinterpret_tensor(buf703, (4, 32, 7, 7), (37632, 49, 7, 1), 26656)  # alias
        buf710 = reinterpret_tensor(buf718, (4, 32, 7, 7), (39200, 49, 7, 1), 26656)  # alias
        # Source Nodes: [cat_69, cat_70, cat_71, cat_72, cat_73, cat_74], Original ATen: [aten.cat]
        triton_poi_fused_cat_87.run(buf637, buf650, buf660, buf671, buf683, buf696, buf710, 6272, grid=grid(6272), stream=stream0)
        buf651 = reinterpret_tensor(buf653, (4, 32, 7, 7), (31360, 49, 7, 1), 28224)  # alias
        buf661 = reinterpret_tensor(buf664, (4, 32, 7, 7), (32928, 49, 7, 1), 28224)  # alias
        buf672 = reinterpret_tensor(buf676, (4, 32, 7, 7), (34496, 49, 7, 1), 28224)  # alias
        buf684 = reinterpret_tensor(buf689, (4, 32, 7, 7), (36064, 49, 7, 1), 28224)  # alias
        buf697 = reinterpret_tensor(buf703, (4, 32, 7, 7), (37632, 49, 7, 1), 28224)  # alias
        buf711 = reinterpret_tensor(buf718, (4, 32, 7, 7), (39200, 49, 7, 1), 28224)  # alias
        # Source Nodes: [cat_69, cat_70, cat_71, cat_72, cat_73, cat_74], Original ATen: [aten.cat]
        triton_poi_fused_cat_87.run(buf642, buf651, buf661, buf672, buf684, buf697, buf711, 6272, grid=grid(6272), stream=stream0)
        buf652 = reinterpret_tensor(buf653, (4, 32, 7, 7), (31360, 49, 7, 1), 29792)  # alias
        buf662 = reinterpret_tensor(buf664, (4, 32, 7, 7), (32928, 49, 7, 1), 29792)  # alias
        buf673 = reinterpret_tensor(buf676, (4, 32, 7, 7), (34496, 49, 7, 1), 29792)  # alias
        buf685 = reinterpret_tensor(buf689, (4, 32, 7, 7), (36064, 49, 7, 1), 29792)  # alias
        buf698 = reinterpret_tensor(buf703, (4, 32, 7, 7), (37632, 49, 7, 1), 29792)  # alias
        buf712 = reinterpret_tensor(buf718, (4, 32, 7, 7), (39200, 49, 7, 1), 29792)  # alias
        # Source Nodes: [cat_69, cat_70, cat_71, cat_72, cat_73, cat_74], Original ATen: [aten.cat]
        triton_poi_fused_cat_87.run(buf647, buf652, buf662, buf673, buf685, buf698, buf712, 6272, grid=grid(6272), stream=stream0)
        buf654 = buf653; del buf653  # reuse
        # Source Nodes: [bottleneck_output_92, l__mod___features_denseblock4_denselayer5_norm1, l__mod___features_denseblock4_denselayer5_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_88.run(buf654, arg652_1, arg653_1, arg288_1, arg289_1, 125440, grid=grid(125440), stream=stream0)
        del arg288_1
        del arg289_1
        del arg652_1
        del arg653_1
        del buf648
        del buf649
        del buf650
        del buf651
        del buf652
        # Source Nodes: [bottleneck_output_92, l__mod___features_denseblock4_denselayer5_norm1, l__mod___features_denseblock4_denselayer5_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf655 = extern_kernels.convolution(buf654, arg290_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf655, (4, 128, 7, 7), (6272, 49, 7, 1))
        del arg290_1
        del buf654
        buf656 = buf655; del buf655  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer5_norm2, l__mod___features_denseblock4_denselayer5_relu2, new_features_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_83.run(buf656, arg655_1, arg656_1, arg291_1, arg292_1, 25088, grid=grid(25088), stream=stream0)
        del arg291_1
        del arg292_1
        del arg655_1
        del arg656_1
        # Source Nodes: [l__mod___features_denseblock4_denselayer5_norm2, l__mod___features_denseblock4_denselayer5_relu2, new_features_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf657 = extern_kernels.convolution(buf656, arg293_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf657, (4, 32, 7, 7), (1568, 49, 7, 1))
        del arg293_1
        del buf656
        buf663 = reinterpret_tensor(buf664, (4, 32, 7, 7), (32928, 49, 7, 1), 31360)  # alias
        buf674 = reinterpret_tensor(buf676, (4, 32, 7, 7), (34496, 49, 7, 1), 31360)  # alias
        buf686 = reinterpret_tensor(buf689, (4, 32, 7, 7), (36064, 49, 7, 1), 31360)  # alias
        buf699 = reinterpret_tensor(buf703, (4, 32, 7, 7), (37632, 49, 7, 1), 31360)  # alias
        buf713 = reinterpret_tensor(buf718, (4, 32, 7, 7), (39200, 49, 7, 1), 31360)  # alias
        # Source Nodes: [cat_69, cat_70, cat_71, cat_72, cat_73], Original ATen: [aten.cat]
        triton_poi_fused_cat_89.run(buf657, buf663, buf674, buf686, buf699, buf713, 6272, grid=grid(6272), stream=stream0)
        buf665 = buf664; del buf664  # reuse
        # Source Nodes: [bottleneck_output_94, l__mod___features_denseblock4_denselayer6_norm1, l__mod___features_denseblock4_denselayer6_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_90.run(buf665, arg658_1, arg659_1, arg294_1, arg295_1, 131712, grid=grid(131712), stream=stream0)
        del arg294_1
        del arg295_1
        del arg658_1
        del arg659_1
        del buf658
        del buf659
        del buf660
        del buf661
        del buf662
        del buf663
        # Source Nodes: [bottleneck_output_94, l__mod___features_denseblock4_denselayer6_norm1, l__mod___features_denseblock4_denselayer6_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf666 = extern_kernels.convolution(buf665, arg296_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf666, (4, 128, 7, 7), (6272, 49, 7, 1))
        del arg296_1
        del buf665
        buf667 = buf666; del buf666  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer6_norm2, l__mod___features_denseblock4_denselayer6_relu2, new_features_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_83.run(buf667, arg661_1, arg662_1, arg297_1, arg298_1, 25088, grid=grid(25088), stream=stream0)
        del arg297_1
        del arg298_1
        del arg661_1
        del arg662_1
        # Source Nodes: [l__mod___features_denseblock4_denselayer6_norm2, l__mod___features_denseblock4_denselayer6_relu2, new_features_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf668 = extern_kernels.convolution(buf667, arg299_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf668, (4, 32, 7, 7), (1568, 49, 7, 1))
        del arg299_1
        del buf667
        buf675 = reinterpret_tensor(buf676, (4, 32, 7, 7), (34496, 49, 7, 1), 32928)  # alias
        buf687 = reinterpret_tensor(buf689, (4, 32, 7, 7), (36064, 49, 7, 1), 32928)  # alias
        buf700 = reinterpret_tensor(buf703, (4, 32, 7, 7), (37632, 49, 7, 1), 32928)  # alias
        buf714 = reinterpret_tensor(buf718, (4, 32, 7, 7), (39200, 49, 7, 1), 32928)  # alias
        buf729 = reinterpret_tensor(buf734, (4, 32, 7, 7), (40768, 49, 7, 1), 32928)  # alias
        # Source Nodes: [cat_68, cat_69, cat_70, cat_71, cat_72], Original ATen: [aten.cat]
        triton_poi_fused_cat_91.run(buf668, buf675, buf687, buf700, buf714, buf729, 6272, grid=grid(6272), stream=stream0)
        buf677 = buf676; del buf676  # reuse
        # Source Nodes: [bottleneck_output_96, l__mod___features_denseblock4_denselayer7_norm1, l__mod___features_denseblock4_denselayer7_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_92.run(buf677, arg664_1, arg665_1, arg300_1, arg301_1, 137984, grid=grid(137984), stream=stream0)
        del arg300_1
        del arg301_1
        del arg664_1
        del arg665_1
        del buf669
        del buf670
        del buf671
        del buf672
        del buf673
        del buf674
        del buf675
        # Source Nodes: [bottleneck_output_96, l__mod___features_denseblock4_denselayer7_norm1, l__mod___features_denseblock4_denselayer7_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf678 = extern_kernels.convolution(buf677, arg302_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf678, (4, 128, 7, 7), (6272, 49, 7, 1))
        del arg302_1
        del buf677
        buf679 = buf678; del buf678  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer7_norm2, l__mod___features_denseblock4_denselayer7_relu2, new_features_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_83.run(buf679, arg667_1, arg668_1, arg303_1, arg304_1, 25088, grid=grid(25088), stream=stream0)
        del arg303_1
        del arg304_1
        del arg667_1
        del arg668_1
        # Source Nodes: [l__mod___features_denseblock4_denselayer7_norm2, l__mod___features_denseblock4_denselayer7_relu2, new_features_96], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf680 = extern_kernels.convolution(buf679, arg305_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf680, (4, 32, 7, 7), (1568, 49, 7, 1))
        del arg305_1
        del buf679
        buf688 = reinterpret_tensor(buf689, (4, 32, 7, 7), (36064, 49, 7, 1), 34496)  # alias
        buf701 = reinterpret_tensor(buf703, (4, 32, 7, 7), (37632, 49, 7, 1), 34496)  # alias
        buf715 = reinterpret_tensor(buf718, (4, 32, 7, 7), (39200, 49, 7, 1), 34496)  # alias
        buf730 = reinterpret_tensor(buf734, (4, 32, 7, 7), (40768, 49, 7, 1), 34496)  # alias
        buf746 = reinterpret_tensor(buf751, (4, 32, 7, 7), (42336, 49, 7, 1), 34496)  # alias
        # Source Nodes: [cat_67, cat_68, cat_69, cat_70, cat_71], Original ATen: [aten.cat]
        triton_poi_fused_cat_93.run(buf680, buf688, buf701, buf715, buf730, buf746, 6272, grid=grid(6272), stream=stream0)
        buf690 = buf689; del buf689  # reuse
        # Source Nodes: [bottleneck_output_98, l__mod___features_denseblock4_denselayer8_norm1, l__mod___features_denseblock4_denselayer8_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_94.run(buf690, arg670_1, arg671_1, arg306_1, arg307_1, 144256, grid=grid(144256), stream=stream0)
        del arg306_1
        del arg307_1
        del arg670_1
        del arg671_1
        del buf681
        del buf682
        del buf683
        del buf684
        del buf685
        del buf686
        del buf687
        del buf688
        # Source Nodes: [bottleneck_output_98, l__mod___features_denseblock4_denselayer8_norm1, l__mod___features_denseblock4_denselayer8_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf691 = extern_kernels.convolution(buf690, arg308_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf691, (4, 128, 7, 7), (6272, 49, 7, 1))
        del arg308_1
        del buf690
        buf692 = buf691; del buf691  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer8_norm2, l__mod___features_denseblock4_denselayer8_relu2, new_features_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_83.run(buf692, arg673_1, arg674_1, arg309_1, arg310_1, 25088, grid=grid(25088), stream=stream0)
        del arg309_1
        del arg310_1
        del arg673_1
        del arg674_1
        # Source Nodes: [l__mod___features_denseblock4_denselayer8_norm2, l__mod___features_denseblock4_denselayer8_relu2, new_features_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf693 = extern_kernels.convolution(buf692, arg311_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf693, (4, 32, 7, 7), (1568, 49, 7, 1))
        del arg311_1
        del buf692
        buf702 = reinterpret_tensor(buf703, (4, 32, 7, 7), (37632, 49, 7, 1), 36064)  # alias
        buf716 = reinterpret_tensor(buf718, (4, 32, 7, 7), (39200, 49, 7, 1), 36064)  # alias
        buf731 = reinterpret_tensor(buf734, (4, 32, 7, 7), (40768, 49, 7, 1), 36064)  # alias
        buf747 = reinterpret_tensor(buf751, (4, 32, 7, 7), (42336, 49, 7, 1), 36064)  # alias
        buf764 = reinterpret_tensor(buf769, (4, 32, 7, 7), (43904, 49, 7, 1), 36064)  # alias
        # Source Nodes: [cat_66, cat_67, cat_68, cat_69, cat_70], Original ATen: [aten.cat]
        triton_poi_fused_cat_95.run(buf693, buf702, buf716, buf731, buf747, buf764, 6272, grid=grid(6272), stream=stream0)
        buf704 = buf703; del buf703  # reuse
        # Source Nodes: [bottleneck_output_100, l__mod___features_denseblock4_denselayer9_norm1, l__mod___features_denseblock4_denselayer9_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_96.run(buf704, arg676_1, arg677_1, arg312_1, arg313_1, 150528, grid=grid(150528), stream=stream0)
        del arg312_1
        del arg313_1
        del arg676_1
        del arg677_1
        del buf694
        del buf695
        del buf696
        del buf697
        del buf698
        del buf699
        del buf700
        del buf701
        del buf702
        # Source Nodes: [bottleneck_output_100, l__mod___features_denseblock4_denselayer9_norm1, l__mod___features_denseblock4_denselayer9_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf705 = extern_kernels.convolution(buf704, arg314_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf705, (4, 128, 7, 7), (6272, 49, 7, 1))
        del arg314_1
        del buf704
        buf706 = buf705; del buf705  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer9_norm2, l__mod___features_denseblock4_denselayer9_relu2, new_features_100], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_83.run(buf706, arg679_1, arg680_1, arg315_1, arg316_1, 25088, grid=grid(25088), stream=stream0)
        del arg315_1
        del arg316_1
        del arg679_1
        del arg680_1
        # Source Nodes: [l__mod___features_denseblock4_denselayer9_norm2, l__mod___features_denseblock4_denselayer9_relu2, new_features_100], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf707 = extern_kernels.convolution(buf706, arg317_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf707, (4, 32, 7, 7), (1568, 49, 7, 1))
        del arg317_1
        del buf706
        buf717 = reinterpret_tensor(buf718, (4, 32, 7, 7), (39200, 49, 7, 1), 37632)  # alias
        buf732 = reinterpret_tensor(buf734, (4, 32, 7, 7), (40768, 49, 7, 1), 37632)  # alias
        buf748 = reinterpret_tensor(buf751, (4, 32, 7, 7), (42336, 49, 7, 1), 37632)  # alias
        buf765 = reinterpret_tensor(buf769, (4, 32, 7, 7), (43904, 49, 7, 1), 37632)  # alias
        # Source Nodes: [cat_66, cat_67, cat_68, cat_69], Original ATen: [aten.cat]
        triton_poi_fused_cat_97.run(buf707, buf717, buf732, buf748, buf765, 6272, grid=grid(6272), stream=stream0)
        buf719 = buf718; del buf718  # reuse
        # Source Nodes: [bottleneck_output_102, l__mod___features_denseblock4_denselayer10_norm1, l__mod___features_denseblock4_denselayer10_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_98.run(buf719, arg682_1, arg683_1, arg318_1, arg319_1, 156800, grid=grid(156800), stream=stream0)
        del arg318_1
        del arg319_1
        del arg682_1
        del arg683_1
        del buf708
        del buf709
        del buf710
        del buf711
        del buf712
        del buf713
        del buf714
        del buf715
        del buf716
        del buf717
        # Source Nodes: [bottleneck_output_102, l__mod___features_denseblock4_denselayer10_norm1, l__mod___features_denseblock4_denselayer10_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf720 = extern_kernels.convolution(buf719, arg320_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf720, (4, 128, 7, 7), (6272, 49, 7, 1))
        del arg320_1
        del buf719
        buf721 = buf720; del buf720  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer10_norm2, l__mod___features_denseblock4_denselayer10_relu2, new_features_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_83.run(buf721, arg685_1, arg686_1, arg321_1, arg322_1, 25088, grid=grid(25088), stream=stream0)
        del arg321_1
        del arg322_1
        del arg685_1
        del arg686_1
        # Source Nodes: [l__mod___features_denseblock4_denselayer10_norm2, l__mod___features_denseblock4_denselayer10_relu2, new_features_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf722 = extern_kernels.convolution(buf721, arg323_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf722, (4, 32, 7, 7), (1568, 49, 7, 1))
        del arg323_1
        del buf721
        buf724 = reinterpret_tensor(buf734, (4, 32, 7, 7), (40768, 49, 7, 1), 25088)  # alias
        buf740 = reinterpret_tensor(buf751, (4, 32, 7, 7), (42336, 49, 7, 1), 25088)  # alias
        buf757 = reinterpret_tensor(buf769, (4, 32, 7, 7), (43904, 49, 7, 1), 25088)  # alias
        buf775 = reinterpret_tensor(buf788, (4, 32, 7, 7), (45472, 49, 7, 1), 25088)  # alias
        # Source Nodes: [cat_65, cat_66, cat_67, cat_68], Original ATen: [aten.cat]
        triton_poi_fused_cat_99.run(buf632, buf724, buf740, buf757, buf775, 6272, grid=grid(6272), stream=stream0)
        buf725 = reinterpret_tensor(buf734, (4, 32, 7, 7), (40768, 49, 7, 1), 26656)  # alias
        buf741 = reinterpret_tensor(buf751, (4, 32, 7, 7), (42336, 49, 7, 1), 26656)  # alias
        buf758 = reinterpret_tensor(buf769, (4, 32, 7, 7), (43904, 49, 7, 1), 26656)  # alias
        buf776 = reinterpret_tensor(buf788, (4, 32, 7, 7), (45472, 49, 7, 1), 26656)  # alias
        # Source Nodes: [cat_65, cat_66, cat_67, cat_68], Original ATen: [aten.cat]
        triton_poi_fused_cat_99.run(buf637, buf725, buf741, buf758, buf776, 6272, grid=grid(6272), stream=stream0)
        buf726 = reinterpret_tensor(buf734, (4, 32, 7, 7), (40768, 49, 7, 1), 28224)  # alias
        buf742 = reinterpret_tensor(buf751, (4, 32, 7, 7), (42336, 49, 7, 1), 28224)  # alias
        buf759 = reinterpret_tensor(buf769, (4, 32, 7, 7), (43904, 49, 7, 1), 28224)  # alias
        buf777 = reinterpret_tensor(buf788, (4, 32, 7, 7), (45472, 49, 7, 1), 28224)  # alias
        # Source Nodes: [cat_65, cat_66, cat_67, cat_68], Original ATen: [aten.cat]
        triton_poi_fused_cat_99.run(buf642, buf726, buf742, buf759, buf777, 6272, grid=grid(6272), stream=stream0)
        buf727 = reinterpret_tensor(buf734, (4, 32, 7, 7), (40768, 49, 7, 1), 29792)  # alias
        buf743 = reinterpret_tensor(buf751, (4, 32, 7, 7), (42336, 49, 7, 1), 29792)  # alias
        buf760 = reinterpret_tensor(buf769, (4, 32, 7, 7), (43904, 49, 7, 1), 29792)  # alias
        buf778 = reinterpret_tensor(buf788, (4, 32, 7, 7), (45472, 49, 7, 1), 29792)  # alias
        # Source Nodes: [cat_65, cat_66, cat_67, cat_68], Original ATen: [aten.cat]
        triton_poi_fused_cat_99.run(buf647, buf727, buf743, buf760, buf778, 6272, grid=grid(6272), stream=stream0)
        buf728 = reinterpret_tensor(buf734, (4, 32, 7, 7), (40768, 49, 7, 1), 31360)  # alias
        buf744 = reinterpret_tensor(buf751, (4, 32, 7, 7), (42336, 49, 7, 1), 31360)  # alias
        buf761 = reinterpret_tensor(buf769, (4, 32, 7, 7), (43904, 49, 7, 1), 31360)  # alias
        buf779 = reinterpret_tensor(buf788, (4, 32, 7, 7), (45472, 49, 7, 1), 31360)  # alias
        # Source Nodes: [cat_65, cat_66, cat_67, cat_68], Original ATen: [aten.cat]
        triton_poi_fused_cat_99.run(buf657, buf728, buf744, buf761, buf779, 6272, grid=grid(6272), stream=stream0)
        buf733 = reinterpret_tensor(buf734, (4, 32, 7, 7), (40768, 49, 7, 1), 39200)  # alias
        buf749 = reinterpret_tensor(buf751, (4, 32, 7, 7), (42336, 49, 7, 1), 39200)  # alias
        buf766 = reinterpret_tensor(buf769, (4, 32, 7, 7), (43904, 49, 7, 1), 39200)  # alias
        buf784 = reinterpret_tensor(buf788, (4, 32, 7, 7), (45472, 49, 7, 1), 39200)  # alias
        # Source Nodes: [cat_65, cat_66, cat_67, cat_68], Original ATen: [aten.cat]
        triton_poi_fused_cat_99.run(buf722, buf733, buf749, buf766, buf784, 6272, grid=grid(6272), stream=stream0)
        buf735 = buf734; del buf734  # reuse
        # Source Nodes: [bottleneck_output_104, l__mod___features_denseblock4_denselayer11_norm1, l__mod___features_denseblock4_denselayer11_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_100.run(buf735, arg688_1, arg689_1, arg324_1, arg325_1, 163072, grid=grid(163072), stream=stream0)
        del arg324_1
        del arg325_1
        del arg688_1
        del arg689_1
        del buf723
        del buf724
        del buf725
        del buf726
        del buf727
        del buf728
        del buf729
        del buf730
        del buf731
        del buf732
        del buf733
        # Source Nodes: [bottleneck_output_104, l__mod___features_denseblock4_denselayer11_norm1, l__mod___features_denseblock4_denselayer11_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf736 = extern_kernels.convolution(buf735, arg326_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf736, (4, 128, 7, 7), (6272, 49, 7, 1))
        del arg326_1
        del buf735
        buf737 = buf736; del buf736  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer11_norm2, l__mod___features_denseblock4_denselayer11_relu2, new_features_104], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_83.run(buf737, arg691_1, arg692_1, arg327_1, arg328_1, 25088, grid=grid(25088), stream=stream0)
        del arg327_1
        del arg328_1
        del arg691_1
        del arg692_1
        # Source Nodes: [l__mod___features_denseblock4_denselayer11_norm2, l__mod___features_denseblock4_denselayer11_relu2, new_features_104], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf738 = extern_kernels.convolution(buf737, arg329_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf738, (4, 32, 7, 7), (1568, 49, 7, 1))
        del arg329_1
        del buf737
        buf745 = reinterpret_tensor(buf751, (4, 32, 7, 7), (42336, 49, 7, 1), 32928)  # alias
        buf762 = reinterpret_tensor(buf769, (4, 32, 7, 7), (43904, 49, 7, 1), 32928)  # alias
        buf780 = reinterpret_tensor(buf788, (4, 32, 7, 7), (45472, 49, 7, 1), 32928)  # alias
        buf799 = reinterpret_tensor(buf808, (4, 32, 7, 7), (47040, 49, 7, 1), 32928)  # alias
        # Source Nodes: [cat_64, cat_65, cat_66, cat_67], Original ATen: [aten.cat]
        triton_poi_fused_cat_101.run(buf668, buf745, buf762, buf780, buf799, 6272, grid=grid(6272), stream=stream0)
        buf750 = reinterpret_tensor(buf751, (4, 32, 7, 7), (42336, 49, 7, 1), 40768)  # alias
        buf767 = reinterpret_tensor(buf769, (4, 32, 7, 7), (43904, 49, 7, 1), 40768)  # alias
        buf785 = reinterpret_tensor(buf788, (4, 32, 7, 7), (45472, 49, 7, 1), 40768)  # alias
        buf804 = reinterpret_tensor(buf808, (4, 32, 7, 7), (47040, 49, 7, 1), 40768)  # alias
        # Source Nodes: [cat_64, cat_65, cat_66, cat_67], Original ATen: [aten.cat]
        triton_poi_fused_cat_101.run(buf738, buf750, buf767, buf785, buf804, 6272, grid=grid(6272), stream=stream0)
        buf752 = buf751; del buf751  # reuse
        # Source Nodes: [bottleneck_output_106, l__mod___features_denseblock4_denselayer12_norm1, l__mod___features_denseblock4_denselayer12_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_102.run(buf752, arg694_1, arg695_1, arg330_1, arg331_1, 169344, grid=grid(169344), stream=stream0)
        del arg330_1
        del arg331_1
        del arg694_1
        del arg695_1
        del buf739
        del buf740
        del buf741
        del buf742
        del buf743
        del buf744
        del buf745
        del buf746
        del buf747
        del buf748
        del buf749
        del buf750
        # Source Nodes: [bottleneck_output_106, l__mod___features_denseblock4_denselayer12_norm1, l__mod___features_denseblock4_denselayer12_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf753 = extern_kernels.convolution(buf752, arg332_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf753, (4, 128, 7, 7), (6272, 49, 7, 1))
        del arg332_1
        del buf752
        buf754 = buf753; del buf753  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer12_norm2, l__mod___features_denseblock4_denselayer12_relu2, new_features_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_83.run(buf754, arg697_1, arg698_1, arg333_1, arg334_1, 25088, grid=grid(25088), stream=stream0)
        del arg333_1
        del arg334_1
        del arg697_1
        del arg698_1
        # Source Nodes: [l__mod___features_denseblock4_denselayer12_norm2, l__mod___features_denseblock4_denselayer12_relu2, new_features_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf755 = extern_kernels.convolution(buf754, arg335_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf755, (4, 32, 7, 7), (1568, 49, 7, 1))
        del arg335_1
        del buf754
        buf763 = reinterpret_tensor(buf769, (4, 32, 7, 7), (43904, 49, 7, 1), 34496)  # alias
        buf781 = reinterpret_tensor(buf788, (4, 32, 7, 7), (45472, 49, 7, 1), 34496)  # alias
        buf800 = reinterpret_tensor(buf808, (4, 32, 7, 7), (47040, 49, 7, 1), 34496)  # alias
        buf820 = reinterpret_tensor(buf829, (4, 32, 7, 7), (48608, 49, 7, 1), 34496)  # alias
        # Source Nodes: [cat_63, cat_64, cat_65, cat_66], Original ATen: [aten.cat]
        triton_poi_fused_cat_103.run(buf680, buf763, buf781, buf800, buf820, 6272, grid=grid(6272), stream=stream0)
        buf768 = reinterpret_tensor(buf769, (4, 32, 7, 7), (43904, 49, 7, 1), 42336)  # alias
        buf786 = reinterpret_tensor(buf788, (4, 32, 7, 7), (45472, 49, 7, 1), 42336)  # alias
        buf805 = reinterpret_tensor(buf808, (4, 32, 7, 7), (47040, 49, 7, 1), 42336)  # alias
        buf825 = reinterpret_tensor(buf829, (4, 32, 7, 7), (48608, 49, 7, 1), 42336)  # alias
        # Source Nodes: [cat_63, cat_64, cat_65, cat_66], Original ATen: [aten.cat]
        triton_poi_fused_cat_103.run(buf755, buf768, buf786, buf805, buf825, 6272, grid=grid(6272), stream=stream0)
        buf770 = buf769; del buf769  # reuse
        # Source Nodes: [bottleneck_output_108, l__mod___features_denseblock4_denselayer13_norm1, l__mod___features_denseblock4_denselayer13_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_104.run(buf770, arg700_1, arg701_1, arg336_1, arg337_1, 175616, grid=grid(175616), stream=stream0)
        del arg336_1
        del arg337_1
        del arg700_1
        del arg701_1
        del buf756
        del buf757
        del buf758
        del buf759
        del buf760
        del buf761
        del buf762
        del buf763
        del buf764
        del buf765
        del buf766
        del buf767
        del buf768
        # Source Nodes: [bottleneck_output_108, l__mod___features_denseblock4_denselayer13_norm1, l__mod___features_denseblock4_denselayer13_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf771 = extern_kernels.convolution(buf770, arg338_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf771, (4, 128, 7, 7), (6272, 49, 7, 1))
        del arg338_1
        del buf770
        buf772 = buf771; del buf771  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer13_norm2, l__mod___features_denseblock4_denselayer13_relu2, new_features_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_83.run(buf772, arg703_1, arg704_1, arg339_1, arg340_1, 25088, grid=grid(25088), stream=stream0)
        del arg339_1
        del arg340_1
        del arg703_1
        del arg704_1
        # Source Nodes: [l__mod___features_denseblock4_denselayer13_norm2, l__mod___features_denseblock4_denselayer13_relu2, new_features_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf773 = extern_kernels.convolution(buf772, arg341_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf773, (4, 32, 7, 7), (1568, 49, 7, 1))
        del arg341_1
        del buf772
        buf782 = reinterpret_tensor(buf788, (4, 32, 7, 7), (45472, 49, 7, 1), 36064)  # alias
        buf801 = reinterpret_tensor(buf808, (4, 32, 7, 7), (47040, 49, 7, 1), 36064)  # alias
        buf821 = reinterpret_tensor(buf829, (4, 32, 7, 7), (48608, 49, 7, 1), 36064)  # alias
        buf842 = reinterpret_tensor(buf851, (4, 32, 7, 7), (50176, 49, 7, 1), 36064)  # alias
        # Source Nodes: [cat_62, cat_63, cat_64, cat_65], Original ATen: [aten.cat]
        triton_poi_fused_cat_105.run(buf693, buf782, buf801, buf821, buf842, 6272, grid=grid(6272), stream=stream0)
        del buf693
        buf783 = reinterpret_tensor(buf788, (4, 32, 7, 7), (45472, 49, 7, 1), 37632)  # alias
        buf802 = reinterpret_tensor(buf808, (4, 32, 7, 7), (47040, 49, 7, 1), 37632)  # alias
        buf822 = reinterpret_tensor(buf829, (4, 32, 7, 7), (48608, 49, 7, 1), 37632)  # alias
        buf843 = reinterpret_tensor(buf851, (4, 32, 7, 7), (50176, 49, 7, 1), 37632)  # alias
        # Source Nodes: [cat_62, cat_63, cat_64, cat_65], Original ATen: [aten.cat]
        triton_poi_fused_cat_105.run(buf707, buf783, buf802, buf822, buf843, 6272, grid=grid(6272), stream=stream0)
        del buf707
        buf787 = reinterpret_tensor(buf788, (4, 32, 7, 7), (45472, 49, 7, 1), 43904)  # alias
        buf806 = reinterpret_tensor(buf808, (4, 32, 7, 7), (47040, 49, 7, 1), 43904)  # alias
        buf826 = reinterpret_tensor(buf829, (4, 32, 7, 7), (48608, 49, 7, 1), 43904)  # alias
        buf847 = reinterpret_tensor(buf851, (4, 32, 7, 7), (50176, 49, 7, 1), 43904)  # alias
        # Source Nodes: [cat_62, cat_63, cat_64, cat_65], Original ATen: [aten.cat]
        triton_poi_fused_cat_105.run(buf773, buf787, buf806, buf826, buf847, 6272, grid=grid(6272), stream=stream0)
        del buf773
        buf789 = buf788; del buf788  # reuse
        # Source Nodes: [bottleneck_output_110, l__mod___features_denseblock4_denselayer14_norm1, l__mod___features_denseblock4_denselayer14_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_106.run(buf789, arg706_1, arg707_1, arg342_1, arg343_1, 181888, grid=grid(181888), stream=stream0)
        del arg342_1
        del arg343_1
        del arg706_1
        del arg707_1
        del buf774
        del buf775
        del buf776
        del buf777
        del buf778
        del buf779
        del buf780
        del buf781
        del buf782
        del buf783
        del buf784
        del buf785
        del buf786
        del buf787
        # Source Nodes: [bottleneck_output_110, l__mod___features_denseblock4_denselayer14_norm1, l__mod___features_denseblock4_denselayer14_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf790 = extern_kernels.convolution(buf789, arg344_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf790, (4, 128, 7, 7), (6272, 49, 7, 1))
        del arg344_1
        del buf789
        buf791 = buf790; del buf790  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer14_norm2, l__mod___features_denseblock4_denselayer14_relu2, new_features_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_83.run(buf791, arg709_1, arg710_1, arg345_1, arg346_1, 25088, grid=grid(25088), stream=stream0)
        del arg345_1
        del arg346_1
        del arg709_1
        del arg710_1
        # Source Nodes: [l__mod___features_denseblock4_denselayer14_norm2, l__mod___features_denseblock4_denselayer14_relu2, new_features_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf792 = extern_kernels.convolution(buf791, arg347_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf792, (4, 32, 7, 7), (1568, 49, 7, 1))
        del arg347_1
        del buf791
        buf794 = reinterpret_tensor(buf808, (4, 32, 7, 7), (47040, 49, 7, 1), 25088)  # alias
        buf814 = reinterpret_tensor(buf829, (4, 32, 7, 7), (48608, 49, 7, 1), 25088)  # alias
        buf835 = reinterpret_tensor(buf851, (4, 32, 7, 7), (50176, 49, 7, 1), 25088)  # alias
        # Source Nodes: [cat_62, cat_63, cat_64], Original ATen: [aten.cat]
        triton_poi_fused_cat_107.run(buf632, buf794, buf814, buf835, 6272, grid=grid(6272), stream=stream0)
        del buf632
        buf795 = reinterpret_tensor(buf808, (4, 32, 7, 7), (47040, 49, 7, 1), 26656)  # alias
        buf815 = reinterpret_tensor(buf829, (4, 32, 7, 7), (48608, 49, 7, 1), 26656)  # alias
        buf836 = reinterpret_tensor(buf851, (4, 32, 7, 7), (50176, 49, 7, 1), 26656)  # alias
        # Source Nodes: [cat_62, cat_63, cat_64], Original ATen: [aten.cat]
        triton_poi_fused_cat_107.run(buf637, buf795, buf815, buf836, 6272, grid=grid(6272), stream=stream0)
        del buf637
        buf796 = reinterpret_tensor(buf808, (4, 32, 7, 7), (47040, 49, 7, 1), 28224)  # alias
        buf816 = reinterpret_tensor(buf829, (4, 32, 7, 7), (48608, 49, 7, 1), 28224)  # alias
        buf837 = reinterpret_tensor(buf851, (4, 32, 7, 7), (50176, 49, 7, 1), 28224)  # alias
        # Source Nodes: [cat_62, cat_63, cat_64], Original ATen: [aten.cat]
        triton_poi_fused_cat_107.run(buf642, buf796, buf816, buf837, 6272, grid=grid(6272), stream=stream0)
        del buf642
        buf797 = reinterpret_tensor(buf808, (4, 32, 7, 7), (47040, 49, 7, 1), 29792)  # alias
        buf817 = reinterpret_tensor(buf829, (4, 32, 7, 7), (48608, 49, 7, 1), 29792)  # alias
        buf838 = reinterpret_tensor(buf851, (4, 32, 7, 7), (50176, 49, 7, 1), 29792)  # alias
        # Source Nodes: [cat_62, cat_63, cat_64], Original ATen: [aten.cat]
        triton_poi_fused_cat_107.run(buf647, buf797, buf817, buf838, 6272, grid=grid(6272), stream=stream0)
        del buf647
        buf798 = reinterpret_tensor(buf808, (4, 32, 7, 7), (47040, 49, 7, 1), 31360)  # alias
        buf818 = reinterpret_tensor(buf829, (4, 32, 7, 7), (48608, 49, 7, 1), 31360)  # alias
        buf839 = reinterpret_tensor(buf851, (4, 32, 7, 7), (50176, 49, 7, 1), 31360)  # alias
        # Source Nodes: [cat_62, cat_63, cat_64], Original ATen: [aten.cat]
        triton_poi_fused_cat_107.run(buf657, buf798, buf818, buf839, 6272, grid=grid(6272), stream=stream0)
        del buf657
        buf803 = reinterpret_tensor(buf808, (4, 32, 7, 7), (47040, 49, 7, 1), 39200)  # alias
        buf823 = reinterpret_tensor(buf829, (4, 32, 7, 7), (48608, 49, 7, 1), 39200)  # alias
        buf844 = reinterpret_tensor(buf851, (4, 32, 7, 7), (50176, 49, 7, 1), 39200)  # alias
        # Source Nodes: [cat_62, cat_63, cat_64], Original ATen: [aten.cat]
        triton_poi_fused_cat_107.run(buf722, buf803, buf823, buf844, 6272, grid=grid(6272), stream=stream0)
        del buf722
        buf807 = reinterpret_tensor(buf808, (4, 32, 7, 7), (47040, 49, 7, 1), 45472)  # alias
        buf827 = reinterpret_tensor(buf829, (4, 32, 7, 7), (48608, 49, 7, 1), 45472)  # alias
        buf848 = reinterpret_tensor(buf851, (4, 32, 7, 7), (50176, 49, 7, 1), 45472)  # alias
        # Source Nodes: [cat_62, cat_63, cat_64], Original ATen: [aten.cat]
        triton_poi_fused_cat_107.run(buf792, buf807, buf827, buf848, 6272, grid=grid(6272), stream=stream0)
        del buf792
        buf809 = buf808; del buf808  # reuse
        # Source Nodes: [bottleneck_output_112, l__mod___features_denseblock4_denselayer15_norm1, l__mod___features_denseblock4_denselayer15_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_108.run(buf809, arg712_1, arg713_1, arg348_1, arg349_1, 188160, grid=grid(188160), stream=stream0)
        del arg348_1
        del arg349_1
        del arg712_1
        del arg713_1
        del buf793
        del buf794
        del buf795
        del buf796
        del buf797
        del buf798
        del buf799
        del buf800
        del buf801
        del buf802
        del buf803
        del buf804
        del buf805
        del buf806
        del buf807
        # Source Nodes: [bottleneck_output_112, l__mod___features_denseblock4_denselayer15_norm1, l__mod___features_denseblock4_denselayer15_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf810 = extern_kernels.convolution(buf809, arg350_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf810, (4, 128, 7, 7), (6272, 49, 7, 1))
        del arg350_1
        del buf809
        buf811 = buf810; del buf810  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer15_norm2, l__mod___features_denseblock4_denselayer15_relu2, new_features_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_83.run(buf811, arg715_1, arg716_1, arg351_1, arg352_1, 25088, grid=grid(25088), stream=stream0)
        del arg351_1
        del arg352_1
        del arg715_1
        del arg716_1
        # Source Nodes: [l__mod___features_denseblock4_denselayer15_norm2, l__mod___features_denseblock4_denselayer15_relu2, new_features_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf812 = extern_kernels.convolution(buf811, arg353_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf812, (4, 32, 7, 7), (1568, 49, 7, 1))
        del arg353_1
        del buf811
        buf819 = reinterpret_tensor(buf829, (4, 32, 7, 7), (48608, 49, 7, 1), 32928)  # alias
        buf840 = reinterpret_tensor(buf851, (4, 32, 7, 7), (50176, 49, 7, 1), 32928)  # alias
        # Source Nodes: [cat_62, cat_63], Original ATen: [aten.cat]
        triton_poi_fused_cat_109.run(buf668, buf819, buf840, 6272, grid=grid(6272), stream=stream0)
        del buf668
        buf824 = reinterpret_tensor(buf829, (4, 32, 7, 7), (48608, 49, 7, 1), 40768)  # alias
        buf845 = reinterpret_tensor(buf851, (4, 32, 7, 7), (50176, 49, 7, 1), 40768)  # alias
        # Source Nodes: [cat_62, cat_63], Original ATen: [aten.cat]
        triton_poi_fused_cat_109.run(buf738, buf824, buf845, 6272, grid=grid(6272), stream=stream0)
        del buf738
        buf828 = reinterpret_tensor(buf829, (4, 32, 7, 7), (48608, 49, 7, 1), 47040)  # alias
        buf849 = reinterpret_tensor(buf851, (4, 32, 7, 7), (50176, 49, 7, 1), 47040)  # alias
        # Source Nodes: [cat_62, cat_63], Original ATen: [aten.cat]
        triton_poi_fused_cat_109.run(buf812, buf828, buf849, 6272, grid=grid(6272), stream=stream0)
        del buf812
        buf830 = buf829; del buf829  # reuse
        # Source Nodes: [bottleneck_output_114, l__mod___features_denseblock4_denselayer16_norm1, l__mod___features_denseblock4_denselayer16_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_110.run(buf830, arg718_1, arg719_1, arg354_1, arg355_1, 194432, grid=grid(194432), stream=stream0)
        del arg354_1
        del arg355_1
        del arg718_1
        del arg719_1
        del buf813
        del buf814
        del buf815
        del buf816
        del buf817
        del buf818
        del buf819
        del buf820
        del buf821
        del buf822
        del buf823
        del buf824
        del buf825
        del buf826
        del buf827
        del buf828
        # Source Nodes: [bottleneck_output_114, l__mod___features_denseblock4_denselayer16_norm1, l__mod___features_denseblock4_denselayer16_relu1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf831 = extern_kernels.convolution(buf830, arg356_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf831, (4, 128, 7, 7), (6272, 49, 7, 1))
        del arg356_1
        del buf830
        buf832 = buf831; del buf831  # reuse
        # Source Nodes: [l__mod___features_denseblock4_denselayer16_norm2, l__mod___features_denseblock4_denselayer16_relu2, new_features_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_83.run(buf832, arg721_1, arg722_1, arg357_1, arg358_1, 25088, grid=grid(25088), stream=stream0)
        del arg357_1
        del arg358_1
        del arg721_1
        del arg722_1
        # Source Nodes: [l__mod___features_denseblock4_denselayer16_norm2, l__mod___features_denseblock4_denselayer16_relu2, new_features_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf833 = extern_kernels.convolution(buf832, arg359_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf833, (4, 32, 7, 7), (1568, 49, 7, 1))
        del arg359_1
        del buf832
        buf841 = reinterpret_tensor(buf851, (4, 32, 7, 7), (50176, 49, 7, 1), 34496)  # alias
        # Source Nodes: [cat_62], Original ATen: [aten.cat]
        triton_poi_fused_cat_111.run(buf680, buf841, 6272, grid=grid(6272), stream=stream0)
        del buf680
        buf846 = reinterpret_tensor(buf851, (4, 32, 7, 7), (50176, 49, 7, 1), 42336)  # alias
        # Source Nodes: [cat_62], Original ATen: [aten.cat]
        triton_poi_fused_cat_111.run(buf755, buf846, 6272, grid=grid(6272), stream=stream0)
        del buf755
        buf850 = reinterpret_tensor(buf851, (4, 32, 7, 7), (50176, 49, 7, 1), 48608)  # alias
        # Source Nodes: [cat_62], Original ATen: [aten.cat]
        triton_poi_fused_cat_111.run(buf833, buf850, 6272, grid=grid(6272), stream=stream0)
        del buf833
        buf852 = empty_strided((4, 1024, 1, 1), (1024, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf853 = reinterpret_tensor(buf852, (4, 1024, 1, 1), (1024, 1, 1, 1), 0); del buf852  # reuse
        # Source Nodes: [features, out, out_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_112.run(buf853, buf851, arg724_1, arg725_1, arg360_1, arg361_1, 4096, 49, grid=grid(4096), stream=stream0)
        del arg360_1
        del arg361_1
        del arg724_1
        del arg725_1
        del buf834
        del buf835
        del buf836
        del buf837
        del buf838
        del buf839
        del buf840
        del buf841
        del buf842
        del buf843
        del buf844
        del buf845
        del buf846
        del buf847
        del buf848
        del buf849
        del buf850
        del buf851
        buf854 = empty((4, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_3], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg363_1, reinterpret_tensor(buf853, (4, 1024), (1024, 1), 0), reinterpret_tensor(arg362_1, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf854)
        del arg362_1
        del arg363_1
        return (buf854, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((128, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((128, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((128, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((128, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, 224, 1, 1), (224, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((128, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((128, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((128, 352, 1, 1), (352, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((128, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((128, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((128, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((128, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((128, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((128, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((128, 352, 1, 1), (352, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((128, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((128, 416, 1, 1), (416, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((128, 448, 1, 1), (448, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((128, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((544, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((544, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((128, 544, 1, 1), (544, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((128, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((128, 608, 1, 1), (608, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((128, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((128, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((128, 704, 1, 1), (704, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((128, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((128, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((128, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((128, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((128, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((928, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((928, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((128, 928, 1, 1), (928, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((128, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((992, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((992, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((128, 992, 1, 1), (992, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((128, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((544, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((544, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((128, 544, 1, 1), (544, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((128, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((128, 608, 1, 1), (608, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((128, 640, 1, 1), (640, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((128, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((128, 704, 1, 1), (704, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((128, 736, 1, 1), (736, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((128, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((128, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((128, 864, 1, 1), (864, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((128, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((928, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((928, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((128, 928, 1, 1), (928, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((128, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((992, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((992, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((128, 992, 1, 1), (992, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((32, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg367_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg370_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg373_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg376_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg379_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg382_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg385_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg388_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg391_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg394_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg397_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg400_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg403_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg406_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg409_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg412_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg415_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg418_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg421_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg424_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg427_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg430_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg433_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg436_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg439_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg442_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg445_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg448_1 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg451_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg454_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg457_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg460_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg463_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg466_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg469_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg472_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg475_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg478_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg481_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg484_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg487_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg490_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg493_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg496_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg499_1 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((352, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg502_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg505_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg508_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg511_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((416, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg514_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg515_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg517_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg518_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg520_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg521_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg522_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg523_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg524_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg525_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg526_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg527_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg528_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg529_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg530_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg531_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg532_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg533_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg534_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg535_1 = rand_strided((544, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg536_1 = rand_strided((544, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg537_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg538_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg539_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg540_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg541_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg542_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg543_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg544_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg545_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg546_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg547_1 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg548_1 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg549_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg550_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg551_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg552_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg553_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg554_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg555_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg556_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg557_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg558_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg559_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg560_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg561_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg562_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg563_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg564_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg565_1 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg566_1 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg567_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg568_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg569_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg570_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg571_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg572_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg573_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg574_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg575_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg576_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg577_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg578_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg579_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg580_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg581_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg582_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg583_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg584_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg585_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg586_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg587_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg588_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg589_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg590_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg591_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg592_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg593_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg594_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg595_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg596_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg597_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg598_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg599_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg600_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg601_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg602_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg603_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg604_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg605_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg606_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg607_1 = rand_strided((928, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg608_1 = rand_strided((928, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg609_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg610_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg611_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg612_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg613_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg614_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg615_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg616_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg617_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg618_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg619_1 = rand_strided((992, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg620_1 = rand_strided((992, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg621_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg622_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg623_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg624_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg625_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg626_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg627_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg628_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg629_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg630_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg631_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg632_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg633_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg634_1 = rand_strided((544, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg635_1 = rand_strided((544, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg636_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg637_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg638_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg639_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg640_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg641_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg642_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg643_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg644_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg645_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg646_1 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg647_1 = rand_strided((608, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg648_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg649_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg650_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg651_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg652_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg653_1 = rand_strided((640, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg654_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg655_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg656_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg657_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg658_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg659_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg660_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg661_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg662_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg663_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg664_1 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg665_1 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg666_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg667_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg668_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg669_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg670_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg671_1 = rand_strided((736, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg672_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg673_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg674_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg675_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg676_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg677_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg678_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg679_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg680_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg681_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg682_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg683_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg684_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg685_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg686_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg687_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg688_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg689_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg690_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg691_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg692_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg693_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg694_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg695_1 = rand_strided((864, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg696_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg697_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg698_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg699_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg700_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg701_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg702_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg703_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg704_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg705_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg706_1 = rand_strided((928, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg707_1 = rand_strided((928, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg708_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg709_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg710_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg711_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg712_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg713_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg714_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg715_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg716_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg717_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg718_1 = rand_strided((992, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg719_1 = rand_strided((992, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg720_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg721_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg722_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg723_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg724_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg725_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg726_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg727_1 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1, arg627_1, arg628_1, arg629_1, arg630_1, arg631_1, arg632_1, arg633_1, arg634_1, arg635_1, arg636_1, arg637_1, arg638_1, arg639_1, arg640_1, arg641_1, arg642_1, arg643_1, arg644_1, arg645_1, arg646_1, arg647_1, arg648_1, arg649_1, arg650_1, arg651_1, arg652_1, arg653_1, arg654_1, arg655_1, arg656_1, arg657_1, arg658_1, arg659_1, arg660_1, arg661_1, arg662_1, arg663_1, arg664_1, arg665_1, arg666_1, arg667_1, arg668_1, arg669_1, arg670_1, arg671_1, arg672_1, arg673_1, arg674_1, arg675_1, arg676_1, arg677_1, arg678_1, arg679_1, arg680_1, arg681_1, arg682_1, arg683_1, arg684_1, arg685_1, arg686_1, arg687_1, arg688_1, arg689_1, arg690_1, arg691_1, arg692_1, arg693_1, arg694_1, arg695_1, arg696_1, arg697_1, arg698_1, arg699_1, arg700_1, arg701_1, arg702_1, arg703_1, arg704_1, arg705_1, arg706_1, arg707_1, arg708_1, arg709_1, arg710_1, arg711_1, arg712_1, arg713_1, arg714_1, arg715_1, arg716_1, arg717_1, arg718_1, arg719_1, arg720_1, arg721_1, arg722_1, arg723_1, arg724_1, arg725_1, arg726_1, arg727_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('densenet121', benchmark_compiled_module)
