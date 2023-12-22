
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


# kernel path: /tmp/torchinductor_youkaichao/ka/ckarwkj4q4vxl6z7x4ide7th74rm535cb5vgk6qwkftvaw4ulm5g.py
# Source Nodes: [x_1, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_1 => add_1, mul_1, mul_2, sub
# x_4 => relu
triton_poi_fused__native_batch_norm_legit_no_training_relu_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_0', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/3q/c3qda3utxmc5yyfhxxsvmzq3dfrowpy2v3cvlnhw6yihh7mpzuax.py
# Source Nodes: [x_1, x_10, x_4, x_5, x_7, x_8, x_in, x_in_1, x_s], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# x_1 => add_1, mul_1, mul_2, sub
# x_10 => relu_2
# x_4 => relu
# x_5 => add_3, mul_4, mul_5, sub_1
# x_7 => relu_1
# x_8 => add_5, mul_7, mul_8, sub_2
# x_in => max_pool2d_with_indices
# x_in_1 => convolution_2
# x_s => convolution_1
triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 56) % 56
    x0 = xindex % 56
    x3 = (xindex // 56)
    x4 = xindex
    x6 = (xindex // 3136) % 128
    tmp70 = tl.load(in_ptr1 + (x6), None, eviction_policy='evict_last')
    tmp72 = tl.load(in_ptr2 + (x6), None, eviction_policy='evict_last')
    tmp80 = tl.load(in_ptr3 + (x6), None, eviction_policy='evict_last')
    tmp82 = tl.load(in_ptr4 + (x6), None, eviction_policy='evict_last')
    tmp85 = tl.load(in_ptr5 + (x6), None, eviction_policy='evict_last')
    tmp87 = tl.load(in_ptr6 + (x6), None, eviction_policy='evict_last')
    tmp93 = tl.load(in_ptr7 + (x6), None, eviction_policy='evict_last')
    tmp95 = tl.load(in_ptr8 + (x6), None, eviction_policy='evict_last')
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
    tmp71 = tmp69 - tmp70
    tmp73 = 0.001
    tmp74 = tmp72 + tmp73
    tmp75 = tl.sqrt(tmp74)
    tmp76 = 1 / tmp75
    tmp77 = 1.0
    tmp78 = tmp76 * tmp77
    tmp79 = tmp71 * tmp78
    tmp81 = tmp79 * tmp80
    tmp83 = tmp81 + tmp82
    tmp84 = triton_helpers.maximum(0, tmp83)
    tmp86 = tmp69 - tmp85
    tmp88 = tmp87 + tmp73
    tmp89 = tl.sqrt(tmp88)
    tmp90 = 1 / tmp89
    tmp91 = tmp90 * tmp77
    tmp92 = tmp86 * tmp91
    tmp94 = tmp92 * tmp93
    tmp96 = tmp94 + tmp95
    tmp97 = triton_helpers.maximum(0, tmp96)
    tl.store(out_ptr1 + (x4), tmp84, None)
    tl.store(out_ptr2 + (x4), tmp97, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/nt/cnt7idwwckpbedmkfuvaog7vu774nl3hjcb5wrhfdwe7xioggrs7.py
# Source Nodes: [x_11, x_13, x_in_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_11 => add_7, mul_10, mul_11, sub_3
# x_13 => relu_3
# x_in_2 => convolution_3
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5017600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 200
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/6o/c6ommec3vpravsbqi5tkwrzrlggvz3jqamaixcjq24retmr2u6v4.py
# Source Nodes: [cat_138, x_17, x_19, x_in_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_138 => cat_1
# x_17 => add_12, mul_16, mul_17, sub_5
# x_19 => relu_5
# x_in_5 => convolution_5
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7927808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3136) % 316
    x2 = (xindex // 990976)
    x3 = xindex % 990976
    x4 = xindex
    tmp32 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
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
    tmp33 = tmp31 - tmp32
    tmp35 = 0.001
    tmp36 = tmp34 + tmp35
    tmp37 = tl.sqrt(tmp36)
    tmp38 = 1 / tmp37
    tmp39 = 1.0
    tmp40 = tmp38 * tmp39
    tmp41 = tmp33 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = triton_helpers.maximum(0, tmp45)
    tl.store(out_ptr0 + (x4), tmp46, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5b/c5bujle5hchiu75a7dm7necv5wczrsguhpdtkeb3wrn27eyqmzw2.py
# Source Nodes: [cat_136, x_26, x_28, x_in_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_136 => cat_3
# x_26 => add_19, mul_25, mul_26, sub_8
# x_28 => relu_8
# x_in_9 => convolution_8
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8429568
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3136) % 336
    x2 = (xindex // 1053696)
    x3 = xindex % 1053696
    x4 = xindex
    tmp45 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
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
    tmp46 = tmp44 - tmp45
    tmp48 = 0.001
    tmp49 = tmp47 + tmp48
    tmp50 = tl.sqrt(tmp49)
    tmp51 = 1 / tmp50
    tmp52 = 1.0
    tmp53 = tmp51 * tmp52
    tmp54 = tmp46 * tmp53
    tmp56 = tmp54 * tmp55
    tmp58 = tmp56 + tmp57
    tmp59 = triton_helpers.maximum(0, tmp58)
    tl.store(in_out_ptr0 + (x4), tmp59, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mt/cmtfnezn7ytzz5nr5tqtipqrw5b64zfcw36qinnuvafxos7yxuks.py
# Source Nodes: [cat_134, x_35, x_37, x_in_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_134 => cat_5
# x_35 => add_26, mul_34, mul_35, sub_11
# x_37 => relu_11
# x_in_13 => convolution_11
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 8931328
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3136) % 356
    x2 = (xindex // 1116416)
    x3 = xindex % 1116416
    x4 = xindex
    tmp58 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
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
    tmp59 = tmp57 - tmp58
    tmp61 = 0.001
    tmp62 = tmp60 + tmp61
    tmp63 = tl.sqrt(tmp62)
    tmp64 = 1 / tmp63
    tmp65 = 1.0
    tmp66 = tmp64 * tmp65
    tmp67 = tmp59 * tmp66
    tmp69 = tmp67 * tmp68
    tmp71 = tmp69 + tmp70
    tmp72 = triton_helpers.maximum(0, tmp71)
    tl.store(in_out_ptr0 + (x4), tmp72, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ie/cievvf2hqlb3olnrlvwsjxnw4bwhicquzuwsfapmnp7jmbk5nit6.py
# Source Nodes: [cat_133], Original ATen: [aten.cat]
# cat_133 => cat_6
triton_poi_fused_cat_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_6', 'mutated_arg_names': []},
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
    x4 = xindex
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
    tl.store(out_ptr0 + (x4), tmp47, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6n/c6nbpmputv7jdyeijwpzddhhhwecq4xz2scxbe4kynnxkfzne5dv.py
# Source Nodes: [cat_132, x_44, x_46, x_47, x_49, x_in_17, x_s_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_132 => cat_7
# x_44 => add_33, mul_43, mul_44, sub_14
# x_46 => relu_14
# x_47 => add_35, mul_46, mul_47, sub_15
# x_49 => relu_15
# x_in_17 => convolution_15
# x_s_1 => convolution_14
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9433088
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3136) % 376
    x2 = (xindex // 1179136)
    x3 = xindex % 1179136
    x4 = xindex
    tmp23 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
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
    tmp12 = tl.load(in_ptr4 + (x3 + (865536*x2)), tmp4, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1], 376, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr5 + ((-802816) + x3 + (376320*x2)), tmp16, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp16, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp15, tmp21)
    tmp24 = tmp22 - tmp23
    tmp26 = 0.001
    tmp27 = tmp25 + tmp26
    tmp28 = tl.sqrt(tmp27)
    tmp29 = 1 / tmp28
    tmp30 = 1.0
    tmp31 = tmp29 * tmp30
    tmp32 = tmp24 * tmp31
    tmp34 = tmp32 * tmp33
    tmp36 = tmp34 + tmp35
    tmp37 = triton_helpers.maximum(0, tmp36)
    tmp39 = tmp22 - tmp38
    tmp41 = tmp40 + tmp26
    tmp42 = tl.sqrt(tmp41)
    tmp43 = 1 / tmp42
    tmp44 = tmp43 * tmp30
    tmp45 = tmp39 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tmp50 = triton_helpers.maximum(0, tmp49)
    tl.store(out_ptr1 + (x4), tmp37, None)
    tl.store(out_ptr2 + (x4), tmp50, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ai/caiurzkecovzhdz5m7vn6swajpsiephmxofbrf5vb6az5seea3gz.py
# Source Nodes: [x_50, x_52, x_in_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_50 => add_37, mul_49, mul_50, sub_16
# x_52 => relu_16
# x_in_18 => convolution_16
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10035200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 400
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/or/corjjd73bnomctg6tel7ps3k3so6pms5zw4jj72vhahddio46vuu.py
# Source Nodes: [x_53, x_55, x_in_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_53 => add_39, mul_52, mul_53, sub_17
# x_55 => relu_17
# x_in_19 => convolution_17
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
    xnumel = 2508800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 400
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/57/c57krpzq53l5fcvn4ro6h3s6zlrq5brgel5bn5v6mgp5zefxdv4n.py
# Source Nodes: [cat_130, x_56, x_58, x_in_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_130 => cat_9
# x_56 => add_42, mul_55, mul_56, sub_18
# x_58 => relu_18
# x_in_21 => convolution_18
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4415488
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 704
    x2 = (xindex // 551936)
    x3 = xindex % 551936
    x4 = xindex
    tmp32 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
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
    tmp33 = tmp31 - tmp32
    tmp35 = 0.001
    tmp36 = tmp34 + tmp35
    tmp37 = tl.sqrt(tmp36)
    tmp38 = 1 / tmp37
    tmp39 = 1.0
    tmp40 = tmp38 * tmp39
    tmp41 = tmp33 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = triton_helpers.maximum(0, tmp45)
    tl.store(out_ptr0 + (x4), tmp46, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hi/chinxnmiomjvo3ybkevlnry6bhsv64g26iywpcqw6wkpw72lhuqh.py
# Source Nodes: [cat_128, x_65, x_67, x_in_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_128 => cat_11
# x_65 => add_49, mul_64, mul_65, sub_21
# x_67 => relu_21
# x_in_25 => convolution_21
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 768
    x2 = (xindex // 602112)
    x3 = xindex % 602112
    x4 = xindex
    tmp45 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
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
    tmp46 = tmp44 - tmp45
    tmp48 = 0.001
    tmp49 = tmp47 + tmp48
    tmp50 = tl.sqrt(tmp49)
    tmp51 = 1 / tmp50
    tmp52 = 1.0
    tmp53 = tmp51 * tmp52
    tmp54 = tmp46 * tmp53
    tmp56 = tmp54 * tmp55
    tmp58 = tmp56 + tmp57
    tmp59 = triton_helpers.maximum(0, tmp58)
    tl.store(in_out_ptr0 + (x4), tmp59, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4n/c4nkahzymcwokupg7y5bbdp7yxbn3qrpw7gg5bwg7nuxy5afoee2.py
# Source Nodes: [cat_126, x_74, x_76, x_in_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_126 => cat_13
# x_74 => add_56, mul_73, mul_74, sub_24
# x_76 => relu_24
# x_in_29 => convolution_24
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5218304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 832
    x2 = (xindex // 652288)
    x3 = xindex % 652288
    x4 = xindex
    tmp58 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
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
    tmp59 = tmp57 - tmp58
    tmp61 = 0.001
    tmp62 = tmp60 + tmp61
    tmp63 = tl.sqrt(tmp62)
    tmp64 = 1 / tmp63
    tmp65 = 1.0
    tmp66 = tmp64 * tmp65
    tmp67 = tmp59 * tmp66
    tmp69 = tmp67 * tmp68
    tmp71 = tmp69 + tmp70
    tmp72 = triton_helpers.maximum(0, tmp71)
    tl.store(in_out_ptr0 + (x4), tmp72, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kd/ckdvadl5yhtnh42tlc2xnkw3ps63fziprfusnoh3tzmuhquoddvh.py
# Source Nodes: [x_s1_5, x_s1_6, x_s1_7, x_s1_8], Original ATen: [aten.add]
# x_s1_5 => add_40
# x_s1_6 => add_47
# x_s1_7 => add_54
# x_s1_8 => add_61
triton_poi_fused_add_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_13', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/2i/c2iy4k3yvasl22fjmwzqox52bbn57agwewue2hri2ztckwnia6be.py
# Source Nodes: [cat_125], Original ATen: [aten.cat]
# cat_125 => cat_14
triton_poi_fused_cat_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_14', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/6a/c6azpipuwnzazwzq3pqole45u4oncjjflk6jdcakkjpfl45goelw.py
# Source Nodes: [x_83, x_85, x_in_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_83 => add_63, mul_82, mul_83, sub_27
# x_85 => relu_27
# x_in_33 => convolution_27
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15', 'mutated_arg_names': []},
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
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/ly/clyuv6bcribqsdicascxjz6mh6wpus2vxsk2krjrzj5flqnfxv3x.py
# Source Nodes: [cat_122, x_92, x_94, x_in_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_122 => cat_17
# x_92 => add_70, mul_91, mul_92, sub_30
# x_94 => relu_30
# x_in_37 => convolution_30
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6021120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 960
    x2 = (xindex // 752640)
    x3 = xindex % 752640
    x4 = xindex
    tmp32 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
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
    tmp33 = tmp31 - tmp32
    tmp35 = 0.001
    tmp36 = tmp34 + tmp35
    tmp37 = tl.sqrt(tmp36)
    tmp38 = 1 / tmp37
    tmp39 = 1.0
    tmp40 = tmp38 * tmp39
    tmp41 = tmp33 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = triton_helpers.maximum(0, tmp45)
    tl.store(out_ptr0 + (x4), tmp46, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/od/codogmbwasgqzfczndtjalvcsdkcqrq7mfejegnmccjo5xmeinyd.py
# Source Nodes: [cat_120, x_101, x_103, x_in_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_120 => cat_19
# x_101 => add_77, mul_100, mul_101, sub_33
# x_103 => relu_33
# x_in_41 => convolution_33
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 1024
    x2 = (xindex // 802816)
    x3 = xindex % 802816
    x4 = xindex
    tmp44 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
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
    tmp45 = tmp43 - tmp44
    tmp47 = 0.001
    tmp48 = tmp46 + tmp47
    tmp49 = tl.sqrt(tmp48)
    tmp50 = 1 / tmp49
    tmp51 = 1.0
    tmp52 = tmp50 * tmp51
    tmp53 = tmp45 * tmp52
    tmp55 = tmp53 * tmp54
    tmp57 = tmp55 + tmp56
    tmp58 = triton_helpers.maximum(0, tmp57)
    tl.store(in_out_ptr0 + (x4), tmp58, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qf/cqfkbnlv2ewppe2m6b7tz5s42u4gh2glqhsx72iucmrptfrnpd7v.py
# Source Nodes: [cat_118, x_110, x_112, x_in_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_118 => cat_21
# x_110 => add_84, mul_109, mul_110, sub_36
# x_112 => relu_36
# x_in_45 => convolution_36
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6823936
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 1088
    x2 = (xindex // 852992)
    x3 = xindex % 852992
    x4 = xindex
    tmp57 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
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
    tmp58 = tmp56 - tmp57
    tmp60 = 0.001
    tmp61 = tmp59 + tmp60
    tmp62 = tl.sqrt(tmp61)
    tmp63 = 1 / tmp62
    tmp64 = 1.0
    tmp65 = tmp63 * tmp64
    tmp66 = tmp58 * tmp65
    tmp68 = tmp66 * tmp67
    tmp70 = tmp68 + tmp69
    tmp71 = triton_helpers.maximum(0, tmp70)
    tl.store(in_out_ptr0 + (x4), tmp71, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4f/c4f6d37y2mvffgrh4ahanbrwj2vz2uhsgd6fakwbwoxzggrh5lxj.py
# Source Nodes: [cat_117], Original ATen: [aten.cat]
# cat_117 => cat_22
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
    xnumel = 4014080
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 640
    x2 = (xindex // 501760)
    x3 = xindex % 501760
    x4 = xindex
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
    tl.store(out_ptr0 + (x4), tmp47, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hz/chz2yx62f7cscphhlmbx7zzlrbgtk66hyrxcepwz2phem4t3yptn.py
# Source Nodes: [cat_116, x_119, x_121, x_122, x_124, x_in_49, x_s_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_116 => cat_23
# x_119 => add_91, mul_118, mul_119, sub_39
# x_121 => relu_39
# x_122 => add_93, mul_121, mul_122, sub_40
# x_124 => relu_40
# x_in_49 => convolution_40
# x_s_2 => convolution_39
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7225344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 1152
    x2 = (xindex // 903168)
    x3 = xindex % 903168
    x4 = xindex
    tmp23 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
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
    tmp12 = tl.load(in_ptr4 + (x3 + (451584*x2)), tmp4, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1], 1152, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr5 + ((-401408) + x3 + (501760*x2)), tmp16, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp16, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp15, tmp21)
    tmp24 = tmp22 - tmp23
    tmp26 = 0.001
    tmp27 = tmp25 + tmp26
    tmp28 = tl.sqrt(tmp27)
    tmp29 = 1 / tmp28
    tmp30 = 1.0
    tmp31 = tmp29 * tmp30
    tmp32 = tmp24 * tmp31
    tmp34 = tmp32 * tmp33
    tmp36 = tmp34 + tmp35
    tmp37 = triton_helpers.maximum(0, tmp36)
    tmp39 = tmp22 - tmp38
    tmp41 = tmp40 + tmp26
    tmp42 = tl.sqrt(tmp41)
    tmp43 = 1 / tmp42
    tmp44 = tmp43 * tmp30
    tmp45 = tmp39 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tmp50 = triton_helpers.maximum(0, tmp49)
    tl.store(out_ptr1 + (x4), tmp37, None)
    tl.store(out_ptr2 + (x4), tmp50, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wp/cwptgvpsssvy54ppduqnwmj5r3nhtjfdcopml27my5lbu4gbyeix.py
# Source Nodes: [x_125, x_127, x_in_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_125 => add_95, mul_124, mul_125, sub_41
# x_127 => relu_41
# x_in_50 => convolution_41
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5017600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 800
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/rw/crwpj3mqno7zljzypt2uj2btqouvsn2y5qsvk65mnwg5evba57cj.py
# Source Nodes: [x_128, x_130, x_in_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_128 => add_97, mul_127, mul_128, sub_42
# x_130 => relu_42
# x_in_51 => convolution_42
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1254400
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/bs/cbsvu3qcogg7rfhsafvqflxg6mg3gnys5h3sqcb2hbxkhubs5asf.py
# Source Nodes: [cat_114, x_131, x_133, x_in_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_114 => cat_25
# x_131 => add_100, mul_130, mul_131, sub_43
# x_133 => relu_43
# x_in_53 => convolution_43
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1906688
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1216
    x2 = (xindex // 238336)
    x3 = xindex % 238336
    x4 = xindex
    tmp32 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
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
    tmp33 = tmp31 - tmp32
    tmp35 = 0.001
    tmp36 = tmp34 + tmp35
    tmp37 = tl.sqrt(tmp36)
    tmp38 = 1 / tmp37
    tmp39 = 1.0
    tmp40 = tmp38 * tmp39
    tmp41 = tmp33 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = triton_helpers.maximum(0, tmp45)
    tl.store(out_ptr0 + (x4), tmp46, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6a/c6acrvoak7ap7ml46gyminmrrt2mxa7mcddrktkqanni7ox6mglb.py
# Source Nodes: [cat_112, x_140, x_142, x_in_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_112 => cat_27
# x_140 => add_107, mul_139, mul_140, sub_46
# x_142 => relu_46
# x_in_57 => convolution_46
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2007040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1280
    x2 = (xindex // 250880)
    x3 = xindex % 250880
    x4 = xindex
    tmp45 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
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
    tmp46 = tmp44 - tmp45
    tmp48 = 0.001
    tmp49 = tmp47 + tmp48
    tmp50 = tl.sqrt(tmp49)
    tmp51 = 1 / tmp50
    tmp52 = 1.0
    tmp53 = tmp51 * tmp52
    tmp54 = tmp46 * tmp53
    tmp56 = tmp54 * tmp55
    tmp58 = tmp56 + tmp57
    tmp59 = triton_helpers.maximum(0, tmp58)
    tl.store(in_out_ptr0 + (x4), tmp59, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sv/csvlnxrc6zwugvefzmngauvhb5khorlkhf36y24gwmnopx7y4ixw.py
# Source Nodes: [cat_110, x_149, x_151, x_in_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_110 => cat_29
# x_149 => add_114, mul_148, mul_149, sub_49
# x_151 => relu_49
# x_in_61 => convolution_49
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2107392
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1344
    x2 = (xindex // 263424)
    x3 = xindex % 263424
    x4 = xindex
    tmp58 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
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
    tmp59 = tmp57 - tmp58
    tmp61 = 0.001
    tmp62 = tmp60 + tmp61
    tmp63 = tl.sqrt(tmp62)
    tmp64 = 1 / tmp63
    tmp65 = 1.0
    tmp66 = tmp64 * tmp65
    tmp67 = tmp59 * tmp66
    tmp69 = tmp67 * tmp68
    tmp71 = tmp69 + tmp70
    tmp72 = triton_helpers.maximum(0, tmp71)
    tl.store(in_out_ptr0 + (x4), tmp72, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5r/c5rbdh5bjm5mrcurm3aydq2jqn7m2txsihsqjrchsdlyjypnbtvh.py
# Source Nodes: [x_s1_13, x_s1_14, x_s1_15, x_s1_16], Original ATen: [aten.add]
# x_s1_13 => add_98
# x_s1_14 => add_105
# x_s1_15 => add_112
# x_s1_16 => add_119
triton_poi_fused_add_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_26', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/de/cdewb2dwqqo47jgcan7beayaohiocniuwgbmn5vzaikw2uwdhw5q.py
# Source Nodes: [cat_109], Original ATen: [aten.cat]
# cat_109 => cat_30
triton_poi_fused_cat_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_27', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/yi/cyig6hj4nauawjd2dyohdhxfcq4ilkesbo3r3wcpexfygiwbkjng.py
# Source Nodes: [x_158, x_160, x_in_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_158 => add_121, mul_157, mul_158, sub_52
# x_160 => relu_52
# x_in_65 => convolution_52
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28', 'mutated_arg_names': []},
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
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/vq/cvqa4o5riikpvo3r6hsubusgmnezqwrey6xrl4rq5haxyw76ewnp.py
# Source Nodes: [cat_106, x_167, x_169, x_in_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_106 => cat_33
# x_167 => add_128, mul_166, mul_167, sub_55
# x_169 => relu_55
# x_in_69 => convolution_55
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2308096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1472
    x2 = (xindex // 288512)
    x3 = xindex % 288512
    x4 = xindex
    tmp32 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
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
    tmp33 = tmp31 - tmp32
    tmp35 = 0.001
    tmp36 = tmp34 + tmp35
    tmp37 = tl.sqrt(tmp36)
    tmp38 = 1 / tmp37
    tmp39 = 1.0
    tmp40 = tmp38 * tmp39
    tmp41 = tmp33 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = triton_helpers.maximum(0, tmp45)
    tl.store(out_ptr0 + (x4), tmp46, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sv/csvyammhkhhzjmpatwyztrpeilf63xxhtjw6sgayb42aydjygtyf.py
# Source Nodes: [cat_104, x_176, x_178, x_in_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_104 => cat_35
# x_176 => add_135, mul_175, mul_176, sub_58
# x_178 => relu_58
# x_in_73 => convolution_58
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_30', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2408448
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1536
    x2 = (xindex // 301056)
    x3 = xindex % 301056
    x4 = xindex
    tmp45 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
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
    tmp46 = tmp44 - tmp45
    tmp48 = 0.001
    tmp49 = tmp47 + tmp48
    tmp50 = tl.sqrt(tmp49)
    tmp51 = 1 / tmp50
    tmp52 = 1.0
    tmp53 = tmp51 * tmp52
    tmp54 = tmp46 * tmp53
    tmp56 = tmp54 * tmp55
    tmp58 = tmp56 + tmp57
    tmp59 = triton_helpers.maximum(0, tmp58)
    tl.store(in_out_ptr0 + (x4), tmp59, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hk/chkqyoxubifhocpjcqgzaavv72dev7xns6rkv2arlpnb6fdfwzew.py
# Source Nodes: [cat_102, x_185, x_187, x_in_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_102 => cat_37
# x_185 => add_142, mul_184, mul_185, sub_61
# x_187 => relu_61
# x_in_77 => convolution_61
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_31 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_31', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2508800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1600
    x2 = (xindex // 313600)
    x3 = xindex % 313600
    x4 = xindex
    tmp58 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
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
    tmp59 = tmp57 - tmp58
    tmp61 = 0.001
    tmp62 = tmp60 + tmp61
    tmp63 = tl.sqrt(tmp62)
    tmp64 = 1 / tmp63
    tmp65 = 1.0
    tmp66 = tmp64 * tmp65
    tmp67 = tmp59 * tmp66
    tmp69 = tmp67 * tmp68
    tmp71 = tmp69 + tmp70
    tmp72 = triton_helpers.maximum(0, tmp71)
    tl.store(in_out_ptr0 + (x4), tmp72, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/va/cva4tolgvgbbra7ehdd2icutdkdhx4ppyjp5uawfaytn7h33rplg.py
# Source Nodes: [x_s1_17, x_s1_18, x_s1_19, x_s1_20], Original ATen: [aten.add]
# x_s1_17 => add_126
# x_s1_18 => add_133
# x_s1_19 => add_140
# x_s1_20 => add_147
triton_poi_fused_add_32 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_32', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/cp/ccpb72czl5burnl55ob2qtxop3fmx7tsxtncdpw3flekgfkf2jnv.py
# Source Nodes: [cat_101], Original ATen: [aten.cat]
# cat_101 => cat_38
triton_poi_fused_cat_33 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_33', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/3w/c3w6wuen3xgcrz5dppv3z3l7gbcssn74jj244lg674t66glakgcq.py
# Source Nodes: [x_194, x_196, x_in_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_194 => add_149, mul_193, mul_194, sub_64
# x_196 => relu_64
# x_in_81 => convolution_64
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34', 'mutated_arg_names': []},
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
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/fj/cfjooldpkemcql3btjbtu5gwy6pvlsfl4vmpzf3k2qzrntiogo2c.py
# Source Nodes: [cat_98, x_203, x_205, x_in_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_98 => cat_41
# x_203 => add_156, mul_202, mul_203, sub_67
# x_205 => relu_67
# x_in_85 => convolution_67
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2709504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1728
    x2 = (xindex // 338688)
    x3 = xindex % 338688
    x4 = xindex
    tmp32 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
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
    tmp33 = tmp31 - tmp32
    tmp35 = 0.001
    tmp36 = tmp34 + tmp35
    tmp37 = tl.sqrt(tmp36)
    tmp38 = 1 / tmp37
    tmp39 = 1.0
    tmp40 = tmp38 * tmp39
    tmp41 = tmp33 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = triton_helpers.maximum(0, tmp45)
    tl.store(out_ptr0 + (x4), tmp46, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ib/ciby2xmynjqigbe3vnin5idry2ezm22pjxih6kfzxs7q7sqsne3i.py
# Source Nodes: [cat_96, x_212, x_214, x_in_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_96 => cat_43
# x_212 => add_163, mul_211, mul_212, sub_70
# x_214 => relu_70
# x_in_89 => convolution_70
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_36 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_36', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2809856
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1792
    x2 = (xindex // 351232)
    x3 = xindex % 351232
    x4 = xindex
    tmp45 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
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
    tmp46 = tmp44 - tmp45
    tmp48 = 0.001
    tmp49 = tmp47 + tmp48
    tmp50 = tl.sqrt(tmp49)
    tmp51 = 1 / tmp50
    tmp52 = 1.0
    tmp53 = tmp51 * tmp52
    tmp54 = tmp46 * tmp53
    tmp56 = tmp54 * tmp55
    tmp58 = tmp56 + tmp57
    tmp59 = triton_helpers.maximum(0, tmp58)
    tl.store(in_out_ptr0 + (x4), tmp59, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/d5/cd5o2i5ynaniahy3eb6qaeri246qh6zmyla3ejh6kh24alzbpx4t.py
# Source Nodes: [cat_94, x_221, x_223, x_in_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_94 => cat_45
# x_221 => add_170, mul_220, mul_221, sub_73
# x_223 => relu_73
# x_in_93 => convolution_73
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_37 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_37', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2910208
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1856
    x2 = (xindex // 363776)
    x3 = xindex % 363776
    x4 = xindex
    tmp58 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
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
    tmp59 = tmp57 - tmp58
    tmp61 = 0.001
    tmp62 = tmp60 + tmp61
    tmp63 = tl.sqrt(tmp62)
    tmp64 = 1 / tmp63
    tmp65 = 1.0
    tmp66 = tmp64 * tmp65
    tmp67 = tmp59 * tmp66
    tmp69 = tmp67 * tmp68
    tmp71 = tmp69 + tmp70
    tmp72 = triton_helpers.maximum(0, tmp71)
    tl.store(in_out_ptr0 + (x4), tmp72, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/rv/crvdoygb5n2t666v7gasig4kej6tt65hsumdpq5bogzan5osfpnx.py
# Source Nodes: [x_s1_21, x_s1_22, x_s1_23, x_s1_24], Original ATen: [aten.add]
# x_s1_21 => add_154
# x_s1_22 => add_161
# x_s1_23 => add_168
# x_s1_24 => add_175
triton_poi_fused_add_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_38', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/3t/c3t4r2biilwfwc4qupya2isnkbjnszg3awmsrut5blt5rdfqvhec.py
# Source Nodes: [cat_93], Original ATen: [aten.cat]
# cat_93 => cat_46
triton_poi_fused_cat_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_39', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/gl/cgl5fitqlmq33tajv2dvurdutkysmrimtlykro2g465npc5b4ww2.py
# Source Nodes: [x_230, x_232, x_in_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_230 => add_177, mul_229, mul_230, sub_76
# x_232 => relu_76
# x_in_97 => convolution_76
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_40 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_40', 'mutated_arg_names': []},
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
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/gb/cgbyks4ngainhicis35kwrgdf2mwcuiyywugh3b6reibi5syktxl.py
# Source Nodes: [cat_90, x_239, x_241, x_in_101], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_90 => cat_49
# x_239 => add_184, mul_238, mul_239, sub_79
# x_241 => relu_79
# x_in_101 => convolution_79
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3110912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 1984
    x2 = (xindex // 388864)
    x3 = xindex % 388864
    x4 = xindex
    tmp32 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
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
    tmp33 = tmp31 - tmp32
    tmp35 = 0.001
    tmp36 = tmp34 + tmp35
    tmp37 = tl.sqrt(tmp36)
    tmp38 = 1 / tmp37
    tmp39 = 1.0
    tmp40 = tmp38 * tmp39
    tmp41 = tmp33 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = triton_helpers.maximum(0, tmp45)
    tl.store(out_ptr0 + (x4), tmp46, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ws/cwspzpbw3jb4z3nuuusdgmz5ynev3wjw4bvbyplfj6fpa7gkfnuj.py
# Source Nodes: [cat_88, x_248, x_250, x_in_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_88 => cat_51
# x_248 => add_191, mul_247, mul_248, sub_82
# x_250 => relu_82
# x_in_105 => convolution_82
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_42', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 2048
    x2 = (xindex // 401408)
    x3 = xindex % 401408
    x4 = xindex
    tmp44 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp54 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp56 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
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
    tmp45 = tmp43 - tmp44
    tmp47 = 0.001
    tmp48 = tmp46 + tmp47
    tmp49 = tl.sqrt(tmp48)
    tmp50 = 1 / tmp49
    tmp51 = 1.0
    tmp52 = tmp50 * tmp51
    tmp53 = tmp45 * tmp52
    tmp55 = tmp53 * tmp54
    tmp57 = tmp55 + tmp56
    tmp58 = triton_helpers.maximum(0, tmp57)
    tl.store(in_out_ptr0 + (x4), tmp58, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/72/c723ndkeyh4vovotesbzjgl4xlm3zftuovg22tqkg6vadhqpvkvr.py
# Source Nodes: [cat_86, x_257, x_259, x_in_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_86 => cat_53
# x_257 => add_198, mul_256, mul_257, sub_85
# x_259 => relu_85
# x_in_109 => convolution_85
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_43', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3311616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 2112
    x2 = (xindex // 413952)
    x3 = xindex % 413952
    x4 = xindex
    tmp57 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp59 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp67 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp69 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
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
    tmp58 = tmp56 - tmp57
    tmp60 = 0.001
    tmp61 = tmp59 + tmp60
    tmp62 = tl.sqrt(tmp61)
    tmp63 = 1 / tmp62
    tmp64 = 1.0
    tmp65 = tmp63 * tmp64
    tmp66 = tmp58 * tmp65
    tmp68 = tmp66 * tmp67
    tmp70 = tmp68 + tmp69
    tmp71 = triton_helpers.maximum(0, tmp70)
    tl.store(in_out_ptr0 + (x4), tmp71, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3i/c3ijoee6fjhmzu3q2rzxkv6vuhqz2suuugxlxxarp7u6pkgo6stm.py
# Source Nodes: [x_s1_25, x_s1_26, x_s1_27, x_s1_28], Original ATen: [aten.add]
# x_s1_25 => add_182
# x_s1_26 => add_189
# x_s1_27 => add_196
# x_s1_28 => add_203
triton_poi_fused_add_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_44', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ns/cnsbpat4il7p6eknx5hsygfrcnyqe2ymch5lcwc7qa22nvfuw7z5.py
# Source Nodes: [cat_85], Original ATen: [aten.cat]
# cat_85 => cat_54
triton_poi_fused_cat_45 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_45', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ng/cngoga37ua3gqfu2e7c6flen7nhyonk2iw6ozf6qfya4grhmracf.py
# Source Nodes: [x_266, x_268, x_in_113], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_266 => add_205, mul_265, mul_266, sub_88
# x_268 => relu_88
# x_in_113 => convolution_88
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_46', 'mutated_arg_names': []},
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
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/gn/cgnh6b4as7fissfvq4tby5shnyhmuezhhczpe3chh2jnrism3eek.py
# Source Nodes: [cat_82, x_275, x_277, x_in_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_82 => cat_57
# x_275 => add_212, mul_274, mul_275, sub_91
# x_277 => relu_91
# x_in_117 => convolution_91
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_47 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3512320
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 2240
    x2 = (xindex // 439040)
    x3 = xindex % 439040
    x4 = xindex
    tmp32 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
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
    tmp33 = tmp31 - tmp32
    tmp35 = 0.001
    tmp36 = tmp34 + tmp35
    tmp37 = tl.sqrt(tmp36)
    tmp38 = 1 / tmp37
    tmp39 = 1.0
    tmp40 = tmp38 * tmp39
    tmp41 = tmp33 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = triton_helpers.maximum(0, tmp45)
    tl.store(out_ptr0 + (x4), tmp46, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/d3/cd3nv6rxexxwx7rj6iog3gx62ruhovesrdxskg4zetgntv7og6ba.py
# Source Nodes: [cat_80, x_284, x_286, x_in_121], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_80 => cat_59
# x_284 => add_219, mul_283, mul_284, sub_94
# x_286 => relu_94
# x_in_121 => convolution_94
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_48 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_48', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3612672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 2304
    x2 = (xindex // 451584)
    x3 = xindex % 451584
    x4 = xindex
    tmp45 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
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
    tmp46 = tmp44 - tmp45
    tmp48 = 0.001
    tmp49 = tmp47 + tmp48
    tmp50 = tl.sqrt(tmp49)
    tmp51 = 1 / tmp50
    tmp52 = 1.0
    tmp53 = tmp51 * tmp52
    tmp54 = tmp46 * tmp53
    tmp56 = tmp54 * tmp55
    tmp58 = tmp56 + tmp57
    tmp59 = triton_helpers.maximum(0, tmp58)
    tl.store(in_out_ptr0 + (x4), tmp59, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/g6/cg6k6fr4eqodzuukjtyo2cy4ympa2shjpxdnir2ve3t3pvcj3c3u.py
# Source Nodes: [cat_78, x_293, x_295, x_in_125], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_78 => cat_61
# x_293 => add_226, mul_292, mul_293, sub_97
# x_295 => relu_97
# x_in_125 => convolution_97
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_49 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_49', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3713024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 2368
    x2 = (xindex // 464128)
    x3 = xindex % 464128
    x4 = xindex
    tmp58 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
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
    tmp59 = tmp57 - tmp58
    tmp61 = 0.001
    tmp62 = tmp60 + tmp61
    tmp63 = tl.sqrt(tmp62)
    tmp64 = 1 / tmp63
    tmp65 = 1.0
    tmp66 = tmp64 * tmp65
    tmp67 = tmp59 * tmp66
    tmp69 = tmp67 * tmp68
    tmp71 = tmp69 + tmp70
    tmp72 = triton_helpers.maximum(0, tmp71)
    tl.store(in_out_ptr0 + (x4), tmp72, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zu/czul4msvb3xm5dhck62qdat4v4ojca2janjoeek5kab2btzlvihi.py
# Source Nodes: [cat_77], Original ATen: [aten.cat]
# cat_77 => cat_62
triton_poi_fused_cat_50 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_50', 'mutated_arg_names': []},
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
    x4 = xindex
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
    tl.store(out_ptr0 + (x4), tmp47, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4e/c4eiyfnexooytvuqfo77grhge7d2zfvjxnqbciqdqj2rnxlqggym.py
# Source Nodes: [cat_76, x_302, x_304, x_305, x_307, x_in_129, x_s_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_76 => cat_63
# x_302 => add_233, mul_301, mul_302, sub_100
# x_304 => relu_100
# x_305 => add_235, mul_304, mul_305, sub_101
# x_307 => relu_101
# x_in_129 => convolution_101
# x_s_3 => convolution_100
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(16,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3813376
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 2432
    x2 = (xindex // 476672)
    x3 = xindex % 476672
    x4 = xindex
    tmp23 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp25 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp33 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp35 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr10 + (x1), None, eviction_policy='evict_last')
    tmp40 = tl.load(in_ptr11 + (x1), None, eviction_policy='evict_last')
    tmp46 = tl.load(in_ptr12 + (x1), None, eviction_policy='evict_last')
    tmp48 = tl.load(in_ptr13 + (x1), None, eviction_policy='evict_last')
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
    tmp12 = tl.load(in_ptr4 + (x3 + (213248*x2)), tmp4, other=0.0)
    tmp13 = tmp11 + tmp12
    tmp14 = tl.full(tmp13.shape, 0.0, tmp13.dtype)
    tmp15 = tl.where(tmp4, tmp13, tmp14)
    tmp16 = tmp0 >= tmp3
    tmp17 = tl.full([1], 2432, tl.int64)
    tmp18 = tmp0 < tmp17
    tmp19 = tl.load(in_ptr5 + ((-200704) + x3 + (275968*x2)), tmp16, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp16, tmp19, tmp20)
    tmp22 = tl.where(tmp4, tmp15, tmp21)
    tmp24 = tmp22 - tmp23
    tmp26 = 0.001
    tmp27 = tmp25 + tmp26
    tmp28 = tl.sqrt(tmp27)
    tmp29 = 1 / tmp28
    tmp30 = 1.0
    tmp31 = tmp29 * tmp30
    tmp32 = tmp24 * tmp31
    tmp34 = tmp32 * tmp33
    tmp36 = tmp34 + tmp35
    tmp37 = triton_helpers.maximum(0, tmp36)
    tmp39 = tmp22 - tmp38
    tmp41 = tmp40 + tmp26
    tmp42 = tl.sqrt(tmp41)
    tmp43 = 1 / tmp42
    tmp44 = tmp43 * tmp30
    tmp45 = tmp39 * tmp44
    tmp47 = tmp45 * tmp46
    tmp49 = tmp47 + tmp48
    tmp50 = triton_helpers.maximum(0, tmp49)
    tl.store(out_ptr1 + (x4), tmp37, None)
    tl.store(out_ptr2 + (x4), tmp50, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xl/cxlimg3mchax63h22kezxlhdpfhhorolri3xp5tvjmvqfm3wamno.py
# Source Nodes: [x_308, x_310, x_in_130], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_308 => add_237, mul_307, mul_308, sub_102
# x_310 => relu_102
# x_in_130 => convolution_102
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_52 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_52', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2508800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 1600
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/fr/cfremjuei2vktd3tkoklxa7264fqxd3noy2wxncwnnvpapuif6rs.py
# Source Nodes: [x_311, x_313, x_in_131], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_311 => add_239, mul_310, mul_311, sub_103
# x_313 => relu_103
# x_in_131 => convolution_103
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_53 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_53', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 627200
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 1600
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 - tmp1
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/s2/cs26qvmerhv5z5texa25z2prstvqxyx23ickseve5lbdxyifwzef.py
# Source Nodes: [cat_74, x_314, x_316, x_in_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_74 => cat_65
# x_314 => add_242, mul_313, mul_314, sub_104
# x_316 => relu_104
# x_in_133 => convolution_104
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_54 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_54', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 953344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 49) % 2432
    x2 = (xindex // 119168)
    x3 = xindex % 119168
    x4 = xindex
    tmp32 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp34 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp42 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
    tmp44 = tl.load(in_ptr5 + (x1), xmask, eviction_policy='evict_last')
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
    tmp33 = tmp31 - tmp32
    tmp35 = 0.001
    tmp36 = tmp34 + tmp35
    tmp37 = tl.sqrt(tmp36)
    tmp38 = 1 / tmp37
    tmp39 = 1.0
    tmp40 = tmp38 * tmp39
    tmp41 = tmp33 * tmp40
    tmp43 = tmp41 * tmp42
    tmp45 = tmp43 + tmp44
    tmp46 = triton_helpers.maximum(0, tmp45)
    tl.store(out_ptr0 + (x4), tmp46, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/th/cthiyr67vlf4l7btdcrpgwfdu5evdv6ubpdtgsnijrjieyrv4qis.py
# Source Nodes: [cat_72, x_323, x_325, x_in_137], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
# cat_72 => cat_67
# x_323 => add_249, mul_322, mul_323, sub_107
# x_325 => relu_107
# x_in_137 => convolution_107
triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_55 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_55', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1003520
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 49) % 2560
    x2 = (xindex // 125440)
    x3 = xindex % 125440
    x4 = xindex
    tmp45 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp47 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp55 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp57 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
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
    tmp46 = tmp44 - tmp45
    tmp48 = 0.001
    tmp49 = tmp47 + tmp48
    tmp50 = tl.sqrt(tmp49)
    tmp51 = 1 / tmp50
    tmp52 = 1.0
    tmp53 = tmp51 * tmp52
    tmp54 = tmp46 * tmp53
    tmp56 = tmp54 * tmp55
    tmp58 = tmp56 + tmp57
    tmp59 = triton_helpers.maximum(0, tmp58)
    tl.store(in_out_ptr0 + (x4), tmp59, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/o2/co2qu3fb47ehki5vcdqb5ju4ycwfnwzi7tw5zr5ibt24wsyi6s35.py
# Source Nodes: [cat_70, x_333, x_336, x_337, x_340], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.mean, aten.relu]
# cat_70 => cat_69
# x_333 => add_256, mul_331, mul_332, sub_110
# x_336 => relu_110
# x_337 => mean
# x_340 => convolution_110
triton_per_fused__native_batch_norm_legit_no_training_cat_convolution_mean_relu_56 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_cat_convolution_mean_relu_56', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 21504
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    x0 = xindex % 2688
    r2 = rindex
    x1 = (xindex // 2688)
    x3 = xindex
    tmp58 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
    tmp60 = tl.load(in_ptr5 + (x0), xmask, eviction_policy='evict_last')
    tmp68 = tl.load(in_ptr6 + (x0), xmask, eviction_policy='evict_last')
    tmp70 = tl.load(in_ptr7 + (x0), xmask, eviction_policy='evict_last')
    tmp0 = x0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 2048, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (r2 + (49*x0) + (112896*x1)), rmask & tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (r2 + (49*x0) + (106624*x1)), rmask & tmp4 & xmask, other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = tl.load(in_ptr2 + (r2 + (49*x0) + (106624*x1)), rmask & tmp4 & xmask, other=0.0)
    tmp9 = tmp7 + tmp8
    tmp10 = tl.load(in_ptr3 + (r2 + (49*x0) + (106624*x1)), rmask & tmp4 & xmask, other=0.0)
    tmp11 = tmp9 + tmp10
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp4, tmp11, tmp12)
    tmp14 = tmp0 >= tmp3
    tmp15 = tl.full([1, 1], 2688, tl.int64)
    tmp16 = tmp0 < tmp15
    tmp17 = tl.broadcast_to((-2048) + x0, [XBLOCK, RBLOCK])
    tmp18 = tmp17 >= tmp1
    tmp19 = tl.full([1, 1], 512, tl.int64)
    tmp20 = tmp17 < tmp19
    tmp21 = tmp20 & tmp14
    tmp22 = tl.full([1, 1], 384, tl.int64)
    tmp23 = tmp17 < tmp22
    tmp24 = tmp23 & tmp21
    tmp25 = tl.full([1, 1], 256, tl.int64)
    tmp26 = tmp17 < tmp25
    tmp27 = tmp26 & tmp24
    tmp28 = tl.load(in_ptr0 + (r2 + (49*x0) + (112896*x1)), rmask & tmp27 & xmask, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp17 >= tmp25
    tmp32 = tmp31 & tmp24
    tmp33 = tl.load(in_ptr1 + ((-12544) + r2 + (49*x0) + (106624*x1)), rmask & tmp32 & xmask, other=0.0)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp32, tmp33, tmp34)
    tmp36 = tl.where(tmp26, tmp30, tmp35)
    tmp37 = tl.full(tmp36.shape, 0.0, tmp36.dtype)
    tmp38 = tl.where(tmp24, tmp36, tmp37)
    tmp39 = tmp17 >= tmp22
    tmp40 = tmp39 & tmp21
    tmp41 = tl.load(in_ptr2 + ((-18816) + r2 + (49*x0) + (106624*x1)), rmask & tmp40 & xmask, other=0.0)
    tmp42 = tl.full(tmp41.shape, 0.0, tmp41.dtype)
    tmp43 = tl.where(tmp40, tmp41, tmp42)
    tmp44 = tl.where(tmp23, tmp38, tmp43)
    tmp45 = tl.full(tmp44.shape, 0.0, tmp44.dtype)
    tmp46 = tl.where(tmp21, tmp44, tmp45)
    tmp47 = tmp17 >= tmp19
    tmp48 = tl.full([1, 1], 640, tl.int64)
    tmp49 = tmp17 < tmp48
    tmp50 = tmp47 & tmp14
    tmp51 = tl.load(in_ptr3 + ((-25088) + r2 + (49*x0) + (106624*x1)), rmask & tmp50 & xmask, other=0.0)
    tmp52 = tl.full(tmp51.shape, 0.0, tmp51.dtype)
    tmp53 = tl.where(tmp50, tmp51, tmp52)
    tmp54 = tl.where(tmp20, tmp46, tmp53)
    tmp55 = tl.full(tmp54.shape, 0.0, tmp54.dtype)
    tmp56 = tl.where(tmp14, tmp54, tmp55)
    tmp57 = tl.where(tmp4, tmp13, tmp56)
    tmp59 = tmp57 - tmp58
    tmp61 = 0.001
    tmp62 = tmp60 + tmp61
    tmp63 = tl.sqrt(tmp62)
    tmp64 = 1 / tmp63
    tmp65 = 1.0
    tmp66 = tmp64 * tmp65
    tmp67 = tmp59 * tmp66
    tmp69 = tmp67 * tmp68
    tmp71 = tmp69 + tmp70
    tmp72 = triton_helpers.maximum(0, tmp71)
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK, RBLOCK])
    tmp75 = tl.where(rmask & xmask, tmp73, 0)
    tmp76 = tl.sum(tmp75, 1)[:, None]
    tmp77 = 49.0
    tmp78 = tmp76 / tmp77
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp78, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ph/cphiuszxxer7rr4kdkgdcyzq2bjevd4xchimdw4sxin3ypxxehc2.py
# Source Nodes: [x_333, x_336, x_337, x_340, x_341], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.view]
# x_333 => add_256, mul_331, mul_332, sub_110
# x_336 => relu_110
# x_337 => mean
# x_340 => convolution_110
# x_341 => view
triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_view_57 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_view_57', 'mutated_arg_names': ['in_out_ptr0']},
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


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1 = args
    args.clear()
    assert_size_stride(arg0_1, (128, ), (1, ))
    assert_size_stride(arg1_1, (128, ), (1, ))
    assert_size_stride(arg2_1, (128, ), (1, ))
    assert_size_stride(arg3_1, (128, ), (1, ))
    assert_size_stride(arg4_1, (128, ), (1, ))
    assert_size_stride(arg5_1, (128, ), (1, ))
    assert_size_stride(arg6_1, (200, ), (1, ))
    assert_size_stride(arg7_1, (200, ), (1, ))
    assert_size_stride(arg8_1, (200, ), (1, ))
    assert_size_stride(arg9_1, (200, ), (1, ))
    assert_size_stride(arg10_1, (316, ), (1, ))
    assert_size_stride(arg11_1, (316, ), (1, ))
    assert_size_stride(arg12_1, (200, ), (1, ))
    assert_size_stride(arg13_1, (200, ), (1, ))
    assert_size_stride(arg14_1, (200, ), (1, ))
    assert_size_stride(arg15_1, (200, ), (1, ))
    assert_size_stride(arg16_1, (336, ), (1, ))
    assert_size_stride(arg17_1, (336, ), (1, ))
    assert_size_stride(arg18_1, (200, ), (1, ))
    assert_size_stride(arg19_1, (200, ), (1, ))
    assert_size_stride(arg20_1, (200, ), (1, ))
    assert_size_stride(arg21_1, (200, ), (1, ))
    assert_size_stride(arg22_1, (356, ), (1, ))
    assert_size_stride(arg23_1, (356, ), (1, ))
    assert_size_stride(arg24_1, (200, ), (1, ))
    assert_size_stride(arg25_1, (200, ), (1, ))
    assert_size_stride(arg26_1, (200, ), (1, ))
    assert_size_stride(arg27_1, (200, ), (1, ))
    assert_size_stride(arg28_1, (376, ), (1, ))
    assert_size_stride(arg29_1, (376, ), (1, ))
    assert_size_stride(arg30_1, (376, ), (1, ))
    assert_size_stride(arg31_1, (376, ), (1, ))
    assert_size_stride(arg32_1, (400, ), (1, ))
    assert_size_stride(arg33_1, (400, ), (1, ))
    assert_size_stride(arg34_1, (400, ), (1, ))
    assert_size_stride(arg35_1, (400, ), (1, ))
    assert_size_stride(arg36_1, (704, ), (1, ))
    assert_size_stride(arg37_1, (704, ), (1, ))
    assert_size_stride(arg38_1, (400, ), (1, ))
    assert_size_stride(arg39_1, (400, ), (1, ))
    assert_size_stride(arg40_1, (400, ), (1, ))
    assert_size_stride(arg41_1, (400, ), (1, ))
    assert_size_stride(arg42_1, (768, ), (1, ))
    assert_size_stride(arg43_1, (768, ), (1, ))
    assert_size_stride(arg44_1, (400, ), (1, ))
    assert_size_stride(arg45_1, (400, ), (1, ))
    assert_size_stride(arg46_1, (400, ), (1, ))
    assert_size_stride(arg47_1, (400, ), (1, ))
    assert_size_stride(arg48_1, (832, ), (1, ))
    assert_size_stride(arg49_1, (832, ), (1, ))
    assert_size_stride(arg50_1, (400, ), (1, ))
    assert_size_stride(arg51_1, (400, ), (1, ))
    assert_size_stride(arg52_1, (400, ), (1, ))
    assert_size_stride(arg53_1, (400, ), (1, ))
    assert_size_stride(arg54_1, (896, ), (1, ))
    assert_size_stride(arg55_1, (896, ), (1, ))
    assert_size_stride(arg56_1, (400, ), (1, ))
    assert_size_stride(arg57_1, (400, ), (1, ))
    assert_size_stride(arg58_1, (400, ), (1, ))
    assert_size_stride(arg59_1, (400, ), (1, ))
    assert_size_stride(arg60_1, (960, ), (1, ))
    assert_size_stride(arg61_1, (960, ), (1, ))
    assert_size_stride(arg62_1, (400, ), (1, ))
    assert_size_stride(arg63_1, (400, ), (1, ))
    assert_size_stride(arg64_1, (400, ), (1, ))
    assert_size_stride(arg65_1, (400, ), (1, ))
    assert_size_stride(arg66_1, (1024, ), (1, ))
    assert_size_stride(arg67_1, (1024, ), (1, ))
    assert_size_stride(arg68_1, (400, ), (1, ))
    assert_size_stride(arg69_1, (400, ), (1, ))
    assert_size_stride(arg70_1, (400, ), (1, ))
    assert_size_stride(arg71_1, (400, ), (1, ))
    assert_size_stride(arg72_1, (1088, ), (1, ))
    assert_size_stride(arg73_1, (1088, ), (1, ))
    assert_size_stride(arg74_1, (400, ), (1, ))
    assert_size_stride(arg75_1, (400, ), (1, ))
    assert_size_stride(arg76_1, (400, ), (1, ))
    assert_size_stride(arg77_1, (400, ), (1, ))
    assert_size_stride(arg78_1, (1152, ), (1, ))
    assert_size_stride(arg79_1, (1152, ), (1, ))
    assert_size_stride(arg80_1, (1152, ), (1, ))
    assert_size_stride(arg81_1, (1152, ), (1, ))
    assert_size_stride(arg82_1, (800, ), (1, ))
    assert_size_stride(arg83_1, (800, ), (1, ))
    assert_size_stride(arg84_1, (800, ), (1, ))
    assert_size_stride(arg85_1, (800, ), (1, ))
    assert_size_stride(arg86_1, (1216, ), (1, ))
    assert_size_stride(arg87_1, (1216, ), (1, ))
    assert_size_stride(arg88_1, (800, ), (1, ))
    assert_size_stride(arg89_1, (800, ), (1, ))
    assert_size_stride(arg90_1, (800, ), (1, ))
    assert_size_stride(arg91_1, (800, ), (1, ))
    assert_size_stride(arg92_1, (1280, ), (1, ))
    assert_size_stride(arg93_1, (1280, ), (1, ))
    assert_size_stride(arg94_1, (800, ), (1, ))
    assert_size_stride(arg95_1, (800, ), (1, ))
    assert_size_stride(arg96_1, (800, ), (1, ))
    assert_size_stride(arg97_1, (800, ), (1, ))
    assert_size_stride(arg98_1, (1344, ), (1, ))
    assert_size_stride(arg99_1, (1344, ), (1, ))
    assert_size_stride(arg100_1, (800, ), (1, ))
    assert_size_stride(arg101_1, (800, ), (1, ))
    assert_size_stride(arg102_1, (800, ), (1, ))
    assert_size_stride(arg103_1, (800, ), (1, ))
    assert_size_stride(arg104_1, (1408, ), (1, ))
    assert_size_stride(arg105_1, (1408, ), (1, ))
    assert_size_stride(arg106_1, (800, ), (1, ))
    assert_size_stride(arg107_1, (800, ), (1, ))
    assert_size_stride(arg108_1, (800, ), (1, ))
    assert_size_stride(arg109_1, (800, ), (1, ))
    assert_size_stride(arg110_1, (1472, ), (1, ))
    assert_size_stride(arg111_1, (1472, ), (1, ))
    assert_size_stride(arg112_1, (800, ), (1, ))
    assert_size_stride(arg113_1, (800, ), (1, ))
    assert_size_stride(arg114_1, (800, ), (1, ))
    assert_size_stride(arg115_1, (800, ), (1, ))
    assert_size_stride(arg116_1, (1536, ), (1, ))
    assert_size_stride(arg117_1, (1536, ), (1, ))
    assert_size_stride(arg118_1, (800, ), (1, ))
    assert_size_stride(arg119_1, (800, ), (1, ))
    assert_size_stride(arg120_1, (800, ), (1, ))
    assert_size_stride(arg121_1, (800, ), (1, ))
    assert_size_stride(arg122_1, (1600, ), (1, ))
    assert_size_stride(arg123_1, (1600, ), (1, ))
    assert_size_stride(arg124_1, (800, ), (1, ))
    assert_size_stride(arg125_1, (800, ), (1, ))
    assert_size_stride(arg126_1, (800, ), (1, ))
    assert_size_stride(arg127_1, (800, ), (1, ))
    assert_size_stride(arg128_1, (1664, ), (1, ))
    assert_size_stride(arg129_1, (1664, ), (1, ))
    assert_size_stride(arg130_1, (800, ), (1, ))
    assert_size_stride(arg131_1, (800, ), (1, ))
    assert_size_stride(arg132_1, (800, ), (1, ))
    assert_size_stride(arg133_1, (800, ), (1, ))
    assert_size_stride(arg134_1, (1728, ), (1, ))
    assert_size_stride(arg135_1, (1728, ), (1, ))
    assert_size_stride(arg136_1, (800, ), (1, ))
    assert_size_stride(arg137_1, (800, ), (1, ))
    assert_size_stride(arg138_1, (800, ), (1, ))
    assert_size_stride(arg139_1, (800, ), (1, ))
    assert_size_stride(arg140_1, (1792, ), (1, ))
    assert_size_stride(arg141_1, (1792, ), (1, ))
    assert_size_stride(arg142_1, (800, ), (1, ))
    assert_size_stride(arg143_1, (800, ), (1, ))
    assert_size_stride(arg144_1, (800, ), (1, ))
    assert_size_stride(arg145_1, (800, ), (1, ))
    assert_size_stride(arg146_1, (1856, ), (1, ))
    assert_size_stride(arg147_1, (1856, ), (1, ))
    assert_size_stride(arg148_1, (800, ), (1, ))
    assert_size_stride(arg149_1, (800, ), (1, ))
    assert_size_stride(arg150_1, (800, ), (1, ))
    assert_size_stride(arg151_1, (800, ), (1, ))
    assert_size_stride(arg152_1, (1920, ), (1, ))
    assert_size_stride(arg153_1, (1920, ), (1, ))
    assert_size_stride(arg154_1, (800, ), (1, ))
    assert_size_stride(arg155_1, (800, ), (1, ))
    assert_size_stride(arg156_1, (800, ), (1, ))
    assert_size_stride(arg157_1, (800, ), (1, ))
    assert_size_stride(arg158_1, (1984, ), (1, ))
    assert_size_stride(arg159_1, (1984, ), (1, ))
    assert_size_stride(arg160_1, (800, ), (1, ))
    assert_size_stride(arg161_1, (800, ), (1, ))
    assert_size_stride(arg162_1, (800, ), (1, ))
    assert_size_stride(arg163_1, (800, ), (1, ))
    assert_size_stride(arg164_1, (2048, ), (1, ))
    assert_size_stride(arg165_1, (2048, ), (1, ))
    assert_size_stride(arg166_1, (800, ), (1, ))
    assert_size_stride(arg167_1, (800, ), (1, ))
    assert_size_stride(arg168_1, (800, ), (1, ))
    assert_size_stride(arg169_1, (800, ), (1, ))
    assert_size_stride(arg170_1, (2112, ), (1, ))
    assert_size_stride(arg171_1, (2112, ), (1, ))
    assert_size_stride(arg172_1, (800, ), (1, ))
    assert_size_stride(arg173_1, (800, ), (1, ))
    assert_size_stride(arg174_1, (800, ), (1, ))
    assert_size_stride(arg175_1, (800, ), (1, ))
    assert_size_stride(arg176_1, (2176, ), (1, ))
    assert_size_stride(arg177_1, (2176, ), (1, ))
    assert_size_stride(arg178_1, (800, ), (1, ))
    assert_size_stride(arg179_1, (800, ), (1, ))
    assert_size_stride(arg180_1, (800, ), (1, ))
    assert_size_stride(arg181_1, (800, ), (1, ))
    assert_size_stride(arg182_1, (2240, ), (1, ))
    assert_size_stride(arg183_1, (2240, ), (1, ))
    assert_size_stride(arg184_1, (800, ), (1, ))
    assert_size_stride(arg185_1, (800, ), (1, ))
    assert_size_stride(arg186_1, (800, ), (1, ))
    assert_size_stride(arg187_1, (800, ), (1, ))
    assert_size_stride(arg188_1, (2304, ), (1, ))
    assert_size_stride(arg189_1, (2304, ), (1, ))
    assert_size_stride(arg190_1, (800, ), (1, ))
    assert_size_stride(arg191_1, (800, ), (1, ))
    assert_size_stride(arg192_1, (800, ), (1, ))
    assert_size_stride(arg193_1, (800, ), (1, ))
    assert_size_stride(arg194_1, (2368, ), (1, ))
    assert_size_stride(arg195_1, (2368, ), (1, ))
    assert_size_stride(arg196_1, (800, ), (1, ))
    assert_size_stride(arg197_1, (800, ), (1, ))
    assert_size_stride(arg198_1, (800, ), (1, ))
    assert_size_stride(arg199_1, (800, ), (1, ))
    assert_size_stride(arg200_1, (2432, ), (1, ))
    assert_size_stride(arg201_1, (2432, ), (1, ))
    assert_size_stride(arg202_1, (2432, ), (1, ))
    assert_size_stride(arg203_1, (2432, ), (1, ))
    assert_size_stride(arg204_1, (1600, ), (1, ))
    assert_size_stride(arg205_1, (1600, ), (1, ))
    assert_size_stride(arg206_1, (1600, ), (1, ))
    assert_size_stride(arg207_1, (1600, ), (1, ))
    assert_size_stride(arg208_1, (2432, ), (1, ))
    assert_size_stride(arg209_1, (2432, ), (1, ))
    assert_size_stride(arg210_1, (1600, ), (1, ))
    assert_size_stride(arg211_1, (1600, ), (1, ))
    assert_size_stride(arg212_1, (1600, ), (1, ))
    assert_size_stride(arg213_1, (1600, ), (1, ))
    assert_size_stride(arg214_1, (2560, ), (1, ))
    assert_size_stride(arg215_1, (2560, ), (1, ))
    assert_size_stride(arg216_1, (1600, ), (1, ))
    assert_size_stride(arg217_1, (1600, ), (1, ))
    assert_size_stride(arg218_1, (1600, ), (1, ))
    assert_size_stride(arg219_1, (1600, ), (1, ))
    assert_size_stride(arg220_1, (2688, ), (1, ))
    assert_size_stride(arg221_1, (2688, ), (1, ))
    assert_size_stride(arg222_1, (128, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg223_1, (296, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg224_1, (200, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg225_1, (200, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg226_1, (276, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(arg227_1, (200, 316, 1, 1), (316, 1, 1, 1))
    assert_size_stride(arg228_1, (200, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg229_1, (276, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(arg230_1, (200, 336, 1, 1), (336, 1, 1, 1))
    assert_size_stride(arg231_1, (200, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg232_1, (276, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(arg233_1, (200, 356, 1, 1), (356, 1, 1, 1))
    assert_size_stride(arg234_1, (200, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(arg235_1, (276, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(arg236_1, (640, 376, 1, 1), (376, 1, 1, 1))
    assert_size_stride(arg237_1, (400, 376, 1, 1), (376, 1, 1, 1))
    assert_size_stride(arg238_1, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg239_1, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(arg240_1, (400, 704, 1, 1), (704, 1, 1, 1))
    assert_size_stride(arg241_1, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg242_1, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(arg243_1, (400, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg244_1, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg245_1, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(arg246_1, (400, 832, 1, 1), (832, 1, 1, 1))
    assert_size_stride(arg247_1, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg248_1, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(arg249_1, (400, 896, 1, 1), (896, 1, 1, 1))
    assert_size_stride(arg250_1, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg251_1, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(arg252_1, (400, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg253_1, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg254_1, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(arg255_1, (400, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg256_1, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg257_1, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(arg258_1, (400, 1088, 1, 1), (1088, 1, 1, 1))
    assert_size_stride(arg259_1, (400, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(arg260_1, (576, 400, 1, 1), (400, 1, 1, 1))
    assert_size_stride(arg261_1, (1152, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg262_1, (800, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg263_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg264_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg265_1, (800, 1216, 1, 1), (1216, 1, 1, 1))
    assert_size_stride(arg266_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg267_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg268_1, (800, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(arg269_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg270_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg271_1, (800, 1344, 1, 1), (1344, 1, 1, 1))
    assert_size_stride(arg272_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg273_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg274_1, (800, 1408, 1, 1), (1408, 1, 1, 1))
    assert_size_stride(arg275_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg276_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg277_1, (800, 1472, 1, 1), (1472, 1, 1, 1))
    assert_size_stride(arg278_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg279_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg280_1, (800, 1536, 1, 1), (1536, 1, 1, 1))
    assert_size_stride(arg281_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg282_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg283_1, (800, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(arg284_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg285_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg286_1, (800, 1664, 1, 1), (1664, 1, 1, 1))
    assert_size_stride(arg287_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg288_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg289_1, (800, 1728, 1, 1), (1728, 1, 1, 1))
    assert_size_stride(arg290_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg291_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg292_1, (800, 1792, 1, 1), (1792, 1, 1, 1))
    assert_size_stride(arg293_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg294_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg295_1, (800, 1856, 1, 1), (1856, 1, 1, 1))
    assert_size_stride(arg296_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg297_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg298_1, (800, 1920, 1, 1), (1920, 1, 1, 1))
    assert_size_stride(arg299_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg300_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg301_1, (800, 1984, 1, 1), (1984, 1, 1, 1))
    assert_size_stride(arg302_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg303_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg304_1, (800, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg305_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg306_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg307_1, (800, 2112, 1, 1), (2112, 1, 1, 1))
    assert_size_stride(arg308_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg309_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg310_1, (800, 2176, 1, 1), (2176, 1, 1, 1))
    assert_size_stride(arg311_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg312_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg313_1, (800, 2240, 1, 1), (2240, 1, 1, 1))
    assert_size_stride(arg314_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg315_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg316_1, (800, 2304, 1, 1), (2304, 1, 1, 1))
    assert_size_stride(arg317_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg318_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg319_1, (800, 2368, 1, 1), (2368, 1, 1, 1))
    assert_size_stride(arg320_1, (800, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg321_1, (1088, 800, 1, 1), (800, 1, 1, 1))
    assert_size_stride(arg322_1, (2304, 2432, 1, 1), (2432, 1, 1, 1))
    assert_size_stride(arg323_1, (1600, 2432, 1, 1), (2432, 1, 1, 1))
    assert_size_stride(arg324_1, (1600, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg325_1, (2176, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(arg326_1, (1600, 2432, 1, 1), (2432, 1, 1, 1))
    assert_size_stride(arg327_1, (1600, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg328_1, (2176, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(arg329_1, (1600, 2560, 1, 1), (2560, 1, 1, 1))
    assert_size_stride(arg330_1, (1600, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg331_1, (2176, 1600, 1, 1), (1600, 1, 1, 1))
    assert_size_stride(arg332_1, (1000, 2688, 1, 1), (2688, 1, 1, 1))
    assert_size_stride(arg333_1, (1000, ), (1, ))
    assert_size_stride(arg334_1, (128, ), (1, ))
    assert_size_stride(arg335_1, (128, ), (1, ))
    assert_size_stride(arg336_1, (128, ), (1, ))
    assert_size_stride(arg337_1, (128, ), (1, ))
    assert_size_stride(arg338_1, (128, ), (1, ))
    assert_size_stride(arg339_1, (128, ), (1, ))
    assert_size_stride(arg340_1, (200, ), (1, ))
    assert_size_stride(arg341_1, (200, ), (1, ))
    assert_size_stride(arg342_1, (200, ), (1, ))
    assert_size_stride(arg343_1, (200, ), (1, ))
    assert_size_stride(arg344_1, (316, ), (1, ))
    assert_size_stride(arg345_1, (316, ), (1, ))
    assert_size_stride(arg346_1, (200, ), (1, ))
    assert_size_stride(arg347_1, (200, ), (1, ))
    assert_size_stride(arg348_1, (200, ), (1, ))
    assert_size_stride(arg349_1, (200, ), (1, ))
    assert_size_stride(arg350_1, (336, ), (1, ))
    assert_size_stride(arg351_1, (336, ), (1, ))
    assert_size_stride(arg352_1, (200, ), (1, ))
    assert_size_stride(arg353_1, (200, ), (1, ))
    assert_size_stride(arg354_1, (200, ), (1, ))
    assert_size_stride(arg355_1, (200, ), (1, ))
    assert_size_stride(arg356_1, (356, ), (1, ))
    assert_size_stride(arg357_1, (356, ), (1, ))
    assert_size_stride(arg358_1, (200, ), (1, ))
    assert_size_stride(arg359_1, (200, ), (1, ))
    assert_size_stride(arg360_1, (200, ), (1, ))
    assert_size_stride(arg361_1, (200, ), (1, ))
    assert_size_stride(arg362_1, (376, ), (1, ))
    assert_size_stride(arg363_1, (376, ), (1, ))
    assert_size_stride(arg364_1, (376, ), (1, ))
    assert_size_stride(arg365_1, (376, ), (1, ))
    assert_size_stride(arg366_1, (400, ), (1, ))
    assert_size_stride(arg367_1, (400, ), (1, ))
    assert_size_stride(arg368_1, (400, ), (1, ))
    assert_size_stride(arg369_1, (400, ), (1, ))
    assert_size_stride(arg370_1, (704, ), (1, ))
    assert_size_stride(arg371_1, (704, ), (1, ))
    assert_size_stride(arg372_1, (400, ), (1, ))
    assert_size_stride(arg373_1, (400, ), (1, ))
    assert_size_stride(arg374_1, (400, ), (1, ))
    assert_size_stride(arg375_1, (400, ), (1, ))
    assert_size_stride(arg376_1, (768, ), (1, ))
    assert_size_stride(arg377_1, (768, ), (1, ))
    assert_size_stride(arg378_1, (400, ), (1, ))
    assert_size_stride(arg379_1, (400, ), (1, ))
    assert_size_stride(arg380_1, (400, ), (1, ))
    assert_size_stride(arg381_1, (400, ), (1, ))
    assert_size_stride(arg382_1, (832, ), (1, ))
    assert_size_stride(arg383_1, (832, ), (1, ))
    assert_size_stride(arg384_1, (400, ), (1, ))
    assert_size_stride(arg385_1, (400, ), (1, ))
    assert_size_stride(arg386_1, (400, ), (1, ))
    assert_size_stride(arg387_1, (400, ), (1, ))
    assert_size_stride(arg388_1, (896, ), (1, ))
    assert_size_stride(arg389_1, (896, ), (1, ))
    assert_size_stride(arg390_1, (400, ), (1, ))
    assert_size_stride(arg391_1, (400, ), (1, ))
    assert_size_stride(arg392_1, (400, ), (1, ))
    assert_size_stride(arg393_1, (400, ), (1, ))
    assert_size_stride(arg394_1, (960, ), (1, ))
    assert_size_stride(arg395_1, (960, ), (1, ))
    assert_size_stride(arg396_1, (400, ), (1, ))
    assert_size_stride(arg397_1, (400, ), (1, ))
    assert_size_stride(arg398_1, (400, ), (1, ))
    assert_size_stride(arg399_1, (400, ), (1, ))
    assert_size_stride(arg400_1, (1024, ), (1, ))
    assert_size_stride(arg401_1, (1024, ), (1, ))
    assert_size_stride(arg402_1, (400, ), (1, ))
    assert_size_stride(arg403_1, (400, ), (1, ))
    assert_size_stride(arg404_1, (400, ), (1, ))
    assert_size_stride(arg405_1, (400, ), (1, ))
    assert_size_stride(arg406_1, (1088, ), (1, ))
    assert_size_stride(arg407_1, (1088, ), (1, ))
    assert_size_stride(arg408_1, (400, ), (1, ))
    assert_size_stride(arg409_1, (400, ), (1, ))
    assert_size_stride(arg410_1, (400, ), (1, ))
    assert_size_stride(arg411_1, (400, ), (1, ))
    assert_size_stride(arg412_1, (1152, ), (1, ))
    assert_size_stride(arg413_1, (1152, ), (1, ))
    assert_size_stride(arg414_1, (1152, ), (1, ))
    assert_size_stride(arg415_1, (1152, ), (1, ))
    assert_size_stride(arg416_1, (800, ), (1, ))
    assert_size_stride(arg417_1, (800, ), (1, ))
    assert_size_stride(arg418_1, (800, ), (1, ))
    assert_size_stride(arg419_1, (800, ), (1, ))
    assert_size_stride(arg420_1, (1216, ), (1, ))
    assert_size_stride(arg421_1, (1216, ), (1, ))
    assert_size_stride(arg422_1, (800, ), (1, ))
    assert_size_stride(arg423_1, (800, ), (1, ))
    assert_size_stride(arg424_1, (800, ), (1, ))
    assert_size_stride(arg425_1, (800, ), (1, ))
    assert_size_stride(arg426_1, (1280, ), (1, ))
    assert_size_stride(arg427_1, (1280, ), (1, ))
    assert_size_stride(arg428_1, (800, ), (1, ))
    assert_size_stride(arg429_1, (800, ), (1, ))
    assert_size_stride(arg430_1, (800, ), (1, ))
    assert_size_stride(arg431_1, (800, ), (1, ))
    assert_size_stride(arg432_1, (1344, ), (1, ))
    assert_size_stride(arg433_1, (1344, ), (1, ))
    assert_size_stride(arg434_1, (800, ), (1, ))
    assert_size_stride(arg435_1, (800, ), (1, ))
    assert_size_stride(arg436_1, (800, ), (1, ))
    assert_size_stride(arg437_1, (800, ), (1, ))
    assert_size_stride(arg438_1, (1408, ), (1, ))
    assert_size_stride(arg439_1, (1408, ), (1, ))
    assert_size_stride(arg440_1, (800, ), (1, ))
    assert_size_stride(arg441_1, (800, ), (1, ))
    assert_size_stride(arg442_1, (800, ), (1, ))
    assert_size_stride(arg443_1, (800, ), (1, ))
    assert_size_stride(arg444_1, (1472, ), (1, ))
    assert_size_stride(arg445_1, (1472, ), (1, ))
    assert_size_stride(arg446_1, (800, ), (1, ))
    assert_size_stride(arg447_1, (800, ), (1, ))
    assert_size_stride(arg448_1, (800, ), (1, ))
    assert_size_stride(arg449_1, (800, ), (1, ))
    assert_size_stride(arg450_1, (1536, ), (1, ))
    assert_size_stride(arg451_1, (1536, ), (1, ))
    assert_size_stride(arg452_1, (800, ), (1, ))
    assert_size_stride(arg453_1, (800, ), (1, ))
    assert_size_stride(arg454_1, (800, ), (1, ))
    assert_size_stride(arg455_1, (800, ), (1, ))
    assert_size_stride(arg456_1, (1600, ), (1, ))
    assert_size_stride(arg457_1, (1600, ), (1, ))
    assert_size_stride(arg458_1, (800, ), (1, ))
    assert_size_stride(arg459_1, (800, ), (1, ))
    assert_size_stride(arg460_1, (800, ), (1, ))
    assert_size_stride(arg461_1, (800, ), (1, ))
    assert_size_stride(arg462_1, (1664, ), (1, ))
    assert_size_stride(arg463_1, (1664, ), (1, ))
    assert_size_stride(arg464_1, (800, ), (1, ))
    assert_size_stride(arg465_1, (800, ), (1, ))
    assert_size_stride(arg466_1, (800, ), (1, ))
    assert_size_stride(arg467_1, (800, ), (1, ))
    assert_size_stride(arg468_1, (1728, ), (1, ))
    assert_size_stride(arg469_1, (1728, ), (1, ))
    assert_size_stride(arg470_1, (800, ), (1, ))
    assert_size_stride(arg471_1, (800, ), (1, ))
    assert_size_stride(arg472_1, (800, ), (1, ))
    assert_size_stride(arg473_1, (800, ), (1, ))
    assert_size_stride(arg474_1, (1792, ), (1, ))
    assert_size_stride(arg475_1, (1792, ), (1, ))
    assert_size_stride(arg476_1, (800, ), (1, ))
    assert_size_stride(arg477_1, (800, ), (1, ))
    assert_size_stride(arg478_1, (800, ), (1, ))
    assert_size_stride(arg479_1, (800, ), (1, ))
    assert_size_stride(arg480_1, (1856, ), (1, ))
    assert_size_stride(arg481_1, (1856, ), (1, ))
    assert_size_stride(arg482_1, (800, ), (1, ))
    assert_size_stride(arg483_1, (800, ), (1, ))
    assert_size_stride(arg484_1, (800, ), (1, ))
    assert_size_stride(arg485_1, (800, ), (1, ))
    assert_size_stride(arg486_1, (1920, ), (1, ))
    assert_size_stride(arg487_1, (1920, ), (1, ))
    assert_size_stride(arg488_1, (800, ), (1, ))
    assert_size_stride(arg489_1, (800, ), (1, ))
    assert_size_stride(arg490_1, (800, ), (1, ))
    assert_size_stride(arg491_1, (800, ), (1, ))
    assert_size_stride(arg492_1, (1984, ), (1, ))
    assert_size_stride(arg493_1, (1984, ), (1, ))
    assert_size_stride(arg494_1, (800, ), (1, ))
    assert_size_stride(arg495_1, (800, ), (1, ))
    assert_size_stride(arg496_1, (800, ), (1, ))
    assert_size_stride(arg497_1, (800, ), (1, ))
    assert_size_stride(arg498_1, (2048, ), (1, ))
    assert_size_stride(arg499_1, (2048, ), (1, ))
    assert_size_stride(arg500_1, (800, ), (1, ))
    assert_size_stride(arg501_1, (800, ), (1, ))
    assert_size_stride(arg502_1, (800, ), (1, ))
    assert_size_stride(arg503_1, (800, ), (1, ))
    assert_size_stride(arg504_1, (2112, ), (1, ))
    assert_size_stride(arg505_1, (2112, ), (1, ))
    assert_size_stride(arg506_1, (800, ), (1, ))
    assert_size_stride(arg507_1, (800, ), (1, ))
    assert_size_stride(arg508_1, (800, ), (1, ))
    assert_size_stride(arg509_1, (800, ), (1, ))
    assert_size_stride(arg510_1, (2176, ), (1, ))
    assert_size_stride(arg511_1, (2176, ), (1, ))
    assert_size_stride(arg512_1, (800, ), (1, ))
    assert_size_stride(arg513_1, (800, ), (1, ))
    assert_size_stride(arg514_1, (800, ), (1, ))
    assert_size_stride(arg515_1, (800, ), (1, ))
    assert_size_stride(arg516_1, (2240, ), (1, ))
    assert_size_stride(arg517_1, (2240, ), (1, ))
    assert_size_stride(arg518_1, (800, ), (1, ))
    assert_size_stride(arg519_1, (800, ), (1, ))
    assert_size_stride(arg520_1, (800, ), (1, ))
    assert_size_stride(arg521_1, (800, ), (1, ))
    assert_size_stride(arg522_1, (2304, ), (1, ))
    assert_size_stride(arg523_1, (2304, ), (1, ))
    assert_size_stride(arg524_1, (800, ), (1, ))
    assert_size_stride(arg525_1, (800, ), (1, ))
    assert_size_stride(arg526_1, (800, ), (1, ))
    assert_size_stride(arg527_1, (800, ), (1, ))
    assert_size_stride(arg528_1, (2368, ), (1, ))
    assert_size_stride(arg529_1, (2368, ), (1, ))
    assert_size_stride(arg530_1, (800, ), (1, ))
    assert_size_stride(arg531_1, (800, ), (1, ))
    assert_size_stride(arg532_1, (800, ), (1, ))
    assert_size_stride(arg533_1, (800, ), (1, ))
    assert_size_stride(arg534_1, (2432, ), (1, ))
    assert_size_stride(arg535_1, (2432, ), (1, ))
    assert_size_stride(arg536_1, (2432, ), (1, ))
    assert_size_stride(arg537_1, (2432, ), (1, ))
    assert_size_stride(arg538_1, (1600, ), (1, ))
    assert_size_stride(arg539_1, (1600, ), (1, ))
    assert_size_stride(arg540_1, (1600, ), (1, ))
    assert_size_stride(arg541_1, (1600, ), (1, ))
    assert_size_stride(arg542_1, (2432, ), (1, ))
    assert_size_stride(arg543_1, (2432, ), (1, ))
    assert_size_stride(arg544_1, (1600, ), (1, ))
    assert_size_stride(arg545_1, (1600, ), (1, ))
    assert_size_stride(arg546_1, (1600, ), (1, ))
    assert_size_stride(arg547_1, (1600, ), (1, ))
    assert_size_stride(arg548_1, (2560, ), (1, ))
    assert_size_stride(arg549_1, (2560, ), (1, ))
    assert_size_stride(arg550_1, (1600, ), (1, ))
    assert_size_stride(arg551_1, (1600, ), (1, ))
    assert_size_stride(arg552_1, (1600, ), (1, ))
    assert_size_stride(arg553_1, (1600, ), (1, ))
    assert_size_stride(arg554_1, (2688, ), (1, ))
    assert_size_stride(arg555_1, (2688, ), (1, ))
    assert_size_stride(arg556_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg556_1, arg222_1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 128, 112, 112), (1605632, 12544, 112, 1))
        del arg222_1
        del arg556_1
        buf1 = buf0; del buf0  # reuse
        # Source Nodes: [x_1, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf1, arg334_1, arg335_1, arg0_1, arg1_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg0_1
        del arg1_1
        del arg334_1
        del arg335_1
        buf3 = empty((8, 128, 56, 56), device='cuda', dtype=torch.float32)
        buf5 = empty((8, 128, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1, x_10, x_4, x_5, x_7, x_8, x_in, x_in_1, x_s], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_1.run(buf1, arg336_1, arg337_1, arg2_1, arg3_1, arg338_1, arg339_1, arg4_1, arg5_1, buf3, buf5, 3211264, grid=grid(3211264), stream=stream0)
        del arg2_1
        del arg336_1
        del arg337_1
        del arg338_1
        del arg339_1
        del arg3_1
        del arg4_1
        del arg5_1
        del buf1
        # Source Nodes: [x_5, x_7, x_s], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf4 = extern_kernels.convolution(buf3, arg223_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (8, 296, 56, 56), (928256, 3136, 56, 1))
        del arg223_1
        del buf3
        # Source Nodes: [x_10, x_8, x_in_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf6 = extern_kernels.convolution(buf5, arg224_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 200, 56, 56), (627200, 3136, 56, 1))
        del arg224_1
        buf7 = buf6; del buf6  # reuse
        # Source Nodes: [x_11, x_13, x_in_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf7, arg340_1, arg341_1, arg6_1, arg7_1, 5017600, grid=grid(5017600), stream=stream0)
        del arg340_1
        del arg341_1
        del arg6_1
        del arg7_1
        # Source Nodes: [x_11, x_13, x_in_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf8 = extern_kernels.convolution(buf7, arg225_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf8, (8, 200, 56, 56), (627200, 3136, 56, 1))
        del arg225_1
        del buf7
        buf9 = buf8; del buf8  # reuse
        # Source Nodes: [x_14, x_16, x_in_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf9, arg342_1, arg343_1, arg8_1, arg9_1, 5017600, grid=grid(5017600), stream=stream0)
        del arg342_1
        del arg343_1
        del arg8_1
        del arg9_1
        # Source Nodes: [x_14, x_16, x_in_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf10 = extern_kernels.convolution(buf9, arg226_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (8, 276, 56, 56), (865536, 3136, 56, 1))
        del arg226_1
        del buf9
        buf11 = empty((8, 316, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_138, x_17, x_19, x_in_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_3.run(buf4, buf10, arg344_1, arg345_1, arg10_1, arg11_1, buf11, 7927808, grid=grid(7927808), stream=stream0)
        del arg10_1
        del arg11_1
        del arg344_1
        del arg345_1
        # Source Nodes: [cat_138, x_17, x_19, x_in_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        buf12 = extern_kernels.convolution(buf11, arg227_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (8, 200, 56, 56), (627200, 3136, 56, 1))
        del arg227_1
        del buf11
        buf13 = buf12; del buf12  # reuse
        # Source Nodes: [x_20, x_22, x_in_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf13, arg346_1, arg347_1, arg12_1, arg13_1, 5017600, grid=grid(5017600), stream=stream0)
        del arg12_1
        del arg13_1
        del arg346_1
        del arg347_1
        # Source Nodes: [x_20, x_22, x_in_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf14 = extern_kernels.convolution(buf13, arg228_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf14, (8, 200, 56, 56), (627200, 3136, 56, 1))
        del arg228_1
        del buf13
        buf15 = buf14; del buf14  # reuse
        # Source Nodes: [x_23, x_25, x_in_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf15, arg348_1, arg349_1, arg14_1, arg15_1, 5017600, grid=grid(5017600), stream=stream0)
        del arg14_1
        del arg15_1
        del arg348_1
        del arg349_1
        # Source Nodes: [x_23, x_25, x_in_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf16 = extern_kernels.convolution(buf15, arg229_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (8, 276, 56, 56), (865536, 3136, 56, 1))
        del arg229_1
        del buf15
        buf17 = empty((8, 336, 56, 56), device='cuda', dtype=torch.float32)
        buf18 = buf17; del buf17  # reuse
        # Source Nodes: [cat_136, x_26, x_28, x_in_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_4.run(buf18, buf4, buf10, buf16, arg350_1, arg351_1, arg16_1, arg17_1, 8429568, grid=grid(8429568), stream=stream0)
        del arg16_1
        del arg17_1
        del arg350_1
        del arg351_1
        # Source Nodes: [x_26, x_28, x_in_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf19 = extern_kernels.convolution(buf18, arg230_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (8, 200, 56, 56), (627200, 3136, 56, 1))
        del arg230_1
        del buf18
        buf20 = buf19; del buf19  # reuse
        # Source Nodes: [x_29, x_31, x_in_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf20, arg352_1, arg353_1, arg18_1, arg19_1, 5017600, grid=grid(5017600), stream=stream0)
        del arg18_1
        del arg19_1
        del arg352_1
        del arg353_1
        # Source Nodes: [x_29, x_31, x_in_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf21 = extern_kernels.convolution(buf20, arg231_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf21, (8, 200, 56, 56), (627200, 3136, 56, 1))
        del arg231_1
        del buf20
        buf22 = buf21; del buf21  # reuse
        # Source Nodes: [x_32, x_34, x_in_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf22, arg354_1, arg355_1, arg20_1, arg21_1, 5017600, grid=grid(5017600), stream=stream0)
        del arg20_1
        del arg21_1
        del arg354_1
        del arg355_1
        # Source Nodes: [x_32, x_34, x_in_11], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf23 = extern_kernels.convolution(buf22, arg232_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (8, 276, 56, 56), (865536, 3136, 56, 1))
        del arg232_1
        del buf22
        buf24 = empty((8, 356, 56, 56), device='cuda', dtype=torch.float32)
        buf25 = buf24; del buf24  # reuse
        # Source Nodes: [cat_134, x_35, x_37, x_in_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_5.run(buf25, buf4, buf10, buf16, buf23, arg356_1, arg357_1, arg22_1, arg23_1, 8931328, grid=grid(8931328), stream=stream0)
        del arg22_1
        del arg23_1
        del arg356_1
        del arg357_1
        # Source Nodes: [x_35, x_37, x_in_13], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf26 = extern_kernels.convolution(buf25, arg233_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 200, 56, 56), (627200, 3136, 56, 1))
        del arg233_1
        del buf25
        buf27 = buf26; del buf26  # reuse
        # Source Nodes: [x_38, x_40, x_in_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf27, arg358_1, arg359_1, arg24_1, arg25_1, 5017600, grid=grid(5017600), stream=stream0)
        del arg24_1
        del arg25_1
        del arg358_1
        del arg359_1
        # Source Nodes: [x_38, x_40, x_in_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf28 = extern_kernels.convolution(buf27, arg234_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf28, (8, 200, 56, 56), (627200, 3136, 56, 1))
        del arg234_1
        del buf27
        buf29 = buf28; del buf28  # reuse
        # Source Nodes: [x_41, x_43, x_in_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf29, arg360_1, arg361_1, arg26_1, arg27_1, 5017600, grid=grid(5017600), stream=stream0)
        del arg26_1
        del arg27_1
        del arg360_1
        del arg361_1
        # Source Nodes: [x_41, x_43, x_in_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf30 = extern_kernels.convolution(buf29, arg235_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 276, 56, 56), (865536, 3136, 56, 1))
        del arg235_1
        del buf29
        buf31 = empty((8, 120, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_133], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf4, buf10, buf16, buf23, buf30, buf31, 3010560, grid=grid(3010560), stream=stream0)
        buf33 = empty((8, 376, 56, 56), device='cuda', dtype=torch.float32)
        buf35 = empty((8, 376, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_132, x_44, x_46, x_47, x_49, x_in_17, x_s_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_7.run(buf4, buf10, buf16, buf23, buf30, buf31, arg362_1, arg363_1, arg28_1, arg29_1, arg364_1, arg365_1, arg30_1, arg31_1, buf33, buf35, 9433088, grid=grid(9433088), stream=stream0)
        del arg28_1
        del arg29_1
        del arg30_1
        del arg31_1
        del arg362_1
        del arg363_1
        del arg364_1
        del arg365_1
        del buf10
        del buf16
        del buf23
        del buf30
        del buf4
        # Source Nodes: [x_44, x_46, x_s_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf34 = extern_kernels.convolution(buf33, arg236_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (8, 640, 28, 28), (501760, 784, 28, 1))
        del arg236_1
        del buf33
        # Source Nodes: [x_47, x_49, x_in_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf36 = extern_kernels.convolution(buf35, arg237_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (8, 400, 56, 56), (1254400, 3136, 56, 1))
        del arg237_1
        del buf35
        buf37 = buf36; del buf36  # reuse
        # Source Nodes: [x_50, x_52, x_in_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_8.run(buf37, arg366_1, arg367_1, arg32_1, arg33_1, 10035200, grid=grid(10035200), stream=stream0)
        del arg32_1
        del arg33_1
        del arg366_1
        del arg367_1
        # Source Nodes: [x_50, x_52, x_in_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf38 = extern_kernels.convolution(buf37, arg238_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf38, (8, 400, 28, 28), (313600, 784, 28, 1))
        del arg238_1
        del buf37
        buf39 = buf38; del buf38  # reuse
        # Source Nodes: [x_53, x_55, x_in_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf39, arg368_1, arg369_1, arg34_1, arg35_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg34_1
        del arg35_1
        del arg368_1
        del arg369_1
        # Source Nodes: [x_53, x_55, x_in_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf40 = extern_kernels.convolution(buf39, arg239_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (8, 576, 28, 28), (451584, 784, 28, 1))
        del arg239_1
        del buf39
        buf41 = empty((8, 704, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_130, x_56, x_58, x_in_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_10.run(buf34, buf40, arg370_1, arg371_1, arg36_1, arg37_1, buf41, 4415488, grid=grid(4415488), stream=stream0)
        del arg36_1
        del arg370_1
        del arg371_1
        del arg37_1
        # Source Nodes: [cat_130, x_56, x_58, x_in_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        buf42 = extern_kernels.convolution(buf41, arg240_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 400, 28, 28), (313600, 784, 28, 1))
        del arg240_1
        del buf41
        buf43 = buf42; del buf42  # reuse
        # Source Nodes: [x_59, x_61, x_in_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf43, arg372_1, arg373_1, arg38_1, arg39_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg372_1
        del arg373_1
        del arg38_1
        del arg39_1
        # Source Nodes: [x_59, x_61, x_in_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf44 = extern_kernels.convolution(buf43, arg241_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf44, (8, 400, 28, 28), (313600, 784, 28, 1))
        del arg241_1
        del buf43
        buf45 = buf44; del buf44  # reuse
        # Source Nodes: [x_62, x_64, x_in_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf45, arg374_1, arg375_1, arg40_1, arg41_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg374_1
        del arg375_1
        del arg40_1
        del arg41_1
        # Source Nodes: [x_62, x_64, x_in_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf46 = extern_kernels.convolution(buf45, arg242_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (8, 576, 28, 28), (451584, 784, 28, 1))
        del arg242_1
        del buf45
        buf47 = empty((8, 768, 28, 28), device='cuda', dtype=torch.float32)
        buf48 = buf47; del buf47  # reuse
        # Source Nodes: [cat_128, x_65, x_67, x_in_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_11.run(buf48, buf34, buf40, buf46, arg376_1, arg377_1, arg42_1, arg43_1, 4816896, grid=grid(4816896), stream=stream0)
        del arg376_1
        del arg377_1
        del arg42_1
        del arg43_1
        # Source Nodes: [x_65, x_67, x_in_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf49 = extern_kernels.convolution(buf48, arg243_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (8, 400, 28, 28), (313600, 784, 28, 1))
        del arg243_1
        del buf48
        buf50 = buf49; del buf49  # reuse
        # Source Nodes: [x_68, x_70, x_in_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf50, arg378_1, arg379_1, arg44_1, arg45_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg378_1
        del arg379_1
        del arg44_1
        del arg45_1
        # Source Nodes: [x_68, x_70, x_in_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf51 = extern_kernels.convolution(buf50, arg244_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf51, (8, 400, 28, 28), (313600, 784, 28, 1))
        del arg244_1
        del buf50
        buf52 = buf51; del buf51  # reuse
        # Source Nodes: [x_71, x_73, x_in_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf52, arg380_1, arg381_1, arg46_1, arg47_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg380_1
        del arg381_1
        del arg46_1
        del arg47_1
        # Source Nodes: [x_71, x_73, x_in_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf53 = extern_kernels.convolution(buf52, arg245_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (8, 576, 28, 28), (451584, 784, 28, 1))
        del arg245_1
        del buf52
        buf54 = empty((8, 832, 28, 28), device='cuda', dtype=torch.float32)
        buf55 = buf54; del buf54  # reuse
        # Source Nodes: [cat_126, x_74, x_76, x_in_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_12.run(buf55, buf34, buf40, buf46, buf53, arg382_1, arg383_1, arg48_1, arg49_1, 5218304, grid=grid(5218304), stream=stream0)
        del arg382_1
        del arg383_1
        del arg48_1
        del arg49_1
        # Source Nodes: [x_74, x_76, x_in_29], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf56 = extern_kernels.convolution(buf55, arg246_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 400, 28, 28), (313600, 784, 28, 1))
        del arg246_1
        del buf55
        buf57 = buf56; del buf56  # reuse
        # Source Nodes: [x_77, x_79, x_in_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf57, arg384_1, arg385_1, arg50_1, arg51_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg384_1
        del arg385_1
        del arg50_1
        del arg51_1
        # Source Nodes: [x_77, x_79, x_in_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf58 = extern_kernels.convolution(buf57, arg247_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf58, (8, 400, 28, 28), (313600, 784, 28, 1))
        del arg247_1
        del buf57
        buf59 = buf58; del buf58  # reuse
        # Source Nodes: [x_80, x_82, x_in_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf59, arg386_1, arg387_1, arg52_1, arg53_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg386_1
        del arg387_1
        del arg52_1
        del arg53_1
        # Source Nodes: [x_80, x_82, x_in_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf60 = extern_kernels.convolution(buf59, arg248_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (8, 576, 28, 28), (451584, 784, 28, 1))
        del arg248_1
        del buf59
        buf63 = empty((8, 896, 28, 28), device='cuda', dtype=torch.float32)
        buf61 = reinterpret_tensor(buf63, (8, 512, 28, 28), (702464, 784, 28, 1), 0)  # alias
        # Source Nodes: [x_s1_5, x_s1_6, x_s1_7, x_s1_8], Original ATen: [aten.add]
        triton_poi_fused_add_13.run(buf34, buf40, buf46, buf53, buf60, buf61, 3211264, grid=grid(3211264), stream=stream0)
        buf62 = reinterpret_tensor(buf63, (8, 384, 28, 28), (702464, 784, 28, 1), 401408)  # alias
        # Source Nodes: [cat_125], Original ATen: [aten.cat]
        triton_poi_fused_cat_14.run(buf34, buf40, buf46, buf53, buf60, buf62, 2408448, grid=grid(2408448), stream=stream0)
        del buf40
        del buf46
        del buf53
        del buf60
        buf64 = empty((8, 896, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_83, x_85, x_in_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf63, arg388_1, arg389_1, arg54_1, arg55_1, buf64, 5619712, grid=grid(5619712), stream=stream0)
        del arg388_1
        del arg389_1
        del arg54_1
        del arg55_1
        # Source Nodes: [x_83, x_85, x_in_33], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf65 = extern_kernels.convolution(buf64, arg249_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (8, 400, 28, 28), (313600, 784, 28, 1))
        del arg249_1
        del buf64
        buf66 = buf65; del buf65  # reuse
        # Source Nodes: [x_86, x_88, x_in_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf66, arg390_1, arg391_1, arg56_1, arg57_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg390_1
        del arg391_1
        del arg56_1
        del arg57_1
        # Source Nodes: [x_86, x_88, x_in_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf67 = extern_kernels.convolution(buf66, arg250_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf67, (8, 400, 28, 28), (313600, 784, 28, 1))
        del arg250_1
        del buf66
        buf68 = buf67; del buf67  # reuse
        # Source Nodes: [x_89, x_91, x_in_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf68, arg392_1, arg393_1, arg58_1, arg59_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg392_1
        del arg393_1
        del arg58_1
        del arg59_1
        # Source Nodes: [x_89, x_91, x_in_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf69 = extern_kernels.convolution(buf68, arg251_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (8, 576, 28, 28), (451584, 784, 28, 1))
        del arg251_1
        del buf68
        buf70 = empty((8, 960, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_122, x_92, x_94, x_in_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_16.run(buf61, buf69, buf62, arg394_1, arg395_1, arg60_1, arg61_1, buf70, 6021120, grid=grid(6021120), stream=stream0)
        del arg394_1
        del arg395_1
        del arg60_1
        del arg61_1
        # Source Nodes: [cat_122, x_92, x_94, x_in_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        buf71 = extern_kernels.convolution(buf70, arg252_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (8, 400, 28, 28), (313600, 784, 28, 1))
        del arg252_1
        del buf70
        buf72 = buf71; del buf71  # reuse
        # Source Nodes: [x_95, x_97, x_in_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf72, arg396_1, arg397_1, arg62_1, arg63_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg396_1
        del arg397_1
        del arg62_1
        del arg63_1
        # Source Nodes: [x_95, x_97, x_in_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf73 = extern_kernels.convolution(buf72, arg253_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf73, (8, 400, 28, 28), (313600, 784, 28, 1))
        del arg253_1
        del buf72
        buf74 = buf73; del buf73  # reuse
        # Source Nodes: [x_100, x_98, x_in_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf74, arg398_1, arg399_1, arg64_1, arg65_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg398_1
        del arg399_1
        del arg64_1
        del arg65_1
        # Source Nodes: [x_100, x_98, x_in_39], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf75 = extern_kernels.convolution(buf74, arg254_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (8, 576, 28, 28), (451584, 784, 28, 1))
        del arg254_1
        del buf74
        buf76 = empty((8, 1024, 28, 28), device='cuda', dtype=torch.float32)
        buf77 = buf76; del buf76  # reuse
        # Source Nodes: [cat_120, x_101, x_103, x_in_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_17.run(buf77, buf61, buf69, buf75, buf62, arg400_1, arg401_1, arg66_1, arg67_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg400_1
        del arg401_1
        del arg66_1
        del arg67_1
        # Source Nodes: [x_101, x_103, x_in_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf78 = extern_kernels.convolution(buf77, arg255_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (8, 400, 28, 28), (313600, 784, 28, 1))
        del arg255_1
        del buf77
        buf79 = buf78; del buf78  # reuse
        # Source Nodes: [x_104, x_106, x_in_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf79, arg402_1, arg403_1, arg68_1, arg69_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg402_1
        del arg403_1
        del arg68_1
        del arg69_1
        # Source Nodes: [x_104, x_106, x_in_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf80 = extern_kernels.convolution(buf79, arg256_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf80, (8, 400, 28, 28), (313600, 784, 28, 1))
        del arg256_1
        del buf79
        buf81 = buf80; del buf80  # reuse
        # Source Nodes: [x_107, x_109, x_in_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf81, arg404_1, arg405_1, arg70_1, arg71_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg404_1
        del arg405_1
        del arg70_1
        del arg71_1
        # Source Nodes: [x_107, x_109, x_in_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf82 = extern_kernels.convolution(buf81, arg257_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (8, 576, 28, 28), (451584, 784, 28, 1))
        del arg257_1
        del buf81
        buf83 = empty((8, 1088, 28, 28), device='cuda', dtype=torch.float32)
        buf84 = buf83; del buf83  # reuse
        # Source Nodes: [cat_118, x_110, x_112, x_in_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_18.run(buf84, buf61, buf69, buf75, buf82, buf62, arg406_1, arg407_1, arg72_1, arg73_1, 6823936, grid=grid(6823936), stream=stream0)
        del arg406_1
        del arg407_1
        del arg72_1
        del arg73_1
        # Source Nodes: [x_110, x_112, x_in_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf85 = extern_kernels.convolution(buf84, arg258_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (8, 400, 28, 28), (313600, 784, 28, 1))
        del arg258_1
        del buf84
        buf86 = buf85; del buf85  # reuse
        # Source Nodes: [x_113, x_115, x_in_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf86, arg408_1, arg409_1, arg74_1, arg75_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg408_1
        del arg409_1
        del arg74_1
        del arg75_1
        # Source Nodes: [x_113, x_115, x_in_46], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf87 = extern_kernels.convolution(buf86, arg259_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf87, (8, 400, 28, 28), (313600, 784, 28, 1))
        del arg259_1
        del buf86
        buf88 = buf87; del buf87  # reuse
        # Source Nodes: [x_116, x_118, x_in_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf88, arg410_1, arg411_1, arg76_1, arg77_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg410_1
        del arg411_1
        del arg76_1
        del arg77_1
        # Source Nodes: [x_116, x_118, x_in_47], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf89 = extern_kernels.convolution(buf88, arg260_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (8, 576, 28, 28), (451584, 784, 28, 1))
        del arg260_1
        buf90 = buf34; del buf34  # reuse
        # Source Nodes: [cat_117], Original ATen: [aten.cat]
        triton_poi_fused_cat_19.run(buf62, buf69, buf75, buf82, buf89, buf90, 4014080, grid=grid(4014080), stream=stream0)
        del buf62
        buf92 = empty((8, 1152, 28, 28), device='cuda', dtype=torch.float32)
        buf94 = empty((8, 1152, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_116, x_119, x_121, x_122, x_124, x_in_49, x_s_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_20.run(buf61, buf69, buf75, buf82, buf89, buf90, arg412_1, arg413_1, arg78_1, arg79_1, arg414_1, arg415_1, arg80_1, arg81_1, buf92, buf94, 7225344, grid=grid(7225344), stream=stream0)
        del arg412_1
        del arg413_1
        del arg414_1
        del arg415_1
        del arg78_1
        del arg79_1
        del arg80_1
        del arg81_1
        del buf61
        del buf63
        del buf69
        del buf75
        del buf82
        del buf90
        # Source Nodes: [x_119, x_121, x_s_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf93 = extern_kernels.convolution(buf92, arg261_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (8, 1152, 14, 14), (225792, 196, 14, 1))
        del arg261_1
        del buf92
        # Source Nodes: [x_122, x_124, x_in_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf95 = extern_kernels.convolution(buf94, arg262_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (8, 800, 28, 28), (627200, 784, 28, 1))
        del arg262_1
        del buf94
        buf96 = buf95; del buf95  # reuse
        # Source Nodes: [x_125, x_127, x_in_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21.run(buf96, arg416_1, arg417_1, arg82_1, arg83_1, 5017600, grid=grid(5017600), stream=stream0)
        del arg416_1
        del arg417_1
        del arg82_1
        del arg83_1
        # Source Nodes: [x_125, x_127, x_in_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf97 = extern_kernels.convolution(buf96, arg263_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf97, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg263_1
        del buf96
        buf98 = buf97; del buf97  # reuse
        # Source Nodes: [x_128, x_130, x_in_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf98, arg418_1, arg419_1, arg84_1, arg85_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg418_1
        del arg419_1
        del arg84_1
        del arg85_1
        # Source Nodes: [x_128, x_130, x_in_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf99 = extern_kernels.convolution(buf98, arg264_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (8, 1088, 14, 14), (213248, 196, 14, 1))
        del arg264_1
        del buf98
        buf100 = empty((8, 1216, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_114, x_131, x_133, x_in_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_23.run(buf93, buf99, arg420_1, arg421_1, arg86_1, arg87_1, buf100, 1906688, grid=grid(1906688), stream=stream0)
        del arg420_1
        del arg421_1
        del arg86_1
        del arg87_1
        # Source Nodes: [cat_114, x_131, x_133, x_in_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        buf101 = extern_kernels.convolution(buf100, arg265_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg265_1
        del buf100
        buf102 = buf101; del buf101  # reuse
        # Source Nodes: [x_134, x_136, x_in_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf102, arg422_1, arg423_1, arg88_1, arg89_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg422_1
        del arg423_1
        del arg88_1
        del arg89_1
        # Source Nodes: [x_134, x_136, x_in_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf103 = extern_kernels.convolution(buf102, arg266_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf103, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg266_1
        del buf102
        buf104 = buf103; del buf103  # reuse
        # Source Nodes: [x_137, x_139, x_in_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf104, arg424_1, arg425_1, arg90_1, arg91_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg424_1
        del arg425_1
        del arg90_1
        del arg91_1
        # Source Nodes: [x_137, x_139, x_in_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf105 = extern_kernels.convolution(buf104, arg267_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (8, 1088, 14, 14), (213248, 196, 14, 1))
        del arg267_1
        del buf104
        buf106 = empty((8, 1280, 14, 14), device='cuda', dtype=torch.float32)
        buf107 = buf106; del buf106  # reuse
        # Source Nodes: [cat_112, x_140, x_142, x_in_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_24.run(buf107, buf93, buf99, buf105, arg426_1, arg427_1, arg92_1, arg93_1, 2007040, grid=grid(2007040), stream=stream0)
        del arg426_1
        del arg427_1
        del arg92_1
        del arg93_1
        # Source Nodes: [x_140, x_142, x_in_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf108 = extern_kernels.convolution(buf107, arg268_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg268_1
        del buf107
        buf109 = buf108; del buf108  # reuse
        # Source Nodes: [x_143, x_145, x_in_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf109, arg428_1, arg429_1, arg94_1, arg95_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg428_1
        del arg429_1
        del arg94_1
        del arg95_1
        # Source Nodes: [x_143, x_145, x_in_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf110 = extern_kernels.convolution(buf109, arg269_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf110, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg269_1
        del buf109
        buf111 = buf110; del buf110  # reuse
        # Source Nodes: [x_146, x_148, x_in_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf111, arg430_1, arg431_1, arg96_1, arg97_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg430_1
        del arg431_1
        del arg96_1
        del arg97_1
        # Source Nodes: [x_146, x_148, x_in_59], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf112 = extern_kernels.convolution(buf111, arg270_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf112, (8, 1088, 14, 14), (213248, 196, 14, 1))
        del arg270_1
        del buf111
        buf113 = empty((8, 1344, 14, 14), device='cuda', dtype=torch.float32)
        buf114 = buf113; del buf113  # reuse
        # Source Nodes: [cat_110, x_149, x_151, x_in_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_25.run(buf114, buf93, buf99, buf105, buf112, arg432_1, arg433_1, arg98_1, arg99_1, 2107392, grid=grid(2107392), stream=stream0)
        del arg432_1
        del arg433_1
        del arg98_1
        del arg99_1
        # Source Nodes: [x_149, x_151, x_in_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf115 = extern_kernels.convolution(buf114, arg271_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg271_1
        del buf114
        buf116 = buf115; del buf115  # reuse
        # Source Nodes: [x_152, x_154, x_in_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf116, arg434_1, arg435_1, arg100_1, arg101_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg100_1
        del arg101_1
        del arg434_1
        del arg435_1
        # Source Nodes: [x_152, x_154, x_in_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf117 = extern_kernels.convolution(buf116, arg272_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf117, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg272_1
        del buf116
        buf118 = buf117; del buf117  # reuse
        # Source Nodes: [x_155, x_157, x_in_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf118, arg436_1, arg437_1, arg102_1, arg103_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg102_1
        del arg103_1
        del arg436_1
        del arg437_1
        # Source Nodes: [x_155, x_157, x_in_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf119 = extern_kernels.convolution(buf118, arg273_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (8, 1088, 14, 14), (213248, 196, 14, 1))
        del arg273_1
        del buf118
        buf122 = empty((8, 1408, 14, 14), device='cuda', dtype=torch.float32)
        buf120 = reinterpret_tensor(buf122, (8, 1024, 14, 14), (275968, 196, 14, 1), 0)  # alias
        # Source Nodes: [x_s1_13, x_s1_14, x_s1_15, x_s1_16], Original ATen: [aten.add]
        triton_poi_fused_add_26.run(buf93, buf99, buf105, buf112, buf119, buf120, 1605632, grid=grid(1605632), stream=stream0)
        buf121 = reinterpret_tensor(buf122, (8, 384, 14, 14), (275968, 196, 14, 1), 200704)  # alias
        # Source Nodes: [cat_109], Original ATen: [aten.cat]
        triton_poi_fused_cat_27.run(buf93, buf99, buf105, buf112, buf119, buf121, 602112, grid=grid(602112), stream=stream0)
        del buf105
        del buf112
        del buf119
        del buf93
        del buf99
        buf123 = empty((8, 1408, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_158, x_160, x_in_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_28.run(buf122, arg438_1, arg439_1, arg104_1, arg105_1, buf123, 2207744, grid=grid(2207744), stream=stream0)
        del arg104_1
        del arg105_1
        del arg438_1
        del arg439_1
        # Source Nodes: [x_158, x_160, x_in_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf124 = extern_kernels.convolution(buf123, arg274_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg274_1
        del buf123
        buf125 = buf124; del buf124  # reuse
        # Source Nodes: [x_161, x_163, x_in_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf125, arg440_1, arg441_1, arg106_1, arg107_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg106_1
        del arg107_1
        del arg440_1
        del arg441_1
        # Source Nodes: [x_161, x_163, x_in_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf126 = extern_kernels.convolution(buf125, arg275_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf126, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg275_1
        del buf125
        buf127 = buf126; del buf126  # reuse
        # Source Nodes: [x_164, x_166, x_in_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf127, arg442_1, arg443_1, arg108_1, arg109_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg108_1
        del arg109_1
        del arg442_1
        del arg443_1
        # Source Nodes: [x_164, x_166, x_in_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf128 = extern_kernels.convolution(buf127, arg276_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (8, 1088, 14, 14), (213248, 196, 14, 1))
        del arg276_1
        del buf127
        buf129 = empty((8, 1472, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_106, x_167, x_169, x_in_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_29.run(buf120, buf128, buf121, arg444_1, arg445_1, arg110_1, arg111_1, buf129, 2308096, grid=grid(2308096), stream=stream0)
        del arg110_1
        del arg111_1
        del arg444_1
        del arg445_1
        # Source Nodes: [cat_106, x_167, x_169, x_in_69], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        buf130 = extern_kernels.convolution(buf129, arg277_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg277_1
        del buf129
        buf131 = buf130; del buf130  # reuse
        # Source Nodes: [x_170, x_172, x_in_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf131, arg446_1, arg447_1, arg112_1, arg113_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg112_1
        del arg113_1
        del arg446_1
        del arg447_1
        # Source Nodes: [x_170, x_172, x_in_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf132 = extern_kernels.convolution(buf131, arg278_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf132, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg278_1
        del buf131
        buf133 = buf132; del buf132  # reuse
        # Source Nodes: [x_173, x_175, x_in_71], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf133, arg448_1, arg449_1, arg114_1, arg115_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg114_1
        del arg115_1
        del arg448_1
        del arg449_1
        # Source Nodes: [x_173, x_175, x_in_71], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf134 = extern_kernels.convolution(buf133, arg279_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (8, 1088, 14, 14), (213248, 196, 14, 1))
        del arg279_1
        del buf133
        buf135 = empty((8, 1536, 14, 14), device='cuda', dtype=torch.float32)
        buf136 = buf135; del buf135  # reuse
        # Source Nodes: [cat_104, x_176, x_178, x_in_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_30.run(buf136, buf120, buf128, buf134, buf121, arg450_1, arg451_1, arg116_1, arg117_1, 2408448, grid=grid(2408448), stream=stream0)
        del arg116_1
        del arg117_1
        del arg450_1
        del arg451_1
        # Source Nodes: [x_176, x_178, x_in_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf137 = extern_kernels.convolution(buf136, arg280_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg280_1
        del buf136
        buf138 = buf137; del buf137  # reuse
        # Source Nodes: [x_179, x_181, x_in_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf138, arg452_1, arg453_1, arg118_1, arg119_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg118_1
        del arg119_1
        del arg452_1
        del arg453_1
        # Source Nodes: [x_179, x_181, x_in_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf139 = extern_kernels.convolution(buf138, arg281_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf139, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg281_1
        del buf138
        buf140 = buf139; del buf139  # reuse
        # Source Nodes: [x_182, x_184, x_in_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf140, arg454_1, arg455_1, arg120_1, arg121_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg120_1
        del arg121_1
        del arg454_1
        del arg455_1
        # Source Nodes: [x_182, x_184, x_in_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf141 = extern_kernels.convolution(buf140, arg282_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (8, 1088, 14, 14), (213248, 196, 14, 1))
        del arg282_1
        del buf140
        buf142 = reinterpret_tensor(buf88, (8, 1600, 14, 14), (313600, 196, 14, 1), 0); del buf88  # reuse
        buf143 = buf142; del buf142  # reuse
        # Source Nodes: [cat_102, x_185, x_187, x_in_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_31.run(buf143, buf120, buf128, buf134, buf141, buf121, arg456_1, arg457_1, arg122_1, arg123_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg122_1
        del arg123_1
        del arg456_1
        del arg457_1
        # Source Nodes: [x_185, x_187, x_in_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf144 = extern_kernels.convolution(buf143, arg283_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf144, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg283_1
        del buf143
        buf145 = buf144; del buf144  # reuse
        # Source Nodes: [x_188, x_190, x_in_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf145, arg458_1, arg459_1, arg124_1, arg125_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg124_1
        del arg125_1
        del arg458_1
        del arg459_1
        # Source Nodes: [x_188, x_190, x_in_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf146 = extern_kernels.convolution(buf145, arg284_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf146, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg284_1
        del buf145
        buf147 = buf146; del buf146  # reuse
        # Source Nodes: [x_191, x_193, x_in_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf147, arg460_1, arg461_1, arg126_1, arg127_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg126_1
        del arg127_1
        del arg460_1
        del arg461_1
        # Source Nodes: [x_191, x_193, x_in_79], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf148 = extern_kernels.convolution(buf147, arg285_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (8, 1088, 14, 14), (213248, 196, 14, 1))
        del arg285_1
        del buf147
        buf151 = empty((8, 1664, 14, 14), device='cuda', dtype=torch.float32)
        buf149 = reinterpret_tensor(buf151, (8, 1024, 14, 14), (326144, 196, 14, 1), 0)  # alias
        # Source Nodes: [x_s1_17, x_s1_18, x_s1_19, x_s1_20], Original ATen: [aten.add]
        triton_poi_fused_add_32.run(buf120, buf128, buf134, buf141, buf148, buf149, 1605632, grid=grid(1605632), stream=stream0)
        del buf120
        buf150 = reinterpret_tensor(buf151, (8, 640, 14, 14), (326144, 196, 14, 1), 200704)  # alias
        # Source Nodes: [cat_101], Original ATen: [aten.cat]
        triton_poi_fused_cat_33.run(buf121, buf128, buf134, buf141, buf148, buf150, 1003520, grid=grid(1003520), stream=stream0)
        del buf121
        del buf128
        del buf134
        del buf141
        del buf148
        buf152 = empty((8, 1664, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_194, x_196, x_in_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34.run(buf151, arg462_1, arg463_1, arg128_1, arg129_1, buf152, 2609152, grid=grid(2609152), stream=stream0)
        del arg128_1
        del arg129_1
        del arg462_1
        del arg463_1
        # Source Nodes: [x_194, x_196, x_in_81], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf153 = extern_kernels.convolution(buf152, arg286_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg286_1
        del buf152
        buf154 = buf153; del buf153  # reuse
        # Source Nodes: [x_197, x_199, x_in_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf154, arg464_1, arg465_1, arg130_1, arg131_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg130_1
        del arg131_1
        del arg464_1
        del arg465_1
        # Source Nodes: [x_197, x_199, x_in_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf155 = extern_kernels.convolution(buf154, arg287_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf155, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg287_1
        del buf154
        buf156 = buf155; del buf155  # reuse
        # Source Nodes: [x_200, x_202, x_in_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf156, arg466_1, arg467_1, arg132_1, arg133_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg132_1
        del arg133_1
        del arg466_1
        del arg467_1
        # Source Nodes: [x_200, x_202, x_in_83], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf157 = extern_kernels.convolution(buf156, arg288_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (8, 1088, 14, 14), (213248, 196, 14, 1))
        del arg288_1
        del buf156
        buf158 = empty((8, 1728, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_98, x_203, x_205, x_in_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_35.run(buf149, buf157, buf150, arg468_1, arg469_1, arg134_1, arg135_1, buf158, 2709504, grid=grid(2709504), stream=stream0)
        del arg134_1
        del arg135_1
        del arg468_1
        del arg469_1
        # Source Nodes: [cat_98, x_203, x_205, x_in_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        buf159 = extern_kernels.convolution(buf158, arg289_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg289_1
        del buf158
        buf160 = buf159; del buf159  # reuse
        # Source Nodes: [x_206, x_208, x_in_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf160, arg470_1, arg471_1, arg136_1, arg137_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg136_1
        del arg137_1
        del arg470_1
        del arg471_1
        # Source Nodes: [x_206, x_208, x_in_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf161 = extern_kernels.convolution(buf160, arg290_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf161, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg290_1
        del buf160
        buf162 = buf161; del buf161  # reuse
        # Source Nodes: [x_209, x_211, x_in_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf162, arg472_1, arg473_1, arg138_1, arg139_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg138_1
        del arg139_1
        del arg472_1
        del arg473_1
        # Source Nodes: [x_209, x_211, x_in_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf163 = extern_kernels.convolution(buf162, arg291_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (8, 1088, 14, 14), (213248, 196, 14, 1))
        del arg291_1
        del buf162
        buf164 = empty((8, 1792, 14, 14), device='cuda', dtype=torch.float32)
        buf165 = buf164; del buf164  # reuse
        # Source Nodes: [cat_96, x_212, x_214, x_in_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_36.run(buf165, buf149, buf157, buf163, buf150, arg474_1, arg475_1, arg140_1, arg141_1, 2809856, grid=grid(2809856), stream=stream0)
        del arg140_1
        del arg141_1
        del arg474_1
        del arg475_1
        # Source Nodes: [x_212, x_214, x_in_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf166 = extern_kernels.convolution(buf165, arg292_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf166, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg292_1
        del buf165
        buf167 = buf166; del buf166  # reuse
        # Source Nodes: [x_215, x_217, x_in_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf167, arg476_1, arg477_1, arg142_1, arg143_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg142_1
        del arg143_1
        del arg476_1
        del arg477_1
        # Source Nodes: [x_215, x_217, x_in_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf168 = extern_kernels.convolution(buf167, arg293_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf168, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg293_1
        del buf167
        buf169 = buf168; del buf168  # reuse
        # Source Nodes: [x_218, x_220, x_in_91], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf169, arg478_1, arg479_1, arg144_1, arg145_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg144_1
        del arg145_1
        del arg478_1
        del arg479_1
        # Source Nodes: [x_218, x_220, x_in_91], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf170 = extern_kernels.convolution(buf169, arg294_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (8, 1088, 14, 14), (213248, 196, 14, 1))
        del arg294_1
        del buf169
        buf171 = empty((8, 1856, 14, 14), device='cuda', dtype=torch.float32)
        buf172 = buf171; del buf171  # reuse
        # Source Nodes: [cat_94, x_221, x_223, x_in_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_37.run(buf172, buf149, buf157, buf163, buf170, buf150, arg480_1, arg481_1, arg146_1, arg147_1, 2910208, grid=grid(2910208), stream=stream0)
        del arg146_1
        del arg147_1
        del arg480_1
        del arg481_1
        # Source Nodes: [x_221, x_223, x_in_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf173 = extern_kernels.convolution(buf172, arg295_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf173, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg295_1
        del buf172
        buf174 = buf173; del buf173  # reuse
        # Source Nodes: [x_224, x_226, x_in_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf174, arg482_1, arg483_1, arg148_1, arg149_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg148_1
        del arg149_1
        del arg482_1
        del arg483_1
        # Source Nodes: [x_224, x_226, x_in_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf175 = extern_kernels.convolution(buf174, arg296_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf175, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg296_1
        del buf174
        buf176 = buf175; del buf175  # reuse
        # Source Nodes: [x_227, x_229, x_in_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf176, arg484_1, arg485_1, arg150_1, arg151_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg150_1
        del arg151_1
        del arg484_1
        del arg485_1
        # Source Nodes: [x_227, x_229, x_in_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf177 = extern_kernels.convolution(buf176, arg297_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (8, 1088, 14, 14), (213248, 196, 14, 1))
        del arg297_1
        del buf176
        buf180 = reinterpret_tensor(buf31, (8, 1920, 14, 14), (376320, 196, 14, 1), 0); del buf31  # reuse
        buf178 = reinterpret_tensor(buf180, (8, 1024, 14, 14), (376320, 196, 14, 1), 0)  # alias
        # Source Nodes: [x_s1_21, x_s1_22, x_s1_23, x_s1_24], Original ATen: [aten.add]
        triton_poi_fused_add_38.run(buf149, buf157, buf163, buf170, buf177, buf178, 1605632, grid=grid(1605632), stream=stream0)
        del buf149
        buf179 = reinterpret_tensor(buf180, (8, 896, 14, 14), (376320, 196, 14, 1), 200704)  # alias
        # Source Nodes: [cat_93], Original ATen: [aten.cat]
        triton_poi_fused_cat_39.run(buf150, buf157, buf163, buf170, buf177, buf179, 1404928, grid=grid(1404928), stream=stream0)
        del buf150
        del buf151
        del buf157
        del buf163
        del buf170
        del buf177
        buf181 = empty((8, 1920, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_230, x_232, x_in_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_40.run(buf180, arg486_1, arg487_1, arg152_1, arg153_1, buf181, 3010560, grid=grid(3010560), stream=stream0)
        del arg152_1
        del arg153_1
        del arg486_1
        del arg487_1
        # Source Nodes: [x_230, x_232, x_in_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf182 = extern_kernels.convolution(buf181, arg298_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg298_1
        del buf181
        buf183 = buf182; del buf182  # reuse
        # Source Nodes: [x_233, x_235, x_in_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf183, arg488_1, arg489_1, arg154_1, arg155_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg154_1
        del arg155_1
        del arg488_1
        del arg489_1
        # Source Nodes: [x_233, x_235, x_in_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf184 = extern_kernels.convolution(buf183, arg299_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf184, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg299_1
        del buf183
        buf185 = buf184; del buf184  # reuse
        # Source Nodes: [x_236, x_238, x_in_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf185, arg490_1, arg491_1, arg156_1, arg157_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg156_1
        del arg157_1
        del arg490_1
        del arg491_1
        # Source Nodes: [x_236, x_238, x_in_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf186 = extern_kernels.convolution(buf185, arg300_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf186, (8, 1088, 14, 14), (213248, 196, 14, 1))
        del arg300_1
        del buf185
        buf187 = empty((8, 1984, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_90, x_239, x_241, x_in_101], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_41.run(buf178, buf186, buf179, arg492_1, arg493_1, arg158_1, arg159_1, buf187, 3110912, grid=grid(3110912), stream=stream0)
        del arg158_1
        del arg159_1
        del arg492_1
        del arg493_1
        # Source Nodes: [cat_90, x_239, x_241, x_in_101], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        buf188 = extern_kernels.convolution(buf187, arg301_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg301_1
        del buf187
        buf189 = buf188; del buf188  # reuse
        # Source Nodes: [x_242, x_244, x_in_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf189, arg494_1, arg495_1, arg160_1, arg161_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg160_1
        del arg161_1
        del arg494_1
        del arg495_1
        # Source Nodes: [x_242, x_244, x_in_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf190 = extern_kernels.convolution(buf189, arg302_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf190, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg302_1
        del buf189
        buf191 = buf190; del buf190  # reuse
        # Source Nodes: [x_245, x_247, x_in_103], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf191, arg496_1, arg497_1, arg162_1, arg163_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg162_1
        del arg163_1
        del arg496_1
        del arg497_1
        # Source Nodes: [x_245, x_247, x_in_103], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf192 = extern_kernels.convolution(buf191, arg303_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf192, (8, 1088, 14, 14), (213248, 196, 14, 1))
        del arg303_1
        del buf191
        buf193 = reinterpret_tensor(buf5, (8, 2048, 14, 14), (401408, 196, 14, 1), 0); del buf5  # reuse
        buf194 = buf193; del buf193  # reuse
        # Source Nodes: [cat_88, x_248, x_250, x_in_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_42.run(buf194, buf178, buf186, buf192, buf179, arg498_1, arg499_1, arg164_1, arg165_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg164_1
        del arg165_1
        del arg498_1
        del arg499_1
        # Source Nodes: [x_248, x_250, x_in_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf195 = extern_kernels.convolution(buf194, arg304_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf195, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg304_1
        del buf194
        buf196 = buf195; del buf195  # reuse
        # Source Nodes: [x_251, x_253, x_in_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf196, arg500_1, arg501_1, arg166_1, arg167_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg166_1
        del arg167_1
        del arg500_1
        del arg501_1
        # Source Nodes: [x_251, x_253, x_in_106], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf197 = extern_kernels.convolution(buf196, arg305_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf197, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg305_1
        del buf196
        buf198 = buf197; del buf197  # reuse
        # Source Nodes: [x_254, x_256, x_in_107], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf198, arg502_1, arg503_1, arg168_1, arg169_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg168_1
        del arg169_1
        del arg502_1
        del arg503_1
        # Source Nodes: [x_254, x_256, x_in_107], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf199 = extern_kernels.convolution(buf198, arg306_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf199, (8, 1088, 14, 14), (213248, 196, 14, 1))
        del arg306_1
        del buf198
        buf200 = empty((8, 2112, 14, 14), device='cuda', dtype=torch.float32)
        buf201 = buf200; del buf200  # reuse
        # Source Nodes: [cat_86, x_257, x_259, x_in_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_43.run(buf201, buf178, buf186, buf192, buf199, buf179, arg504_1, arg505_1, arg170_1, arg171_1, 3311616, grid=grid(3311616), stream=stream0)
        del arg170_1
        del arg171_1
        del arg504_1
        del arg505_1
        # Source Nodes: [x_257, x_259, x_in_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf202 = extern_kernels.convolution(buf201, arg307_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf202, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg307_1
        del buf201
        buf203 = buf202; del buf202  # reuse
        # Source Nodes: [x_260, x_262, x_in_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf203, arg506_1, arg507_1, arg172_1, arg173_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg172_1
        del arg173_1
        del arg506_1
        del arg507_1
        # Source Nodes: [x_260, x_262, x_in_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf204 = extern_kernels.convolution(buf203, arg308_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf204, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg308_1
        del buf203
        buf205 = buf204; del buf204  # reuse
        # Source Nodes: [x_263, x_265, x_in_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf205, arg508_1, arg509_1, arg174_1, arg175_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg174_1
        del arg175_1
        del arg508_1
        del arg509_1
        # Source Nodes: [x_263, x_265, x_in_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf206 = extern_kernels.convolution(buf205, arg309_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (8, 1088, 14, 14), (213248, 196, 14, 1))
        del arg309_1
        del buf205
        buf209 = empty((8, 2176, 14, 14), device='cuda', dtype=torch.float32)
        buf207 = reinterpret_tensor(buf209, (8, 1024, 14, 14), (426496, 196, 14, 1), 0)  # alias
        # Source Nodes: [x_s1_25, x_s1_26, x_s1_27, x_s1_28], Original ATen: [aten.add]
        triton_poi_fused_add_44.run(buf178, buf186, buf192, buf199, buf206, buf207, 1605632, grid=grid(1605632), stream=stream0)
        del buf178
        buf208 = reinterpret_tensor(buf209, (8, 1152, 14, 14), (426496, 196, 14, 1), 200704)  # alias
        # Source Nodes: [cat_85], Original ATen: [aten.cat]
        triton_poi_fused_cat_45.run(buf179, buf186, buf192, buf199, buf206, buf208, 1806336, grid=grid(1806336), stream=stream0)
        del buf179
        del buf180
        del buf186
        del buf192
        del buf199
        del buf206
        buf210 = empty((8, 2176, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_266, x_268, x_in_113], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_46.run(buf209, arg510_1, arg511_1, arg176_1, arg177_1, buf210, 3411968, grid=grid(3411968), stream=stream0)
        del arg176_1
        del arg177_1
        del arg510_1
        del arg511_1
        # Source Nodes: [x_266, x_268, x_in_113], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf211 = extern_kernels.convolution(buf210, arg310_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg310_1
        del buf210
        buf212 = buf211; del buf211  # reuse
        # Source Nodes: [x_269, x_271, x_in_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf212, arg512_1, arg513_1, arg178_1, arg179_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg178_1
        del arg179_1
        del arg512_1
        del arg513_1
        # Source Nodes: [x_269, x_271, x_in_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf213 = extern_kernels.convolution(buf212, arg311_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf213, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg311_1
        del buf212
        buf214 = buf213; del buf213  # reuse
        # Source Nodes: [x_272, x_274, x_in_115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf214, arg514_1, arg515_1, arg180_1, arg181_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg180_1
        del arg181_1
        del arg514_1
        del arg515_1
        # Source Nodes: [x_272, x_274, x_in_115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf215 = extern_kernels.convolution(buf214, arg312_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf215, (8, 1088, 14, 14), (213248, 196, 14, 1))
        del arg312_1
        del buf214
        buf216 = empty((8, 2240, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_82, x_275, x_277, x_in_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_47.run(buf207, buf215, buf208, arg516_1, arg517_1, arg182_1, arg183_1, buf216, 3512320, grid=grid(3512320), stream=stream0)
        del arg182_1
        del arg183_1
        del arg516_1
        del arg517_1
        # Source Nodes: [cat_82, x_275, x_277, x_in_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        buf217 = extern_kernels.convolution(buf216, arg313_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf217, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg313_1
        del buf216
        buf218 = buf217; del buf217  # reuse
        # Source Nodes: [x_278, x_280, x_in_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf218, arg518_1, arg519_1, arg184_1, arg185_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg184_1
        del arg185_1
        del arg518_1
        del arg519_1
        # Source Nodes: [x_278, x_280, x_in_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf219 = extern_kernels.convolution(buf218, arg314_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf219, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg314_1
        del buf218
        buf220 = buf219; del buf219  # reuse
        # Source Nodes: [x_281, x_283, x_in_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf220, arg520_1, arg521_1, arg186_1, arg187_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg186_1
        del arg187_1
        del arg520_1
        del arg521_1
        # Source Nodes: [x_281, x_283, x_in_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf221 = extern_kernels.convolution(buf220, arg315_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (8, 1088, 14, 14), (213248, 196, 14, 1))
        del arg315_1
        del buf220
        buf222 = reinterpret_tensor(buf89, (8, 2304, 14, 14), (451584, 196, 14, 1), 0); del buf89  # reuse
        buf223 = buf222; del buf222  # reuse
        # Source Nodes: [cat_80, x_284, x_286, x_in_121], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_48.run(buf223, buf207, buf215, buf221, buf208, arg522_1, arg523_1, arg188_1, arg189_1, 3612672, grid=grid(3612672), stream=stream0)
        del arg188_1
        del arg189_1
        del arg522_1
        del arg523_1
        # Source Nodes: [x_284, x_286, x_in_121], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf224 = extern_kernels.convolution(buf223, arg316_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg316_1
        del buf223
        buf225 = buf224; del buf224  # reuse
        # Source Nodes: [x_287, x_289, x_in_122], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf225, arg524_1, arg525_1, arg190_1, arg191_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg190_1
        del arg191_1
        del arg524_1
        del arg525_1
        # Source Nodes: [x_287, x_289, x_in_122], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf226 = extern_kernels.convolution(buf225, arg317_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf226, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg317_1
        del buf225
        buf227 = buf226; del buf226  # reuse
        # Source Nodes: [x_290, x_292, x_in_123], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf227, arg526_1, arg527_1, arg192_1, arg193_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg192_1
        del arg193_1
        del arg526_1
        del arg527_1
        # Source Nodes: [x_290, x_292, x_in_123], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf228 = extern_kernels.convolution(buf227, arg318_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf228, (8, 1088, 14, 14), (213248, 196, 14, 1))
        del arg318_1
        del buf227
        buf229 = empty((8, 2368, 14, 14), device='cuda', dtype=torch.float32)
        buf230 = buf229; del buf229  # reuse
        # Source Nodes: [cat_78, x_293, x_295, x_in_125], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_49.run(buf230, buf207, buf215, buf221, buf228, buf208, arg528_1, arg529_1, arg194_1, arg195_1, 3713024, grid=grid(3713024), stream=stream0)
        del arg194_1
        del arg195_1
        del arg528_1
        del arg529_1
        # Source Nodes: [x_293, x_295, x_in_125], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf231 = extern_kernels.convolution(buf230, arg319_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf231, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg319_1
        del buf230
        buf232 = buf231; del buf231  # reuse
        # Source Nodes: [x_296, x_298, x_in_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf232, arg530_1, arg531_1, arg196_1, arg197_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg196_1
        del arg197_1
        del arg530_1
        del arg531_1
        # Source Nodes: [x_296, x_298, x_in_126], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf233 = extern_kernels.convolution(buf232, arg320_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf233, (8, 800, 14, 14), (156800, 196, 14, 1))
        del arg320_1
        del buf232
        buf234 = buf233; del buf233  # reuse
        # Source Nodes: [x_299, x_301, x_in_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_22.run(buf234, arg532_1, arg533_1, arg198_1, arg199_1, 1254400, grid=grid(1254400), stream=stream0)
        del arg198_1
        del arg199_1
        del arg532_1
        del arg533_1
        # Source Nodes: [x_299, x_301, x_in_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf235 = extern_kernels.convolution(buf234, arg321_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (8, 1088, 14, 14), (213248, 196, 14, 1))
        del arg321_1
        del buf234
        buf236 = buf122; del buf122  # reuse
        # Source Nodes: [cat_77], Original ATen: [aten.cat]
        triton_poi_fused_cat_50.run(buf208, buf215, buf221, buf228, buf235, buf236, 2207744, grid=grid(2207744), stream=stream0)
        del buf208
        buf238 = empty((8, 2432, 14, 14), device='cuda', dtype=torch.float32)
        buf240 = empty((8, 2432, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_76, x_302, x_304, x_305, x_307, x_in_129, x_s_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_51.run(buf207, buf215, buf221, buf228, buf235, buf236, arg534_1, arg535_1, arg200_1, arg201_1, arg536_1, arg537_1, arg202_1, arg203_1, buf238, buf240, 3813376, grid=grid(3813376), stream=stream0)
        del arg200_1
        del arg201_1
        del arg202_1
        del arg203_1
        del arg534_1
        del arg535_1
        del arg536_1
        del arg537_1
        del buf207
        del buf209
        del buf215
        del buf221
        del buf228
        del buf235
        del buf236
        # Source Nodes: [x_302, x_304, x_s_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf239 = extern_kernels.convolution(buf238, arg322_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf239, (8, 2304, 7, 7), (112896, 49, 7, 1))
        del arg322_1
        del buf238
        # Source Nodes: [x_305, x_307, x_in_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf241 = extern_kernels.convolution(buf240, arg323_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf241, (8, 1600, 14, 14), (313600, 196, 14, 1))
        del arg323_1
        del buf240
        buf242 = buf241; del buf241  # reuse
        # Source Nodes: [x_308, x_310, x_in_130], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_52.run(buf242, arg538_1, arg539_1, arg204_1, arg205_1, 2508800, grid=grid(2508800), stream=stream0)
        del arg204_1
        del arg205_1
        del arg538_1
        del arg539_1
        # Source Nodes: [x_308, x_310, x_in_130], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf243 = extern_kernels.convolution(buf242, arg324_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf243, (8, 1600, 7, 7), (78400, 49, 7, 1))
        del arg324_1
        del buf242
        buf244 = buf243; del buf243  # reuse
        # Source Nodes: [x_311, x_313, x_in_131], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_53.run(buf244, arg540_1, arg541_1, arg206_1, arg207_1, 627200, grid=grid(627200), stream=stream0)
        del arg206_1
        del arg207_1
        del arg540_1
        del arg541_1
        # Source Nodes: [x_311, x_313, x_in_131], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf245 = extern_kernels.convolution(buf244, arg325_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf245, (8, 2176, 7, 7), (106624, 49, 7, 1))
        del arg325_1
        del buf244
        buf246 = empty((8, 2432, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_74, x_314, x_316, x_in_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_54.run(buf239, buf245, arg542_1, arg543_1, arg208_1, arg209_1, buf246, 953344, grid=grid(953344), stream=stream0)
        del arg208_1
        del arg209_1
        del arg542_1
        del arg543_1
        # Source Nodes: [cat_74, x_314, x_316, x_in_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        buf247 = extern_kernels.convolution(buf246, arg326_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf247, (8, 1600, 7, 7), (78400, 49, 7, 1))
        del arg326_1
        del buf246
        buf248 = buf247; del buf247  # reuse
        # Source Nodes: [x_317, x_319, x_in_134], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_53.run(buf248, arg544_1, arg545_1, arg210_1, arg211_1, 627200, grid=grid(627200), stream=stream0)
        del arg210_1
        del arg211_1
        del arg544_1
        del arg545_1
        # Source Nodes: [x_317, x_319, x_in_134], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf249 = extern_kernels.convolution(buf248, arg327_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf249, (8, 1600, 7, 7), (78400, 49, 7, 1))
        del arg327_1
        del buf248
        buf250 = buf249; del buf249  # reuse
        # Source Nodes: [x_320, x_322, x_in_135], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_53.run(buf250, arg546_1, arg547_1, arg212_1, arg213_1, 627200, grid=grid(627200), stream=stream0)
        del arg212_1
        del arg213_1
        del arg546_1
        del arg547_1
        # Source Nodes: [x_320, x_322, x_in_135], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf251 = extern_kernels.convolution(buf250, arg328_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf251, (8, 2176, 7, 7), (106624, 49, 7, 1))
        del arg328_1
        del buf250
        buf252 = empty((8, 2560, 7, 7), device='cuda', dtype=torch.float32)
        buf253 = buf252; del buf252  # reuse
        # Source Nodes: [cat_72, x_323, x_325, x_in_137], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_convolution_relu_55.run(buf253, buf239, buf245, buf251, arg548_1, arg549_1, arg214_1, arg215_1, 1003520, grid=grid(1003520), stream=stream0)
        del arg214_1
        del arg215_1
        del arg548_1
        del arg549_1
        # Source Nodes: [x_323, x_325, x_in_137], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf254 = extern_kernels.convolution(buf253, arg329_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf254, (8, 1600, 7, 7), (78400, 49, 7, 1))
        del arg329_1
        del buf253
        buf255 = buf254; del buf254  # reuse
        # Source Nodes: [x_326, x_328, x_in_138], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_53.run(buf255, arg550_1, arg551_1, arg216_1, arg217_1, 627200, grid=grid(627200), stream=stream0)
        del arg216_1
        del arg217_1
        del arg550_1
        del arg551_1
        # Source Nodes: [x_326, x_328, x_in_138], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf256 = extern_kernels.convolution(buf255, arg330_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=50, bias=None)
        assert_size_stride(buf256, (8, 1600, 7, 7), (78400, 49, 7, 1))
        del arg330_1
        del buf255
        buf257 = buf256; del buf256  # reuse
        # Source Nodes: [x_329, x_331, x_in_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_53.run(buf257, arg552_1, arg553_1, arg218_1, arg219_1, 627200, grid=grid(627200), stream=stream0)
        del arg218_1
        del arg219_1
        del arg552_1
        del arg553_1
        # Source Nodes: [x_329, x_331, x_in_139], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf258 = extern_kernels.convolution(buf257, arg331_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf258, (8, 2176, 7, 7), (106624, 49, 7, 1))
        del arg331_1
        del buf257
        buf260 = empty_strided((8, 2688, 1, 1), (2688, 1, 21504, 21504), device='cuda', dtype=torch.float32)
        buf261 = reinterpret_tensor(buf260, (8, 2688, 1, 1), (2688, 1, 1, 1), 0); del buf260  # reuse
        # Source Nodes: [cat_70, x_333, x_336, x_337, x_340], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.convolution, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_cat_convolution_mean_relu_56.run(buf261, buf239, buf245, buf251, buf258, arg554_1, arg555_1, arg220_1, arg221_1, 21504, 49, grid=grid(21504), stream=stream0)
        del arg220_1
        del arg221_1
        del arg554_1
        del arg555_1
        del buf239
        del buf245
        del buf251
        del buf258
        # Source Nodes: [x_333, x_336, x_337, x_340], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu]
        buf262 = extern_kernels.convolution(buf261, arg332_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf262, (8, 1000, 1, 1), (1000, 1, 1, 1))
        del arg332_1
        del buf261
        buf263 = reinterpret_tensor(buf262, (8, 1000), (1000, 1), 0); del buf262  # reuse
        # Source Nodes: [x_333, x_336, x_337, x_340, x_341], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.mean, aten.relu, aten.view]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_mean_relu_view_57.run(buf263, arg333_1, 8000, grid=grid(8000), stream=stream0)
        del arg333_1
        return (buf263, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((316, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((316, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((356, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((356, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((1216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((1216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((1472, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((1472, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((1664, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((1664, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((1728, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1728, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1856, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((1856, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((1984, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((1984, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((2112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((2112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((2176, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((2176, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((2368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((2368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((2688, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((2688, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((128, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((296, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((200, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((200, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((276, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((200, 316, 1, 1), (316, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((200, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((276, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((200, 336, 1, 1), (336, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((200, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((276, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((200, 356, 1, 1), (356, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((200, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((276, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((640, 376, 1, 1), (376, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((400, 376, 1, 1), (376, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((400, 704, 1, 1), (704, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((400, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((400, 832, 1, 1), (832, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((400, 896, 1, 1), (896, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((400, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((400, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((400, 1088, 1, 1), (1088, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((400, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((576, 400, 1, 1), (400, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1152, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((800, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((800, 1216, 1, 1), (1216, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((800, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((800, 1344, 1, 1), (1344, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((800, 1408, 1, 1), (1408, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((800, 1472, 1, 1), (1472, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((800, 1536, 1, 1), (1536, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((800, 1600, 1, 1), (1600, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((800, 1664, 1, 1), (1664, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((800, 1728, 1, 1), (1728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((800, 1792, 1, 1), (1792, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((800, 1856, 1, 1), (1856, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((800, 1920, 1, 1), (1920, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((800, 1984, 1, 1), (1984, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((800, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((800, 2112, 1, 1), (2112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((800, 2176, 1, 1), (2176, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((800, 2240, 1, 1), (2240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((800, 2304, 1, 1), (2304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((800, 2368, 1, 1), (2368, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((800, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((1088, 800, 1, 1), (800, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((2304, 2432, 1, 1), (2432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((1600, 2432, 1, 1), (2432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((1600, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((2176, 1600, 1, 1), (1600, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((1600, 2432, 1, 1), (2432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((1600, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((2176, 1600, 1, 1), (1600, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((1600, 2560, 1, 1), (2560, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((1600, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((2176, 1600, 1, 1), (1600, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((1000, 2688, 1, 1), (2688, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((316, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((316, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((336, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((356, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((356, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((376, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((704, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((832, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((896, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((1088, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((400, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((1216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((1216, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((1344, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((1408, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((1472, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((1472, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((1536, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((1664, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((1664, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((1728, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((1728, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg473_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((1792, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg476_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg479_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((1856, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((1856, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg482_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg485_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((1920, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg488_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg491_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((1984, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((1984, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg494_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg497_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg500_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg503_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((2112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((2112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg506_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg509_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((2176, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((2176, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg512_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg515_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((2240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg518_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg521_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg522_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg523_1 = rand_strided((2304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg524_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg525_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg526_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg527_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg528_1 = rand_strided((2368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg529_1 = rand_strided((2368, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg530_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg531_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg532_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg533_1 = rand_strided((800, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg534_1 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg535_1 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg536_1 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg537_1 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg538_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg539_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg540_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg541_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg542_1 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg543_1 = rand_strided((2432, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg544_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg545_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg546_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg547_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg548_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg549_1 = rand_strided((2560, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg550_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg551_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg552_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg553_1 = rand_strided((1600, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg554_1 = rand_strided((2688, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg555_1 = rand_strided((2688, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg556_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('dpn107', benchmark_compiled_module)
