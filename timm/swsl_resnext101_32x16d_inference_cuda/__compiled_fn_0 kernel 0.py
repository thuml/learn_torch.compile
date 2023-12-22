
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


# kernel path: /tmp/torchinductor_youkaichao/63/c63tcifdfouzgxewkdgeawixxyvaylaegdnohhvu7jymv2qzuy4c.py
# Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_1 => add_1, mul_1, mul_2, sub
# x_2 => relu
triton_poi_fused__native_batch_norm_legit_no_training_relu_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_0', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
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


# kernel path: /tmp/torchinductor_youkaichao/sv/csvbla5mz5jgtaoekyjsm4ld7uiquhja3wn5hcce4sp6hhuvfsh5.py
# Source Nodes: [shortcut, x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.max_pool2d_with_indices, aten.relu]
# shortcut => max_pool2d_with_indices
# x_1 => add_1, mul_1, mul_2, sub
# x_2 => relu
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x4), tmp69, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/jo/cjoxxhogqac2digoaha276vixaa5tctcglngn7pey37veggu7qfz.py
# Source Nodes: [x_5, x_6, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_5 => add_3, mul_4, mul_5, sub_1
# x_6 => relu_1
# x_7 => convolution_2
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 512
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


# kernel path: /tmp/torchinductor_youkaichao/3g/c3gubb7b4awp2cj3tmt2ssicznxtqriuqnrj2bagklczxruwnvuk.py
# Source Nodes: [shortcut_1, shortcut_2, x_13, x_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_1 => add_9, mul_13, mul_14, sub_4
# shortcut_2 => relu_3
# x_13 => add_7, mul_10, mul_11, sub_3
# x_14 => add_10
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
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
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
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
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(in_out_ptr0 + (x3), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hg/chgzlrw5xgosl2awvm4duezxhlr63w4pnk4hyanl7oczyldn43vx.py
# Source Nodes: [shortcut_3, x_25, x_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_3 => relu_6
# x_25 => add_16, mul_22, mul_23, sub_7
# x_26 => add_17
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
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
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/4t/c4tdasap54zr4dy5oc3sdgjlgwjwnidzp2r6upsvvz4lptwt6f32.py
# Source Nodes: [x_42, x_43, x_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_42 => add_26, mul_34, mul_35, sub_11
# x_43 => relu_10
# x_44 => convolution_12
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 25690112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 1024
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


# kernel path: /tmp/torchinductor_youkaichao/oc/coczwxriatukbcgorkv3mlxrb4hs2y4oclhhgeujy4zdqryqdbuo.py
# Source Nodes: [x_45, x_47, x_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_45 => add_28, mul_37, mul_38, sub_12
# x_47 => relu_11
# x_49 => convolution_13
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 1024
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


# kernel path: /tmp/torchinductor_youkaichao/ax/caxpqm4ytyfffsgw4fwak6ido6p2r5q2d53li7u2ftdsasqwqwnj.py
# Source Nodes: [shortcut_5, shortcut_6, x_50, x_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_5 => add_32, mul_43, mul_44, sub_14
# shortcut_6 => relu_12
# x_50 => add_30, mul_40, mul_41, sub_13
# x_51 => add_33
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
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
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
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
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(in_out_ptr0 + (x3), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/yf/cyfdmn3g6rhap7pmogdqllxyabhqfjbthbvqgw73ej425ajdndmh.py
# Source Nodes: [shortcut_7, x_62, x_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_7 => relu_15
# x_62 => add_39, mul_52, mul_53, sub_17
# x_63 => add_40
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
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
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/be/cbeojiiaibq4hgoebicj4nhhaj2txqzoguhvni7zx2rv4wyjzlax.py
# Source Nodes: [x_91, x_92, x_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_91 => add_56, mul_73, mul_74, sub_24
# x_92 => relu_22
# x_93 => convolution_25
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 2048
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


# kernel path: /tmp/torchinductor_youkaichao/yq/cyqbtih32h3h7kxeq4qonkj6ogblt6ta3sjerqmi3kj6frujuoj2.py
# Source Nodes: [x_94, x_96, x_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_94 => add_58, mul_76, mul_77, sub_25
# x_96 => relu_23
# x_98 => convolution_26
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 2048
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


# kernel path: /tmp/torchinductor_youkaichao/3n/c3ngyfrwn6xtsi6twqv4xwneqln5od6t3a7vtcnxciyfq5sdktzc.py
# Source Nodes: [shortcut_10, shortcut_11, x_100, x_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_10 => add_62, mul_82, mul_83, sub_27
# shortcut_11 => relu_24
# x_100 => add_63
# x_99 => add_60, mul_79, mul_80, sub_26
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
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
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
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
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(in_out_ptr0 + (x3), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ns/cnscqij5yg4ih34yt47wbwj6uov7mhfiaxsp5z4afjk57mv27m3l.py
# Source Nodes: [shortcut_12, x_111, x_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_12 => relu_27
# x_111 => add_69, mul_91, mul_92, sub_30
# x_112 => add_70
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
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
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qv/cqvvcz5j7fzunbndx6y6vqkzh6oc7fkldolzopq4yhd3mpug3jre.py
# Source Nodes: [shortcut_19, x_195, x_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_19 => relu_48
# x_195 => add_118, mul_154, mul_155, sub_51
# x_196 => add_119
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
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
    tmp15 = tl.load(in_ptr4 + (x3), None)
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xf/cxfod5wx3d4dwd5osx3ypcpf5g27gteecp6hobxd7qw7tznbgv7b.py
# Source Nodes: [x_368, x_369, x_370], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_368 => add_219, mul_283, mul_284, sub_94
# x_369 => relu_91
# x_370 => convolution_95
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 4096
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


# kernel path: /tmp/torchinductor_youkaichao/ku/cku27ubs2nfemvvujp3dvfciiq22qug2pmhagwmgjfeehk4kscje.py
# Source Nodes: [x_371, x_373, x_375], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_371 => add_221, mul_286, mul_287, sub_95
# x_373 => relu_92
# x_375 => convolution_96
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 4096
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


# kernel path: /tmp/torchinductor_youkaichao/wa/cwaj24mfbn6v77t4l7mpu3gl53vdtq7rmwkagsbalasbgjeqywir.py
# Source Nodes: [shortcut_34, shortcut_35, x_376, x_377], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_34 => add_225, mul_292, mul_293, sub_97
# shortcut_35 => relu_93
# x_376 => add_223, mul_289, mul_290, sub_96
# x_377 => add_226
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 2048
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x3), None)
    tmp16 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
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
    tmp17 = tmp15 - tmp16
    tmp19 = tmp18 + tmp4
    tmp20 = tl.sqrt(tmp19)
    tmp21 = 1 / tmp20
    tmp22 = tmp21 * tmp8
    tmp23 = tmp17 * tmp22
    tmp25 = tmp23 * tmp24
    tmp27 = tmp25 + tmp26
    tmp28 = tmp14 + tmp27
    tmp29 = triton_helpers.maximum(0, tmp28)
    tl.store(in_out_ptr0 + (x3), tmp29, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/cl/ccl4oa2rjdixigze7q7quswnzqpoiibqi6rppog7iizukfcduooo.py
# Source Nodes: [shortcut_36, x_388, x_389], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# shortcut_36 => relu_96
# x_388 => add_232, mul_301, mul_302, sub_100
# x_389 => add_233
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 2048
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x3), None)
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/br/cbr7vsmobefhp7yfs36hrmf6lohxdukqe5hyau3qntppbnutal3u.py
# Source Nodes: [x_400, x_401, x_404, x_405], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.relu]
# x_400 => add_239, mul_310, mul_311, sub_103
# x_401 => add_240
# x_404 => relu_99
# x_405 => mean
triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_18', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (r2 + (49*x3)), rmask, other=0.0)
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
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp20 = tl.where(rmask, tmp18, 0)
    tmp21 = tl.sum(tmp20, 1)[:, None]
    tmp22 = 49.0
    tmp23 = tmp21 / tmp22
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp23, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (512, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg4_1, (512, ), (1, ))
    assert_size_stride(arg5_1, (512, ), (1, ))
    assert_size_stride(arg6_1, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg7_1, (512, ), (1, ))
    assert_size_stride(arg8_1, (512, ), (1, ))
    assert_size_stride(arg9_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg10_1, (256, ), (1, ))
    assert_size_stride(arg11_1, (256, ), (1, ))
    assert_size_stride(arg12_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg13_1, (256, ), (1, ))
    assert_size_stride(arg14_1, (256, ), (1, ))
    assert_size_stride(arg15_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg16_1, (512, ), (1, ))
    assert_size_stride(arg17_1, (512, ), (1, ))
    assert_size_stride(arg18_1, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg19_1, (512, ), (1, ))
    assert_size_stride(arg20_1, (512, ), (1, ))
    assert_size_stride(arg21_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (256, ), (1, ))
    assert_size_stride(arg24_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg25_1, (512, ), (1, ))
    assert_size_stride(arg26_1, (512, ), (1, ))
    assert_size_stride(arg27_1, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg28_1, (512, ), (1, ))
    assert_size_stride(arg29_1, (512, ), (1, ))
    assert_size_stride(arg30_1, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg31_1, (256, ), (1, ))
    assert_size_stride(arg32_1, (256, ), (1, ))
    assert_size_stride(arg33_1, (1024, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg34_1, (1024, ), (1, ))
    assert_size_stride(arg35_1, (1024, ), (1, ))
    assert_size_stride(arg36_1, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg37_1, (1024, ), (1, ))
    assert_size_stride(arg38_1, (1024, ), (1, ))
    assert_size_stride(arg39_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg40_1, (512, ), (1, ))
    assert_size_stride(arg41_1, (512, ), (1, ))
    assert_size_stride(arg42_1, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg43_1, (512, ), (1, ))
    assert_size_stride(arg44_1, (512, ), (1, ))
    assert_size_stride(arg45_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg46_1, (1024, ), (1, ))
    assert_size_stride(arg47_1, (1024, ), (1, ))
    assert_size_stride(arg48_1, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg49_1, (1024, ), (1, ))
    assert_size_stride(arg50_1, (1024, ), (1, ))
    assert_size_stride(arg51_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg52_1, (512, ), (1, ))
    assert_size_stride(arg53_1, (512, ), (1, ))
    assert_size_stride(arg54_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg55_1, (1024, ), (1, ))
    assert_size_stride(arg56_1, (1024, ), (1, ))
    assert_size_stride(arg57_1, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg58_1, (1024, ), (1, ))
    assert_size_stride(arg59_1, (1024, ), (1, ))
    assert_size_stride(arg60_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg61_1, (512, ), (1, ))
    assert_size_stride(arg62_1, (512, ), (1, ))
    assert_size_stride(arg63_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg64_1, (1024, ), (1, ))
    assert_size_stride(arg65_1, (1024, ), (1, ))
    assert_size_stride(arg66_1, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg67_1, (1024, ), (1, ))
    assert_size_stride(arg68_1, (1024, ), (1, ))
    assert_size_stride(arg69_1, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg70_1, (512, ), (1, ))
    assert_size_stride(arg71_1, (512, ), (1, ))
    assert_size_stride(arg72_1, (2048, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg73_1, (2048, ), (1, ))
    assert_size_stride(arg74_1, (2048, ), (1, ))
    assert_size_stride(arg75_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg76_1, (2048, ), (1, ))
    assert_size_stride(arg77_1, (2048, ), (1, ))
    assert_size_stride(arg78_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg79_1, (1024, ), (1, ))
    assert_size_stride(arg80_1, (1024, ), (1, ))
    assert_size_stride(arg81_1, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg82_1, (1024, ), (1, ))
    assert_size_stride(arg83_1, (1024, ), (1, ))
    assert_size_stride(arg84_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg85_1, (2048, ), (1, ))
    assert_size_stride(arg86_1, (2048, ), (1, ))
    assert_size_stride(arg87_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg88_1, (2048, ), (1, ))
    assert_size_stride(arg89_1, (2048, ), (1, ))
    assert_size_stride(arg90_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg91_1, (1024, ), (1, ))
    assert_size_stride(arg92_1, (1024, ), (1, ))
    assert_size_stride(arg93_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg94_1, (2048, ), (1, ))
    assert_size_stride(arg95_1, (2048, ), (1, ))
    assert_size_stride(arg96_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg97_1, (2048, ), (1, ))
    assert_size_stride(arg98_1, (2048, ), (1, ))
    assert_size_stride(arg99_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg100_1, (1024, ), (1, ))
    assert_size_stride(arg101_1, (1024, ), (1, ))
    assert_size_stride(arg102_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg103_1, (2048, ), (1, ))
    assert_size_stride(arg104_1, (2048, ), (1, ))
    assert_size_stride(arg105_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg106_1, (2048, ), (1, ))
    assert_size_stride(arg107_1, (2048, ), (1, ))
    assert_size_stride(arg108_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg109_1, (1024, ), (1, ))
    assert_size_stride(arg110_1, (1024, ), (1, ))
    assert_size_stride(arg111_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg112_1, (2048, ), (1, ))
    assert_size_stride(arg113_1, (2048, ), (1, ))
    assert_size_stride(arg114_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg115_1, (2048, ), (1, ))
    assert_size_stride(arg116_1, (2048, ), (1, ))
    assert_size_stride(arg117_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg118_1, (1024, ), (1, ))
    assert_size_stride(arg119_1, (1024, ), (1, ))
    assert_size_stride(arg120_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg121_1, (2048, ), (1, ))
    assert_size_stride(arg122_1, (2048, ), (1, ))
    assert_size_stride(arg123_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg124_1, (2048, ), (1, ))
    assert_size_stride(arg125_1, (2048, ), (1, ))
    assert_size_stride(arg126_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg127_1, (1024, ), (1, ))
    assert_size_stride(arg128_1, (1024, ), (1, ))
    assert_size_stride(arg129_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg130_1, (2048, ), (1, ))
    assert_size_stride(arg131_1, (2048, ), (1, ))
    assert_size_stride(arg132_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg133_1, (2048, ), (1, ))
    assert_size_stride(arg134_1, (2048, ), (1, ))
    assert_size_stride(arg135_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg136_1, (1024, ), (1, ))
    assert_size_stride(arg137_1, (1024, ), (1, ))
    assert_size_stride(arg138_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg139_1, (2048, ), (1, ))
    assert_size_stride(arg140_1, (2048, ), (1, ))
    assert_size_stride(arg141_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg142_1, (2048, ), (1, ))
    assert_size_stride(arg143_1, (2048, ), (1, ))
    assert_size_stride(arg144_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg145_1, (1024, ), (1, ))
    assert_size_stride(arg146_1, (1024, ), (1, ))
    assert_size_stride(arg147_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg148_1, (2048, ), (1, ))
    assert_size_stride(arg149_1, (2048, ), (1, ))
    assert_size_stride(arg150_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg151_1, (2048, ), (1, ))
    assert_size_stride(arg152_1, (2048, ), (1, ))
    assert_size_stride(arg153_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg154_1, (1024, ), (1, ))
    assert_size_stride(arg155_1, (1024, ), (1, ))
    assert_size_stride(arg156_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg157_1, (2048, ), (1, ))
    assert_size_stride(arg158_1, (2048, ), (1, ))
    assert_size_stride(arg159_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg160_1, (2048, ), (1, ))
    assert_size_stride(arg161_1, (2048, ), (1, ))
    assert_size_stride(arg162_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg163_1, (1024, ), (1, ))
    assert_size_stride(arg164_1, (1024, ), (1, ))
    assert_size_stride(arg165_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg166_1, (2048, ), (1, ))
    assert_size_stride(arg167_1, (2048, ), (1, ))
    assert_size_stride(arg168_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg169_1, (2048, ), (1, ))
    assert_size_stride(arg170_1, (2048, ), (1, ))
    assert_size_stride(arg171_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg172_1, (1024, ), (1, ))
    assert_size_stride(arg173_1, (1024, ), (1, ))
    assert_size_stride(arg174_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg175_1, (2048, ), (1, ))
    assert_size_stride(arg176_1, (2048, ), (1, ))
    assert_size_stride(arg177_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg178_1, (2048, ), (1, ))
    assert_size_stride(arg179_1, (2048, ), (1, ))
    assert_size_stride(arg180_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg181_1, (1024, ), (1, ))
    assert_size_stride(arg182_1, (1024, ), (1, ))
    assert_size_stride(arg183_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg184_1, (2048, ), (1, ))
    assert_size_stride(arg185_1, (2048, ), (1, ))
    assert_size_stride(arg186_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg187_1, (2048, ), (1, ))
    assert_size_stride(arg188_1, (2048, ), (1, ))
    assert_size_stride(arg189_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg190_1, (1024, ), (1, ))
    assert_size_stride(arg191_1, (1024, ), (1, ))
    assert_size_stride(arg192_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg193_1, (2048, ), (1, ))
    assert_size_stride(arg194_1, (2048, ), (1, ))
    assert_size_stride(arg195_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg196_1, (2048, ), (1, ))
    assert_size_stride(arg197_1, (2048, ), (1, ))
    assert_size_stride(arg198_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg199_1, (1024, ), (1, ))
    assert_size_stride(arg200_1, (1024, ), (1, ))
    assert_size_stride(arg201_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg202_1, (2048, ), (1, ))
    assert_size_stride(arg203_1, (2048, ), (1, ))
    assert_size_stride(arg204_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg205_1, (2048, ), (1, ))
    assert_size_stride(arg206_1, (2048, ), (1, ))
    assert_size_stride(arg207_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg208_1, (1024, ), (1, ))
    assert_size_stride(arg209_1, (1024, ), (1, ))
    assert_size_stride(arg210_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg211_1, (2048, ), (1, ))
    assert_size_stride(arg212_1, (2048, ), (1, ))
    assert_size_stride(arg213_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg214_1, (2048, ), (1, ))
    assert_size_stride(arg215_1, (2048, ), (1, ))
    assert_size_stride(arg216_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg217_1, (1024, ), (1, ))
    assert_size_stride(arg218_1, (1024, ), (1, ))
    assert_size_stride(arg219_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg220_1, (2048, ), (1, ))
    assert_size_stride(arg221_1, (2048, ), (1, ))
    assert_size_stride(arg222_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg223_1, (2048, ), (1, ))
    assert_size_stride(arg224_1, (2048, ), (1, ))
    assert_size_stride(arg225_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg226_1, (1024, ), (1, ))
    assert_size_stride(arg227_1, (1024, ), (1, ))
    assert_size_stride(arg228_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg229_1, (2048, ), (1, ))
    assert_size_stride(arg230_1, (2048, ), (1, ))
    assert_size_stride(arg231_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg232_1, (2048, ), (1, ))
    assert_size_stride(arg233_1, (2048, ), (1, ))
    assert_size_stride(arg234_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg235_1, (1024, ), (1, ))
    assert_size_stride(arg236_1, (1024, ), (1, ))
    assert_size_stride(arg237_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg238_1, (2048, ), (1, ))
    assert_size_stride(arg239_1, (2048, ), (1, ))
    assert_size_stride(arg240_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg241_1, (2048, ), (1, ))
    assert_size_stride(arg242_1, (2048, ), (1, ))
    assert_size_stride(arg243_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg244_1, (1024, ), (1, ))
    assert_size_stride(arg245_1, (1024, ), (1, ))
    assert_size_stride(arg246_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg247_1, (2048, ), (1, ))
    assert_size_stride(arg248_1, (2048, ), (1, ))
    assert_size_stride(arg249_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg250_1, (2048, ), (1, ))
    assert_size_stride(arg251_1, (2048, ), (1, ))
    assert_size_stride(arg252_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg253_1, (1024, ), (1, ))
    assert_size_stride(arg254_1, (1024, ), (1, ))
    assert_size_stride(arg255_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg256_1, (2048, ), (1, ))
    assert_size_stride(arg257_1, (2048, ), (1, ))
    assert_size_stride(arg258_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg259_1, (2048, ), (1, ))
    assert_size_stride(arg260_1, (2048, ), (1, ))
    assert_size_stride(arg261_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg262_1, (1024, ), (1, ))
    assert_size_stride(arg263_1, (1024, ), (1, ))
    assert_size_stride(arg264_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg265_1, (2048, ), (1, ))
    assert_size_stride(arg266_1, (2048, ), (1, ))
    assert_size_stride(arg267_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg268_1, (2048, ), (1, ))
    assert_size_stride(arg269_1, (2048, ), (1, ))
    assert_size_stride(arg270_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg271_1, (1024, ), (1, ))
    assert_size_stride(arg272_1, (1024, ), (1, ))
    assert_size_stride(arg273_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg274_1, (2048, ), (1, ))
    assert_size_stride(arg275_1, (2048, ), (1, ))
    assert_size_stride(arg276_1, (2048, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg277_1, (2048, ), (1, ))
    assert_size_stride(arg278_1, (2048, ), (1, ))
    assert_size_stride(arg279_1, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg280_1, (1024, ), (1, ))
    assert_size_stride(arg281_1, (1024, ), (1, ))
    assert_size_stride(arg282_1, (4096, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg283_1, (4096, ), (1, ))
    assert_size_stride(arg284_1, (4096, ), (1, ))
    assert_size_stride(arg285_1, (4096, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg286_1, (4096, ), (1, ))
    assert_size_stride(arg287_1, (4096, ), (1, ))
    assert_size_stride(arg288_1, (2048, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(arg289_1, (2048, ), (1, ))
    assert_size_stride(arg290_1, (2048, ), (1, ))
    assert_size_stride(arg291_1, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(arg292_1, (2048, ), (1, ))
    assert_size_stride(arg293_1, (2048, ), (1, ))
    assert_size_stride(arg294_1, (4096, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg295_1, (4096, ), (1, ))
    assert_size_stride(arg296_1, (4096, ), (1, ))
    assert_size_stride(arg297_1, (4096, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg298_1, (4096, ), (1, ))
    assert_size_stride(arg299_1, (4096, ), (1, ))
    assert_size_stride(arg300_1, (2048, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(arg301_1, (2048, ), (1, ))
    assert_size_stride(arg302_1, (2048, ), (1, ))
    assert_size_stride(arg303_1, (4096, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg304_1, (4096, ), (1, ))
    assert_size_stride(arg305_1, (4096, ), (1, ))
    assert_size_stride(arg306_1, (4096, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg307_1, (4096, ), (1, ))
    assert_size_stride(arg308_1, (4096, ), (1, ))
    assert_size_stride(arg309_1, (2048, 4096, 1, 1), (4096, 1, 1, 1))
    assert_size_stride(arg310_1, (2048, ), (1, ))
    assert_size_stride(arg311_1, (2048, ), (1, ))
    assert_size_stride(arg312_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg313_1, (1000, ), (1, ))
    assert_size_stride(arg314_1, (64, ), (1, ))
    assert_size_stride(arg315_1, (64, ), (1, ))
    assert_size_stride(arg316_1, (), ())
    assert_size_stride(arg317_1, (512, ), (1, ))
    assert_size_stride(arg318_1, (512, ), (1, ))
    assert_size_stride(arg319_1, (), ())
    assert_size_stride(arg320_1, (512, ), (1, ))
    assert_size_stride(arg321_1, (512, ), (1, ))
    assert_size_stride(arg322_1, (), ())
    assert_size_stride(arg323_1, (256, ), (1, ))
    assert_size_stride(arg324_1, (256, ), (1, ))
    assert_size_stride(arg325_1, (), ())
    assert_size_stride(arg326_1, (256, ), (1, ))
    assert_size_stride(arg327_1, (256, ), (1, ))
    assert_size_stride(arg328_1, (), ())
    assert_size_stride(arg329_1, (512, ), (1, ))
    assert_size_stride(arg330_1, (512, ), (1, ))
    assert_size_stride(arg331_1, (), ())
    assert_size_stride(arg332_1, (512, ), (1, ))
    assert_size_stride(arg333_1, (512, ), (1, ))
    assert_size_stride(arg334_1, (), ())
    assert_size_stride(arg335_1, (256, ), (1, ))
    assert_size_stride(arg336_1, (256, ), (1, ))
    assert_size_stride(arg337_1, (), ())
    assert_size_stride(arg338_1, (512, ), (1, ))
    assert_size_stride(arg339_1, (512, ), (1, ))
    assert_size_stride(arg340_1, (), ())
    assert_size_stride(arg341_1, (512, ), (1, ))
    assert_size_stride(arg342_1, (512, ), (1, ))
    assert_size_stride(arg343_1, (), ())
    assert_size_stride(arg344_1, (256, ), (1, ))
    assert_size_stride(arg345_1, (256, ), (1, ))
    assert_size_stride(arg346_1, (), ())
    assert_size_stride(arg347_1, (1024, ), (1, ))
    assert_size_stride(arg348_1, (1024, ), (1, ))
    assert_size_stride(arg349_1, (), ())
    assert_size_stride(arg350_1, (1024, ), (1, ))
    assert_size_stride(arg351_1, (1024, ), (1, ))
    assert_size_stride(arg352_1, (), ())
    assert_size_stride(arg353_1, (512, ), (1, ))
    assert_size_stride(arg354_1, (512, ), (1, ))
    assert_size_stride(arg355_1, (), ())
    assert_size_stride(arg356_1, (512, ), (1, ))
    assert_size_stride(arg357_1, (512, ), (1, ))
    assert_size_stride(arg358_1, (), ())
    assert_size_stride(arg359_1, (1024, ), (1, ))
    assert_size_stride(arg360_1, (1024, ), (1, ))
    assert_size_stride(arg361_1, (), ())
    assert_size_stride(arg362_1, (1024, ), (1, ))
    assert_size_stride(arg363_1, (1024, ), (1, ))
    assert_size_stride(arg364_1, (), ())
    assert_size_stride(arg365_1, (512, ), (1, ))
    assert_size_stride(arg366_1, (512, ), (1, ))
    assert_size_stride(arg367_1, (), ())
    assert_size_stride(arg368_1, (1024, ), (1, ))
    assert_size_stride(arg369_1, (1024, ), (1, ))
    assert_size_stride(arg370_1, (), ())
    assert_size_stride(arg371_1, (1024, ), (1, ))
    assert_size_stride(arg372_1, (1024, ), (1, ))
    assert_size_stride(arg373_1, (), ())
    assert_size_stride(arg374_1, (512, ), (1, ))
    assert_size_stride(arg375_1, (512, ), (1, ))
    assert_size_stride(arg376_1, (), ())
    assert_size_stride(arg377_1, (1024, ), (1, ))
    assert_size_stride(arg378_1, (1024, ), (1, ))
    assert_size_stride(arg379_1, (), ())
    assert_size_stride(arg380_1, (1024, ), (1, ))
    assert_size_stride(arg381_1, (1024, ), (1, ))
    assert_size_stride(arg382_1, (), ())
    assert_size_stride(arg383_1, (512, ), (1, ))
    assert_size_stride(arg384_1, (512, ), (1, ))
    assert_size_stride(arg385_1, (), ())
    assert_size_stride(arg386_1, (2048, ), (1, ))
    assert_size_stride(arg387_1, (2048, ), (1, ))
    assert_size_stride(arg388_1, (), ())
    assert_size_stride(arg389_1, (2048, ), (1, ))
    assert_size_stride(arg390_1, (2048, ), (1, ))
    assert_size_stride(arg391_1, (), ())
    assert_size_stride(arg392_1, (1024, ), (1, ))
    assert_size_stride(arg393_1, (1024, ), (1, ))
    assert_size_stride(arg394_1, (), ())
    assert_size_stride(arg395_1, (1024, ), (1, ))
    assert_size_stride(arg396_1, (1024, ), (1, ))
    assert_size_stride(arg397_1, (), ())
    assert_size_stride(arg398_1, (2048, ), (1, ))
    assert_size_stride(arg399_1, (2048, ), (1, ))
    assert_size_stride(arg400_1, (), ())
    assert_size_stride(arg401_1, (2048, ), (1, ))
    assert_size_stride(arg402_1, (2048, ), (1, ))
    assert_size_stride(arg403_1, (), ())
    assert_size_stride(arg404_1, (1024, ), (1, ))
    assert_size_stride(arg405_1, (1024, ), (1, ))
    assert_size_stride(arg406_1, (), ())
    assert_size_stride(arg407_1, (2048, ), (1, ))
    assert_size_stride(arg408_1, (2048, ), (1, ))
    assert_size_stride(arg409_1, (), ())
    assert_size_stride(arg410_1, (2048, ), (1, ))
    assert_size_stride(arg411_1, (2048, ), (1, ))
    assert_size_stride(arg412_1, (), ())
    assert_size_stride(arg413_1, (1024, ), (1, ))
    assert_size_stride(arg414_1, (1024, ), (1, ))
    assert_size_stride(arg415_1, (), ())
    assert_size_stride(arg416_1, (2048, ), (1, ))
    assert_size_stride(arg417_1, (2048, ), (1, ))
    assert_size_stride(arg418_1, (), ())
    assert_size_stride(arg419_1, (2048, ), (1, ))
    assert_size_stride(arg420_1, (2048, ), (1, ))
    assert_size_stride(arg421_1, (), ())
    assert_size_stride(arg422_1, (1024, ), (1, ))
    assert_size_stride(arg423_1, (1024, ), (1, ))
    assert_size_stride(arg424_1, (), ())
    assert_size_stride(arg425_1, (2048, ), (1, ))
    assert_size_stride(arg426_1, (2048, ), (1, ))
    assert_size_stride(arg427_1, (), ())
    assert_size_stride(arg428_1, (2048, ), (1, ))
    assert_size_stride(arg429_1, (2048, ), (1, ))
    assert_size_stride(arg430_1, (), ())
    assert_size_stride(arg431_1, (1024, ), (1, ))
    assert_size_stride(arg432_1, (1024, ), (1, ))
    assert_size_stride(arg433_1, (), ())
    assert_size_stride(arg434_1, (2048, ), (1, ))
    assert_size_stride(arg435_1, (2048, ), (1, ))
    assert_size_stride(arg436_1, (), ())
    assert_size_stride(arg437_1, (2048, ), (1, ))
    assert_size_stride(arg438_1, (2048, ), (1, ))
    assert_size_stride(arg439_1, (), ())
    assert_size_stride(arg440_1, (1024, ), (1, ))
    assert_size_stride(arg441_1, (1024, ), (1, ))
    assert_size_stride(arg442_1, (), ())
    assert_size_stride(arg443_1, (2048, ), (1, ))
    assert_size_stride(arg444_1, (2048, ), (1, ))
    assert_size_stride(arg445_1, (), ())
    assert_size_stride(arg446_1, (2048, ), (1, ))
    assert_size_stride(arg447_1, (2048, ), (1, ))
    assert_size_stride(arg448_1, (), ())
    assert_size_stride(arg449_1, (1024, ), (1, ))
    assert_size_stride(arg450_1, (1024, ), (1, ))
    assert_size_stride(arg451_1, (), ())
    assert_size_stride(arg452_1, (2048, ), (1, ))
    assert_size_stride(arg453_1, (2048, ), (1, ))
    assert_size_stride(arg454_1, (), ())
    assert_size_stride(arg455_1, (2048, ), (1, ))
    assert_size_stride(arg456_1, (2048, ), (1, ))
    assert_size_stride(arg457_1, (), ())
    assert_size_stride(arg458_1, (1024, ), (1, ))
    assert_size_stride(arg459_1, (1024, ), (1, ))
    assert_size_stride(arg460_1, (), ())
    assert_size_stride(arg461_1, (2048, ), (1, ))
    assert_size_stride(arg462_1, (2048, ), (1, ))
    assert_size_stride(arg463_1, (), ())
    assert_size_stride(arg464_1, (2048, ), (1, ))
    assert_size_stride(arg465_1, (2048, ), (1, ))
    assert_size_stride(arg466_1, (), ())
    assert_size_stride(arg467_1, (1024, ), (1, ))
    assert_size_stride(arg468_1, (1024, ), (1, ))
    assert_size_stride(arg469_1, (), ())
    assert_size_stride(arg470_1, (2048, ), (1, ))
    assert_size_stride(arg471_1, (2048, ), (1, ))
    assert_size_stride(arg472_1, (), ())
    assert_size_stride(arg473_1, (2048, ), (1, ))
    assert_size_stride(arg474_1, (2048, ), (1, ))
    assert_size_stride(arg475_1, (), ())
    assert_size_stride(arg476_1, (1024, ), (1, ))
    assert_size_stride(arg477_1, (1024, ), (1, ))
    assert_size_stride(arg478_1, (), ())
    assert_size_stride(arg479_1, (2048, ), (1, ))
    assert_size_stride(arg480_1, (2048, ), (1, ))
    assert_size_stride(arg481_1, (), ())
    assert_size_stride(arg482_1, (2048, ), (1, ))
    assert_size_stride(arg483_1, (2048, ), (1, ))
    assert_size_stride(arg484_1, (), ())
    assert_size_stride(arg485_1, (1024, ), (1, ))
    assert_size_stride(arg486_1, (1024, ), (1, ))
    assert_size_stride(arg487_1, (), ())
    assert_size_stride(arg488_1, (2048, ), (1, ))
    assert_size_stride(arg489_1, (2048, ), (1, ))
    assert_size_stride(arg490_1, (), ())
    assert_size_stride(arg491_1, (2048, ), (1, ))
    assert_size_stride(arg492_1, (2048, ), (1, ))
    assert_size_stride(arg493_1, (), ())
    assert_size_stride(arg494_1, (1024, ), (1, ))
    assert_size_stride(arg495_1, (1024, ), (1, ))
    assert_size_stride(arg496_1, (), ())
    assert_size_stride(arg497_1, (2048, ), (1, ))
    assert_size_stride(arg498_1, (2048, ), (1, ))
    assert_size_stride(arg499_1, (), ())
    assert_size_stride(arg500_1, (2048, ), (1, ))
    assert_size_stride(arg501_1, (2048, ), (1, ))
    assert_size_stride(arg502_1, (), ())
    assert_size_stride(arg503_1, (1024, ), (1, ))
    assert_size_stride(arg504_1, (1024, ), (1, ))
    assert_size_stride(arg505_1, (), ())
    assert_size_stride(arg506_1, (2048, ), (1, ))
    assert_size_stride(arg507_1, (2048, ), (1, ))
    assert_size_stride(arg508_1, (), ())
    assert_size_stride(arg509_1, (2048, ), (1, ))
    assert_size_stride(arg510_1, (2048, ), (1, ))
    assert_size_stride(arg511_1, (), ())
    assert_size_stride(arg512_1, (1024, ), (1, ))
    assert_size_stride(arg513_1, (1024, ), (1, ))
    assert_size_stride(arg514_1, (), ())
    assert_size_stride(arg515_1, (2048, ), (1, ))
    assert_size_stride(arg516_1, (2048, ), (1, ))
    assert_size_stride(arg517_1, (), ())
    assert_size_stride(arg518_1, (2048, ), (1, ))
    assert_size_stride(arg519_1, (2048, ), (1, ))
    assert_size_stride(arg520_1, (), ())
    assert_size_stride(arg521_1, (1024, ), (1, ))
    assert_size_stride(arg522_1, (1024, ), (1, ))
    assert_size_stride(arg523_1, (), ())
    assert_size_stride(arg524_1, (2048, ), (1, ))
    assert_size_stride(arg525_1, (2048, ), (1, ))
    assert_size_stride(arg526_1, (), ())
    assert_size_stride(arg527_1, (2048, ), (1, ))
    assert_size_stride(arg528_1, (2048, ), (1, ))
    assert_size_stride(arg529_1, (), ())
    assert_size_stride(arg530_1, (1024, ), (1, ))
    assert_size_stride(arg531_1, (1024, ), (1, ))
    assert_size_stride(arg532_1, (), ())
    assert_size_stride(arg533_1, (2048, ), (1, ))
    assert_size_stride(arg534_1, (2048, ), (1, ))
    assert_size_stride(arg535_1, (), ())
    assert_size_stride(arg536_1, (2048, ), (1, ))
    assert_size_stride(arg537_1, (2048, ), (1, ))
    assert_size_stride(arg538_1, (), ())
    assert_size_stride(arg539_1, (1024, ), (1, ))
    assert_size_stride(arg540_1, (1024, ), (1, ))
    assert_size_stride(arg541_1, (), ())
    assert_size_stride(arg542_1, (2048, ), (1, ))
    assert_size_stride(arg543_1, (2048, ), (1, ))
    assert_size_stride(arg544_1, (), ())
    assert_size_stride(arg545_1, (2048, ), (1, ))
    assert_size_stride(arg546_1, (2048, ), (1, ))
    assert_size_stride(arg547_1, (), ())
    assert_size_stride(arg548_1, (1024, ), (1, ))
    assert_size_stride(arg549_1, (1024, ), (1, ))
    assert_size_stride(arg550_1, (), ())
    assert_size_stride(arg551_1, (2048, ), (1, ))
    assert_size_stride(arg552_1, (2048, ), (1, ))
    assert_size_stride(arg553_1, (), ())
    assert_size_stride(arg554_1, (2048, ), (1, ))
    assert_size_stride(arg555_1, (2048, ), (1, ))
    assert_size_stride(arg556_1, (), ())
    assert_size_stride(arg557_1, (1024, ), (1, ))
    assert_size_stride(arg558_1, (1024, ), (1, ))
    assert_size_stride(arg559_1, (), ())
    assert_size_stride(arg560_1, (2048, ), (1, ))
    assert_size_stride(arg561_1, (2048, ), (1, ))
    assert_size_stride(arg562_1, (), ())
    assert_size_stride(arg563_1, (2048, ), (1, ))
    assert_size_stride(arg564_1, (2048, ), (1, ))
    assert_size_stride(arg565_1, (), ())
    assert_size_stride(arg566_1, (1024, ), (1, ))
    assert_size_stride(arg567_1, (1024, ), (1, ))
    assert_size_stride(arg568_1, (), ())
    assert_size_stride(arg569_1, (2048, ), (1, ))
    assert_size_stride(arg570_1, (2048, ), (1, ))
    assert_size_stride(arg571_1, (), ())
    assert_size_stride(arg572_1, (2048, ), (1, ))
    assert_size_stride(arg573_1, (2048, ), (1, ))
    assert_size_stride(arg574_1, (), ())
    assert_size_stride(arg575_1, (1024, ), (1, ))
    assert_size_stride(arg576_1, (1024, ), (1, ))
    assert_size_stride(arg577_1, (), ())
    assert_size_stride(arg578_1, (2048, ), (1, ))
    assert_size_stride(arg579_1, (2048, ), (1, ))
    assert_size_stride(arg580_1, (), ())
    assert_size_stride(arg581_1, (2048, ), (1, ))
    assert_size_stride(arg582_1, (2048, ), (1, ))
    assert_size_stride(arg583_1, (), ())
    assert_size_stride(arg584_1, (1024, ), (1, ))
    assert_size_stride(arg585_1, (1024, ), (1, ))
    assert_size_stride(arg586_1, (), ())
    assert_size_stride(arg587_1, (2048, ), (1, ))
    assert_size_stride(arg588_1, (2048, ), (1, ))
    assert_size_stride(arg589_1, (), ())
    assert_size_stride(arg590_1, (2048, ), (1, ))
    assert_size_stride(arg591_1, (2048, ), (1, ))
    assert_size_stride(arg592_1, (), ())
    assert_size_stride(arg593_1, (1024, ), (1, ))
    assert_size_stride(arg594_1, (1024, ), (1, ))
    assert_size_stride(arg595_1, (), ())
    assert_size_stride(arg596_1, (4096, ), (1, ))
    assert_size_stride(arg597_1, (4096, ), (1, ))
    assert_size_stride(arg598_1, (), ())
    assert_size_stride(arg599_1, (4096, ), (1, ))
    assert_size_stride(arg600_1, (4096, ), (1, ))
    assert_size_stride(arg601_1, (), ())
    assert_size_stride(arg602_1, (2048, ), (1, ))
    assert_size_stride(arg603_1, (2048, ), (1, ))
    assert_size_stride(arg604_1, (), ())
    assert_size_stride(arg605_1, (2048, ), (1, ))
    assert_size_stride(arg606_1, (2048, ), (1, ))
    assert_size_stride(arg607_1, (), ())
    assert_size_stride(arg608_1, (4096, ), (1, ))
    assert_size_stride(arg609_1, (4096, ), (1, ))
    assert_size_stride(arg610_1, (), ())
    assert_size_stride(arg611_1, (4096, ), (1, ))
    assert_size_stride(arg612_1, (4096, ), (1, ))
    assert_size_stride(arg613_1, (), ())
    assert_size_stride(arg614_1, (2048, ), (1, ))
    assert_size_stride(arg615_1, (2048, ), (1, ))
    assert_size_stride(arg616_1, (), ())
    assert_size_stride(arg617_1, (4096, ), (1, ))
    assert_size_stride(arg618_1, (4096, ), (1, ))
    assert_size_stride(arg619_1, (), ())
    assert_size_stride(arg620_1, (4096, ), (1, ))
    assert_size_stride(arg621_1, (4096, ), (1, ))
    assert_size_stride(arg622_1, (), ())
    assert_size_stride(arg623_1, (2048, ), (1, ))
    assert_size_stride(arg624_1, (2048, ), (1, ))
    assert_size_stride(arg625_1, (), ())
    assert_size_stride(arg626_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg626_1, arg0_1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 64, 112, 112), (802816, 12544, 112, 1))
        del arg0_1
        del arg626_1
        buf1 = buf0; del buf0  # reuse
        # Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf1, arg314_1, arg315_1, arg1_1, arg2_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg1_1
        del arg2_1
        del arg314_1
        del arg315_1
        buf2 = empty((8, 64, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_1.run(buf1, buf2, 1605632, grid=grid(1605632), stream=stream0)
        del buf1
        # Source Nodes: [x_4], Original ATen: [aten.convolution]
        buf3 = extern_kernels.convolution(buf2, arg3_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf3, (8, 512, 56, 56), (1605632, 3136, 56, 1))
        del arg3_1
        buf4 = buf3; del buf3  # reuse
        # Source Nodes: [x_5, x_6, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf4, arg317_1, arg318_1, arg4_1, arg5_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg317_1
        del arg318_1
        del arg4_1
        del arg5_1
        # Source Nodes: [x_5, x_6, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf5 = extern_kernels.convolution(buf4, arg6_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf5, (8, 512, 56, 56), (1605632, 3136, 56, 1))
        del arg6_1
        del buf4
        buf6 = buf5; del buf5  # reuse
        # Source Nodes: [x_10, x_12, x_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf6, arg320_1, arg321_1, arg7_1, arg8_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg320_1
        del arg321_1
        del arg7_1
        del arg8_1
        # Source Nodes: [x_10, x_12, x_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf7 = extern_kernels.convolution(buf6, arg9_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (8, 256, 56, 56), (802816, 3136, 56, 1))
        del arg9_1
        del buf6
        # Source Nodes: [getattr_l__mod___layer1___0___downsample_0], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf2, arg12_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (8, 256, 56, 56), (802816, 3136, 56, 1))
        del arg12_1
        del buf2
        buf9 = buf7; del buf7  # reuse
        buf10 = buf9; del buf9  # reuse
        # Source Nodes: [shortcut_1, shortcut_2, x_13, x_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3.run(buf10, arg323_1, arg324_1, arg10_1, arg11_1, buf8, arg326_1, arg327_1, arg13_1, arg14_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg10_1
        del arg11_1
        del arg13_1
        del arg14_1
        del arg323_1
        del arg324_1
        del arg326_1
        del arg327_1
        del buf8
        # Source Nodes: [x_16], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, arg15_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (8, 512, 56, 56), (1605632, 3136, 56, 1))
        del arg15_1
        buf12 = buf11; del buf11  # reuse
        # Source Nodes: [x_17, x_18, x_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf12, arg329_1, arg330_1, arg16_1, arg17_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg16_1
        del arg17_1
        del arg329_1
        del arg330_1
        # Source Nodes: [x_17, x_18, x_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf13 = extern_kernels.convolution(buf12, arg18_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf13, (8, 512, 56, 56), (1605632, 3136, 56, 1))
        del arg18_1
        del buf12
        buf14 = buf13; del buf13  # reuse
        # Source Nodes: [x_20, x_22, x_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf14, arg332_1, arg333_1, arg19_1, arg20_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg19_1
        del arg20_1
        del arg332_1
        del arg333_1
        # Source Nodes: [x_20, x_22, x_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf15 = extern_kernels.convolution(buf14, arg21_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (8, 256, 56, 56), (802816, 3136, 56, 1))
        del arg21_1
        del buf14
        buf16 = buf10; del buf10  # reuse
        # Source Nodes: [shortcut_3, x_25, x_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4.run(buf16, buf15, arg335_1, arg336_1, arg22_1, arg23_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg22_1
        del arg23_1
        del arg335_1
        del arg336_1
        del buf15
        # Source Nodes: [x_28], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, arg24_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (8, 512, 56, 56), (1605632, 3136, 56, 1))
        del arg24_1
        buf18 = buf17; del buf17  # reuse
        # Source Nodes: [x_29, x_30, x_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf18, arg338_1, arg339_1, arg25_1, arg26_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg25_1
        del arg26_1
        del arg338_1
        del arg339_1
        # Source Nodes: [x_29, x_30, x_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf19 = extern_kernels.convolution(buf18, arg27_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf19, (8, 512, 56, 56), (1605632, 3136, 56, 1))
        del arg27_1
        del buf18
        buf20 = buf19; del buf19  # reuse
        # Source Nodes: [x_32, x_34, x_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_2.run(buf20, arg341_1, arg342_1, arg28_1, arg29_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg28_1
        del arg29_1
        del arg341_1
        del arg342_1
        # Source Nodes: [x_32, x_34, x_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf21 = extern_kernels.convolution(buf20, arg30_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (8, 256, 56, 56), (802816, 3136, 56, 1))
        del arg30_1
        del buf20
        buf22 = buf16; del buf16  # reuse
        # Source Nodes: [shortcut_4, x_37, x_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4.run(buf22, buf21, arg344_1, arg345_1, arg31_1, arg32_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg31_1
        del arg32_1
        del arg344_1
        del arg345_1
        del buf21
        # Source Nodes: [x_41], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, arg33_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (8, 1024, 56, 56), (3211264, 3136, 56, 1))
        del arg33_1
        buf24 = buf23; del buf23  # reuse
        # Source Nodes: [x_42, x_43, x_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(buf24, arg347_1, arg348_1, arg34_1, arg35_1, 25690112, grid=grid(25690112), stream=stream0)
        del arg347_1
        del arg348_1
        del arg34_1
        del arg35_1
        # Source Nodes: [x_42, x_43, x_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf25 = extern_kernels.convolution(buf24, arg36_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf25, (8, 1024, 28, 28), (802816, 784, 28, 1))
        del arg36_1
        del buf24
        buf26 = buf25; del buf25  # reuse
        # Source Nodes: [x_45, x_47, x_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf26, arg350_1, arg351_1, arg37_1, arg38_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg350_1
        del arg351_1
        del arg37_1
        del arg38_1
        # Source Nodes: [x_45, x_47, x_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf27 = extern_kernels.convolution(buf26, arg39_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (8, 512, 28, 28), (401408, 784, 28, 1))
        del arg39_1
        del buf26
        # Source Nodes: [getattr_l__mod___layer2___0___downsample_0], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf22, arg42_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (8, 512, 28, 28), (401408, 784, 28, 1))
        del arg42_1
        del buf22
        buf29 = buf27; del buf27  # reuse
        buf30 = buf29; del buf29  # reuse
        # Source Nodes: [shortcut_5, shortcut_6, x_50, x_51], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf30, arg353_1, arg354_1, arg40_1, arg41_1, buf28, arg356_1, arg357_1, arg43_1, arg44_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg353_1
        del arg354_1
        del arg356_1
        del arg357_1
        del arg40_1
        del arg41_1
        del arg43_1
        del arg44_1
        del buf28
        # Source Nodes: [x_53], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, arg45_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (8, 1024, 28, 28), (802816, 784, 28, 1))
        del arg45_1
        buf32 = buf31; del buf31  # reuse
        # Source Nodes: [x_54, x_55, x_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf32, arg359_1, arg360_1, arg46_1, arg47_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg359_1
        del arg360_1
        del arg46_1
        del arg47_1
        # Source Nodes: [x_54, x_55, x_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf33 = extern_kernels.convolution(buf32, arg48_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf33, (8, 1024, 28, 28), (802816, 784, 28, 1))
        del arg48_1
        del buf32
        buf34 = buf33; del buf33  # reuse
        # Source Nodes: [x_57, x_59, x_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf34, arg362_1, arg363_1, arg49_1, arg50_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg362_1
        del arg363_1
        del arg49_1
        del arg50_1
        # Source Nodes: [x_57, x_59, x_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf35 = extern_kernels.convolution(buf34, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (8, 512, 28, 28), (401408, 784, 28, 1))
        del arg51_1
        del buf34
        buf36 = buf30; del buf30  # reuse
        # Source Nodes: [shortcut_7, x_62, x_63], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8.run(buf36, buf35, arg365_1, arg366_1, arg52_1, arg53_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg365_1
        del arg366_1
        del arg52_1
        del arg53_1
        del buf35
        # Source Nodes: [x_65], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, arg54_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (8, 1024, 28, 28), (802816, 784, 28, 1))
        del arg54_1
        buf38 = buf37; del buf37  # reuse
        # Source Nodes: [x_66, x_67, x_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf38, arg368_1, arg369_1, arg55_1, arg56_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg368_1
        del arg369_1
        del arg55_1
        del arg56_1
        # Source Nodes: [x_66, x_67, x_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf39 = extern_kernels.convolution(buf38, arg57_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf39, (8, 1024, 28, 28), (802816, 784, 28, 1))
        del arg57_1
        del buf38
        buf40 = buf39; del buf39  # reuse
        # Source Nodes: [x_69, x_71, x_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf40, arg371_1, arg372_1, arg58_1, arg59_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg371_1
        del arg372_1
        del arg58_1
        del arg59_1
        # Source Nodes: [x_69, x_71, x_73], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf41 = extern_kernels.convolution(buf40, arg60_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (8, 512, 28, 28), (401408, 784, 28, 1))
        del arg60_1
        del buf40
        buf42 = buf36; del buf36  # reuse
        # Source Nodes: [shortcut_8, x_74, x_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8.run(buf42, buf41, arg374_1, arg375_1, arg61_1, arg62_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg374_1
        del arg375_1
        del arg61_1
        del arg62_1
        del buf41
        # Source Nodes: [x_77], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, arg63_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (8, 1024, 28, 28), (802816, 784, 28, 1))
        del arg63_1
        buf44 = buf43; del buf43  # reuse
        # Source Nodes: [x_78, x_79, x_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf44, arg377_1, arg378_1, arg64_1, arg65_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg377_1
        del arg378_1
        del arg64_1
        del arg65_1
        # Source Nodes: [x_78, x_79, x_80], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf45 = extern_kernels.convolution(buf44, arg66_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf45, (8, 1024, 28, 28), (802816, 784, 28, 1))
        del arg66_1
        del buf44
        buf46 = buf45; del buf45  # reuse
        # Source Nodes: [x_81, x_83, x_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_6.run(buf46, arg380_1, arg381_1, arg67_1, arg68_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg380_1
        del arg381_1
        del arg67_1
        del arg68_1
        # Source Nodes: [x_81, x_83, x_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf47 = extern_kernels.convolution(buf46, arg69_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (8, 512, 28, 28), (401408, 784, 28, 1))
        del arg69_1
        del buf46
        buf48 = buf42; del buf42  # reuse
        # Source Nodes: [shortcut_9, x_86, x_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8.run(buf48, buf47, arg383_1, arg384_1, arg70_1, arg71_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg383_1
        del arg384_1
        del arg70_1
        del arg71_1
        del buf47
        # Source Nodes: [x_90], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, arg72_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (8, 2048, 28, 28), (1605632, 784, 28, 1))
        del arg72_1
        buf50 = buf49; del buf49  # reuse
        # Source Nodes: [x_91, x_92, x_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(buf50, arg386_1, arg387_1, arg73_1, arg74_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg386_1
        del arg387_1
        del arg73_1
        del arg74_1
        # Source Nodes: [x_91, x_92, x_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf51 = extern_kernels.convolution(buf50, arg75_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf51, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg75_1
        del buf50
        buf52 = buf51; del buf51  # reuse
        # Source Nodes: [x_94, x_96, x_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf52, arg389_1, arg390_1, arg76_1, arg77_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg389_1
        del arg390_1
        del arg76_1
        del arg77_1
        # Source Nodes: [x_94, x_96, x_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf53 = extern_kernels.convolution(buf52, arg78_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg78_1
        del buf52
        # Source Nodes: [getattr_l__mod___layer3___0___downsample_0], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf48, arg81_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg81_1
        del buf48
        buf55 = buf53; del buf53  # reuse
        buf56 = buf55; del buf55  # reuse
        # Source Nodes: [shortcut_10, shortcut_11, x_100, x_99], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf56, arg392_1, arg393_1, arg79_1, arg80_1, buf54, arg395_1, arg396_1, arg82_1, arg83_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg392_1
        del arg393_1
        del arg395_1
        del arg396_1
        del arg79_1
        del arg80_1
        del arg82_1
        del arg83_1
        del buf54
        # Source Nodes: [x_102], Original ATen: [aten.convolution]
        buf57 = extern_kernels.convolution(buf56, arg84_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg84_1
        buf58 = buf57; del buf57  # reuse
        # Source Nodes: [x_103, x_104, x_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf58, arg398_1, arg399_1, arg85_1, arg86_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg398_1
        del arg399_1
        del arg85_1
        del arg86_1
        # Source Nodes: [x_103, x_104, x_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf59 = extern_kernels.convolution(buf58, arg87_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf59, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg87_1
        del buf58
        buf60 = buf59; del buf59  # reuse
        # Source Nodes: [x_106, x_108, x_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf60, arg401_1, arg402_1, arg88_1, arg89_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg401_1
        del arg402_1
        del arg88_1
        del arg89_1
        # Source Nodes: [x_106, x_108, x_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf61 = extern_kernels.convolution(buf60, arg90_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf61, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg90_1
        del buf60
        buf62 = buf56; del buf56  # reuse
        # Source Nodes: [shortcut_12, x_111, x_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf62, buf61, arg404_1, arg405_1, arg91_1, arg92_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg404_1
        del arg405_1
        del arg91_1
        del arg92_1
        del buf61
        # Source Nodes: [x_114], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf62, arg93_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg93_1
        buf64 = buf63; del buf63  # reuse
        # Source Nodes: [x_115, x_116, x_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf64, arg407_1, arg408_1, arg94_1, arg95_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg407_1
        del arg408_1
        del arg94_1
        del arg95_1
        # Source Nodes: [x_115, x_116, x_117], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf65 = extern_kernels.convolution(buf64, arg96_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf65, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg96_1
        del buf64
        buf66 = buf65; del buf65  # reuse
        # Source Nodes: [x_118, x_120, x_122], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf66, arg410_1, arg411_1, arg97_1, arg98_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg410_1
        del arg411_1
        del arg97_1
        del arg98_1
        # Source Nodes: [x_118, x_120, x_122], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf67 = extern_kernels.convolution(buf66, arg99_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg99_1
        del buf66
        buf68 = buf62; del buf62  # reuse
        # Source Nodes: [shortcut_13, x_123, x_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf68, buf67, arg413_1, arg414_1, arg100_1, arg101_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg100_1
        del arg101_1
        del arg413_1
        del arg414_1
        del buf67
        # Source Nodes: [x_126], Original ATen: [aten.convolution]
        buf69 = extern_kernels.convolution(buf68, arg102_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf69, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg102_1
        buf70 = buf69; del buf69  # reuse
        # Source Nodes: [x_127, x_128, x_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf70, arg416_1, arg417_1, arg103_1, arg104_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg103_1
        del arg104_1
        del arg416_1
        del arg417_1
        # Source Nodes: [x_127, x_128, x_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf71 = extern_kernels.convolution(buf70, arg105_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf71, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg105_1
        del buf70
        buf72 = buf71; del buf71  # reuse
        # Source Nodes: [x_130, x_132, x_134], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf72, arg419_1, arg420_1, arg106_1, arg107_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg106_1
        del arg107_1
        del arg419_1
        del arg420_1
        # Source Nodes: [x_130, x_132, x_134], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf73 = extern_kernels.convolution(buf72, arg108_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg108_1
        del buf72
        buf74 = buf68; del buf68  # reuse
        # Source Nodes: [shortcut_14, x_135, x_136], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf74, buf73, arg422_1, arg423_1, arg109_1, arg110_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg109_1
        del arg110_1
        del arg422_1
        del arg423_1
        del buf73
        # Source Nodes: [x_138], Original ATen: [aten.convolution]
        buf75 = extern_kernels.convolution(buf74, arg111_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg111_1
        buf76 = buf75; del buf75  # reuse
        # Source Nodes: [x_139, x_140, x_141], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf76, arg425_1, arg426_1, arg112_1, arg113_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg112_1
        del arg113_1
        del arg425_1
        del arg426_1
        # Source Nodes: [x_139, x_140, x_141], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf77 = extern_kernels.convolution(buf76, arg114_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf77, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg114_1
        del buf76
        buf78 = buf77; del buf77  # reuse
        # Source Nodes: [x_142, x_144, x_146], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf78, arg428_1, arg429_1, arg115_1, arg116_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg115_1
        del arg116_1
        del arg428_1
        del arg429_1
        # Source Nodes: [x_142, x_144, x_146], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf79 = extern_kernels.convolution(buf78, arg117_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg117_1
        del buf78
        buf80 = buf74; del buf74  # reuse
        # Source Nodes: [shortcut_15, x_147, x_148], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf80, buf79, arg431_1, arg432_1, arg118_1, arg119_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg118_1
        del arg119_1
        del arg431_1
        del arg432_1
        del buf79
        # Source Nodes: [x_150], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, arg120_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg120_1
        buf82 = buf81; del buf81  # reuse
        # Source Nodes: [x_151, x_152, x_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf82, arg434_1, arg435_1, arg121_1, arg122_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg121_1
        del arg122_1
        del arg434_1
        del arg435_1
        # Source Nodes: [x_151, x_152, x_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf83 = extern_kernels.convolution(buf82, arg123_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf83, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg123_1
        del buf82
        buf84 = buf83; del buf83  # reuse
        # Source Nodes: [x_154, x_156, x_158], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf84, arg437_1, arg438_1, arg124_1, arg125_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg124_1
        del arg125_1
        del arg437_1
        del arg438_1
        # Source Nodes: [x_154, x_156, x_158], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf85 = extern_kernels.convolution(buf84, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg126_1
        del buf84
        buf86 = buf80; del buf80  # reuse
        # Source Nodes: [shortcut_16, x_159, x_160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf86, buf85, arg440_1, arg441_1, arg127_1, arg128_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg127_1
        del arg128_1
        del arg440_1
        del arg441_1
        del buf85
        # Source Nodes: [x_162], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, arg129_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg129_1
        buf88 = buf87; del buf87  # reuse
        # Source Nodes: [x_163, x_164, x_165], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf88, arg443_1, arg444_1, arg130_1, arg131_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg130_1
        del arg131_1
        del arg443_1
        del arg444_1
        # Source Nodes: [x_163, x_164, x_165], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf89 = extern_kernels.convolution(buf88, arg132_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf89, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg132_1
        del buf88
        buf90 = buf89; del buf89  # reuse
        # Source Nodes: [x_166, x_168, x_170], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf90, arg446_1, arg447_1, arg133_1, arg134_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg133_1
        del arg134_1
        del arg446_1
        del arg447_1
        # Source Nodes: [x_166, x_168, x_170], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf91 = extern_kernels.convolution(buf90, arg135_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg135_1
        del buf90
        buf92 = buf86; del buf86  # reuse
        # Source Nodes: [shortcut_17, x_171, x_172], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf92, buf91, arg449_1, arg450_1, arg136_1, arg137_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg136_1
        del arg137_1
        del arg449_1
        del arg450_1
        del buf91
        # Source Nodes: [x_174], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf92, arg138_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg138_1
        buf94 = buf93; del buf93  # reuse
        # Source Nodes: [x_175, x_176, x_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf94, arg452_1, arg453_1, arg139_1, arg140_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg139_1
        del arg140_1
        del arg452_1
        del arg453_1
        # Source Nodes: [x_175, x_176, x_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf95 = extern_kernels.convolution(buf94, arg141_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf95, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg141_1
        del buf94
        buf96 = buf95; del buf95  # reuse
        # Source Nodes: [x_178, x_180, x_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf96, arg455_1, arg456_1, arg142_1, arg143_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg142_1
        del arg143_1
        del arg455_1
        del arg456_1
        # Source Nodes: [x_178, x_180, x_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf97 = extern_kernels.convolution(buf96, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg144_1
        del buf96
        buf98 = buf92; del buf92  # reuse
        # Source Nodes: [shortcut_18, x_183, x_184], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf98, buf97, arg458_1, arg459_1, arg145_1, arg146_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg145_1
        del arg146_1
        del arg458_1
        del arg459_1
        del buf97
        # Source Nodes: [x_186], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf98, arg147_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg147_1
        buf100 = buf99; del buf99  # reuse
        # Source Nodes: [x_187, x_188, x_189], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf100, arg461_1, arg462_1, arg148_1, arg149_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg148_1
        del arg149_1
        del arg461_1
        del arg462_1
        # Source Nodes: [x_187, x_188, x_189], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf101 = extern_kernels.convolution(buf100, arg150_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf101, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg150_1
        del buf100
        buf102 = buf101; del buf101  # reuse
        # Source Nodes: [x_190, x_192, x_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf102, arg464_1, arg465_1, arg151_1, arg152_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg151_1
        del arg152_1
        del arg464_1
        del arg465_1
        # Source Nodes: [x_190, x_192, x_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf103 = extern_kernels.convolution(buf102, arg153_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg153_1
        del buf102
        buf104 = buf103; del buf103  # reuse
        # Source Nodes: [shortcut_19, x_195, x_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_13.run(buf104, arg467_1, arg468_1, arg154_1, arg155_1, buf98, 1605632, grid=grid(1605632), stream=stream0)
        del arg154_1
        del arg155_1
        del arg467_1
        del arg468_1
        del buf98
        # Source Nodes: [x_198], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, arg156_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg156_1
        buf106 = buf105; del buf105  # reuse
        # Source Nodes: [x_199, x_200, x_201], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf106, arg470_1, arg471_1, arg157_1, arg158_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg157_1
        del arg158_1
        del arg470_1
        del arg471_1
        # Source Nodes: [x_199, x_200, x_201], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf107 = extern_kernels.convolution(buf106, arg159_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf107, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg159_1
        del buf106
        buf108 = buf107; del buf107  # reuse
        # Source Nodes: [x_202, x_204, x_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf108, arg473_1, arg474_1, arg160_1, arg161_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg160_1
        del arg161_1
        del arg473_1
        del arg474_1
        # Source Nodes: [x_202, x_204, x_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf109 = extern_kernels.convolution(buf108, arg162_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg162_1
        del buf108
        buf110 = buf104; del buf104  # reuse
        # Source Nodes: [shortcut_20, x_207, x_208], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf110, buf109, arg476_1, arg477_1, arg163_1, arg164_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg163_1
        del arg164_1
        del arg476_1
        del arg477_1
        del buf109
        # Source Nodes: [x_210], Original ATen: [aten.convolution]
        buf111 = extern_kernels.convolution(buf110, arg165_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg165_1
        buf112 = buf111; del buf111  # reuse
        # Source Nodes: [x_211, x_212, x_213], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf112, arg479_1, arg480_1, arg166_1, arg167_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg166_1
        del arg167_1
        del arg479_1
        del arg480_1
        # Source Nodes: [x_211, x_212, x_213], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf113 = extern_kernels.convolution(buf112, arg168_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf113, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg168_1
        del buf112
        buf114 = buf113; del buf113  # reuse
        # Source Nodes: [x_214, x_216, x_218], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf114, arg482_1, arg483_1, arg169_1, arg170_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg169_1
        del arg170_1
        del arg482_1
        del arg483_1
        # Source Nodes: [x_214, x_216, x_218], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf115 = extern_kernels.convolution(buf114, arg171_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg171_1
        del buf114
        buf116 = buf110; del buf110  # reuse
        # Source Nodes: [shortcut_21, x_219, x_220], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf116, buf115, arg485_1, arg486_1, arg172_1, arg173_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg172_1
        del arg173_1
        del arg485_1
        del arg486_1
        del buf115
        # Source Nodes: [x_222], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf116, arg174_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg174_1
        buf118 = buf117; del buf117  # reuse
        # Source Nodes: [x_223, x_224, x_225], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf118, arg488_1, arg489_1, arg175_1, arg176_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg175_1
        del arg176_1
        del arg488_1
        del arg489_1
        # Source Nodes: [x_223, x_224, x_225], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf119 = extern_kernels.convolution(buf118, arg177_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf119, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg177_1
        del buf118
        buf120 = buf119; del buf119  # reuse
        # Source Nodes: [x_226, x_228, x_230], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf120, arg491_1, arg492_1, arg178_1, arg179_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg178_1
        del arg179_1
        del arg491_1
        del arg492_1
        # Source Nodes: [x_226, x_228, x_230], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf121 = extern_kernels.convolution(buf120, arg180_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg180_1
        del buf120
        buf122 = buf116; del buf116  # reuse
        # Source Nodes: [shortcut_22, x_231, x_232], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf122, buf121, arg494_1, arg495_1, arg181_1, arg182_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg181_1
        del arg182_1
        del arg494_1
        del arg495_1
        del buf121
        # Source Nodes: [x_234], Original ATen: [aten.convolution]
        buf123 = extern_kernels.convolution(buf122, arg183_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf123, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg183_1
        buf124 = buf123; del buf123  # reuse
        # Source Nodes: [x_235, x_236, x_237], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf124, arg497_1, arg498_1, arg184_1, arg185_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg184_1
        del arg185_1
        del arg497_1
        del arg498_1
        # Source Nodes: [x_235, x_236, x_237], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf125 = extern_kernels.convolution(buf124, arg186_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf125, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg186_1
        del buf124
        buf126 = buf125; del buf125  # reuse
        # Source Nodes: [x_238, x_240, x_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf126, arg500_1, arg501_1, arg187_1, arg188_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg187_1
        del arg188_1
        del arg500_1
        del arg501_1
        # Source Nodes: [x_238, x_240, x_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf127 = extern_kernels.convolution(buf126, arg189_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg189_1
        del buf126
        buf128 = buf122; del buf122  # reuse
        # Source Nodes: [shortcut_23, x_243, x_244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf128, buf127, arg503_1, arg504_1, arg190_1, arg191_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg190_1
        del arg191_1
        del arg503_1
        del arg504_1
        del buf127
        # Source Nodes: [x_246], Original ATen: [aten.convolution]
        buf129 = extern_kernels.convolution(buf128, arg192_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf129, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg192_1
        buf130 = buf129; del buf129  # reuse
        # Source Nodes: [x_247, x_248, x_249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf130, arg506_1, arg507_1, arg193_1, arg194_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg193_1
        del arg194_1
        del arg506_1
        del arg507_1
        # Source Nodes: [x_247, x_248, x_249], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf131 = extern_kernels.convolution(buf130, arg195_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf131, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg195_1
        del buf130
        buf132 = buf131; del buf131  # reuse
        # Source Nodes: [x_250, x_252, x_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf132, arg509_1, arg510_1, arg196_1, arg197_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg196_1
        del arg197_1
        del arg509_1
        del arg510_1
        # Source Nodes: [x_250, x_252, x_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf133 = extern_kernels.convolution(buf132, arg198_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg198_1
        del buf132
        buf134 = buf128; del buf128  # reuse
        # Source Nodes: [shortcut_24, x_255, x_256], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf134, buf133, arg512_1, arg513_1, arg199_1, arg200_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg199_1
        del arg200_1
        del arg512_1
        del arg513_1
        del buf133
        # Source Nodes: [x_258], Original ATen: [aten.convolution]
        buf135 = extern_kernels.convolution(buf134, arg201_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg201_1
        buf136 = buf135; del buf135  # reuse
        # Source Nodes: [x_259, x_260, x_261], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf136, arg515_1, arg516_1, arg202_1, arg203_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg202_1
        del arg203_1
        del arg515_1
        del arg516_1
        # Source Nodes: [x_259, x_260, x_261], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf137 = extern_kernels.convolution(buf136, arg204_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf137, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg204_1
        del buf136
        buf138 = buf137; del buf137  # reuse
        # Source Nodes: [x_262, x_264, x_266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf138, arg518_1, arg519_1, arg205_1, arg206_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg205_1
        del arg206_1
        del arg518_1
        del arg519_1
        # Source Nodes: [x_262, x_264, x_266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf139 = extern_kernels.convolution(buf138, arg207_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf139, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg207_1
        del buf138
        buf140 = buf134; del buf134  # reuse
        # Source Nodes: [shortcut_25, x_267, x_268], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf140, buf139, arg521_1, arg522_1, arg208_1, arg209_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg208_1
        del arg209_1
        del arg521_1
        del arg522_1
        del buf139
        # Source Nodes: [x_270], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, arg210_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg210_1
        buf142 = buf141; del buf141  # reuse
        # Source Nodes: [x_271, x_272, x_273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf142, arg524_1, arg525_1, arg211_1, arg212_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg211_1
        del arg212_1
        del arg524_1
        del arg525_1
        # Source Nodes: [x_271, x_272, x_273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf143 = extern_kernels.convolution(buf142, arg213_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf143, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg213_1
        del buf142
        buf144 = buf143; del buf143  # reuse
        # Source Nodes: [x_274, x_276, x_278], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf144, arg527_1, arg528_1, arg214_1, arg215_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg214_1
        del arg215_1
        del arg527_1
        del arg528_1
        # Source Nodes: [x_274, x_276, x_278], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf145 = extern_kernels.convolution(buf144, arg216_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf145, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg216_1
        del buf144
        buf146 = buf140; del buf140  # reuse
        # Source Nodes: [shortcut_26, x_279, x_280], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf146, buf145, arg530_1, arg531_1, arg217_1, arg218_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg217_1
        del arg218_1
        del arg530_1
        del arg531_1
        del buf145
        # Source Nodes: [x_282], Original ATen: [aten.convolution]
        buf147 = extern_kernels.convolution(buf146, arg219_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf147, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg219_1
        buf148 = buf147; del buf147  # reuse
        # Source Nodes: [x_283, x_284, x_285], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf148, arg533_1, arg534_1, arg220_1, arg221_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg220_1
        del arg221_1
        del arg533_1
        del arg534_1
        # Source Nodes: [x_283, x_284, x_285], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf149 = extern_kernels.convolution(buf148, arg222_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf149, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg222_1
        del buf148
        buf150 = buf149; del buf149  # reuse
        # Source Nodes: [x_286, x_288, x_290], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf150, arg536_1, arg537_1, arg223_1, arg224_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg223_1
        del arg224_1
        del arg536_1
        del arg537_1
        # Source Nodes: [x_286, x_288, x_290], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf151 = extern_kernels.convolution(buf150, arg225_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg225_1
        del buf150
        buf152 = buf146; del buf146  # reuse
        # Source Nodes: [shortcut_27, x_291, x_292], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf152, buf151, arg539_1, arg540_1, arg226_1, arg227_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg226_1
        del arg227_1
        del arg539_1
        del arg540_1
        del buf151
        # Source Nodes: [x_294], Original ATen: [aten.convolution]
        buf153 = extern_kernels.convolution(buf152, arg228_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf153, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg228_1
        buf154 = buf153; del buf153  # reuse
        # Source Nodes: [x_295, x_296, x_297], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf154, arg542_1, arg543_1, arg229_1, arg230_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg229_1
        del arg230_1
        del arg542_1
        del arg543_1
        # Source Nodes: [x_295, x_296, x_297], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf155 = extern_kernels.convolution(buf154, arg231_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf155, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg231_1
        del buf154
        buf156 = buf155; del buf155  # reuse
        # Source Nodes: [x_298, x_300, x_302], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf156, arg545_1, arg546_1, arg232_1, arg233_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg232_1
        del arg233_1
        del arg545_1
        del arg546_1
        # Source Nodes: [x_298, x_300, x_302], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf157 = extern_kernels.convolution(buf156, arg234_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg234_1
        del buf156
        buf158 = buf152; del buf152  # reuse
        # Source Nodes: [shortcut_28, x_303, x_304], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf158, buf157, arg548_1, arg549_1, arg235_1, arg236_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg235_1
        del arg236_1
        del arg548_1
        del arg549_1
        del buf157
        # Source Nodes: [x_306], Original ATen: [aten.convolution]
        buf159 = extern_kernels.convolution(buf158, arg237_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf159, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg237_1
        buf160 = buf159; del buf159  # reuse
        # Source Nodes: [x_307, x_308, x_309], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf160, arg551_1, arg552_1, arg238_1, arg239_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg238_1
        del arg239_1
        del arg551_1
        del arg552_1
        # Source Nodes: [x_307, x_308, x_309], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf161 = extern_kernels.convolution(buf160, arg240_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf161, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg240_1
        del buf160
        buf162 = buf161; del buf161  # reuse
        # Source Nodes: [x_310, x_312, x_314], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf162, arg554_1, arg555_1, arg241_1, arg242_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg241_1
        del arg242_1
        del arg554_1
        del arg555_1
        # Source Nodes: [x_310, x_312, x_314], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf163 = extern_kernels.convolution(buf162, arg243_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf163, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg243_1
        del buf162
        buf164 = buf158; del buf158  # reuse
        # Source Nodes: [shortcut_29, x_315, x_316], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf164, buf163, arg557_1, arg558_1, arg244_1, arg245_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg244_1
        del arg245_1
        del arg557_1
        del arg558_1
        del buf163
        # Source Nodes: [x_318], Original ATen: [aten.convolution]
        buf165 = extern_kernels.convolution(buf164, arg246_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf165, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg246_1
        buf166 = buf165; del buf165  # reuse
        # Source Nodes: [x_319, x_320, x_321], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf166, arg560_1, arg561_1, arg247_1, arg248_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg247_1
        del arg248_1
        del arg560_1
        del arg561_1
        # Source Nodes: [x_319, x_320, x_321], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf167 = extern_kernels.convolution(buf166, arg249_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf167, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg249_1
        del buf166
        buf168 = buf167; del buf167  # reuse
        # Source Nodes: [x_322, x_324, x_326], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf168, arg563_1, arg564_1, arg250_1, arg251_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg250_1
        del arg251_1
        del arg563_1
        del arg564_1
        # Source Nodes: [x_322, x_324, x_326], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf169 = extern_kernels.convolution(buf168, arg252_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf169, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg252_1
        del buf168
        buf170 = buf164; del buf164  # reuse
        # Source Nodes: [shortcut_30, x_327, x_328], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf170, buf169, arg566_1, arg567_1, arg253_1, arg254_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg253_1
        del arg254_1
        del arg566_1
        del arg567_1
        del buf169
        # Source Nodes: [x_330], Original ATen: [aten.convolution]
        buf171 = extern_kernels.convolution(buf170, arg255_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf171, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg255_1
        buf172 = buf171; del buf171  # reuse
        # Source Nodes: [x_331, x_332, x_333], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf172, arg569_1, arg570_1, arg256_1, arg257_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg256_1
        del arg257_1
        del arg569_1
        del arg570_1
        # Source Nodes: [x_331, x_332, x_333], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf173 = extern_kernels.convolution(buf172, arg258_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf173, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg258_1
        del buf172
        buf174 = buf173; del buf173  # reuse
        # Source Nodes: [x_334, x_336, x_338], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf174, arg572_1, arg573_1, arg259_1, arg260_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg259_1
        del arg260_1
        del arg572_1
        del arg573_1
        # Source Nodes: [x_334, x_336, x_338], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf175 = extern_kernels.convolution(buf174, arg261_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg261_1
        del buf174
        buf176 = buf170; del buf170  # reuse
        # Source Nodes: [shortcut_31, x_339, x_340], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf176, buf175, arg575_1, arg576_1, arg262_1, arg263_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg262_1
        del arg263_1
        del arg575_1
        del arg576_1
        del buf175
        # Source Nodes: [x_342], Original ATen: [aten.convolution]
        buf177 = extern_kernels.convolution(buf176, arg264_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf177, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg264_1
        buf178 = buf177; del buf177  # reuse
        # Source Nodes: [x_343, x_344, x_345], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf178, arg578_1, arg579_1, arg265_1, arg266_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg265_1
        del arg266_1
        del arg578_1
        del arg579_1
        # Source Nodes: [x_343, x_344, x_345], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf179 = extern_kernels.convolution(buf178, arg267_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf179, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg267_1
        del buf178
        buf180 = buf179; del buf179  # reuse
        # Source Nodes: [x_346, x_348, x_350], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf180, arg581_1, arg582_1, arg268_1, arg269_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg268_1
        del arg269_1
        del arg581_1
        del arg582_1
        # Source Nodes: [x_346, x_348, x_350], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf181 = extern_kernels.convolution(buf180, arg270_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf181, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg270_1
        del buf180
        buf182 = buf176; del buf176  # reuse
        # Source Nodes: [shortcut_32, x_351, x_352], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf182, buf181, arg584_1, arg585_1, arg271_1, arg272_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg271_1
        del arg272_1
        del arg584_1
        del arg585_1
        del buf181
        # Source Nodes: [x_354], Original ATen: [aten.convolution]
        buf183 = extern_kernels.convolution(buf182, arg273_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf183, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg273_1
        buf184 = buf183; del buf183  # reuse
        # Source Nodes: [x_355, x_356, x_357], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf184, arg587_1, arg588_1, arg274_1, arg275_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg274_1
        del arg275_1
        del arg587_1
        del arg588_1
        # Source Nodes: [x_355, x_356, x_357], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf185 = extern_kernels.convolution(buf184, arg276_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf185, (8, 2048, 14, 14), (401408, 196, 14, 1))
        del arg276_1
        del buf184
        buf186 = buf185; del buf185  # reuse
        # Source Nodes: [x_358, x_360, x_362], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf186, arg590_1, arg591_1, arg277_1, arg278_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg277_1
        del arg278_1
        del arg590_1
        del arg591_1
        # Source Nodes: [x_358, x_360, x_362], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf187 = extern_kernels.convolution(buf186, arg279_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf187, (8, 1024, 14, 14), (200704, 196, 14, 1))
        del arg279_1
        del buf186
        buf188 = buf182; del buf182  # reuse
        # Source Nodes: [shortcut_33, x_363, x_364], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf188, buf187, arg593_1, arg594_1, arg280_1, arg281_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg280_1
        del arg281_1
        del arg593_1
        del arg594_1
        del buf187
        # Source Nodes: [x_367], Original ATen: [aten.convolution]
        buf189 = extern_kernels.convolution(buf188, arg282_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf189, (8, 4096, 14, 14), (802816, 196, 14, 1))
        del arg282_1
        buf190 = buf189; del buf189  # reuse
        # Source Nodes: [x_368, x_369, x_370], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_14.run(buf190, arg596_1, arg597_1, arg283_1, arg284_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg283_1
        del arg284_1
        del arg596_1
        del arg597_1
        # Source Nodes: [x_368, x_369, x_370], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf191 = extern_kernels.convolution(buf190, arg285_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf191, (8, 4096, 7, 7), (200704, 49, 7, 1))
        del arg285_1
        del buf190
        buf192 = buf191; del buf191  # reuse
        # Source Nodes: [x_371, x_373, x_375], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf192, arg599_1, arg600_1, arg286_1, arg287_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg286_1
        del arg287_1
        del arg599_1
        del arg600_1
        # Source Nodes: [x_371, x_373, x_375], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf193 = extern_kernels.convolution(buf192, arg288_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (8, 2048, 7, 7), (100352, 49, 7, 1))
        del arg288_1
        del buf192
        # Source Nodes: [getattr_l__mod___layer4___0___downsample_0], Original ATen: [aten.convolution]
        buf194 = extern_kernels.convolution(buf188, arg291_1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf194, (8, 2048, 7, 7), (100352, 49, 7, 1))
        del arg291_1
        del buf188
        buf195 = buf193; del buf193  # reuse
        buf196 = buf195; del buf195  # reuse
        # Source Nodes: [shortcut_34, shortcut_35, x_376, x_377], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf196, arg602_1, arg603_1, arg289_1, arg290_1, buf194, arg605_1, arg606_1, arg292_1, arg293_1, 802816, grid=grid(802816), stream=stream0)
        del arg289_1
        del arg290_1
        del arg292_1
        del arg293_1
        del arg602_1
        del arg603_1
        del arg605_1
        del arg606_1
        del buf194
        # Source Nodes: [x_379], Original ATen: [aten.convolution]
        buf197 = extern_kernels.convolution(buf196, arg294_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf197, (8, 4096, 7, 7), (200704, 49, 7, 1))
        del arg294_1
        buf198 = buf197; del buf197  # reuse
        # Source Nodes: [x_380, x_381, x_382], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf198, arg608_1, arg609_1, arg295_1, arg296_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg295_1
        del arg296_1
        del arg608_1
        del arg609_1
        # Source Nodes: [x_380, x_381, x_382], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf199 = extern_kernels.convolution(buf198, arg297_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf199, (8, 4096, 7, 7), (200704, 49, 7, 1))
        del arg297_1
        del buf198
        buf200 = buf199; del buf199  # reuse
        # Source Nodes: [x_383, x_385, x_387], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf200, arg611_1, arg612_1, arg298_1, arg299_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg298_1
        del arg299_1
        del arg611_1
        del arg612_1
        # Source Nodes: [x_383, x_385, x_387], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf201 = extern_kernels.convolution(buf200, arg300_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf201, (8, 2048, 7, 7), (100352, 49, 7, 1))
        del arg300_1
        del buf200
        buf202 = buf196; del buf196  # reuse
        # Source Nodes: [shortcut_36, x_388, x_389], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_17.run(buf202, buf201, arg614_1, arg615_1, arg301_1, arg302_1, 802816, grid=grid(802816), stream=stream0)
        del arg301_1
        del arg302_1
        del arg614_1
        del arg615_1
        del buf201
        # Source Nodes: [x_391], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, arg303_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (8, 4096, 7, 7), (200704, 49, 7, 1))
        del arg303_1
        buf204 = buf203; del buf203  # reuse
        # Source Nodes: [x_392, x_393, x_394], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf204, arg617_1, arg618_1, arg304_1, arg305_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg304_1
        del arg305_1
        del arg617_1
        del arg618_1
        # Source Nodes: [x_392, x_393, x_394], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf205 = extern_kernels.convolution(buf204, arg306_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf205, (8, 4096, 7, 7), (200704, 49, 7, 1))
        del arg306_1
        del buf204
        buf206 = buf205; del buf205  # reuse
        # Source Nodes: [x_395, x_397, x_399], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf206, arg620_1, arg621_1, arg307_1, arg308_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg307_1
        del arg308_1
        del arg620_1
        del arg621_1
        # Source Nodes: [x_395, x_397, x_399], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf207 = extern_kernels.convolution(buf206, arg309_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf207, (8, 2048, 7, 7), (100352, 49, 7, 1))
        del arg309_1
        del buf206
        buf208 = empty_strided((8, 2048, 1, 1), (2048, 1, 16384, 16384), device='cuda', dtype=torch.float32)
        buf209 = reinterpret_tensor(buf208, (8, 2048, 1, 1), (2048, 1, 1, 1), 0); del buf208  # reuse
        # Source Nodes: [x_400, x_401, x_404, x_405], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_18.run(buf209, buf207, arg623_1, arg624_1, arg310_1, arg311_1, buf202, 16384, 49, grid=grid(16384), stream=stream0)
        del arg310_1
        del arg311_1
        del arg623_1
        del arg624_1
        del buf202
        del buf207
        buf210 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_408], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg313_1, reinterpret_tensor(buf209, (8, 2048), (2048, 1), 0), reinterpret_tensor(arg312_1, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf210)
        del arg312_1
        del arg313_1
        return (buf210, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((512, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((1024, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((2048, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((2048, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((4096, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((4096, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((2048, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((4096, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((4096, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((2048, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((4096, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((4096, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((2048, 4096, 1, 1), (4096, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg317_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg320_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg323_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg326_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg329_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg332_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg335_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg338_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg341_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg344_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg347_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg350_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg353_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg356_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg359_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg362_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg365_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg368_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg371_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg374_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg377_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg380_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg383_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg386_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg389_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg392_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg395_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg398_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg401_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg404_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg407_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg410_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg413_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg416_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg419_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg422_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg425_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg428_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg431_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg434_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg437_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg440_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg443_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg446_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg449_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg452_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg455_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg458_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg461_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg464_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg467_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg470_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg473_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg474_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg475_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg476_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg477_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg478_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg479_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg480_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg481_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg482_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg483_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg484_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg485_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg486_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg487_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg488_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg489_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg490_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg491_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg492_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg493_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg494_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg495_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg496_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg497_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg498_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg499_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg500_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg501_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg502_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg503_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg504_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg505_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg506_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg507_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg508_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg509_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg510_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg511_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg512_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg513_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg514_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg515_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg516_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg517_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg518_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg519_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg520_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg521_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg522_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg523_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg524_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg525_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg526_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg527_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg528_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg529_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg530_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg531_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg532_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg533_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg534_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg535_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg536_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg537_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg538_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg539_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg540_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg541_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg542_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg543_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg544_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg545_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg546_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg547_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg548_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg549_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg550_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg551_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg552_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg553_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg554_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg555_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg556_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg557_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg558_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg559_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg560_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg561_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg562_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg563_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg564_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg565_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg566_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg567_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg568_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg569_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg570_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg571_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg572_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg573_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg574_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg575_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg576_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg577_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg578_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg579_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg580_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg581_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg582_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg583_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg584_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg585_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg586_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg587_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg588_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg589_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg590_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg591_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg592_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg593_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg594_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg595_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg596_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg597_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg598_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg599_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg600_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg601_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg602_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg603_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg604_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg605_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg606_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg607_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg608_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg609_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg610_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg611_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg612_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg613_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg614_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg615_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg616_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg617_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg618_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg619_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg620_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg621_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg622_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg623_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg624_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg625_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg626_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1, arg473_1, arg474_1, arg475_1, arg476_1, arg477_1, arg478_1, arg479_1, arg480_1, arg481_1, arg482_1, arg483_1, arg484_1, arg485_1, arg486_1, arg487_1, arg488_1, arg489_1, arg490_1, arg491_1, arg492_1, arg493_1, arg494_1, arg495_1, arg496_1, arg497_1, arg498_1, arg499_1, arg500_1, arg501_1, arg502_1, arg503_1, arg504_1, arg505_1, arg506_1, arg507_1, arg508_1, arg509_1, arg510_1, arg511_1, arg512_1, arg513_1, arg514_1, arg515_1, arg516_1, arg517_1, arg518_1, arg519_1, arg520_1, arg521_1, arg522_1, arg523_1, arg524_1, arg525_1, arg526_1, arg527_1, arg528_1, arg529_1, arg530_1, arg531_1, arg532_1, arg533_1, arg534_1, arg535_1, arg536_1, arg537_1, arg538_1, arg539_1, arg540_1, arg541_1, arg542_1, arg543_1, arg544_1, arg545_1, arg546_1, arg547_1, arg548_1, arg549_1, arg550_1, arg551_1, arg552_1, arg553_1, arg554_1, arg555_1, arg556_1, arg557_1, arg558_1, arg559_1, arg560_1, arg561_1, arg562_1, arg563_1, arg564_1, arg565_1, arg566_1, arg567_1, arg568_1, arg569_1, arg570_1, arg571_1, arg572_1, arg573_1, arg574_1, arg575_1, arg576_1, arg577_1, arg578_1, arg579_1, arg580_1, arg581_1, arg582_1, arg583_1, arg584_1, arg585_1, arg586_1, arg587_1, arg588_1, arg589_1, arg590_1, arg591_1, arg592_1, arg593_1, arg594_1, arg595_1, arg596_1, arg597_1, arg598_1, arg599_1, arg600_1, arg601_1, arg602_1, arg603_1, arg604_1, arg605_1, arg606_1, arg607_1, arg608_1, arg609_1, arg610_1, arg611_1, arg612_1, arg613_1, arg614_1, arg615_1, arg616_1, arg617_1, arg618_1, arg619_1, arg620_1, arg621_1, arg622_1, arg623_1, arg624_1, arg625_1, arg626_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('swsl_resnext101_32x16d', benchmark_compiled_module)
