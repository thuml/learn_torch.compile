
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


# kernel path: /tmp/torchinductor_youkaichao/tt/ctthslvflc74kcaw2no5sg52uk6vnxrxsr6a4mbzvzu7azvybxuo.py
# Source Nodes: [identity], Original ATen: [aten.max_pool2d_with_indices]
# identity => getitem, getitem_1
triton_poi_fused_max_pool2d_with_indices_1 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
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


# kernel path: /tmp/torchinductor_youkaichao/5w/c5wojlu5g63zdr6jjohtdwe2v52hytiohfaaa37djlgiljiuqafu.py
# Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_1 => add_3, mul_4, mul_5, sub_1
# out_2 => relu_1
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


# kernel path: /tmp/torchinductor_youkaichao/4c/c4ccyzpjrau7pntfjc7z4lkhfmogtqzxobugcgmn3u6qoxxsemto.py
# Source Nodes: [identity_1, identity_2, out_7, out_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# identity_1 => add_9, mul_13, mul_14, sub_4
# identity_2 => relu_3
# out_7 => add_7, mul_10, mul_11, sub_3
# out_8 => add_10
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
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
    tmp15 = tl.load(in_ptr5 + (x3), None)
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/6w/c6ws4bogeafet5vevittw2me7vxcaclsppw33ab52xahrjfomwow.py
# Source Nodes: [identity_3, out_17, out_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# identity_3 => relu_6
# out_17 => add_16, mul_22, mul_23, sub_7
# out_18 => add_17
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp15 = tl.load(in_ptr5 + (x3), None)
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
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bd/cbdppxpz4v4ejgidwhzarggsdlvowoe65l72xxros3z3hzahy5qu.py
# Source Nodes: [out_31, out_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_31 => add_26, mul_34, mul_35, sub_11
# out_32 => relu_10
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/pl/cpl3pn34adli5kkf6kria4suzzd7cgmyhpoguyx6gcrhmbhjrids.py
# Source Nodes: [out_34, out_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_34 => add_28, mul_37, mul_38, sub_12
# out_35 => relu_11
triton_poi_fused__native_batch_norm_legit_no_training_relu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_6', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/be/cbemv44le4xy4psvobvhpyezctfv4xt7g4qedkknmolftowakwex.py
# Source Nodes: [identity_5, identity_6, out_37, out_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# identity_5 => add_32, mul_43, mul_44, sub_14
# identity_6 => relu_12
# out_37 => add_30, mul_40, mul_41, sub_13
# out_38 => add_33
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
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
    tmp15 = tl.load(in_ptr5 + (x3), None)
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/e2/ce2y5x3m4agbfzvnwyv34iajyphbbiixlsz4rfjdc75tct5ok47g.py
# Source Nodes: [identity_7, out_47, out_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# identity_7 => relu_15
# out_47 => add_39, mul_52, mul_53, sub_17
# out_48 => add_40
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp15 = tl.load(in_ptr5 + (x3), None)
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
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ia/ciauqeiaswfutm3ccdnehhiduvzk6sgtnyggm4i2wx2pfyb45upn.py
# Source Nodes: [out_71, out_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_71 => add_56, mul_73, mul_74, sub_24
# out_72 => relu_22
triton_poi_fused__native_batch_norm_legit_no_training_relu_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_9', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/q2/cq2x22legb6f44l6bakd45m4s5onsh2upxu6pxw22u3uigndsrxg.py
# Source Nodes: [out_74, out_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_74 => add_58, mul_76, mul_77, sub_25
# out_75 => relu_23
triton_poi_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_10', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ep/cepizdimpnw4kjs6fqmvq63imr2yvl34tqhywyzvcizc3kkkg2gg.py
# Source Nodes: [identity_10, identity_11, out_77, out_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# identity_10 => add_62, mul_82, mul_83, sub_27
# identity_11 => relu_24
# out_77 => add_60, mul_79, mul_80, sub_26
# out_78 => add_63
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
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
    tmp15 = tl.load(in_ptr5 + (x3), None)
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/zj/czjmhgjrsl435hu4u7auxuy4gxn6bxw7srp3ip4fwps2yiiiqz7u.py
# Source Nodes: [identity_12, out_87, out_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# identity_12 => relu_27
# out_87 => add_69, mul_91, mul_92, sub_30
# out_88 => add_70
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tmp15 = tl.load(in_ptr5 + (x3), None)
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
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/34/c34fiygasoyrz6mdfax6a35npaokrotrgmjwf6m5vydxqyooz5lo.py
# Source Nodes: [out_131, out_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_131 => add_100, mul_130, mul_131, sub_43
# out_132 => relu_40
triton_poi_fused__native_batch_norm_legit_no_training_relu_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_13', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/4z/c4zjiqqu3g24u3mt2d4evwnvrs6cbar5akwrfbigbeqcppclpbbk.py
# Source Nodes: [out_134, out_135], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# out_134 => add_102, mul_133, mul_134, sub_44
# out_135 => relu_41
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 1024
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


# kernel path: /tmp/torchinductor_youkaichao/se/csedlb5tvfk3zwtrcefslfclj625rxs6fwfpcxfelqodflearwsz.py
# Source Nodes: [identity_17, identity_18, out_137, out_138], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# identity_17 => add_106, mul_139, mul_140, sub_46
# identity_18 => relu_42
# out_137 => add_104, mul_136, mul_137, sub_45
# out_138 => add_107
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
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
    tmp15 = tl.load(in_ptr5 + (x3), None)
    tmp16 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/cm/ccmqc7xwlbe6dlaoz6q52ss63ugd74peobm4ch6twq7yiomxqvcy.py
# Source Nodes: [identity_19, out_147, out_148], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
# identity_19 => relu_45
# out_147 => add_113, mul_148, mul_149, sub_49
# out_148 => add_114
triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
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
    tmp15 = tl.load(in_ptr5 + (x3), None)
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
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/i5/ci5euzpumzteak3zvuoi6wjsh7mbygs3ylqbp6zcaccwu7og5yxi.py
# Source Nodes: [out_157, out_158, x_7, x_8, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.relu, aten.threshold_backward, aten.view]
# out_157 => add_120, mul_157, mul_158, sub_52
# out_158 => add_121
# x_7 => relu_48
# x_8 => mean
# x_9 => view
triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_threshold_backward_view_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*i1', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_threshold_backward_view_17', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
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
    tmp18 = 0.0
    tmp19 = tmp17 <= tmp18
    tmp20 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp22 = tl.where(rmask, tmp20, 0)
    tmp23 = tl.sum(tmp22, 1)[:, None]
    tmp24 = 49.0
    tmp25 = tmp23 / tmp24
    tl.store(out_ptr1 + (r2 + (49*x3)), tmp19, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp25, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 7, 7), (147, 49, 7, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_6, (128, ), (1, ))
    assert_size_stride(primals_7, (128, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_10, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_11, (256, ), (1, ))
    assert_size_stride(primals_12, (256, ), (1, ))
    assert_size_stride(primals_13, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_14, (256, ), (1, ))
    assert_size_stride(primals_15, (256, ), (1, ))
    assert_size_stride(primals_16, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_17, (128, ), (1, ))
    assert_size_stride(primals_18, (128, ), (1, ))
    assert_size_stride(primals_19, (128, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_21, (128, ), (1, ))
    assert_size_stride(primals_22, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_23, (256, ), (1, ))
    assert_size_stride(primals_24, (256, ), (1, ))
    assert_size_stride(primals_25, (128, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_27, (128, ), (1, ))
    assert_size_stride(primals_28, (128, 4, 3, 3), (36, 9, 3, 1))
    assert_size_stride(primals_29, (128, ), (1, ))
    assert_size_stride(primals_30, (128, ), (1, ))
    assert_size_stride(primals_31, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_32, (256, ), (1, ))
    assert_size_stride(primals_33, (256, ), (1, ))
    assert_size_stride(primals_34, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_35, (256, ), (1, ))
    assert_size_stride(primals_36, (256, ), (1, ))
    assert_size_stride(primals_37, (256, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_38, (256, ), (1, ))
    assert_size_stride(primals_39, (256, ), (1, ))
    assert_size_stride(primals_40, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_41, (512, ), (1, ))
    assert_size_stride(primals_42, (512, ), (1, ))
    assert_size_stride(primals_43, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_44, (512, ), (1, ))
    assert_size_stride(primals_45, (512, ), (1, ))
    assert_size_stride(primals_46, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_47, (256, ), (1, ))
    assert_size_stride(primals_48, (256, ), (1, ))
    assert_size_stride(primals_49, (256, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_50, (256, ), (1, ))
    assert_size_stride(primals_51, (256, ), (1, ))
    assert_size_stride(primals_52, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_53, (512, ), (1, ))
    assert_size_stride(primals_54, (512, ), (1, ))
    assert_size_stride(primals_55, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_56, (256, ), (1, ))
    assert_size_stride(primals_57, (256, ), (1, ))
    assert_size_stride(primals_58, (256, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_59, (256, ), (1, ))
    assert_size_stride(primals_60, (256, ), (1, ))
    assert_size_stride(primals_61, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_62, (512, ), (1, ))
    assert_size_stride(primals_63, (512, ), (1, ))
    assert_size_stride(primals_64, (256, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_65, (256, ), (1, ))
    assert_size_stride(primals_66, (256, ), (1, ))
    assert_size_stride(primals_67, (256, 8, 3, 3), (72, 9, 3, 1))
    assert_size_stride(primals_68, (256, ), (1, ))
    assert_size_stride(primals_69, (256, ), (1, ))
    assert_size_stride(primals_70, (512, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_71, (512, ), (1, ))
    assert_size_stride(primals_72, (512, ), (1, ))
    assert_size_stride(primals_73, (512, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_74, (512, ), (1, ))
    assert_size_stride(primals_75, (512, ), (1, ))
    assert_size_stride(primals_76, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_77, (512, ), (1, ))
    assert_size_stride(primals_78, (512, ), (1, ))
    assert_size_stride(primals_79, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_80, (1024, ), (1, ))
    assert_size_stride(primals_81, (1024, ), (1, ))
    assert_size_stride(primals_82, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_83, (1024, ), (1, ))
    assert_size_stride(primals_84, (1024, ), (1, ))
    assert_size_stride(primals_85, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_86, (512, ), (1, ))
    assert_size_stride(primals_87, (512, ), (1, ))
    assert_size_stride(primals_88, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_89, (512, ), (1, ))
    assert_size_stride(primals_90, (512, ), (1, ))
    assert_size_stride(primals_91, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_92, (1024, ), (1, ))
    assert_size_stride(primals_93, (1024, ), (1, ))
    assert_size_stride(primals_94, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_95, (512, ), (1, ))
    assert_size_stride(primals_96, (512, ), (1, ))
    assert_size_stride(primals_97, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_98, (512, ), (1, ))
    assert_size_stride(primals_99, (512, ), (1, ))
    assert_size_stride(primals_100, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_101, (1024, ), (1, ))
    assert_size_stride(primals_102, (1024, ), (1, ))
    assert_size_stride(primals_103, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_104, (512, ), (1, ))
    assert_size_stride(primals_105, (512, ), (1, ))
    assert_size_stride(primals_106, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_107, (512, ), (1, ))
    assert_size_stride(primals_108, (512, ), (1, ))
    assert_size_stride(primals_109, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_110, (1024, ), (1, ))
    assert_size_stride(primals_111, (1024, ), (1, ))
    assert_size_stride(primals_112, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_113, (512, ), (1, ))
    assert_size_stride(primals_114, (512, ), (1, ))
    assert_size_stride(primals_115, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_116, (512, ), (1, ))
    assert_size_stride(primals_117, (512, ), (1, ))
    assert_size_stride(primals_118, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_119, (1024, ), (1, ))
    assert_size_stride(primals_120, (1024, ), (1, ))
    assert_size_stride(primals_121, (512, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_122, (512, ), (1, ))
    assert_size_stride(primals_123, (512, ), (1, ))
    assert_size_stride(primals_124, (512, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_125, (512, ), (1, ))
    assert_size_stride(primals_126, (512, ), (1, ))
    assert_size_stride(primals_127, (1024, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_128, (1024, ), (1, ))
    assert_size_stride(primals_129, (1024, ), (1, ))
    assert_size_stride(primals_130, (1024, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_131, (1024, ), (1, ))
    assert_size_stride(primals_132, (1024, ), (1, ))
    assert_size_stride(primals_133, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_134, (1024, ), (1, ))
    assert_size_stride(primals_135, (1024, ), (1, ))
    assert_size_stride(primals_136, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_137, (2048, ), (1, ))
    assert_size_stride(primals_138, (2048, ), (1, ))
    assert_size_stride(primals_139, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_140, (2048, ), (1, ))
    assert_size_stride(primals_141, (2048, ), (1, ))
    assert_size_stride(primals_142, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_143, (1024, ), (1, ))
    assert_size_stride(primals_144, (1024, ), (1, ))
    assert_size_stride(primals_145, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_146, (1024, ), (1, ))
    assert_size_stride(primals_147, (1024, ), (1, ))
    assert_size_stride(primals_148, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_149, (2048, ), (1, ))
    assert_size_stride(primals_150, (2048, ), (1, ))
    assert_size_stride(primals_151, (1024, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(primals_152, (1024, ), (1, ))
    assert_size_stride(primals_153, (1024, ), (1, ))
    assert_size_stride(primals_154, (1024, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_155, (1024, ), (1, ))
    assert_size_stride(primals_156, (1024, ), (1, ))
    assert_size_stride(primals_157, (2048, 1024, 1, 1), (1024, 1, 1, 1))
    assert_size_stride(primals_158, (2048, ), (1, ))
    assert_size_stride(primals_159, (2048, ), (1, ))
    assert_size_stride(primals_160, (1000, 2048), (2048, 1))
    assert_size_stride(primals_161, (1000, ), (1, ))
    assert_size_stride(primals_162, (64, ), (1, ))
    assert_size_stride(primals_163, (64, ), (1, ))
    assert_size_stride(primals_164, (), ())
    assert_size_stride(primals_165, (128, ), (1, ))
    assert_size_stride(primals_166, (128, ), (1, ))
    assert_size_stride(primals_167, (), ())
    assert_size_stride(primals_168, (128, ), (1, ))
    assert_size_stride(primals_169, (128, ), (1, ))
    assert_size_stride(primals_170, (), ())
    assert_size_stride(primals_171, (256, ), (1, ))
    assert_size_stride(primals_172, (256, ), (1, ))
    assert_size_stride(primals_173, (), ())
    assert_size_stride(primals_174, (256, ), (1, ))
    assert_size_stride(primals_175, (256, ), (1, ))
    assert_size_stride(primals_176, (), ())
    assert_size_stride(primals_177, (128, ), (1, ))
    assert_size_stride(primals_178, (128, ), (1, ))
    assert_size_stride(primals_179, (), ())
    assert_size_stride(primals_180, (128, ), (1, ))
    assert_size_stride(primals_181, (128, ), (1, ))
    assert_size_stride(primals_182, (), ())
    assert_size_stride(primals_183, (256, ), (1, ))
    assert_size_stride(primals_184, (256, ), (1, ))
    assert_size_stride(primals_185, (), ())
    assert_size_stride(primals_186, (128, ), (1, ))
    assert_size_stride(primals_187, (128, ), (1, ))
    assert_size_stride(primals_188, (), ())
    assert_size_stride(primals_189, (128, ), (1, ))
    assert_size_stride(primals_190, (128, ), (1, ))
    assert_size_stride(primals_191, (), ())
    assert_size_stride(primals_192, (256, ), (1, ))
    assert_size_stride(primals_193, (256, ), (1, ))
    assert_size_stride(primals_194, (), ())
    assert_size_stride(primals_195, (256, ), (1, ))
    assert_size_stride(primals_196, (256, ), (1, ))
    assert_size_stride(primals_197, (), ())
    assert_size_stride(primals_198, (256, ), (1, ))
    assert_size_stride(primals_199, (256, ), (1, ))
    assert_size_stride(primals_200, (), ())
    assert_size_stride(primals_201, (512, ), (1, ))
    assert_size_stride(primals_202, (512, ), (1, ))
    assert_size_stride(primals_203, (), ())
    assert_size_stride(primals_204, (512, ), (1, ))
    assert_size_stride(primals_205, (512, ), (1, ))
    assert_size_stride(primals_206, (), ())
    assert_size_stride(primals_207, (256, ), (1, ))
    assert_size_stride(primals_208, (256, ), (1, ))
    assert_size_stride(primals_209, (), ())
    assert_size_stride(primals_210, (256, ), (1, ))
    assert_size_stride(primals_211, (256, ), (1, ))
    assert_size_stride(primals_212, (), ())
    assert_size_stride(primals_213, (512, ), (1, ))
    assert_size_stride(primals_214, (512, ), (1, ))
    assert_size_stride(primals_215, (), ())
    assert_size_stride(primals_216, (256, ), (1, ))
    assert_size_stride(primals_217, (256, ), (1, ))
    assert_size_stride(primals_218, (), ())
    assert_size_stride(primals_219, (256, ), (1, ))
    assert_size_stride(primals_220, (256, ), (1, ))
    assert_size_stride(primals_221, (), ())
    assert_size_stride(primals_222, (512, ), (1, ))
    assert_size_stride(primals_223, (512, ), (1, ))
    assert_size_stride(primals_224, (), ())
    assert_size_stride(primals_225, (256, ), (1, ))
    assert_size_stride(primals_226, (256, ), (1, ))
    assert_size_stride(primals_227, (), ())
    assert_size_stride(primals_228, (256, ), (1, ))
    assert_size_stride(primals_229, (256, ), (1, ))
    assert_size_stride(primals_230, (), ())
    assert_size_stride(primals_231, (512, ), (1, ))
    assert_size_stride(primals_232, (512, ), (1, ))
    assert_size_stride(primals_233, (), ())
    assert_size_stride(primals_234, (512, ), (1, ))
    assert_size_stride(primals_235, (512, ), (1, ))
    assert_size_stride(primals_236, (), ())
    assert_size_stride(primals_237, (512, ), (1, ))
    assert_size_stride(primals_238, (512, ), (1, ))
    assert_size_stride(primals_239, (), ())
    assert_size_stride(primals_240, (1024, ), (1, ))
    assert_size_stride(primals_241, (1024, ), (1, ))
    assert_size_stride(primals_242, (), ())
    assert_size_stride(primals_243, (1024, ), (1, ))
    assert_size_stride(primals_244, (1024, ), (1, ))
    assert_size_stride(primals_245, (), ())
    assert_size_stride(primals_246, (512, ), (1, ))
    assert_size_stride(primals_247, (512, ), (1, ))
    assert_size_stride(primals_248, (), ())
    assert_size_stride(primals_249, (512, ), (1, ))
    assert_size_stride(primals_250, (512, ), (1, ))
    assert_size_stride(primals_251, (), ())
    assert_size_stride(primals_252, (1024, ), (1, ))
    assert_size_stride(primals_253, (1024, ), (1, ))
    assert_size_stride(primals_254, (), ())
    assert_size_stride(primals_255, (512, ), (1, ))
    assert_size_stride(primals_256, (512, ), (1, ))
    assert_size_stride(primals_257, (), ())
    assert_size_stride(primals_258, (512, ), (1, ))
    assert_size_stride(primals_259, (512, ), (1, ))
    assert_size_stride(primals_260, (), ())
    assert_size_stride(primals_261, (1024, ), (1, ))
    assert_size_stride(primals_262, (1024, ), (1, ))
    assert_size_stride(primals_263, (), ())
    assert_size_stride(primals_264, (512, ), (1, ))
    assert_size_stride(primals_265, (512, ), (1, ))
    assert_size_stride(primals_266, (), ())
    assert_size_stride(primals_267, (512, ), (1, ))
    assert_size_stride(primals_268, (512, ), (1, ))
    assert_size_stride(primals_269, (), ())
    assert_size_stride(primals_270, (1024, ), (1, ))
    assert_size_stride(primals_271, (1024, ), (1, ))
    assert_size_stride(primals_272, (), ())
    assert_size_stride(primals_273, (512, ), (1, ))
    assert_size_stride(primals_274, (512, ), (1, ))
    assert_size_stride(primals_275, (), ())
    assert_size_stride(primals_276, (512, ), (1, ))
    assert_size_stride(primals_277, (512, ), (1, ))
    assert_size_stride(primals_278, (), ())
    assert_size_stride(primals_279, (1024, ), (1, ))
    assert_size_stride(primals_280, (1024, ), (1, ))
    assert_size_stride(primals_281, (), ())
    assert_size_stride(primals_282, (512, ), (1, ))
    assert_size_stride(primals_283, (512, ), (1, ))
    assert_size_stride(primals_284, (), ())
    assert_size_stride(primals_285, (512, ), (1, ))
    assert_size_stride(primals_286, (512, ), (1, ))
    assert_size_stride(primals_287, (), ())
    assert_size_stride(primals_288, (1024, ), (1, ))
    assert_size_stride(primals_289, (1024, ), (1, ))
    assert_size_stride(primals_290, (), ())
    assert_size_stride(primals_291, (1024, ), (1, ))
    assert_size_stride(primals_292, (1024, ), (1, ))
    assert_size_stride(primals_293, (), ())
    assert_size_stride(primals_294, (1024, ), (1, ))
    assert_size_stride(primals_295, (1024, ), (1, ))
    assert_size_stride(primals_296, (), ())
    assert_size_stride(primals_297, (2048, ), (1, ))
    assert_size_stride(primals_298, (2048, ), (1, ))
    assert_size_stride(primals_299, (), ())
    assert_size_stride(primals_300, (2048, ), (1, ))
    assert_size_stride(primals_301, (2048, ), (1, ))
    assert_size_stride(primals_302, (), ())
    assert_size_stride(primals_303, (1024, ), (1, ))
    assert_size_stride(primals_304, (1024, ), (1, ))
    assert_size_stride(primals_305, (), ())
    assert_size_stride(primals_306, (1024, ), (1, ))
    assert_size_stride(primals_307, (1024, ), (1, ))
    assert_size_stride(primals_308, (), ())
    assert_size_stride(primals_309, (2048, ), (1, ))
    assert_size_stride(primals_310, (2048, ), (1, ))
    assert_size_stride(primals_311, (), ())
    assert_size_stride(primals_312, (1024, ), (1, ))
    assert_size_stride(primals_313, (1024, ), (1, ))
    assert_size_stride(primals_314, (), ())
    assert_size_stride(primals_315, (1024, ), (1, ))
    assert_size_stride(primals_316, (1024, ), (1, ))
    assert_size_stride(primals_317, (), ())
    assert_size_stride(primals_318, (2048, ), (1, ))
    assert_size_stride(primals_319, (2048, ), (1, ))
    assert_size_stride(primals_320, (), ())
    assert_size_stride(primals_321, (4, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_321, primals_1, stride=(2, 2), padding=(3, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 112, 112), (802816, 12544, 112, 1))
        buf1 = empty((4, 64, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1, x_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf0, primals_162, primals_163, primals_2, primals_3, buf1, 3211264, grid=grid(3211264), stream=stream0)
        del primals_3
        buf2 = empty((4, 64, 56, 56), device='cuda', dtype=torch.float32)
        buf3 = empty((4, 64, 56, 56), device='cuda', dtype=torch.int64)
        # Source Nodes: [identity], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_1.run(buf1, buf2, buf3, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [out], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf2, primals_4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 128, 56, 56), (401408, 3136, 56, 1))
        buf5 = empty((4, 128, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf4, primals_165, primals_166, primals_5, primals_6, buf5, 1605632, grid=grid(1605632), stream=stream0)
        del primals_6
        # Source Nodes: [out_3], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf6, (4, 128, 56, 56), (401408, 3136, 56, 1))
        buf7 = empty((4, 128, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_4, out_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf6, primals_168, primals_169, primals_8, primals_9, buf7, 1605632, grid=grid(1605632), stream=stream0)
        del primals_9
        # Source Nodes: [out_6], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_10, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 256, 56, 56), (802816, 3136, 56, 1))
        # Source Nodes: [getattr_l__mod___layer1___0___downsample_0], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf2, primals_13, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 256, 56, 56), (802816, 3136, 56, 1))
        buf10 = empty((4, 256, 56, 56), device='cuda', dtype=torch.float32)
        buf11 = buf10; del buf10  # reuse
        # Source Nodes: [identity_1, identity_2, out_7, out_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_3.run(buf11, buf8, primals_171, primals_172, primals_11, primals_12, buf9, primals_174, primals_175, primals_14, primals_15, 3211264, grid=grid(3211264), stream=stream0)
        del primals_12
        del primals_15
        # Source Nodes: [out_10], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_16, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 128, 56, 56), (401408, 3136, 56, 1))
        buf13 = empty((4, 128, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_11, out_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf12, primals_177, primals_178, primals_17, primals_18, buf13, 1605632, grid=grid(1605632), stream=stream0)
        del primals_18
        # Source Nodes: [out_13], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_19, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf14, (4, 128, 56, 56), (401408, 3136, 56, 1))
        buf15 = empty((4, 128, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_14, out_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf14, primals_180, primals_181, primals_20, primals_21, buf15, 1605632, grid=grid(1605632), stream=stream0)
        del primals_21
        # Source Nodes: [out_16], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_22, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 256, 56, 56), (802816, 3136, 56, 1))
        buf17 = empty((4, 256, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_3, out_17, out_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4.run(buf16, primals_183, primals_184, primals_23, primals_24, buf11, buf17, 3211264, grid=grid(3211264), stream=stream0)
        del primals_24
        # Source Nodes: [out_20], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, primals_25, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 128, 56, 56), (401408, 3136, 56, 1))
        buf19 = empty((4, 128, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_21, out_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf18, primals_186, primals_187, primals_26, primals_27, buf19, 1605632, grid=grid(1605632), stream=stream0)
        del primals_27
        # Source Nodes: [out_23], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, primals_28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf20, (4, 128, 56, 56), (401408, 3136, 56, 1))
        buf21 = empty((4, 128, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_24, out_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf20, primals_189, primals_190, primals_29, primals_30, buf21, 1605632, grid=grid(1605632), stream=stream0)
        del primals_30
        # Source Nodes: [out_26], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_31, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 256, 56, 56), (802816, 3136, 56, 1))
        buf23 = empty((4, 256, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_4, out_27, out_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_4.run(buf22, primals_192, primals_193, primals_32, primals_33, buf17, buf23, 3211264, grid=grid(3211264), stream=stream0)
        del primals_33
        # Source Nodes: [out_30], Original ATen: [aten.convolution]
        buf24 = extern_kernels.convolution(buf23, primals_34, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 256, 56, 56), (802816, 3136, 56, 1))
        buf25 = empty((4, 256, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_31, out_32], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf24, primals_195, primals_196, primals_35, primals_36, buf25, 3211264, grid=grid(3211264), stream=stream0)
        del primals_36
        # Source Nodes: [out_33], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, primals_37, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf26, (4, 256, 28, 28), (200704, 784, 28, 1))
        buf27 = empty((4, 256, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_34, out_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf26, primals_198, primals_199, primals_38, primals_39, buf27, 802816, grid=grid(802816), stream=stream0)
        del primals_39
        # Source Nodes: [out_36], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_40, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 512, 28, 28), (401408, 784, 28, 1))
        # Source Nodes: [getattr_l__mod___layer2___0___downsample_0], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf23, primals_43, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 512, 28, 28), (401408, 784, 28, 1))
        buf30 = empty((4, 512, 28, 28), device='cuda', dtype=torch.float32)
        buf31 = buf30; del buf30  # reuse
        # Source Nodes: [identity_5, identity_6, out_37, out_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_7.run(buf31, buf28, primals_201, primals_202, primals_41, primals_42, buf29, primals_204, primals_205, primals_44, primals_45, 1605632, grid=grid(1605632), stream=stream0)
        del primals_42
        del primals_45
        # Source Nodes: [out_40], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_46, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 256, 28, 28), (200704, 784, 28, 1))
        buf33 = empty((4, 256, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_41, out_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf32, primals_207, primals_208, primals_47, primals_48, buf33, 802816, grid=grid(802816), stream=stream0)
        del primals_48
        # Source Nodes: [out_43], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, primals_49, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf34, (4, 256, 28, 28), (200704, 784, 28, 1))
        buf35 = empty((4, 256, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_44, out_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf34, primals_210, primals_211, primals_50, primals_51, buf35, 802816, grid=grid(802816), stream=stream0)
        del primals_51
        # Source Nodes: [out_46], Original ATen: [aten.convolution]
        buf36 = extern_kernels.convolution(buf35, primals_52, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 512, 28, 28), (401408, 784, 28, 1))
        buf37 = empty((4, 512, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_7, out_47, out_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8.run(buf36, primals_213, primals_214, primals_53, primals_54, buf31, buf37, 1605632, grid=grid(1605632), stream=stream0)
        del primals_54
        # Source Nodes: [out_50], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, primals_55, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 256, 28, 28), (200704, 784, 28, 1))
        buf39 = empty((4, 256, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_51, out_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf38, primals_216, primals_217, primals_56, primals_57, buf39, 802816, grid=grid(802816), stream=stream0)
        del primals_57
        # Source Nodes: [out_53], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, primals_58, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf40, (4, 256, 28, 28), (200704, 784, 28, 1))
        buf41 = empty((4, 256, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_54, out_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf40, primals_219, primals_220, primals_59, primals_60, buf41, 802816, grid=grid(802816), stream=stream0)
        del primals_60
        # Source Nodes: [out_56], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_61, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 512, 28, 28), (401408, 784, 28, 1))
        buf43 = empty((4, 512, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_8, out_57, out_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8.run(buf42, primals_222, primals_223, primals_62, primals_63, buf37, buf43, 1605632, grid=grid(1605632), stream=stream0)
        del primals_63
        # Source Nodes: [out_60], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, primals_64, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 256, 28, 28), (200704, 784, 28, 1))
        buf45 = empty((4, 256, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_61, out_62], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf44, primals_225, primals_226, primals_65, primals_66, buf45, 802816, grid=grid(802816), stream=stream0)
        del primals_66
        # Source Nodes: [out_63], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, primals_67, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf46, (4, 256, 28, 28), (200704, 784, 28, 1))
        buf47 = empty((4, 256, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_64, out_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf46, primals_228, primals_229, primals_68, primals_69, buf47, 802816, grid=grid(802816), stream=stream0)
        del primals_69
        # Source Nodes: [out_66], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, primals_70, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 512, 28, 28), (401408, 784, 28, 1))
        buf49 = empty((4, 512, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_9, out_67, out_68], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_8.run(buf48, primals_231, primals_232, primals_71, primals_72, buf43, buf49, 1605632, grid=grid(1605632), stream=stream0)
        del primals_72
        # Source Nodes: [out_70], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_73, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 512, 28, 28), (401408, 784, 28, 1))
        buf51 = empty((4, 512, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_71, out_72], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf50, primals_234, primals_235, primals_74, primals_75, buf51, 1605632, grid=grid(1605632), stream=stream0)
        del primals_75
        # Source Nodes: [out_73], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_76, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf52, (4, 512, 14, 14), (100352, 196, 14, 1))
        buf53 = empty((4, 512, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_74, out_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf52, primals_237, primals_238, primals_77, primals_78, buf53, 401408, grid=grid(401408), stream=stream0)
        del primals_78
        # Source Nodes: [out_76], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, primals_79, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 1024, 14, 14), (200704, 196, 14, 1))
        # Source Nodes: [getattr_l__mod___layer3___0___downsample_0], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf49, primals_82, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 1024, 14, 14), (200704, 196, 14, 1))
        buf56 = empty((4, 1024, 14, 14), device='cuda', dtype=torch.float32)
        buf57 = buf56; del buf56  # reuse
        # Source Nodes: [identity_10, identity_11, out_77, out_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_11.run(buf57, buf54, primals_240, primals_241, primals_80, primals_81, buf55, primals_243, primals_244, primals_83, primals_84, 802816, grid=grid(802816), stream=stream0)
        del primals_81
        del primals_84
        # Source Nodes: [out_80], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_85, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 512, 14, 14), (100352, 196, 14, 1))
        buf59 = empty((4, 512, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_81, out_82], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf58, primals_246, primals_247, primals_86, primals_87, buf59, 401408, grid=grid(401408), stream=stream0)
        del primals_87
        # Source Nodes: [out_83], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, primals_88, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf60, (4, 512, 14, 14), (100352, 196, 14, 1))
        buf61 = empty((4, 512, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_84, out_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf60, primals_249, primals_250, primals_89, primals_90, buf61, 401408, grid=grid(401408), stream=stream0)
        del primals_90
        # Source Nodes: [out_86], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_91, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 1024, 14, 14), (200704, 196, 14, 1))
        buf63 = empty((4, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_12, out_87, out_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf62, primals_252, primals_253, primals_92, primals_93, buf57, buf63, 802816, grid=grid(802816), stream=stream0)
        del primals_93
        # Source Nodes: [out_90], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_94, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 512, 14, 14), (100352, 196, 14, 1))
        buf65 = empty((4, 512, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_91, out_92], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf64, primals_255, primals_256, primals_95, primals_96, buf65, 401408, grid=grid(401408), stream=stream0)
        del primals_96
        # Source Nodes: [out_93], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf66, (4, 512, 14, 14), (100352, 196, 14, 1))
        buf67 = empty((4, 512, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_94, out_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf66, primals_258, primals_259, primals_98, primals_99, buf67, 401408, grid=grid(401408), stream=stream0)
        del primals_99
        # Source Nodes: [out_96], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_100, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 1024, 14, 14), (200704, 196, 14, 1))
        buf69 = empty((4, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_13, out_97, out_98], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf68, primals_261, primals_262, primals_101, primals_102, buf63, buf69, 802816, grid=grid(802816), stream=stream0)
        del primals_102
        # Source Nodes: [out_100], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, primals_103, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (4, 512, 14, 14), (100352, 196, 14, 1))
        buf71 = empty((4, 512, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_101, out_102], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf70, primals_264, primals_265, primals_104, primals_105, buf71, 401408, grid=grid(401408), stream=stream0)
        del primals_105
        # Source Nodes: [out_103], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, primals_106, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf72, (4, 512, 14, 14), (100352, 196, 14, 1))
        buf73 = empty((4, 512, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_104, out_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf72, primals_267, primals_268, primals_107, primals_108, buf73, 401408, grid=grid(401408), stream=stream0)
        del primals_108
        # Source Nodes: [out_106], Original ATen: [aten.convolution]
        buf74 = extern_kernels.convolution(buf73, primals_109, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 1024, 14, 14), (200704, 196, 14, 1))
        buf75 = empty((4, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_14, out_107, out_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf74, primals_270, primals_271, primals_110, primals_111, buf69, buf75, 802816, grid=grid(802816), stream=stream0)
        del primals_111
        # Source Nodes: [out_110], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, primals_112, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (4, 512, 14, 14), (100352, 196, 14, 1))
        buf77 = empty((4, 512, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_111, out_112], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf76, primals_273, primals_274, primals_113, primals_114, buf77, 401408, grid=grid(401408), stream=stream0)
        del primals_114
        # Source Nodes: [out_113], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, primals_115, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf78, (4, 512, 14, 14), (100352, 196, 14, 1))
        buf79 = empty((4, 512, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_114, out_115], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf78, primals_276, primals_277, primals_116, primals_117, buf79, 401408, grid=grid(401408), stream=stream0)
        del primals_117
        # Source Nodes: [out_116], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_118, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 1024, 14, 14), (200704, 196, 14, 1))
        buf81 = empty((4, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_15, out_117, out_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf80, primals_279, primals_280, primals_119, primals_120, buf75, buf81, 802816, grid=grid(802816), stream=stream0)
        del primals_120
        # Source Nodes: [out_120], Original ATen: [aten.convolution]
        buf82 = extern_kernels.convolution(buf81, primals_121, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf82, (4, 512, 14, 14), (100352, 196, 14, 1))
        buf83 = empty((4, 512, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_121, out_122], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf82, primals_282, primals_283, primals_122, primals_123, buf83, 401408, grid=grid(401408), stream=stream0)
        del primals_123
        # Source Nodes: [out_123], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, primals_124, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf84, (4, 512, 14, 14), (100352, 196, 14, 1))
        buf85 = empty((4, 512, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_124, out_125], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf84, primals_285, primals_286, primals_125, primals_126, buf85, 401408, grid=grid(401408), stream=stream0)
        del primals_126
        # Source Nodes: [out_126], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_127, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (4, 1024, 14, 14), (200704, 196, 14, 1))
        buf87 = empty((4, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_16, out_127, out_128], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_12.run(buf86, primals_288, primals_289, primals_128, primals_129, buf81, buf87, 802816, grid=grid(802816), stream=stream0)
        del primals_129
        # Source Nodes: [out_130], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, primals_130, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 1024, 14, 14), (200704, 196, 14, 1))
        buf89 = empty((4, 1024, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_131, out_132], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf88, primals_291, primals_292, primals_131, primals_132, buf89, 802816, grid=grid(802816), stream=stream0)
        del primals_132
        # Source Nodes: [out_133], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, primals_133, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf90, (4, 1024, 7, 7), (50176, 49, 7, 1))
        buf91 = empty((4, 1024, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_134, out_135], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf90, primals_294, primals_295, primals_134, primals_135, buf91, 200704, grid=grid(200704), stream=stream0)
        del primals_135
        # Source Nodes: [out_136], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, primals_136, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 2048, 7, 7), (100352, 49, 7, 1))
        # Source Nodes: [getattr_l__mod___layer4___0___downsample_0], Original ATen: [aten.convolution]
        buf93 = extern_kernels.convolution(buf87, primals_139, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf93, (4, 2048, 7, 7), (100352, 49, 7, 1))
        buf94 = empty((4, 2048, 7, 7), device='cuda', dtype=torch.float32)
        buf95 = buf94; del buf94  # reuse
        # Source Nodes: [identity_17, identity_18, out_137, out_138], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_15.run(buf95, buf92, primals_297, primals_298, primals_137, primals_138, buf93, primals_300, primals_301, primals_140, primals_141, 401408, grid=grid(401408), stream=stream0)
        del primals_138
        del primals_141
        # Source Nodes: [out_140], Original ATen: [aten.convolution]
        buf96 = extern_kernels.convolution(buf95, primals_142, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 1024, 7, 7), (50176, 49, 7, 1))
        buf97 = empty((4, 1024, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_141, out_142], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf96, primals_303, primals_304, primals_143, primals_144, buf97, 200704, grid=grid(200704), stream=stream0)
        del primals_144
        # Source Nodes: [out_143], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, primals_145, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf98, (4, 1024, 7, 7), (50176, 49, 7, 1))
        buf99 = empty((4, 1024, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_144, out_145], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf98, primals_306, primals_307, primals_146, primals_147, buf99, 200704, grid=grid(200704), stream=stream0)
        del primals_147
        # Source Nodes: [out_146], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, primals_148, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (4, 2048, 7, 7), (100352, 49, 7, 1))
        buf101 = empty((4, 2048, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [identity_19, out_147, out_148], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_add_relu_16.run(buf100, primals_309, primals_310, primals_149, primals_150, buf95, buf101, 401408, grid=grid(401408), stream=stream0)
        del primals_150
        # Source Nodes: [out_150], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, primals_151, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (4, 1024, 7, 7), (50176, 49, 7, 1))
        buf103 = empty((4, 1024, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_151, out_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf102, primals_312, primals_313, primals_152, primals_153, buf103, 200704, grid=grid(200704), stream=stream0)
        del primals_153
        # Source Nodes: [out_153], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, primals_154, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf104, (4, 1024, 7, 7), (50176, 49, 7, 1))
        buf105 = empty((4, 1024, 7, 7), device='cuda', dtype=torch.float32)
        # Source Nodes: [out_154, out_155], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf104, primals_315, primals_316, primals_155, primals_156, buf105, 200704, grid=grid(200704), stream=stream0)
        del primals_156
        # Source Nodes: [out_156], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, primals_157, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (4, 2048, 7, 7), (100352, 49, 7, 1))
        buf111 = empty((4, 2048, 7, 7), device='cuda', dtype=torch.bool)
        buf108 = empty_strided((4, 2048, 1, 1), (2048, 1, 8192, 8192), device='cuda', dtype=torch.float32)
        buf109 = reinterpret_tensor(buf108, (4, 2048), (2048, 1), 0); del buf108  # reuse
        # Source Nodes: [out_157, out_158, x_7, x_8, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.mean, aten.relu, aten.threshold_backward, aten.view]
        triton_per_fused__native_batch_norm_legit_no_training_add_mean_relu_threshold_backward_view_17.run(buf109, buf106, primals_318, primals_319, primals_158, primals_159, buf101, buf111, 8192, 49, grid=grid(8192), stream=stream0)
        del primals_159
        buf110 = empty((4, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_161, buf109, reinterpret_tensor(primals_160, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf110)
        del primals_161
        return (buf110, primals_1, primals_2, primals_4, primals_5, primals_7, primals_8, primals_10, primals_11, primals_13, primals_14, primals_16, primals_17, primals_19, primals_20, primals_22, primals_23, primals_25, primals_26, primals_28, primals_29, primals_31, primals_32, primals_34, primals_35, primals_37, primals_38, primals_40, primals_41, primals_43, primals_44, primals_46, primals_47, primals_49, primals_50, primals_52, primals_53, primals_55, primals_56, primals_58, primals_59, primals_61, primals_62, primals_64, primals_65, primals_67, primals_68, primals_70, primals_71, primals_73, primals_74, primals_76, primals_77, primals_79, primals_80, primals_82, primals_83, primals_85, primals_86, primals_88, primals_89, primals_91, primals_92, primals_94, primals_95, primals_97, primals_98, primals_100, primals_101, primals_103, primals_104, primals_106, primals_107, primals_109, primals_110, primals_112, primals_113, primals_115, primals_116, primals_118, primals_119, primals_121, primals_122, primals_124, primals_125, primals_127, primals_128, primals_130, primals_131, primals_133, primals_134, primals_136, primals_137, primals_139, primals_140, primals_142, primals_143, primals_145, primals_146, primals_148, primals_149, primals_151, primals_152, primals_154, primals_155, primals_157, primals_158, primals_162, primals_163, primals_165, primals_166, primals_168, primals_169, primals_171, primals_172, primals_174, primals_175, primals_177, primals_178, primals_180, primals_181, primals_183, primals_184, primals_186, primals_187, primals_189, primals_190, primals_192, primals_193, primals_195, primals_196, primals_198, primals_199, primals_201, primals_202, primals_204, primals_205, primals_207, primals_208, primals_210, primals_211, primals_213, primals_214, primals_216, primals_217, primals_219, primals_220, primals_222, primals_223, primals_225, primals_226, primals_228, primals_229, primals_231, primals_232, primals_234, primals_235, primals_237, primals_238, primals_240, primals_241, primals_243, primals_244, primals_246, primals_247, primals_249, primals_250, primals_252, primals_253, primals_255, primals_256, primals_258, primals_259, primals_261, primals_262, primals_264, primals_265, primals_267, primals_268, primals_270, primals_271, primals_273, primals_274, primals_276, primals_277, primals_279, primals_280, primals_282, primals_283, primals_285, primals_286, primals_288, primals_289, primals_291, primals_292, primals_294, primals_295, primals_297, primals_298, primals_300, primals_301, primals_303, primals_304, primals_306, primals_307, primals_309, primals_310, primals_312, primals_313, primals_315, primals_316, primals_318, primals_319, primals_321, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf11, buf12, buf13, buf14, buf15, buf16, buf17, buf18, buf19, buf20, buf21, buf22, buf23, buf24, buf25, buf26, buf27, buf28, buf29, buf31, buf32, buf33, buf34, buf35, buf36, buf37, buf38, buf39, buf40, buf41, buf42, buf43, buf44, buf45, buf46, buf47, buf48, buf49, buf50, buf51, buf52, buf53, buf54, buf55, buf57, buf58, buf59, buf60, buf61, buf62, buf63, buf64, buf65, buf66, buf67, buf68, buf69, buf70, buf71, buf72, buf73, buf74, buf75, buf76, buf77, buf78, buf79, buf80, buf81, buf82, buf83, buf84, buf85, buf86, buf87, buf88, buf89, buf90, buf91, buf92, buf93, buf95, buf96, buf97, buf98, buf99, buf100, buf101, buf102, buf103, buf104, buf105, buf106, buf109, reinterpret_tensor(primals_160, (1000, 2048), (2048, 1), 0), buf111, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 7, 7), (147, 49, 7, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((128, 4, 3, 3), (36, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((256, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((256, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((256, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((256, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((256, 8, 3, 3), (72, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((512, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((512, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((512, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((512, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((1024, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((1024, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((1024, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((1024, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((2048, 1024, 1, 1), (1024, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_165 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_168 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_171 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_174 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_177 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_180 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_183 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_186 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_189 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_192 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_195 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_198 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_199 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_200 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_201 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_202 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_203 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_204 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_205 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_206 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_207 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_208 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_209 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_210 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_211 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_212 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_213 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_214 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_215 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_216 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_217 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_218 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_219 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_220 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_221 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_222 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_223 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_224 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_225 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_226 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_227 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_228 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_229 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_230 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_231 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_232 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_233 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_234 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_235 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_236 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_237 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_238 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_239 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_240 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_241 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_242 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_243 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_244 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_245 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_246 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_247 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_248 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_249 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_250 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_251 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_252 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_253 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_254 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_255 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_256 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_257 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_258 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_259 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_260 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_261 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_262 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_263 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_264 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_265 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_266 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_267 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_268 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_269 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_270 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_271 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_272 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_273 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_274 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_275 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_276 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_277 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_278 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_279 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_280 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_281 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_282 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_283 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_284 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_285 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_286 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_287 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_288 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_289 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_290 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_291 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_292 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_293 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_294 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_295 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_296 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_297 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_298 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_299 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_300 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_301 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_302 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_303 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_304 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_305 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_306 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_307 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_308 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_309 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_310 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_311 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_312 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_313 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_314 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_315 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_316 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_317 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_318 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_319 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_320 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    primals_321 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, primals_199, primals_200, primals_201, primals_202, primals_203, primals_204, primals_205, primals_206, primals_207, primals_208, primals_209, primals_210, primals_211, primals_212, primals_213, primals_214, primals_215, primals_216, primals_217, primals_218, primals_219, primals_220, primals_221, primals_222, primals_223, primals_224, primals_225, primals_226, primals_227, primals_228, primals_229, primals_230, primals_231, primals_232, primals_233, primals_234, primals_235, primals_236, primals_237, primals_238, primals_239, primals_240, primals_241, primals_242, primals_243, primals_244, primals_245, primals_246, primals_247, primals_248, primals_249, primals_250, primals_251, primals_252, primals_253, primals_254, primals_255, primals_256, primals_257, primals_258, primals_259, primals_260, primals_261, primals_262, primals_263, primals_264, primals_265, primals_266, primals_267, primals_268, primals_269, primals_270, primals_271, primals_272, primals_273, primals_274, primals_275, primals_276, primals_277, primals_278, primals_279, primals_280, primals_281, primals_282, primals_283, primals_284, primals_285, primals_286, primals_287, primals_288, primals_289, primals_290, primals_291, primals_292, primals_293, primals_294, primals_295, primals_296, primals_297, primals_298, primals_299, primals_300, primals_301, primals_302, primals_303, primals_304, primals_305, primals_306, primals_307, primals_308, primals_309, primals_310, primals_311, primals_312, primals_313, primals_314, primals_315, primals_316, primals_317, primals_318, primals_319, primals_320, primals_321]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('resnext50_32x4d', benchmark_compiled_module)
