
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


# kernel path: /tmp/torchinductor_youkaichao/tl/ctl54c3o2pqmlnegfgededxojug6ulfcvyczwifym64w5g2yq65b.py
# Source Nodes: [cat_11, x_11, x_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_11 => cat
# x_11 => add_5, mul_7, mul_8, sub_2
# x_15 => relu_2
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
    x2 = (xindex // 401408)
    x4 = xindex % 401408
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
    tl.store(out_ptr1 + (x4 + (2408448*x2)), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xe/cxewi5urzqtkcaoq57xzblzk6t5tdfqmaml4lmgn5jrxwf76omop.py
# Source Nodes: [x_37, x_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
# x_37 => add_15, mul_22, mul_23, sub_7
# x_40 => relu_7
triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
    x2 = (xindex // 401408)
    x4 = xindex % 401408
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
    tmp16 = 0.0
    tmp17 = tmp15 <= tmp16
    tl.store(out_ptr0 + (x4 + (2408448*x2)), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qa/cqafpa72ppx23pgeraixdaj7mc3kprq7n36ax23q2ndkslv3wpzw.py
# Source Nodes: [x_44, x_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_44 => add_17, mul_25, mul_26, sub_8
# x_49 => relu_8
triton_poi_fused__native_batch_norm_legit_no_training_relu_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_3', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ip/cip6yqkrvlrkdbzmfr4e7igio6i6tta5fog43cshlds57cycqqt5.py
# Source Nodes: [cat_10, x_50], Original ATen: [aten.cat, aten.max_pool2d_with_indices]
# cat_10 => cat_1
# x_50 => getitem, getitem_1
triton_poi_fused_cat_max_pool2d_with_indices_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_max_pool2d_with_indices_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 28) % 28
    x0 = xindex % 28
    x3 = (xindex // 28)
    x4 = xindex
    x5 = xindex % 200704
    x6 = (xindex // 200704)
    tmp0 = 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 56, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((2*x0) + (112*x3)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 1 + (2*x0)
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + (1 + (2*x0) + (112*x3)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 2 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + (2 + (2*x0) + (112*x3)), tmp27, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 1 + (2*x1)
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + (56 + (2*x0) + (112*x3)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (57 + (2*x0) + (112*x3)), tmp41, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (58 + (2*x0) + (112*x3)), tmp46, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 2 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (112 + (2*x0) + (112*x3)), tmp55, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (113 + (2*x0) + (112*x3)), tmp60, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (114 + (2*x0) + (112*x3)), tmp65, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tmp70 = tmp21 > tmp13
    tmp71 = 1 + (2*x0) + (112*x1)
    tmp72 = (2*x0) + (112*x1)
    tmp73 = tl.where(tmp70, tmp71, tmp72)
    tmp74 = tmp30 > tmp22
    tmp75 = 2 + (2*x0) + (112*x1)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tmp77 = tmp39 > tmp31
    tmp78 = 56 + (2*x0) + (112*x1)
    tmp79 = tl.where(tmp77, tmp78, tmp76)
    tmp80 = tmp44 > tmp40
    tmp81 = 57 + (2*x0) + (112*x1)
    tmp82 = tl.where(tmp80, tmp81, tmp79)
    tmp83 = tmp49 > tmp45
    tmp84 = 58 + (2*x0) + (112*x1)
    tmp85 = tl.where(tmp83, tmp84, tmp82)
    tmp86 = tmp58 > tmp50
    tmp87 = 112 + (2*x0) + (112*x1)
    tmp88 = tl.where(tmp86, tmp87, tmp85)
    tmp89 = tmp63 > tmp59
    tmp90 = 113 + (2*x0) + (112*x1)
    tmp91 = tl.where(tmp89, tmp90, tmp88)
    tmp92 = tmp68 > tmp64
    tmp93 = 114 + (2*x0) + (112*x1)
    tmp94 = tl.where(tmp92, tmp93, tmp91)
    tl.store(out_ptr0 + (x4), tmp69, None)
    tl.store(out_ptr1 + (x4), tmp94, None)
    tl.store(out_ptr2 + (x5 + (827904*x6)), tmp69, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xy/cxyjycmr5de4zot62b7ye2v3yxbx5s37vdvhtmrbhay3xc5s2edk.py
# Source Nodes: [cat_10, x_52, x_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_10 => cat_1
# x_52 => add_19, mul_28, mul_29, sub_9
# x_55 => relu_9
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 160
    x2 = (xindex // 125440)
    x4 = xindex % 125440
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
    tl.store(out_ptr1 + (x4 + (827904*x2)), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/i7/ci7d63htjfoddcicazuunl3u7gjsv274myy7aksq4xgt5tn73pfg.py
# Source Nodes: [x_72, x_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
# x_72 => add_27, mul_40, mul_41, sub_13
# x_75 => relu_13
triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 160
    x2 = (xindex // 125440)
    x4 = xindex % 125440
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
    tmp16 = 0.0
    tmp17 = tmp15 <= tmp16
    tl.store(out_ptr0 + (x4 + (827904*x2)), tmp15, None)
    tl.store(out_ptr1 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/d6/cd6m6ayfhlqimg26wplcmkpw6vlgzub5dbqwcdwq7ns4wokm2j6l.py
# Source Nodes: [x_79, x_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_79 => add_29, mul_43, mul_44, sub_14
# x_84 => relu_14
triton_poi_fused__native_batch_norm_legit_no_training_relu_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_7', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/qe/cqefj3xzfoopkhkfilfquietrpmwto66klq4ewptz36r5dyerjbc.py
# Source Nodes: [cat_9, x_85], Original ATen: [aten.cat, aten.max_pool2d_with_indices]
# cat_9 => cat_2
# x_85 => getitem_2, getitem_3
triton_poi_fused_cat_max_pool2d_with_indices_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_max_pool2d_with_indices_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 14) % 14
    x0 = xindex % 14
    x3 = (xindex // 14)
    x4 = xindex
    x5 = xindex % 100352
    x6 = (xindex // 100352)
    tmp0 = 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 28, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((2*x0) + (56*x3)), tmp10, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 1 + (2*x0)
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + (1 + (2*x0) + (56*x3)), tmp18, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 2 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + (2 + (2*x0) + (56*x3)), tmp27, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 1 + (2*x1)
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + (28 + (2*x0) + (56*x3)), tmp36, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (29 + (2*x0) + (56*x3)), tmp41, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (30 + (2*x0) + (56*x3)), tmp46, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 2 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (56 + (2*x0) + (56*x3)), tmp55, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (57 + (2*x0) + (56*x3)), tmp60, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (58 + (2*x0) + (56*x3)), tmp65, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tmp70 = tmp21 > tmp13
    tmp71 = 1 + (2*x0) + (56*x1)
    tmp72 = (2*x0) + (56*x1)
    tmp73 = tl.where(tmp70, tmp71, tmp72)
    tmp74 = tmp30 > tmp22
    tmp75 = 2 + (2*x0) + (56*x1)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tmp77 = tmp39 > tmp31
    tmp78 = 28 + (2*x0) + (56*x1)
    tmp79 = tl.where(tmp77, tmp78, tmp76)
    tmp80 = tmp44 > tmp40
    tmp81 = 29 + (2*x0) + (56*x1)
    tmp82 = tl.where(tmp80, tmp81, tmp79)
    tmp83 = tmp49 > tmp45
    tmp84 = 30 + (2*x0) + (56*x1)
    tmp85 = tl.where(tmp83, tmp84, tmp82)
    tmp86 = tmp58 > tmp50
    tmp87 = 56 + (2*x0) + (56*x1)
    tmp88 = tl.where(tmp86, tmp87, tmp85)
    tmp89 = tmp63 > tmp59
    tmp90 = 57 + (2*x0) + (56*x1)
    tmp91 = tl.where(tmp89, tmp90, tmp88)
    tmp92 = tmp68 > tmp64
    tmp93 = 58 + (2*x0) + (56*x1)
    tmp94 = tl.where(tmp92, tmp93, tmp91)
    tl.store(out_ptr0 + (x4), tmp69, None)
    tl.store(out_ptr1 + (x4), tmp94, None)
    tl.store(out_ptr2 + (x5 + (288512*x6)), tmp69, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/eu/ceuqy4fmwqld3ccsb36pwvbz6ilatskgtiuzg7azjteuykuu5cbh.py
# Source Nodes: [cat_9, x_87, x_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_9 => cat_2
# x_87 => add_31, mul_46, mul_47, sub_15
# x_90 => relu_15
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 192
    x2 = (xindex // 37632)
    x4 = xindex % 37632
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
    tl.store(out_ptr1 + (x4 + (288512*x2)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jm/cjmhmf4kk4o26hj4j6jhg3q7jtkyigbzyuyenik5mxctlyoqlw4k.py
# Source Nodes: [x_107, x_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
# x_107 => add_39, mul_58, mul_59, sub_19
# x_110 => relu_19
triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 192
    x2 = (xindex // 37632)
    x4 = xindex % 37632
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
    tmp16 = 0.0
    tmp17 = tmp15 <= tmp16
    tl.store(out_ptr0 + (x4 + (288512*x2)), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ry/cryiwdyagxbcrybdjcbjg4tbze5e47hfubkrqfbof633pswovjn7.py
# Source Nodes: [cat_8, x_114, x_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_8 => cat_3
# x_114 => add_41, mul_61, mul_62, sub_20
# x_118 => relu_20
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 768
    x2 = (xindex // 150528)
    x4 = xindex % 150528
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
    tl.store(out_ptr1 + (x4 + (338688*x2)), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ge/cgef7pcpgvedsoh4tg6l66mppvijmycidqsxi4lbgkjgzgjligif.py
# Source Nodes: [cat_8, x_120, x_123], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_8 => cat_3
# x_120 => add_43, mul_64, mul_65, sub_21
# x_123 => relu_21
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 192
    x2 = (xindex // 37632)
    x4 = xindex % 37632
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
    tl.store(out_ptr1 + (x4 + (338688*x2)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oy/coyj536k3uhvz5sq7qmyzihs7ioy4s7jc7n5u4rgrdlseviojs2m.py
# Source Nodes: [x_140, x_143], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
# x_140 => add_51, mul_76, mul_77, sub_25
# x_143 => relu_25
triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 192
    x2 = (xindex // 37632)
    x4 = xindex % 37632
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
    tmp16 = 0.0
    tmp17 = tmp15 <= tmp16
    tl.store(out_ptr0 + (x4 + (338688*x2)), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qd/cqdiii27us4yspusolbwuwfkhps5rar3rydlpuklbkk2qksdgdsa.py
# Source Nodes: [x_147, x_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_147 => add_53, mul_79, mul_80, sub_26
# x_152 => relu_26
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/6x/c6xyysmyegbnwn5ypefphlix2bx6kvwk6qdcprbxzl4bqkkmo6h4.py
# Source Nodes: [cat_7, x_153], Original ATen: [aten.cat, aten.max_pool2d_with_indices]
# cat_7 => cat_4
# x_153 => getitem_4, getitem_5
triton_poi_fused_cat_max_pool2d_with_indices_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: '*fp32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_max_pool2d_with_indices_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 7) % 7
    x0 = xindex % 7
    x3 = (xindex // 7)
    x4 = xindex
    x5 = xindex % 37632
    x6 = (xindex // 37632)
    tmp0 = 2*x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 14, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = 2*x0
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((2*x0) + (28*x3)), tmp10 & xmask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 1 + (2*x0)
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + (1 + (2*x0) + (28*x3)), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 2 + (2*x0)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + (2 + (2*x0) + (28*x3)), tmp27 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 1 + (2*x1)
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + (14 + (2*x0) + (28*x3)), tmp36 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (15 + (2*x0) + (28*x3)), tmp41 & xmask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (16 + (2*x0) + (28*x3)), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 2 + (2*x1)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (28 + (2*x0) + (28*x3)), tmp55 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (29 + (2*x0) + (28*x3)), tmp60 & xmask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (30 + (2*x0) + (28*x3)), tmp65 & xmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tmp70 = tmp21 > tmp13
    tmp71 = 1 + (2*x0) + (28*x1)
    tmp72 = (2*x0) + (28*x1)
    tmp73 = tl.where(tmp70, tmp71, tmp72)
    tmp74 = tmp30 > tmp22
    tmp75 = 2 + (2*x0) + (28*x1)
    tmp76 = tl.where(tmp74, tmp75, tmp73)
    tmp77 = tmp39 > tmp31
    tmp78 = 14 + (2*x0) + (28*x1)
    tmp79 = tl.where(tmp77, tmp78, tmp76)
    tmp80 = tmp44 > tmp40
    tmp81 = 15 + (2*x0) + (28*x1)
    tmp82 = tl.where(tmp80, tmp81, tmp79)
    tmp83 = tmp49 > tmp45
    tmp84 = 16 + (2*x0) + (28*x1)
    tmp85 = tl.where(tmp83, tmp84, tmp82)
    tmp86 = tmp58 > tmp50
    tmp87 = 28 + (2*x0) + (28*x1)
    tmp88 = tl.where(tmp86, tmp87, tmp85)
    tmp89 = tmp63 > tmp59
    tmp90 = 29 + (2*x0) + (28*x1)
    tmp91 = tl.where(tmp89, tmp90, tmp88)
    tmp92 = tmp68 > tmp64
    tmp93 = 30 + (2*x0) + (28*x1)
    tmp94 = tl.where(tmp92, tmp93, tmp91)
    tl.store(out_ptr0 + (x4), tmp69, xmask)
    tl.store(out_ptr1 + (x4), tmp94, xmask)
    tl.store(out_ptr2 + (x5 + (92512*x6)), tmp69, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/te/ctektett4resij5hjydgv7ttt2zyt4v3rleozkxubo5gip2b6ibj.py
# Source Nodes: [cat_7, x_155, x_158], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_7 => cat_4
# x_155 => add_55, mul_82, mul_83, sub_27
# x_158 => relu_27
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 224
    x2 = (xindex // 10976)
    x4 = xindex % 10976
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
    tl.store(out_ptr1 + (x4 + (92512*x2)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sy/csyli7sqw437xvaf3dgla5rlinkuq6rt64qgq7aksm3bsxfswfnu.py
# Source Nodes: [x_175, x_178], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
# x_175 => add_63, mul_94, mul_95, sub_31
# x_178 => relu_31
triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 224
    x2 = (xindex // 10976)
    x4 = xindex % 10976
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
    tmp16 = 0.0
    tmp17 = tmp15 <= tmp16
    tl.store(out_ptr0 + (x4 + (92512*x2)), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ju/cjuqwcijhqwttrolp6bzvb36l7swgfnoa5wlxij75nafubamrhc2.py
# Source Nodes: [cat_6, x_182, x_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_6 => cat_5
# x_182 => add_65, mul_97, mul_98, sub_32
# x_186 => relu_32
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 1024
    x2 = (xindex // 50176)
    x4 = xindex % 50176
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
    tl.store(out_ptr1 + (x4 + (105056*x2)), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ry/cry4ucommqbdeagki3b7m3smi4ayqthspegea4kgfiv2f55umhgi.py
# Source Nodes: [cat_6, x_188, x_191], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
# cat_6 => cat_5
# x_188 => add_67, mul_100, mul_101, sub_33
# x_191 => relu_33
triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 224
    x2 = (xindex // 10976)
    x4 = xindex % 10976
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
    tl.store(out_ptr1 + (x4 + (105056*x2)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vj/cvj7czz42h4wnfccb7h24l3dxw3546yobcxmj4cfvkgpbuw4nwcw.py
# Source Nodes: [x_208, x_211], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
# x_208 => add_75, mul_112, mul_113, sub_37
# x_211 => relu_37
triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*i1', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 224
    x2 = (xindex // 10976)
    x4 = xindex % 10976
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
    tmp16 = 0.0
    tmp17 = tmp15 <= tmp16
    tl.store(out_ptr0 + (x4 + (105056*x2)), tmp15, xmask)
    tl.store(out_ptr1 + (x3), tmp17, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s2/cs25u3lzuycp5hav6z3wgmektox7knsabp2thryeps2we77tq5lu.py
# Source Nodes: [x_215, x_221, x_222, x_224], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu, aten.threshold_backward, aten.view]
# x_215 => add_77, mul_115, mul_116, sub_38
# x_221 => relu_38
# x_222 => mean
# x_224 => view
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_threshold_backward_view_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_threshold_backward_view_21', 'mutated_arg_names': ['in_out_ptr0']}
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
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198 = args
    args.clear()
    assert_size_stride(primals_1, (64, ), (1, ))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (64, ), (1, ))
    assert_size_stride(primals_4, (64, ), (1, ))
    assert_size_stride(primals_5, (128, ), (1, ))
    assert_size_stride(primals_6, (128, ), (1, ))
    assert_size_stride(primals_7, (128, ), (1, ))
    assert_size_stride(primals_8, (128, ), (1, ))
    assert_size_stride(primals_9, (128, ), (1, ))
    assert_size_stride(primals_10, (128, ), (1, ))
    assert_size_stride(primals_11, (128, ), (1, ))
    assert_size_stride(primals_12, (128, ), (1, ))
    assert_size_stride(primals_13, (128, ), (1, ))
    assert_size_stride(primals_14, (128, ), (1, ))
    assert_size_stride(primals_15, (128, ), (1, ))
    assert_size_stride(primals_16, (128, ), (1, ))
    assert_size_stride(primals_17, (256, ), (1, ))
    assert_size_stride(primals_18, (256, ), (1, ))
    assert_size_stride(primals_19, (160, ), (1, ))
    assert_size_stride(primals_20, (160, ), (1, ))
    assert_size_stride(primals_21, (160, ), (1, ))
    assert_size_stride(primals_22, (160, ), (1, ))
    assert_size_stride(primals_23, (160, ), (1, ))
    assert_size_stride(primals_24, (160, ), (1, ))
    assert_size_stride(primals_25, (160, ), (1, ))
    assert_size_stride(primals_26, (160, ), (1, ))
    assert_size_stride(primals_27, (160, ), (1, ))
    assert_size_stride(primals_28, (160, ), (1, ))
    assert_size_stride(primals_29, (512, ), (1, ))
    assert_size_stride(primals_30, (512, ), (1, ))
    assert_size_stride(primals_31, (192, ), (1, ))
    assert_size_stride(primals_32, (192, ), (1, ))
    assert_size_stride(primals_33, (192, ), (1, ))
    assert_size_stride(primals_34, (192, ), (1, ))
    assert_size_stride(primals_35, (192, ), (1, ))
    assert_size_stride(primals_36, (192, ), (1, ))
    assert_size_stride(primals_37, (192, ), (1, ))
    assert_size_stride(primals_38, (192, ), (1, ))
    assert_size_stride(primals_39, (192, ), (1, ))
    assert_size_stride(primals_40, (192, ), (1, ))
    assert_size_stride(primals_41, (768, ), (1, ))
    assert_size_stride(primals_42, (768, ), (1, ))
    assert_size_stride(primals_43, (192, ), (1, ))
    assert_size_stride(primals_44, (192, ), (1, ))
    assert_size_stride(primals_45, (192, ), (1, ))
    assert_size_stride(primals_46, (192, ), (1, ))
    assert_size_stride(primals_47, (192, ), (1, ))
    assert_size_stride(primals_48, (192, ), (1, ))
    assert_size_stride(primals_49, (192, ), (1, ))
    assert_size_stride(primals_50, (192, ), (1, ))
    assert_size_stride(primals_51, (192, ), (1, ))
    assert_size_stride(primals_52, (192, ), (1, ))
    assert_size_stride(primals_53, (768, ), (1, ))
    assert_size_stride(primals_54, (768, ), (1, ))
    assert_size_stride(primals_55, (224, ), (1, ))
    assert_size_stride(primals_56, (224, ), (1, ))
    assert_size_stride(primals_57, (224, ), (1, ))
    assert_size_stride(primals_58, (224, ), (1, ))
    assert_size_stride(primals_59, (224, ), (1, ))
    assert_size_stride(primals_60, (224, ), (1, ))
    assert_size_stride(primals_61, (224, ), (1, ))
    assert_size_stride(primals_62, (224, ), (1, ))
    assert_size_stride(primals_63, (224, ), (1, ))
    assert_size_stride(primals_64, (224, ), (1, ))
    assert_size_stride(primals_65, (1024, ), (1, ))
    assert_size_stride(primals_66, (1024, ), (1, ))
    assert_size_stride(primals_67, (224, ), (1, ))
    assert_size_stride(primals_68, (224, ), (1, ))
    assert_size_stride(primals_69, (224, ), (1, ))
    assert_size_stride(primals_70, (224, ), (1, ))
    assert_size_stride(primals_71, (224, ), (1, ))
    assert_size_stride(primals_72, (224, ), (1, ))
    assert_size_stride(primals_73, (224, ), (1, ))
    assert_size_stride(primals_74, (224, ), (1, ))
    assert_size_stride(primals_75, (224, ), (1, ))
    assert_size_stride(primals_76, (224, ), (1, ))
    assert_size_stride(primals_77, (1024, ), (1, ))
    assert_size_stride(primals_78, (1024, ), (1, ))
    assert_size_stride(primals_79, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_80, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_81, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_82, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_83, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_84, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_85, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_86, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(primals_87, (256, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(primals_88, (160, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_89, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_90, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_91, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_92, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(primals_93, (512, 1056, 1, 1), (1056, 1, 1, 1))
    assert_size_stride(primals_94, (192, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(primals_95, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_96, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_97, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_98, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_99, (768, 1472, 1, 1), (1472, 1, 1, 1))
    assert_size_stride(primals_100, (192, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_101, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_102, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_103, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_104, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_105, (768, 1728, 1, 1), (1728, 1, 1, 1))
    assert_size_stride(primals_106, (224, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(primals_107, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(primals_108, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(primals_109, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(primals_110, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(primals_111, (1024, 1888, 1, 1), (1888, 1, 1, 1))
    assert_size_stride(primals_112, (224, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(primals_113, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(primals_114, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(primals_115, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(primals_116, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(primals_117, (1024, 2144, 1, 1), (2144, 1, 1, 1))
    assert_size_stride(primals_118, (1000, 1024), (1024, 1))
    assert_size_stride(primals_119, (1000, ), (1, ))
    assert_size_stride(primals_120, (64, ), (1, ))
    assert_size_stride(primals_121, (64, ), (1, ))
    assert_size_stride(primals_122, (64, ), (1, ))
    assert_size_stride(primals_123, (64, ), (1, ))
    assert_size_stride(primals_124, (128, ), (1, ))
    assert_size_stride(primals_125, (128, ), (1, ))
    assert_size_stride(primals_126, (128, ), (1, ))
    assert_size_stride(primals_127, (128, ), (1, ))
    assert_size_stride(primals_128, (128, ), (1, ))
    assert_size_stride(primals_129, (128, ), (1, ))
    assert_size_stride(primals_130, (128, ), (1, ))
    assert_size_stride(primals_131, (128, ), (1, ))
    assert_size_stride(primals_132, (128, ), (1, ))
    assert_size_stride(primals_133, (128, ), (1, ))
    assert_size_stride(primals_134, (128, ), (1, ))
    assert_size_stride(primals_135, (128, ), (1, ))
    assert_size_stride(primals_136, (256, ), (1, ))
    assert_size_stride(primals_137, (256, ), (1, ))
    assert_size_stride(primals_138, (160, ), (1, ))
    assert_size_stride(primals_139, (160, ), (1, ))
    assert_size_stride(primals_140, (160, ), (1, ))
    assert_size_stride(primals_141, (160, ), (1, ))
    assert_size_stride(primals_142, (160, ), (1, ))
    assert_size_stride(primals_143, (160, ), (1, ))
    assert_size_stride(primals_144, (160, ), (1, ))
    assert_size_stride(primals_145, (160, ), (1, ))
    assert_size_stride(primals_146, (160, ), (1, ))
    assert_size_stride(primals_147, (160, ), (1, ))
    assert_size_stride(primals_148, (512, ), (1, ))
    assert_size_stride(primals_149, (512, ), (1, ))
    assert_size_stride(primals_150, (192, ), (1, ))
    assert_size_stride(primals_151, (192, ), (1, ))
    assert_size_stride(primals_152, (192, ), (1, ))
    assert_size_stride(primals_153, (192, ), (1, ))
    assert_size_stride(primals_154, (192, ), (1, ))
    assert_size_stride(primals_155, (192, ), (1, ))
    assert_size_stride(primals_156, (192, ), (1, ))
    assert_size_stride(primals_157, (192, ), (1, ))
    assert_size_stride(primals_158, (192, ), (1, ))
    assert_size_stride(primals_159, (192, ), (1, ))
    assert_size_stride(primals_160, (768, ), (1, ))
    assert_size_stride(primals_161, (768, ), (1, ))
    assert_size_stride(primals_162, (192, ), (1, ))
    assert_size_stride(primals_163, (192, ), (1, ))
    assert_size_stride(primals_164, (192, ), (1, ))
    assert_size_stride(primals_165, (192, ), (1, ))
    assert_size_stride(primals_166, (192, ), (1, ))
    assert_size_stride(primals_167, (192, ), (1, ))
    assert_size_stride(primals_168, (192, ), (1, ))
    assert_size_stride(primals_169, (192, ), (1, ))
    assert_size_stride(primals_170, (192, ), (1, ))
    assert_size_stride(primals_171, (192, ), (1, ))
    assert_size_stride(primals_172, (768, ), (1, ))
    assert_size_stride(primals_173, (768, ), (1, ))
    assert_size_stride(primals_174, (224, ), (1, ))
    assert_size_stride(primals_175, (224, ), (1, ))
    assert_size_stride(primals_176, (224, ), (1, ))
    assert_size_stride(primals_177, (224, ), (1, ))
    assert_size_stride(primals_178, (224, ), (1, ))
    assert_size_stride(primals_179, (224, ), (1, ))
    assert_size_stride(primals_180, (224, ), (1, ))
    assert_size_stride(primals_181, (224, ), (1, ))
    assert_size_stride(primals_182, (224, ), (1, ))
    assert_size_stride(primals_183, (224, ), (1, ))
    assert_size_stride(primals_184, (1024, ), (1, ))
    assert_size_stride(primals_185, (1024, ), (1, ))
    assert_size_stride(primals_186, (224, ), (1, ))
    assert_size_stride(primals_187, (224, ), (1, ))
    assert_size_stride(primals_188, (224, ), (1, ))
    assert_size_stride(primals_189, (224, ), (1, ))
    assert_size_stride(primals_190, (224, ), (1, ))
    assert_size_stride(primals_191, (224, ), (1, ))
    assert_size_stride(primals_192, (224, ), (1, ))
    assert_size_stride(primals_193, (224, ), (1, ))
    assert_size_stride(primals_194, (224, ), (1, ))
    assert_size_stride(primals_195, (224, ), (1, ))
    assert_size_stride(primals_196, (1024, ), (1, ))
    assert_size_stride(primals_197, (1024, ), (1, ))
    assert_size_stride(primals_198, (4, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(primals_198, primals_79, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 112, 112), (802816, 12544, 112, 1))
        buf1 = empty((4, 64, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1, x_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf0, primals_120, primals_121, primals_1, primals_2, buf1, 3211264, grid=grid(3211264), stream=stream0)
        del primals_2
        # Source Nodes: [x_5], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf1, primals_80, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 64, 112, 112), (802816, 12544, 112, 1))
        buf3 = empty((4, 64, 112, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_6, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_0.run(buf2, primals_122, primals_123, primals_3, primals_4, buf3, 3211264, grid=grid(3211264), stream=stream0)
        del primals_4
        # Source Nodes: [x_10], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, primals_81, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 128, 56, 56), (401408, 3136, 56, 1))
        buf5 = empty((4, 128, 56, 56), device='cuda', dtype=torch.float32)
        buf21 = empty((4, 768, 56, 56), device='cuda', dtype=torch.float32)
        buf16 = reinterpret_tensor(buf21, (4, 128, 56, 56), (2408448, 3136, 56, 1), 0)  # alias
        # Source Nodes: [cat_11, x_11, x_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_1.run(buf4, primals_124, primals_125, primals_5, primals_6, buf5, buf16, 1605632, grid=grid(1605632), stream=stream0)
        del primals_6
        # Source Nodes: [x_16], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, primals_82, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 128, 56, 56), (401408, 3136, 56, 1))
        buf7 = empty((4, 128, 56, 56), device='cuda', dtype=torch.float32)
        buf17 = reinterpret_tensor(buf21, (4, 128, 56, 56), (2408448, 3136, 56, 1), 401408)  # alias
        # Source Nodes: [cat_11, x_17, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_1.run(buf6, primals_126, primals_127, primals_7, primals_8, buf7, buf17, 1605632, grid=grid(1605632), stream=stream0)
        del primals_8
        # Source Nodes: [x_21], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, primals_83, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 128, 56, 56), (401408, 3136, 56, 1))
        buf9 = empty((4, 128, 56, 56), device='cuda', dtype=torch.float32)
        buf18 = reinterpret_tensor(buf21, (4, 128, 56, 56), (2408448, 3136, 56, 1), 802816)  # alias
        # Source Nodes: [cat_11, x_22, x_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_1.run(buf8, primals_128, primals_129, primals_9, primals_10, buf9, buf18, 1605632, grid=grid(1605632), stream=stream0)
        del primals_10
        # Source Nodes: [x_26], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, primals_84, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 128, 56, 56), (401408, 3136, 56, 1))
        buf11 = empty((4, 128, 56, 56), device='cuda', dtype=torch.float32)
        buf19 = reinterpret_tensor(buf21, (4, 128, 56, 56), (2408448, 3136, 56, 1), 1204224)  # alias
        # Source Nodes: [cat_11, x_27, x_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_1.run(buf10, primals_130, primals_131, primals_11, primals_12, buf11, buf19, 1605632, grid=grid(1605632), stream=stream0)
        del primals_12
        # Source Nodes: [x_31], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, primals_85, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 128, 56, 56), (401408, 3136, 56, 1))
        buf13 = empty((4, 128, 56, 56), device='cuda', dtype=torch.float32)
        buf20 = reinterpret_tensor(buf21, (4, 128, 56, 56), (2408448, 3136, 56, 1), 1605632)  # alias
        # Source Nodes: [cat_11, x_32, x_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_1.run(buf12, primals_132, primals_133, primals_13, primals_14, buf13, buf20, 1605632, grid=grid(1605632), stream=stream0)
        del primals_14
        # Source Nodes: [x_36], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, primals_86, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 128, 56, 56), (401408, 3136, 56, 1))
        buf15 = reinterpret_tensor(buf21, (4, 128, 56, 56), (2408448, 3136, 56, 1), 2007040)  # alias
        buf129 = empty((4, 128, 56, 56), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_37, x_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_2.run(buf14, primals_134, primals_135, primals_15, primals_16, buf15, buf129, 1605632, grid=grid(1605632), stream=stream0)
        del primals_16
        # Source Nodes: [x_43], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, primals_87, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 256, 56, 56), (802816, 3136, 56, 1))
        buf23 = empty((4, 256, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_44, x_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_3.run(buf22, primals_136, primals_137, primals_17, primals_18, buf23, 3211264, grid=grid(3211264), stream=stream0)
        del primals_18
        buf24 = empty((4, 256, 28, 28), device='cuda', dtype=torch.float32)
        buf25 = empty((4, 256, 28, 28), device='cuda', dtype=torch.int64)
        buf41 = empty((4, 1056, 28, 28), device='cuda', dtype=torch.float32)
        buf36 = reinterpret_tensor(buf41, (4, 256, 28, 28), (827904, 784, 28, 1), 0)  # alias
        # Source Nodes: [cat_10, x_50], Original ATen: [aten.cat, aten.max_pool2d_with_indices]
        triton_poi_fused_cat_max_pool2d_with_indices_4.run(buf23, buf24, buf25, buf36, 802816, grid=grid(802816), stream=stream0)
        # Source Nodes: [x_51], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf24, primals_88, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 160, 28, 28), (125440, 784, 28, 1))
        buf27 = empty((4, 160, 28, 28), device='cuda', dtype=torch.float32)
        buf37 = reinterpret_tensor(buf41, (4, 160, 28, 28), (827904, 784, 28, 1), 200704)  # alias
        # Source Nodes: [cat_10, x_52, x_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5.run(buf26, primals_138, primals_139, primals_19, primals_20, buf27, buf37, 501760, grid=grid(501760), stream=stream0)
        del primals_20
        # Source Nodes: [x_56], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_89, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 160, 28, 28), (125440, 784, 28, 1))
        buf29 = empty((4, 160, 28, 28), device='cuda', dtype=torch.float32)
        buf38 = reinterpret_tensor(buf41, (4, 160, 28, 28), (827904, 784, 28, 1), 326144)  # alias
        # Source Nodes: [cat_10, x_57, x_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5.run(buf28, primals_140, primals_141, primals_21, primals_22, buf29, buf38, 501760, grid=grid(501760), stream=stream0)
        del primals_22
        # Source Nodes: [x_61], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, primals_90, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 160, 28, 28), (125440, 784, 28, 1))
        buf31 = empty((4, 160, 28, 28), device='cuda', dtype=torch.float32)
        buf39 = reinterpret_tensor(buf41, (4, 160, 28, 28), (827904, 784, 28, 1), 451584)  # alias
        # Source Nodes: [cat_10, x_62, x_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5.run(buf30, primals_142, primals_143, primals_23, primals_24, buf31, buf39, 501760, grid=grid(501760), stream=stream0)
        del primals_24
        # Source Nodes: [x_66], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, primals_91, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 160, 28, 28), (125440, 784, 28, 1))
        buf33 = empty((4, 160, 28, 28), device='cuda', dtype=torch.float32)
        buf40 = reinterpret_tensor(buf41, (4, 160, 28, 28), (827904, 784, 28, 1), 577024)  # alias
        # Source Nodes: [cat_10, x_67, x_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5.run(buf32, primals_144, primals_145, primals_25, primals_26, buf33, buf40, 501760, grid=grid(501760), stream=stream0)
        del primals_26
        # Source Nodes: [x_71], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf33, primals_92, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 160, 28, 28), (125440, 784, 28, 1))
        buf35 = reinterpret_tensor(buf41, (4, 160, 28, 28), (827904, 784, 28, 1), 702464)  # alias
        buf128 = empty((4, 160, 28, 28), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_72, x_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_6.run(buf34, primals_146, primals_147, primals_27, primals_28, buf35, buf128, 501760, grid=grid(501760), stream=stream0)
        del primals_28
        # Source Nodes: [x_78], Original ATen: [aten.convolution]
        buf42 = extern_kernels.convolution(buf41, primals_93, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 512, 28, 28), (401408, 784, 28, 1))
        buf43 = empty((4, 512, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_79, x_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf42, primals_148, primals_149, primals_29, primals_30, buf43, 1605632, grid=grid(1605632), stream=stream0)
        del primals_30
        buf44 = empty((4, 512, 14, 14), device='cuda', dtype=torch.float32)
        buf45 = empty((4, 512, 14, 14), device='cuda', dtype=torch.int64)
        buf61 = empty((4, 1472, 14, 14), device='cuda', dtype=torch.float32)
        buf56 = reinterpret_tensor(buf61, (4, 512, 14, 14), (288512, 196, 14, 1), 0)  # alias
        # Source Nodes: [cat_9, x_85], Original ATen: [aten.cat, aten.max_pool2d_with_indices]
        triton_poi_fused_cat_max_pool2d_with_indices_8.run(buf43, buf44, buf45, buf56, 401408, grid=grid(401408), stream=stream0)
        # Source Nodes: [x_86], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf44, primals_94, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 192, 14, 14), (37632, 196, 14, 1))
        buf47 = empty((4, 192, 14, 14), device='cuda', dtype=torch.float32)
        buf57 = reinterpret_tensor(buf61, (4, 192, 14, 14), (288512, 196, 14, 1), 100352)  # alias
        # Source Nodes: [cat_9, x_87, x_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9.run(buf46, primals_150, primals_151, primals_31, primals_32, buf47, buf57, 150528, grid=grid(150528), stream=stream0)
        del primals_32
        # Source Nodes: [x_91], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, primals_95, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 192, 14, 14), (37632, 196, 14, 1))
        buf49 = empty((4, 192, 14, 14), device='cuda', dtype=torch.float32)
        buf58 = reinterpret_tensor(buf61, (4, 192, 14, 14), (288512, 196, 14, 1), 137984)  # alias
        # Source Nodes: [cat_9, x_92, x_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9.run(buf48, primals_152, primals_153, primals_33, primals_34, buf49, buf58, 150528, grid=grid(150528), stream=stream0)
        del primals_34
        # Source Nodes: [x_96], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_96, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 192, 14, 14), (37632, 196, 14, 1))
        buf51 = empty((4, 192, 14, 14), device='cuda', dtype=torch.float32)
        buf59 = reinterpret_tensor(buf61, (4, 192, 14, 14), (288512, 196, 14, 1), 175616)  # alias
        # Source Nodes: [cat_9, x_100, x_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9.run(buf50, primals_154, primals_155, primals_35, primals_36, buf51, buf59, 150528, grid=grid(150528), stream=stream0)
        del primals_36
        # Source Nodes: [x_101], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, primals_97, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 192, 14, 14), (37632, 196, 14, 1))
        buf53 = empty((4, 192, 14, 14), device='cuda', dtype=torch.float32)
        buf60 = reinterpret_tensor(buf61, (4, 192, 14, 14), (288512, 196, 14, 1), 213248)  # alias
        # Source Nodes: [cat_9, x_102, x_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9.run(buf52, primals_156, primals_157, primals_37, primals_38, buf53, buf60, 150528, grid=grid(150528), stream=stream0)
        del primals_38
        # Source Nodes: [x_106], Original ATen: [aten.convolution]
        buf54 = extern_kernels.convolution(buf53, primals_98, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 192, 14, 14), (37632, 196, 14, 1))
        buf55 = reinterpret_tensor(buf61, (4, 192, 14, 14), (288512, 196, 14, 1), 250880)  # alias
        buf127 = empty((4, 192, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_107, x_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_10.run(buf54, primals_158, primals_159, primals_39, primals_40, buf55, buf127, 150528, grid=grid(150528), stream=stream0)
        del primals_40
        # Source Nodes: [x_113], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, primals_99, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 768, 14, 14), (150528, 196, 14, 1))
        buf63 = empty((4, 768, 14, 14), device='cuda', dtype=torch.float32)
        buf79 = empty((4, 1728, 14, 14), device='cuda', dtype=torch.float32)
        buf74 = reinterpret_tensor(buf79, (4, 768, 14, 14), (338688, 196, 14, 1), 0)  # alias
        # Source Nodes: [cat_8, x_114, x_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_11.run(buf62, primals_160, primals_161, primals_41, primals_42, buf63, buf74, 602112, grid=grid(602112), stream=stream0)
        del primals_42
        # Source Nodes: [x_119], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, primals_100, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 192, 14, 14), (37632, 196, 14, 1))
        buf65 = empty((4, 192, 14, 14), device='cuda', dtype=torch.float32)
        buf75 = reinterpret_tensor(buf79, (4, 192, 14, 14), (338688, 196, 14, 1), 150528)  # alias
        # Source Nodes: [cat_8, x_120, x_123], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_12.run(buf64, primals_162, primals_163, primals_43, primals_44, buf65, buf75, 150528, grid=grid(150528), stream=stream0)
        del primals_44
        # Source Nodes: [x_124], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, primals_101, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 192, 14, 14), (37632, 196, 14, 1))
        buf67 = empty((4, 192, 14, 14), device='cuda', dtype=torch.float32)
        buf76 = reinterpret_tensor(buf79, (4, 192, 14, 14), (338688, 196, 14, 1), 188160)  # alias
        # Source Nodes: [cat_8, x_125, x_128], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_12.run(buf66, primals_164, primals_165, primals_45, primals_46, buf67, buf76, 150528, grid=grid(150528), stream=stream0)
        del primals_46
        # Source Nodes: [x_129], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, primals_102, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 192, 14, 14), (37632, 196, 14, 1))
        buf69 = empty((4, 192, 14, 14), device='cuda', dtype=torch.float32)
        buf77 = reinterpret_tensor(buf79, (4, 192, 14, 14), (338688, 196, 14, 1), 225792)  # alias
        # Source Nodes: [cat_8, x_130, x_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_12.run(buf68, primals_166, primals_167, primals_47, primals_48, buf69, buf77, 150528, grid=grid(150528), stream=stream0)
        del primals_48
        # Source Nodes: [x_134], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, primals_103, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (4, 192, 14, 14), (37632, 196, 14, 1))
        buf71 = empty((4, 192, 14, 14), device='cuda', dtype=torch.float32)
        buf78 = reinterpret_tensor(buf79, (4, 192, 14, 14), (338688, 196, 14, 1), 263424)  # alias
        # Source Nodes: [cat_8, x_135, x_138], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_12.run(buf70, primals_168, primals_169, primals_49, primals_50, buf71, buf78, 150528, grid=grid(150528), stream=stream0)
        del primals_50
        # Source Nodes: [x_139], Original ATen: [aten.convolution]
        buf72 = extern_kernels.convolution(buf71, primals_104, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 192, 14, 14), (37632, 196, 14, 1))
        buf73 = reinterpret_tensor(buf79, (4, 192, 14, 14), (338688, 196, 14, 1), 301056)  # alias
        buf126 = empty((4, 192, 14, 14), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_140, x_143], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_13.run(buf72, primals_170, primals_171, primals_51, primals_52, buf73, buf126, 150528, grid=grid(150528), stream=stream0)
        del primals_52
        # Source Nodes: [x_146], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, primals_105, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 768, 14, 14), (150528, 196, 14, 1))
        buf81 = empty((4, 768, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_147, x_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf80, primals_172, primals_173, primals_53, primals_54, buf81, 602112, grid=grid(602112), stream=stream0)
        del primals_54
        buf82 = empty((4, 768, 7, 7), device='cuda', dtype=torch.float32)
        buf83 = empty((4, 768, 7, 7), device='cuda', dtype=torch.int64)
        buf99 = empty((4, 1888, 7, 7), device='cuda', dtype=torch.float32)
        buf94 = reinterpret_tensor(buf99, (4, 768, 7, 7), (92512, 49, 7, 1), 0)  # alias
        # Source Nodes: [cat_7, x_153], Original ATen: [aten.cat, aten.max_pool2d_with_indices]
        triton_poi_fused_cat_max_pool2d_with_indices_15.run(buf81, buf82, buf83, buf94, 150528, grid=grid(150528), stream=stream0)
        # Source Nodes: [x_154], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf82, primals_106, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 224, 7, 7), (10976, 49, 7, 1))
        buf85 = empty((4, 224, 7, 7), device='cuda', dtype=torch.float32)
        buf95 = reinterpret_tensor(buf99, (4, 224, 7, 7), (92512, 49, 7, 1), 37632)  # alias
        # Source Nodes: [cat_7, x_155, x_158], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16.run(buf84, primals_174, primals_175, primals_55, primals_56, buf85, buf95, 43904, grid=grid(43904), stream=stream0)
        del primals_56
        # Source Nodes: [x_159], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, primals_107, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (4, 224, 7, 7), (10976, 49, 7, 1))
        buf87 = empty((4, 224, 7, 7), device='cuda', dtype=torch.float32)
        buf96 = reinterpret_tensor(buf99, (4, 224, 7, 7), (92512, 49, 7, 1), 48608)  # alias
        # Source Nodes: [cat_7, x_160, x_163], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16.run(buf86, primals_176, primals_177, primals_57, primals_58, buf87, buf96, 43904, grid=grid(43904), stream=stream0)
        del primals_58
        # Source Nodes: [x_164], Original ATen: [aten.convolution]
        buf88 = extern_kernels.convolution(buf87, primals_108, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 224, 7, 7), (10976, 49, 7, 1))
        buf89 = empty((4, 224, 7, 7), device='cuda', dtype=torch.float32)
        buf97 = reinterpret_tensor(buf99, (4, 224, 7, 7), (92512, 49, 7, 1), 59584)  # alias
        # Source Nodes: [cat_7, x_165, x_168], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16.run(buf88, primals_178, primals_179, primals_59, primals_60, buf89, buf97, 43904, grid=grid(43904), stream=stream0)
        del primals_60
        # Source Nodes: [x_169], Original ATen: [aten.convolution]
        buf90 = extern_kernels.convolution(buf89, primals_109, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (4, 224, 7, 7), (10976, 49, 7, 1))
        buf91 = empty((4, 224, 7, 7), device='cuda', dtype=torch.float32)
        buf98 = reinterpret_tensor(buf99, (4, 224, 7, 7), (92512, 49, 7, 1), 70560)  # alias
        # Source Nodes: [cat_7, x_170, x_173], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16.run(buf90, primals_180, primals_181, primals_61, primals_62, buf91, buf98, 43904, grid=grid(43904), stream=stream0)
        del primals_62
        # Source Nodes: [x_174], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, primals_110, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 224, 7, 7), (10976, 49, 7, 1))
        buf93 = reinterpret_tensor(buf99, (4, 224, 7, 7), (92512, 49, 7, 1), 81536)  # alias
        buf125 = empty((4, 224, 7, 7), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_175, x_178], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_17.run(buf92, primals_182, primals_183, primals_63, primals_64, buf93, buf125, 43904, grid=grid(43904), stream=stream0)
        del primals_64
        # Source Nodes: [x_181], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, primals_111, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (4, 1024, 7, 7), (50176, 49, 7, 1))
        buf101 = empty((4, 1024, 7, 7), device='cuda', dtype=torch.float32)
        buf117 = empty((4, 2144, 7, 7), device='cuda', dtype=torch.float32)
        buf112 = reinterpret_tensor(buf117, (4, 1024, 7, 7), (105056, 49, 7, 1), 0)  # alias
        # Source Nodes: [cat_6, x_182, x_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_18.run(buf100, primals_184, primals_185, primals_65, primals_66, buf101, buf112, 200704, grid=grid(200704), stream=stream0)
        del primals_66
        # Source Nodes: [x_187], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, primals_112, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (4, 224, 7, 7), (10976, 49, 7, 1))
        buf103 = empty((4, 224, 7, 7), device='cuda', dtype=torch.float32)
        buf113 = reinterpret_tensor(buf117, (4, 224, 7, 7), (105056, 49, 7, 1), 50176)  # alias
        # Source Nodes: [cat_6, x_188, x_191], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19.run(buf102, primals_186, primals_187, primals_67, primals_68, buf103, buf113, 43904, grid=grid(43904), stream=stream0)
        del primals_68
        # Source Nodes: [x_192], Original ATen: [aten.convolution]
        buf104 = extern_kernels.convolution(buf103, primals_113, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (4, 224, 7, 7), (10976, 49, 7, 1))
        buf105 = empty((4, 224, 7, 7), device='cuda', dtype=torch.float32)
        buf114 = reinterpret_tensor(buf117, (4, 224, 7, 7), (105056, 49, 7, 1), 61152)  # alias
        # Source Nodes: [cat_6, x_193, x_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19.run(buf104, primals_188, primals_189, primals_69, primals_70, buf105, buf114, 43904, grid=grid(43904), stream=stream0)
        del primals_70
        # Source Nodes: [x_197], Original ATen: [aten.convolution]
        buf106 = extern_kernels.convolution(buf105, primals_114, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (4, 224, 7, 7), (10976, 49, 7, 1))
        buf107 = empty((4, 224, 7, 7), device='cuda', dtype=torch.float32)
        buf115 = reinterpret_tensor(buf117, (4, 224, 7, 7), (105056, 49, 7, 1), 72128)  # alias
        # Source Nodes: [cat_6, x_198, x_201], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19.run(buf106, primals_190, primals_191, primals_71, primals_72, buf107, buf115, 43904, grid=grid(43904), stream=stream0)
        del primals_72
        # Source Nodes: [x_202], Original ATen: [aten.convolution]
        buf108 = extern_kernels.convolution(buf107, primals_115, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf108, (4, 224, 7, 7), (10976, 49, 7, 1))
        buf109 = empty((4, 224, 7, 7), device='cuda', dtype=torch.float32)
        buf116 = reinterpret_tensor(buf117, (4, 224, 7, 7), (105056, 49, 7, 1), 83104)  # alias
        # Source Nodes: [cat_6, x_203, x_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19.run(buf108, primals_192, primals_193, primals_73, primals_74, buf109, buf116, 43904, grid=grid(43904), stream=stream0)
        del primals_74
        # Source Nodes: [x_207], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, primals_116, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (4, 224, 7, 7), (10976, 49, 7, 1))
        buf111 = reinterpret_tensor(buf117, (4, 224, 7, 7), (105056, 49, 7, 1), 94080)  # alias
        buf124 = empty((4, 224, 7, 7), device='cuda', dtype=torch.bool)
        # Source Nodes: [x_208, x_211], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu, aten.threshold_backward]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_threshold_backward_20.run(buf110, primals_194, primals_195, primals_75, primals_76, buf111, buf124, 43904, grid=grid(43904), stream=stream0)
        del primals_76
        # Source Nodes: [x_214], Original ATen: [aten.convolution]
        buf118 = extern_kernels.convolution(buf117, primals_117, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf118, (4, 1024, 7, 7), (50176, 49, 7, 1))
        buf123 = empty((4, 1024, 7, 7), device='cuda', dtype=torch.bool)
        buf120 = empty_strided((4, 1024, 1, 1), (1024, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf121 = reinterpret_tensor(buf120, (4, 1024), (1024, 1), 0); del buf120  # reuse
        # Source Nodes: [x_215, x_221, x_222, x_224], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu, aten.threshold_backward, aten.view]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_threshold_backward_view_21.run(buf121, buf118, primals_196, primals_197, primals_77, primals_78, buf123, 4096, 49, grid=grid(4096), stream=stream0)
        del primals_78
        buf122 = empty((4, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_226], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_119, buf121, reinterpret_tensor(primals_118, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf122)
        del primals_119
        return (buf122, primals_1, primals_3, primals_5, primals_7, primals_9, primals_11, primals_13, primals_15, primals_17, primals_19, primals_21, primals_23, primals_25, primals_27, primals_29, primals_31, primals_33, primals_35, primals_37, primals_39, primals_41, primals_43, primals_45, primals_47, primals_49, primals_51, primals_53, primals_55, primals_57, primals_59, primals_61, primals_63, primals_65, primals_67, primals_69, primals_71, primals_73, primals_75, primals_77, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198, buf0, buf1, buf2, buf3, buf4, buf5, buf6, buf7, buf8, buf9, buf10, buf11, buf12, buf13, buf14, buf21, buf22, buf23, buf24, buf25, buf26, buf27, buf28, buf29, buf30, buf31, buf32, buf33, buf34, buf41, buf42, buf43, buf44, buf45, buf46, buf47, buf48, buf49, buf50, buf51, buf52, buf53, buf54, buf61, buf62, buf63, buf64, buf65, buf66, buf67, buf68, buf69, buf70, buf71, buf72, buf79, buf80, buf81, buf82, buf83, buf84, buf85, buf86, buf87, buf88, buf89, buf90, buf91, buf92, buf99, buf100, buf101, buf102, buf103, buf104, buf105, buf106, buf107, buf108, buf109, buf110, buf117, buf118, buf121, reinterpret_tensor(primals_118, (1000, 1024), (1024, 1), 0), buf123, buf124, buf125, buf126, buf127, buf128, buf129, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_54 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_55 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_56 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_57 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_58 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_59 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_60 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_61 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_62 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_63 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_64 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_65 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_66 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_67 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_68 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_69 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_70 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_71 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_72 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_73 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_74 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_75 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_76 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_77 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_78 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_79 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_80 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_81 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_82 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_83 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_84 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_85 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_86 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_87 = rand_strided((256, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_88 = rand_strided((160, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_89 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_90 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_91 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_92 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_93 = rand_strided((512, 1056, 1, 1), (1056, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_94 = rand_strided((192, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_95 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_96 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_97 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_98 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_99 = rand_strided((768, 1472, 1, 1), (1472, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_100 = rand_strided((192, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_101 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_102 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_103 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_104 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_105 = rand_strided((768, 1728, 1, 1), (1728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_106 = rand_strided((224, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_107 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_108 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_109 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_110 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_111 = rand_strided((1024, 1888, 1, 1), (1888, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_112 = rand_strided((224, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_113 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_114 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_115 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_116 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_117 = rand_strided((1024, 2144, 1, 1), (2144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_118 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    primals_119 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_120 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_121 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_122 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_123 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_124 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_125 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_126 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_127 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_128 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_129 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_130 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_131 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_132 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_133 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_134 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_135 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_136 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_137 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_138 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_139 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_140 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_141 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_142 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_143 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_144 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_145 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_146 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_147 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_148 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_149 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_150 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_151 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_152 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_153 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_154 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_155 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_156 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_157 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_158 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_159 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_160 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_161 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_162 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_163 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_164 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_165 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_166 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_167 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_168 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_169 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_170 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_171 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_172 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_173 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_174 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_175 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_176 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_177 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_178 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_179 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_180 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_181 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_182 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_183 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_184 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_185 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_186 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_187 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_188 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_189 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_190 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_191 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_192 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_193 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_194 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_195 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_196 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_197 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_198 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175, primals_176, primals_177, primals_178, primals_179, primals_180, primals_181, primals_182, primals_183, primals_184, primals_185, primals_186, primals_187, primals_188, primals_189, primals_190, primals_191, primals_192, primals_193, primals_194, primals_195, primals_196, primals_197, primals_198]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('timm_vovnet', benchmark_compiled_module)
