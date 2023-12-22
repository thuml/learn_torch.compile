
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


# kernel path: /tmp/torchinductor_youkaichao/p4/cp4iyzq5k3afr2yb5ks64ma7rjy2vkfsxgkjzzddecv7ytpvgda5.py
# Source Nodes: [l__mod___features_0_conv1_0, l__mod___stem_1, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___features_0_conv1_0 => convolution_1
# l__mod___stem_1 => add_1, mul_1, mul_2, sub
# x => relu
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 32
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


# kernel path: /tmp/torchinductor_youkaichao/rw/crwxcbbcfeaez32rev7l3jqg7nvdiuwyzhdxnf7o5lfjds5sr3p7.py
# Source Nodes: [d1, l__mod___features_0_conv1_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# d1 => relu_1
# l__mod___features_0_conv1_1 => add_3, mul_4, mul_5, sub_1
triton_poi_fused__native_batch_norm_legit_no_training_relu_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 64
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


# kernel path: /tmp/torchinductor_youkaichao/w2/cw2ikhdoc3dl2e6v2q4nrtkrfxmh25t4wctal4fodhkbgmayyqbw.py
# Source Nodes: [d2, l__mod___features_0_conv3_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# d2 => relu_3
# l__mod___features_0_conv3_1 => add_7, mul_10, mul_11, sub_3
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 802816
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 32
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


# kernel path: /tmp/torchinductor_youkaichao/vt/cvtobvqh3aqm5nmsknfiifvdvptkwhax4a27uhaq2icnlcytks4d.py
# Source Nodes: [cat_11, l__mod___features_0_conv6_0], Original ATen: [aten.cat, aten.convolution]
# cat_11 => cat
# l__mod___features_0_conv6_0 => convolution_6
triton_poi_fused_cat_convolution_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3136) % 128
    x2 = (xindex // 401408)
    x3 = xindex % 401408
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (200704*x2)), tmp4, other=0.0)
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
    tmp19 = tl.load(in_ptr3 + ((-96) + x1), tmp15, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 - tmp19
    tmp21 = tl.load(in_ptr4 + ((-96) + x1), tmp15, eviction_policy='evict_last', other=0.0)
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = tl.sqrt(tmp23)
    tmp25 = 1 / tmp24
    tmp26 = 1.0
    tmp27 = tmp25 * tmp26
    tmp28 = tmp20 * tmp27
    tmp29 = tl.load(in_ptr5 + ((-96) + x1), tmp15, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 * tmp29
    tmp31 = tl.load(in_ptr6 + ((-96) + x1), tmp15, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp30 + tmp31
    tmp33 = triton_helpers.maximum(0, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp15, tmp33, tmp34)
    tmp36 = tl.where(tmp11, tmp14, tmp35)
    tmp37 = tl.where(tmp4, tmp7, tmp36)
    tl.store(out_ptr0 + (x4), tmp37, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/n4/cn4txyl7ewclgg6lidmqkmnpoxkuhhr3jw2p6lafco2vrgtoki2s.py
# Source Nodes: [cat_10, l__mod___features_1_conv6_0], Original ATen: [aten.cat, aten.convolution]
# cat_10 => cat_1
# l__mod___features_1_conv6_0 => convolution_12
triton_poi_fused_cat_convolution_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4816896
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3136) % 192
    x2 = (xindex // 602112)
    x3 = xindex % 602112
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (200704*x2)), tmp4, other=0.0)
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
    tmp20 = tl.load(in_ptr3 + ((-96) + x1), tmp18, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 - tmp20
    tmp22 = tl.load(in_ptr4 + ((-96) + x1), tmp18, eviction_policy='evict_last', other=0.0)
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = tl.sqrt(tmp24)
    tmp26 = 1 / tmp25
    tmp27 = 1.0
    tmp28 = tmp26 * tmp27
    tmp29 = tmp21 * tmp28
    tmp30 = tl.load(in_ptr5 + ((-96) + x1), tmp18, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 * tmp30
    tmp32 = tl.load(in_ptr6 + ((-96) + x1), tmp18, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 + tmp32
    tmp34 = triton_helpers.maximum(0, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp18, tmp34, tmp35)
    tmp37 = tmp0 >= tmp16
    tmp38 = tl.full([1], 192, tl.int64)
    tmp39 = tmp0 < tmp38
    tmp40 = tl.load(in_ptr7 + ((-401408) + x3 + (200704*x2)), tmp37, other=0.0)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp37, tmp40, tmp41)
    tmp43 = tl.where(tmp18, tmp36, tmp42)
    tmp44 = tl.where(tmp11, tmp14, tmp43)
    tmp45 = tl.where(tmp4, tmp7, tmp44)
    tl.store(out_ptr0 + (x4), tmp45, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/l5/cl5ypdsc5oz3vetauqurecrbdfilntgu3rrdbxhn6id6hdnwubqv.py
# Source Nodes: [l__mod___features_1_conv6_1, l__mod___features_1_conv6_2, l__mod___features_2_conv1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___features_1_conv6_1 => add_25, mul_37, mul_38, sub_12
# l__mod___features_1_conv6_2 => relu_12
# l__mod___features_2_conv1_0 => convolution_13
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
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


# kernel path: /tmp/torchinductor_youkaichao/4i/c4iz2ugo3gfr5w2dcmzl463h6sr7mol7mbdjw7kzvste6liwnukq.py
# Source Nodes: [d1_2, l__mod___features_2_conv1_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# d1_2 => relu_13
# l__mod___features_2_conv1_1 => add_27, mul_40, mul_41, sub_13
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 903168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 144
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


# kernel path: /tmp/torchinductor_youkaichao/n2/cn26ruevj6j336wzicyrrxoyuetexw6fddm5asflwjxjcmupwaza.py
# Source Nodes: [d2_2, l__mod___features_2_conv3_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# d2_2 => relu_15
# l__mod___features_2_conv3_1 => add_31, mul_46, mul_47, sub_15
triton_poi_fused__native_batch_norm_legit_no_training_relu_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 451584
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 72
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


# kernel path: /tmp/torchinductor_youkaichao/hm/chmeujvoke2igzonihsiuxdqk25wiz7e27qqnzsieuimb57e4efh.py
# Source Nodes: [cat_9, l__mod___features_2_conv6_0], Original ATen: [aten.cat, aten.convolution]
# cat_9 => cat_2
# l__mod___features_2_conv6_0 => convolution_18
triton_poi_fused_cat_convolution_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1806336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 288
    x2 = (xindex // 225792)
    x3 = xindex % 225792
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 144, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (112896*x2)), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 216, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-112896) + x3 + (56448*x2)), tmp11, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 288, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-169344) + x3 + (56448*x2)), tmp15, other=0.0)
    tmp19 = tl.load(in_ptr3 + ((-216) + x1), tmp15, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 - tmp19
    tmp21 = tl.load(in_ptr4 + ((-216) + x1), tmp15, eviction_policy='evict_last', other=0.0)
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = tl.sqrt(tmp23)
    tmp25 = 1 / tmp24
    tmp26 = 1.0
    tmp27 = tmp25 * tmp26
    tmp28 = tmp20 * tmp27
    tmp29 = tl.load(in_ptr5 + ((-216) + x1), tmp15, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 * tmp29
    tmp31 = tl.load(in_ptr6 + ((-216) + x1), tmp15, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp30 + tmp31
    tmp33 = triton_helpers.maximum(0, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp15, tmp33, tmp34)
    tmp36 = tl.where(tmp11, tmp14, tmp35)
    tmp37 = tl.where(tmp4, tmp7, tmp36)
    tl.store(out_ptr0 + (x4), tmp37, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/fh/cfhy26n3ltppbuwwdrqyzpvbserkajvysqnjrsgf422wdxkj6yhz.py
# Source Nodes: [cat_8, l__mod___features_3_conv6_0], Original ATen: [aten.cat, aten.convolution]
# cat_8 => cat_3
# l__mod___features_3_conv6_0 => convolution_24
triton_poi_fused_cat_convolution_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2709504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 432
    x2 = (xindex // 338688)
    x3 = xindex % 338688
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 144, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (112896*x2)), tmp4, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 216, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-112896) + x3 + (56448*x2)), tmp11, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 288, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-169344) + x3 + (56448*x2)), tmp18, other=0.0)
    tmp20 = tl.load(in_ptr3 + ((-216) + x1), tmp18, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 - tmp20
    tmp22 = tl.load(in_ptr4 + ((-216) + x1), tmp18, eviction_policy='evict_last', other=0.0)
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = tl.sqrt(tmp24)
    tmp26 = 1 / tmp25
    tmp27 = 1.0
    tmp28 = tmp26 * tmp27
    tmp29 = tmp21 * tmp28
    tmp30 = tl.load(in_ptr5 + ((-216) + x1), tmp18, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 * tmp30
    tmp32 = tl.load(in_ptr6 + ((-216) + x1), tmp18, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 + tmp32
    tmp34 = triton_helpers.maximum(0, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp18, tmp34, tmp35)
    tmp37 = tmp0 >= tmp16
    tmp38 = tl.full([1], 432, tl.int64)
    tmp39 = tmp0 < tmp38
    tmp40 = tl.load(in_ptr7 + ((-225792) + x3 + (112896*x2)), tmp37, other=0.0)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp37, tmp40, tmp41)
    tmp43 = tl.where(tmp18, tmp36, tmp42)
    tmp44 = tl.where(tmp11, tmp14, tmp43)
    tmp45 = tl.where(tmp4, tmp7, tmp44)
    tl.store(out_ptr0 + (x4), tmp45, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/th/cth5eosyefghtyxgzw3fpj7y46lmgfrn4yowcazlassobd66hsbx.py
# Source Nodes: [l__mod___features_3_conv6_1, l__mod___features_3_conv6_2, l__mod___features_4_conv1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___features_3_conv6_1 => add_49, mul_73, mul_74, sub_24
# l__mod___features_3_conv6_2 => relu_24
# l__mod___features_4_conv1_0 => convolution_25
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1806336
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


# kernel path: /tmp/torchinductor_youkaichao/al/caluwbul4766k65pozsjukm7khsgrrf5odfp3qn4pnto7xsxpd5t.py
# Source Nodes: [d1_4, l__mod___features_4_conv1_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# d1_4 => relu_25
# l__mod___features_4_conv1_1 => add_51, mul_76, mul_77, sub_25
triton_poi_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 476672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 304
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


# kernel path: /tmp/torchinductor_youkaichao/ss/csslysanqrdepnac73oow7r7aan3rz2p6tjd2noqba4gz42n3lth.py
# Source Nodes: [d2_4, l__mod___features_4_conv3_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# d2_4 => relu_27
# l__mod___features_4_conv3_1 => add_55, mul_82, mul_83, sub_27
triton_poi_fused__native_batch_norm_legit_no_training_relu_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 238336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 152
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


# kernel path: /tmp/torchinductor_youkaichao/hb/chbgsgkptcmz75zgpe3f5gauh2gjjmyws6fyglwdwxxtzxvbyoht.py
# Source Nodes: [cat_7, l__mod___features_4_conv6_0], Original ATen: [aten.cat, aten.convolution]
# cat_7 => cat_4
# l__mod___features_4_conv6_0 => convolution_30
triton_poi_fused_cat_convolution_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 953344
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 608
    x2 = (xindex // 119168)
    x3 = xindex % 119168
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 304, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (59584*x2)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 456, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-59584) + x3 + (29792*x2)), tmp11 & xmask, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 608, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tl.load(in_ptr2 + ((-89376) + x3 + (29792*x2)), tmp15 & xmask, other=0.0)
    tmp19 = tl.load(in_ptr3 + ((-456) + x1), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp20 = tmp18 - tmp19
    tmp21 = tl.load(in_ptr4 + ((-456) + x1), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = tl.sqrt(tmp23)
    tmp25 = 1 / tmp24
    tmp26 = 1.0
    tmp27 = tmp25 * tmp26
    tmp28 = tmp20 * tmp27
    tmp29 = tl.load(in_ptr5 + ((-456) + x1), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp28 * tmp29
    tmp31 = tl.load(in_ptr6 + ((-456) + x1), tmp15 & xmask, eviction_policy='evict_last', other=0.0)
    tmp32 = tmp30 + tmp31
    tmp33 = triton_helpers.maximum(0, tmp32)
    tmp34 = tl.full(tmp33.shape, 0.0, tmp33.dtype)
    tmp35 = tl.where(tmp15, tmp33, tmp34)
    tmp36 = tl.where(tmp11, tmp14, tmp35)
    tmp37 = tl.where(tmp4, tmp7, tmp36)
    tl.store(out_ptr0 + (x4), tmp37, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i3/ci3hkwlvjvpkjm7pyly6fg6e3f44x5nd64nway4az6phbbmo74qd.py
# Source Nodes: [cat_6, l__mod___features_5_conv6_0], Original ATen: [aten.cat, aten.convolution]
# cat_6 => cat_5
# l__mod___features_5_conv6_0 => convolution_36
triton_poi_fused_cat_convolution_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1430016
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 912
    x2 = (xindex // 178752)
    x3 = xindex % 178752
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 304, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (59584*x2)), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 456, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tmp8 & tmp10
    tmp12 = tl.load(in_ptr1 + ((-59584) + x3 + (29792*x2)), tmp11 & xmask, other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = tmp0 >= tmp9
    tmp16 = tl.full([1], 608, tl.int64)
    tmp17 = tmp0 < tmp16
    tmp18 = tmp15 & tmp17
    tmp19 = tl.load(in_ptr2 + ((-89376) + x3 + (29792*x2)), tmp18 & xmask, other=0.0)
    tmp20 = tl.load(in_ptr3 + ((-456) + x1), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tmp19 - tmp20
    tmp22 = tl.load(in_ptr4 + ((-456) + x1), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = 1e-05
    tmp24 = tmp22 + tmp23
    tmp25 = tl.sqrt(tmp24)
    tmp26 = 1 / tmp25
    tmp27 = 1.0
    tmp28 = tmp26 * tmp27
    tmp29 = tmp21 * tmp28
    tmp30 = tl.load(in_ptr5 + ((-456) + x1), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp29 * tmp30
    tmp32 = tl.load(in_ptr6 + ((-456) + x1), tmp18 & xmask, eviction_policy='evict_last', other=0.0)
    tmp33 = tmp31 + tmp32
    tmp34 = triton_helpers.maximum(0, tmp33)
    tmp35 = tl.full(tmp34.shape, 0.0, tmp34.dtype)
    tmp36 = tl.where(tmp18, tmp34, tmp35)
    tmp37 = tmp0 >= tmp16
    tmp38 = tl.full([1], 912, tl.int64)
    tmp39 = tmp0 < tmp38
    tmp40 = tl.load(in_ptr7 + ((-119168) + x3 + (59584*x2)), tmp37 & xmask, other=0.0)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp37, tmp40, tmp41)
    tmp43 = tl.where(tmp18, tmp36, tmp42)
    tmp44 = tl.where(tmp11, tmp14, tmp43)
    tmp45 = tl.where(tmp4, tmp7, tmp44)
    tl.store(out_ptr0 + (x4), tmp45, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ed/ced5dpf3cc55mrxcmc4rpcctez7lkgvaoctyqpumjirhvgxh77p2.py
# Source Nodes: [l__mod___features_5_conv6_1, l__mod___features_5_conv6_2, l__mod___head_0_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___features_5_conv6_1 => add_73, mul_109, mul_110, sub_36
# l__mod___features_5_conv6_2 => relu_36
# l__mod___head_0_0 => convolution_37
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 752640
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


# kernel path: /tmp/torchinductor_youkaichao/7l/c7llwvyoivn3hoi7gxl64mlwrutlxnokuopqrr4vhhlr5e4rwpbi.py
# Source Nodes: [l__mod___head_0_1, l__mod___head_0_2, l__mod___head_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___head_0_1 => add_75, mul_112, mul_113, sub_37
# l__mod___head_0_2 => relu_37
# l__mod___head_1_0 => convolution_38
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 376320
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


# kernel path: /tmp/torchinductor_youkaichao/iy/ciy2vtl6wgrnpxx662cmls57l637v24i56nqgrjtzsutzxjee3ax.py
# Source Nodes: [l__mod___head_1_1, l__mod___head_1_2, l__mod___head_2_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___head_1_1 => add_77, mul_115, mul_116, sub_38
# l__mod___head_1_2 => relu_38
# l__mod___head_2_0 => convolution_39
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 1024
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


# kernel path: /tmp/torchinductor_youkaichao/cz/cczdlosym4cj5wfw7f6vc2y4qtasq2zd7k4yreyfxk7wer2wceze.py
# Source Nodes: [l__mod___head_2_1, l__mod___head_2_2, l__mod___head_3_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___head_2_1 => add_79, mul_118, mul_119, sub_39
# l__mod___head_2_2 => relu_39
# l__mod___head_3_0 => convolution_40
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 16) % 1280
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


# kernel path: /tmp/torchinductor_youkaichao/h3/ch3zaafumktmtizwcuu6bnkxfhmld5ipi4o63uyccwt5gc2hlbmv.py
# Source Nodes: [l__mod___head_3_1, x_2, x_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
# l__mod___head_3_1 => add_81, mul_121, mul_122, sub_40
# x_2 => relu_40
# x_3 => mean
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 16],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_19', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 8192
    rnumel = 16
    RBLOCK: tl.constexpr = 16
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1024
    tmp0 = tl.load(in_ptr0 + (r2 + (16*x3)), rmask, other=0.0)
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
    tmp20 = 16.0
    tmp21 = tmp19 / tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp21, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (32, ), (1, ))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg10_1, (32, ), (1, ))
    assert_size_stride(arg11_1, (32, ), (1, ))
    assert_size_stride(arg12_1, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg13_1, (64, ), (1, ))
    assert_size_stride(arg14_1, (64, ), (1, ))
    assert_size_stride(arg15_1, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg16_1, (32, ), (1, ))
    assert_size_stride(arg17_1, (32, ), (1, ))
    assert_size_stride(arg18_1, (64, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg19_1, (64, ), (1, ))
    assert_size_stride(arg20_1, (64, ), (1, ))
    assert_size_stride(arg21_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg22_1, (64, ), (1, ))
    assert_size_stride(arg23_1, (64, ), (1, ))
    assert_size_stride(arg24_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg25_1, (64, ), (1, ))
    assert_size_stride(arg26_1, (64, ), (1, ))
    assert_size_stride(arg27_1, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg28_1, (32, ), (1, ))
    assert_size_stride(arg29_1, (32, ), (1, ))
    assert_size_stride(arg30_1, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg31_1, (64, ), (1, ))
    assert_size_stride(arg32_1, (64, ), (1, ))
    assert_size_stride(arg33_1, (32, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg34_1, (32, ), (1, ))
    assert_size_stride(arg35_1, (32, ), (1, ))
    assert_size_stride(arg36_1, (128, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg37_1, (128, ), (1, ))
    assert_size_stride(arg38_1, (128, ), (1, ))
    assert_size_stride(arg39_1, (144, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg40_1, (144, ), (1, ))
    assert_size_stride(arg41_1, (144, ), (1, ))
    assert_size_stride(arg42_1, (144, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg43_1, (144, ), (1, ))
    assert_size_stride(arg44_1, (144, ), (1, ))
    assert_size_stride(arg45_1, (72, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg46_1, (72, ), (1, ))
    assert_size_stride(arg47_1, (72, ), (1, ))
    assert_size_stride(arg48_1, (144, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg49_1, (144, ), (1, ))
    assert_size_stride(arg50_1, (144, ), (1, ))
    assert_size_stride(arg51_1, (72, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg52_1, (72, ), (1, ))
    assert_size_stride(arg53_1, (72, ), (1, ))
    assert_size_stride(arg54_1, (144, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg55_1, (144, ), (1, ))
    assert_size_stride(arg56_1, (144, ), (1, ))
    assert_size_stride(arg57_1, (144, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg58_1, (144, ), (1, ))
    assert_size_stride(arg59_1, (144, ), (1, ))
    assert_size_stride(arg60_1, (144, 144, 1, 1), (144, 1, 1, 1))
    assert_size_stride(arg61_1, (144, ), (1, ))
    assert_size_stride(arg62_1, (144, ), (1, ))
    assert_size_stride(arg63_1, (72, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg64_1, (72, ), (1, ))
    assert_size_stride(arg65_1, (72, ), (1, ))
    assert_size_stride(arg66_1, (144, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg67_1, (144, ), (1, ))
    assert_size_stride(arg68_1, (144, ), (1, ))
    assert_size_stride(arg69_1, (72, 144, 3, 3), (1296, 9, 3, 1))
    assert_size_stride(arg70_1, (72, ), (1, ))
    assert_size_stride(arg71_1, (72, ), (1, ))
    assert_size_stride(arg72_1, (288, 432, 1, 1), (432, 1, 1, 1))
    assert_size_stride(arg73_1, (288, ), (1, ))
    assert_size_stride(arg74_1, (288, ), (1, ))
    assert_size_stride(arg75_1, (304, 288, 3, 3), (2592, 9, 3, 1))
    assert_size_stride(arg76_1, (304, ), (1, ))
    assert_size_stride(arg77_1, (304, ), (1, ))
    assert_size_stride(arg78_1, (304, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(arg79_1, (304, ), (1, ))
    assert_size_stride(arg80_1, (304, ), (1, ))
    assert_size_stride(arg81_1, (152, 304, 3, 3), (2736, 9, 3, 1))
    assert_size_stride(arg82_1, (152, ), (1, ))
    assert_size_stride(arg83_1, (152, ), (1, ))
    assert_size_stride(arg84_1, (304, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg85_1, (304, ), (1, ))
    assert_size_stride(arg86_1, (304, ), (1, ))
    assert_size_stride(arg87_1, (152, 304, 3, 3), (2736, 9, 3, 1))
    assert_size_stride(arg88_1, (152, ), (1, ))
    assert_size_stride(arg89_1, (152, ), (1, ))
    assert_size_stride(arg90_1, (304, 608, 1, 1), (608, 1, 1, 1))
    assert_size_stride(arg91_1, (304, ), (1, ))
    assert_size_stride(arg92_1, (304, ), (1, ))
    assert_size_stride(arg93_1, (304, 304, 3, 3), (2736, 9, 3, 1))
    assert_size_stride(arg94_1, (304, ), (1, ))
    assert_size_stride(arg95_1, (304, ), (1, ))
    assert_size_stride(arg96_1, (304, 304, 1, 1), (304, 1, 1, 1))
    assert_size_stride(arg97_1, (304, ), (1, ))
    assert_size_stride(arg98_1, (304, ), (1, ))
    assert_size_stride(arg99_1, (152, 304, 3, 3), (2736, 9, 3, 1))
    assert_size_stride(arg100_1, (152, ), (1, ))
    assert_size_stride(arg101_1, (152, ), (1, ))
    assert_size_stride(arg102_1, (304, 152, 1, 1), (152, 1, 1, 1))
    assert_size_stride(arg103_1, (304, ), (1, ))
    assert_size_stride(arg104_1, (304, ), (1, ))
    assert_size_stride(arg105_1, (152, 304, 3, 3), (2736, 9, 3, 1))
    assert_size_stride(arg106_1, (152, ), (1, ))
    assert_size_stride(arg107_1, (152, ), (1, ))
    assert_size_stride(arg108_1, (480, 912, 1, 1), (912, 1, 1, 1))
    assert_size_stride(arg109_1, (480, ), (1, ))
    assert_size_stride(arg110_1, (480, ), (1, ))
    assert_size_stride(arg111_1, (960, 480, 3, 3), (4320, 9, 3, 1))
    assert_size_stride(arg112_1, (960, ), (1, ))
    assert_size_stride(arg113_1, (960, ), (1, ))
    assert_size_stride(arg114_1, (1024, 960, 3, 3), (8640, 9, 3, 1))
    assert_size_stride(arg115_1, (1024, ), (1, ))
    assert_size_stride(arg116_1, (1024, ), (1, ))
    assert_size_stride(arg117_1, (1280, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(arg118_1, (1280, ), (1, ))
    assert_size_stride(arg119_1, (1280, ), (1, ))
    assert_size_stride(arg120_1, (1024, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(arg121_1, (1024, ), (1, ))
    assert_size_stride(arg122_1, (1024, ), (1, ))
    assert_size_stride(arg123_1, (1000, 1024), (1024, 1))
    assert_size_stride(arg124_1, (1000, ), (1, ))
    assert_size_stride(arg125_1, (32, ), (1, ))
    assert_size_stride(arg126_1, (32, ), (1, ))
    assert_size_stride(arg127_1, (), ())
    assert_size_stride(arg128_1, (64, ), (1, ))
    assert_size_stride(arg129_1, (64, ), (1, ))
    assert_size_stride(arg130_1, (), ())
    assert_size_stride(arg131_1, (64, ), (1, ))
    assert_size_stride(arg132_1, (64, ), (1, ))
    assert_size_stride(arg133_1, (), ())
    assert_size_stride(arg134_1, (32, ), (1, ))
    assert_size_stride(arg135_1, (32, ), (1, ))
    assert_size_stride(arg136_1, (), ())
    assert_size_stride(arg137_1, (64, ), (1, ))
    assert_size_stride(arg138_1, (64, ), (1, ))
    assert_size_stride(arg139_1, (), ())
    assert_size_stride(arg140_1, (32, ), (1, ))
    assert_size_stride(arg141_1, (32, ), (1, ))
    assert_size_stride(arg142_1, (), ())
    assert_size_stride(arg143_1, (64, ), (1, ))
    assert_size_stride(arg144_1, (64, ), (1, ))
    assert_size_stride(arg145_1, (), ())
    assert_size_stride(arg146_1, (64, ), (1, ))
    assert_size_stride(arg147_1, (64, ), (1, ))
    assert_size_stride(arg148_1, (), ())
    assert_size_stride(arg149_1, (64, ), (1, ))
    assert_size_stride(arg150_1, (64, ), (1, ))
    assert_size_stride(arg151_1, (), ())
    assert_size_stride(arg152_1, (32, ), (1, ))
    assert_size_stride(arg153_1, (32, ), (1, ))
    assert_size_stride(arg154_1, (), ())
    assert_size_stride(arg155_1, (64, ), (1, ))
    assert_size_stride(arg156_1, (64, ), (1, ))
    assert_size_stride(arg157_1, (), ())
    assert_size_stride(arg158_1, (32, ), (1, ))
    assert_size_stride(arg159_1, (32, ), (1, ))
    assert_size_stride(arg160_1, (), ())
    assert_size_stride(arg161_1, (128, ), (1, ))
    assert_size_stride(arg162_1, (128, ), (1, ))
    assert_size_stride(arg163_1, (), ())
    assert_size_stride(arg164_1, (144, ), (1, ))
    assert_size_stride(arg165_1, (144, ), (1, ))
    assert_size_stride(arg166_1, (), ())
    assert_size_stride(arg167_1, (144, ), (1, ))
    assert_size_stride(arg168_1, (144, ), (1, ))
    assert_size_stride(arg169_1, (), ())
    assert_size_stride(arg170_1, (72, ), (1, ))
    assert_size_stride(arg171_1, (72, ), (1, ))
    assert_size_stride(arg172_1, (), ())
    assert_size_stride(arg173_1, (144, ), (1, ))
    assert_size_stride(arg174_1, (144, ), (1, ))
    assert_size_stride(arg175_1, (), ())
    assert_size_stride(arg176_1, (72, ), (1, ))
    assert_size_stride(arg177_1, (72, ), (1, ))
    assert_size_stride(arg178_1, (), ())
    assert_size_stride(arg179_1, (144, ), (1, ))
    assert_size_stride(arg180_1, (144, ), (1, ))
    assert_size_stride(arg181_1, (), ())
    assert_size_stride(arg182_1, (144, ), (1, ))
    assert_size_stride(arg183_1, (144, ), (1, ))
    assert_size_stride(arg184_1, (), ())
    assert_size_stride(arg185_1, (144, ), (1, ))
    assert_size_stride(arg186_1, (144, ), (1, ))
    assert_size_stride(arg187_1, (), ())
    assert_size_stride(arg188_1, (72, ), (1, ))
    assert_size_stride(arg189_1, (72, ), (1, ))
    assert_size_stride(arg190_1, (), ())
    assert_size_stride(arg191_1, (144, ), (1, ))
    assert_size_stride(arg192_1, (144, ), (1, ))
    assert_size_stride(arg193_1, (), ())
    assert_size_stride(arg194_1, (72, ), (1, ))
    assert_size_stride(arg195_1, (72, ), (1, ))
    assert_size_stride(arg196_1, (), ())
    assert_size_stride(arg197_1, (288, ), (1, ))
    assert_size_stride(arg198_1, (288, ), (1, ))
    assert_size_stride(arg199_1, (), ())
    assert_size_stride(arg200_1, (304, ), (1, ))
    assert_size_stride(arg201_1, (304, ), (1, ))
    assert_size_stride(arg202_1, (), ())
    assert_size_stride(arg203_1, (304, ), (1, ))
    assert_size_stride(arg204_1, (304, ), (1, ))
    assert_size_stride(arg205_1, (), ())
    assert_size_stride(arg206_1, (152, ), (1, ))
    assert_size_stride(arg207_1, (152, ), (1, ))
    assert_size_stride(arg208_1, (), ())
    assert_size_stride(arg209_1, (304, ), (1, ))
    assert_size_stride(arg210_1, (304, ), (1, ))
    assert_size_stride(arg211_1, (), ())
    assert_size_stride(arg212_1, (152, ), (1, ))
    assert_size_stride(arg213_1, (152, ), (1, ))
    assert_size_stride(arg214_1, (), ())
    assert_size_stride(arg215_1, (304, ), (1, ))
    assert_size_stride(arg216_1, (304, ), (1, ))
    assert_size_stride(arg217_1, (), ())
    assert_size_stride(arg218_1, (304, ), (1, ))
    assert_size_stride(arg219_1, (304, ), (1, ))
    assert_size_stride(arg220_1, (), ())
    assert_size_stride(arg221_1, (304, ), (1, ))
    assert_size_stride(arg222_1, (304, ), (1, ))
    assert_size_stride(arg223_1, (), ())
    assert_size_stride(arg224_1, (152, ), (1, ))
    assert_size_stride(arg225_1, (152, ), (1, ))
    assert_size_stride(arg226_1, (), ())
    assert_size_stride(arg227_1, (304, ), (1, ))
    assert_size_stride(arg228_1, (304, ), (1, ))
    assert_size_stride(arg229_1, (), ())
    assert_size_stride(arg230_1, (152, ), (1, ))
    assert_size_stride(arg231_1, (152, ), (1, ))
    assert_size_stride(arg232_1, (), ())
    assert_size_stride(arg233_1, (480, ), (1, ))
    assert_size_stride(arg234_1, (480, ), (1, ))
    assert_size_stride(arg235_1, (), ())
    assert_size_stride(arg236_1, (960, ), (1, ))
    assert_size_stride(arg237_1, (960, ), (1, ))
    assert_size_stride(arg238_1, (), ())
    assert_size_stride(arg239_1, (1024, ), (1, ))
    assert_size_stride(arg240_1, (1024, ), (1, ))
    assert_size_stride(arg241_1, (), ())
    assert_size_stride(arg242_1, (1280, ), (1, ))
    assert_size_stride(arg243_1, (1280, ), (1, ))
    assert_size_stride(arg244_1, (), ())
    assert_size_stride(arg245_1, (1024, ), (1, ))
    assert_size_stride(arg246_1, (1024, ), (1, ))
    assert_size_stride(arg247_1, (), ())
    assert_size_stride(arg248_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [l__mod___stem_0], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg248_1, arg0_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (8, 32, 112, 112), (401408, 12544, 112, 1))
        del arg0_1
        del arg248_1
        buf1 = buf0; del buf0  # reuse
        # Source Nodes: [l__mod___features_0_conv1_0, l__mod___stem_1, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf1, arg125_1, arg126_1, arg1_1, arg2_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg125_1
        del arg126_1
        del arg1_1
        del arg2_1
        # Source Nodes: [l__mod___features_0_conv1_0, l__mod___stem_1, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf2 = extern_kernels.convolution(buf1, arg3_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del arg3_1
        buf3 = buf2; del buf2  # reuse
        # Source Nodes: [d1, l__mod___features_0_conv1_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf3, arg128_1, arg129_1, arg4_1, arg5_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg128_1
        del arg129_1
        del arg4_1
        del arg5_1
        # Source Nodes: [l__mod___features_0_conv2_0], Original ATen: [aten.convolution]
        buf4 = extern_kernels.convolution(buf3, arg6_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del arg6_1
        buf5 = buf4; del buf4  # reuse
        # Source Nodes: [l__mod___features_0_conv2_1, l__mod___features_0_conv2_2, l__mod___features_0_conv3_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf5, arg131_1, arg132_1, arg7_1, arg8_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg131_1
        del arg132_1
        del arg7_1
        del arg8_1
        # Source Nodes: [l__mod___features_0_conv2_1, l__mod___features_0_conv2_2, l__mod___features_0_conv3_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf6 = extern_kernels.convolution(buf5, arg9_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 32, 56, 56), (100352, 3136, 56, 1))
        del arg9_1
        del buf5
        buf7 = buf6; del buf6  # reuse
        # Source Nodes: [d2, l__mod___features_0_conv3_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf7, arg134_1, arg135_1, arg10_1, arg11_1, 802816, grid=grid(802816), stream=stream0)
        del arg10_1
        del arg11_1
        del arg134_1
        del arg135_1
        # Source Nodes: [l__mod___features_0_conv4_0], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, arg12_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del arg12_1
        buf9 = buf8; del buf8  # reuse
        # Source Nodes: [l__mod___features_0_conv4_1, l__mod___features_0_conv4_2, l__mod___features_0_conv5_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf9, arg137_1, arg138_1, arg13_1, arg14_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg137_1
        del arg138_1
        del arg13_1
        del arg14_1
        # Source Nodes: [l__mod___features_0_conv4_1, l__mod___features_0_conv4_2, l__mod___features_0_conv5_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf10 = extern_kernels.convolution(buf9, arg15_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (8, 32, 56, 56), (100352, 3136, 56, 1))
        del arg15_1
        del buf9
        buf11 = reinterpret_tensor(buf1, (8, 128, 56, 56), (401408, 3136, 56, 1), 0); del buf1  # reuse
        # Source Nodes: [cat_11, l__mod___features_0_conv6_0], Original ATen: [aten.cat, aten.convolution]
        triton_poi_fused_cat_convolution_3.run(buf3, buf7, buf10, arg140_1, arg141_1, arg16_1, arg17_1, buf11, 3211264, grid=grid(3211264), stream=stream0)
        del arg140_1
        del arg141_1
        del arg16_1
        del arg17_1
        del buf10
        del buf3
        del buf7
        # Source Nodes: [cat_11, l__mod___features_0_conv6_0], Original ATen: [aten.cat, aten.convolution]
        buf12 = extern_kernels.convolution(buf11, arg18_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del arg18_1
        del buf11
        buf13 = buf12; del buf12  # reuse
        # Source Nodes: [l__mod___features_0_conv6_1, out], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf13, arg143_1, arg144_1, arg19_1, arg20_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg143_1
        del arg144_1
        del arg19_1
        del arg20_1
        # Source Nodes: [l__mod___features_1_conv1_0], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, arg21_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del arg21_1
        buf15 = buf14; del buf14  # reuse
        # Source Nodes: [d1_1, l__mod___features_1_conv1_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf15, arg146_1, arg147_1, arg22_1, arg23_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg146_1
        del arg147_1
        del arg22_1
        del arg23_1
        # Source Nodes: [l__mod___features_1_conv2_0], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg24_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del arg24_1
        buf17 = buf16; del buf16  # reuse
        # Source Nodes: [l__mod___features_1_conv2_1, l__mod___features_1_conv2_2, l__mod___features_1_conv3_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf17, arg149_1, arg150_1, arg25_1, arg26_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg149_1
        del arg150_1
        del arg25_1
        del arg26_1
        # Source Nodes: [l__mod___features_1_conv2_1, l__mod___features_1_conv2_2, l__mod___features_1_conv3_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf18 = extern_kernels.convolution(buf17, arg27_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (8, 32, 56, 56), (100352, 3136, 56, 1))
        del arg27_1
        del buf17
        buf19 = buf18; del buf18  # reuse
        # Source Nodes: [d2_1, l__mod___features_1_conv3_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf19, arg152_1, arg153_1, arg28_1, arg29_1, 802816, grid=grid(802816), stream=stream0)
        del arg152_1
        del arg153_1
        del arg28_1
        del arg29_1
        # Source Nodes: [l__mod___features_1_conv4_0], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, arg30_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del arg30_1
        buf21 = buf20; del buf20  # reuse
        # Source Nodes: [l__mod___features_1_conv4_1, l__mod___features_1_conv4_2, l__mod___features_1_conv5_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_1.run(buf21, arg155_1, arg156_1, arg31_1, arg32_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg155_1
        del arg156_1
        del arg31_1
        del arg32_1
        # Source Nodes: [l__mod___features_1_conv4_1, l__mod___features_1_conv4_2, l__mod___features_1_conv5_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf22 = extern_kernels.convolution(buf21, arg33_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (8, 32, 56, 56), (100352, 3136, 56, 1))
        del arg33_1
        del buf21
        buf23 = empty((8, 192, 56, 56), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_10, l__mod___features_1_conv6_0], Original ATen: [aten.cat, aten.convolution]
        triton_poi_fused_cat_convolution_4.run(buf15, buf19, buf22, arg158_1, arg159_1, arg34_1, arg35_1, buf13, buf23, 4816896, grid=grid(4816896), stream=stream0)
        del arg158_1
        del arg159_1
        del arg34_1
        del arg35_1
        del buf13
        del buf15
        del buf19
        del buf22
        # Source Nodes: [cat_10, l__mod___features_1_conv6_0], Original ATen: [aten.cat, aten.convolution]
        buf24 = extern_kernels.convolution(buf23, arg36_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (8, 128, 56, 56), (401408, 3136, 56, 1))
        del arg36_1
        del buf23
        buf25 = buf24; del buf24  # reuse
        # Source Nodes: [l__mod___features_1_conv6_1, l__mod___features_1_conv6_2, l__mod___features_2_conv1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(buf25, arg161_1, arg162_1, arg37_1, arg38_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg161_1
        del arg162_1
        del arg37_1
        del arg38_1
        # Source Nodes: [l__mod___features_1_conv6_1, l__mod___features_1_conv6_2, l__mod___features_2_conv1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf26 = extern_kernels.convolution(buf25, arg39_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 144, 28, 28), (112896, 784, 28, 1))
        del arg39_1
        del buf25
        buf27 = buf26; del buf26  # reuse
        # Source Nodes: [d1_2, l__mod___features_2_conv1_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf27, arg164_1, arg165_1, arg40_1, arg41_1, 903168, grid=grid(903168), stream=stream0)
        del arg164_1
        del arg165_1
        del arg40_1
        del arg41_1
        # Source Nodes: [l__mod___features_2_conv2_0], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, arg42_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (8, 144, 28, 28), (112896, 784, 28, 1))
        del arg42_1
        buf29 = buf28; del buf28  # reuse
        # Source Nodes: [l__mod___features_2_conv2_1, l__mod___features_2_conv2_2, l__mod___features_2_conv3_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf29, arg167_1, arg168_1, arg43_1, arg44_1, 903168, grid=grid(903168), stream=stream0)
        del arg167_1
        del arg168_1
        del arg43_1
        del arg44_1
        # Source Nodes: [l__mod___features_2_conv2_1, l__mod___features_2_conv2_2, l__mod___features_2_conv3_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf30 = extern_kernels.convolution(buf29, arg45_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 72, 28, 28), (56448, 784, 28, 1))
        del arg45_1
        del buf29
        buf31 = buf30; del buf30  # reuse
        # Source Nodes: [d2_2, l__mod___features_2_conv3_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf31, arg170_1, arg171_1, arg46_1, arg47_1, 451584, grid=grid(451584), stream=stream0)
        del arg170_1
        del arg171_1
        del arg46_1
        del arg47_1
        # Source Nodes: [l__mod___features_2_conv4_0], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, arg48_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 144, 28, 28), (112896, 784, 28, 1))
        del arg48_1
        buf33 = buf32; del buf32  # reuse
        # Source Nodes: [l__mod___features_2_conv4_1, l__mod___features_2_conv4_2, l__mod___features_2_conv5_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf33, arg173_1, arg174_1, arg49_1, arg50_1, 903168, grid=grid(903168), stream=stream0)
        del arg173_1
        del arg174_1
        del arg49_1
        del arg50_1
        # Source Nodes: [l__mod___features_2_conv4_1, l__mod___features_2_conv4_2, l__mod___features_2_conv5_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf34 = extern_kernels.convolution(buf33, arg51_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (8, 72, 28, 28), (56448, 784, 28, 1))
        del arg51_1
        del buf33
        buf35 = empty((8, 288, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_9, l__mod___features_2_conv6_0], Original ATen: [aten.cat, aten.convolution]
        triton_poi_fused_cat_convolution_8.run(buf27, buf31, buf34, arg176_1, arg177_1, arg52_1, arg53_1, buf35, 1806336, grid=grid(1806336), stream=stream0)
        del arg176_1
        del arg177_1
        del arg52_1
        del arg53_1
        del buf27
        del buf31
        del buf34
        # Source Nodes: [cat_9, l__mod___features_2_conv6_0], Original ATen: [aten.cat, aten.convolution]
        buf36 = extern_kernels.convolution(buf35, arg54_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (8, 144, 28, 28), (112896, 784, 28, 1))
        del arg54_1
        del buf35
        buf37 = buf36; del buf36  # reuse
        # Source Nodes: [l__mod___features_2_conv6_1, out_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf37, arg179_1, arg180_1, arg55_1, arg56_1, 903168, grid=grid(903168), stream=stream0)
        del arg179_1
        del arg180_1
        del arg55_1
        del arg56_1
        # Source Nodes: [l__mod___features_3_conv1_0], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, arg57_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 144, 28, 28), (112896, 784, 28, 1))
        del arg57_1
        buf39 = buf38; del buf38  # reuse
        # Source Nodes: [d1_3, l__mod___features_3_conv1_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf39, arg182_1, arg183_1, arg58_1, arg59_1, 903168, grid=grid(903168), stream=stream0)
        del arg182_1
        del arg183_1
        del arg58_1
        del arg59_1
        # Source Nodes: [l__mod___features_3_conv2_0], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, arg60_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (8, 144, 28, 28), (112896, 784, 28, 1))
        del arg60_1
        buf41 = buf40; del buf40  # reuse
        # Source Nodes: [l__mod___features_3_conv2_1, l__mod___features_3_conv2_2, l__mod___features_3_conv3_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf41, arg185_1, arg186_1, arg61_1, arg62_1, 903168, grid=grid(903168), stream=stream0)
        del arg185_1
        del arg186_1
        del arg61_1
        del arg62_1
        # Source Nodes: [l__mod___features_3_conv2_1, l__mod___features_3_conv2_2, l__mod___features_3_conv3_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf42 = extern_kernels.convolution(buf41, arg63_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 72, 28, 28), (56448, 784, 28, 1))
        del arg63_1
        del buf41
        buf43 = buf42; del buf42  # reuse
        # Source Nodes: [d2_3, l__mod___features_3_conv3_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf43, arg188_1, arg189_1, arg64_1, arg65_1, 451584, grid=grid(451584), stream=stream0)
        del arg188_1
        del arg189_1
        del arg64_1
        del arg65_1
        # Source Nodes: [l__mod___features_3_conv4_0], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, arg66_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 144, 28, 28), (112896, 784, 28, 1))
        del arg66_1
        buf45 = buf44; del buf44  # reuse
        # Source Nodes: [l__mod___features_3_conv4_1, l__mod___features_3_conv4_2, l__mod___features_3_conv5_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf45, arg191_1, arg192_1, arg67_1, arg68_1, 903168, grid=grid(903168), stream=stream0)
        del arg191_1
        del arg192_1
        del arg67_1
        del arg68_1
        # Source Nodes: [l__mod___features_3_conv4_1, l__mod___features_3_conv4_2, l__mod___features_3_conv5_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf46 = extern_kernels.convolution(buf45, arg69_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (8, 72, 28, 28), (56448, 784, 28, 1))
        del arg69_1
        del buf45
        buf47 = empty((8, 432, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_8, l__mod___features_3_conv6_0], Original ATen: [aten.cat, aten.convolution]
        triton_poi_fused_cat_convolution_9.run(buf39, buf43, buf46, arg194_1, arg195_1, arg70_1, arg71_1, buf37, buf47, 2709504, grid=grid(2709504), stream=stream0)
        del arg194_1
        del arg195_1
        del arg70_1
        del arg71_1
        del buf37
        del buf39
        del buf43
        del buf46
        # Source Nodes: [cat_8, l__mod___features_3_conv6_0], Original ATen: [aten.cat, aten.convolution]
        buf48 = extern_kernels.convolution(buf47, arg72_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (8, 288, 28, 28), (225792, 784, 28, 1))
        del arg72_1
        del buf47
        buf49 = buf48; del buf48  # reuse
        # Source Nodes: [l__mod___features_3_conv6_1, l__mod___features_3_conv6_2, l__mod___features_4_conv1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_10.run(buf49, arg197_1, arg198_1, arg73_1, arg74_1, 1806336, grid=grid(1806336), stream=stream0)
        del arg197_1
        del arg198_1
        del arg73_1
        del arg74_1
        # Source Nodes: [l__mod___features_3_conv6_1, l__mod___features_3_conv6_2, l__mod___features_4_conv1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf50 = extern_kernels.convolution(buf49, arg75_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (8, 304, 14, 14), (59584, 196, 14, 1))
        del arg75_1
        del buf49
        buf51 = buf50; del buf50  # reuse
        # Source Nodes: [d1_4, l__mod___features_4_conv1_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf51, arg200_1, arg201_1, arg76_1, arg77_1, 476672, grid=grid(476672), stream=stream0)
        del arg200_1
        del arg201_1
        del arg76_1
        del arg77_1
        # Source Nodes: [l__mod___features_4_conv2_0], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, arg78_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (8, 304, 14, 14), (59584, 196, 14, 1))
        del arg78_1
        buf53 = buf52; del buf52  # reuse
        # Source Nodes: [l__mod___features_4_conv2_1, l__mod___features_4_conv2_2, l__mod___features_4_conv3_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf53, arg203_1, arg204_1, arg79_1, arg80_1, 476672, grid=grid(476672), stream=stream0)
        del arg203_1
        del arg204_1
        del arg79_1
        del arg80_1
        # Source Nodes: [l__mod___features_4_conv2_1, l__mod___features_4_conv2_2, l__mod___features_4_conv3_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf54 = extern_kernels.convolution(buf53, arg81_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 152, 14, 14), (29792, 196, 14, 1))
        del arg81_1
        del buf53
        buf55 = buf54; del buf54  # reuse
        # Source Nodes: [d2_4, l__mod___features_4_conv3_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf55, arg206_1, arg207_1, arg82_1, arg83_1, 238336, grid=grid(238336), stream=stream0)
        del arg206_1
        del arg207_1
        del arg82_1
        del arg83_1
        # Source Nodes: [l__mod___features_4_conv4_0], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, arg84_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 304, 14, 14), (59584, 196, 14, 1))
        del arg84_1
        buf57 = buf56; del buf56  # reuse
        # Source Nodes: [l__mod___features_4_conv4_1, l__mod___features_4_conv4_2, l__mod___features_4_conv5_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf57, arg209_1, arg210_1, arg85_1, arg86_1, 476672, grid=grid(476672), stream=stream0)
        del arg209_1
        del arg210_1
        del arg85_1
        del arg86_1
        # Source Nodes: [l__mod___features_4_conv4_1, l__mod___features_4_conv4_2, l__mod___features_4_conv5_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf58 = extern_kernels.convolution(buf57, arg87_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (8, 152, 14, 14), (29792, 196, 14, 1))
        del arg87_1
        del buf57
        buf59 = empty((8, 608, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_7, l__mod___features_4_conv6_0], Original ATen: [aten.cat, aten.convolution]
        triton_poi_fused_cat_convolution_13.run(buf51, buf55, buf58, arg212_1, arg213_1, arg88_1, arg89_1, buf59, 953344, grid=grid(953344), stream=stream0)
        del arg212_1
        del arg213_1
        del arg88_1
        del arg89_1
        del buf51
        del buf55
        del buf58
        # Source Nodes: [cat_7, l__mod___features_4_conv6_0], Original ATen: [aten.cat, aten.convolution]
        buf60 = extern_kernels.convolution(buf59, arg90_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (8, 304, 14, 14), (59584, 196, 14, 1))
        del arg90_1
        del buf59
        buf61 = buf60; del buf60  # reuse
        # Source Nodes: [l__mod___features_4_conv6_1, out_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf61, arg215_1, arg216_1, arg91_1, arg92_1, 476672, grid=grid(476672), stream=stream0)
        del arg215_1
        del arg216_1
        del arg91_1
        del arg92_1
        # Source Nodes: [l__mod___features_5_conv1_0], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, arg93_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 304, 14, 14), (59584, 196, 14, 1))
        del arg93_1
        buf63 = buf62; del buf62  # reuse
        # Source Nodes: [d1_5, l__mod___features_5_conv1_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf63, arg218_1, arg219_1, arg94_1, arg95_1, 476672, grid=grid(476672), stream=stream0)
        del arg218_1
        del arg219_1
        del arg94_1
        del arg95_1
        # Source Nodes: [l__mod___features_5_conv2_0], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (8, 304, 14, 14), (59584, 196, 14, 1))
        del arg96_1
        buf65 = buf64; del buf64  # reuse
        # Source Nodes: [l__mod___features_5_conv2_1, l__mod___features_5_conv2_2, l__mod___features_5_conv3_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf65, arg221_1, arg222_1, arg97_1, arg98_1, 476672, grid=grid(476672), stream=stream0)
        del arg221_1
        del arg222_1
        del arg97_1
        del arg98_1
        # Source Nodes: [l__mod___features_5_conv2_1, l__mod___features_5_conv2_2, l__mod___features_5_conv3_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf66 = extern_kernels.convolution(buf65, arg99_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 152, 14, 14), (29792, 196, 14, 1))
        del arg99_1
        del buf65
        buf67 = buf66; del buf66  # reuse
        # Source Nodes: [d2_5, l__mod___features_5_conv3_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf67, arg224_1, arg225_1, arg100_1, arg101_1, 238336, grid=grid(238336), stream=stream0)
        del arg100_1
        del arg101_1
        del arg224_1
        del arg225_1
        # Source Nodes: [l__mod___features_5_conv4_0], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, arg102_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 304, 14, 14), (59584, 196, 14, 1))
        del arg102_1
        buf69 = buf68; del buf68  # reuse
        # Source Nodes: [l__mod___features_5_conv4_1, l__mod___features_5_conv4_2, l__mod___features_5_conv5_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf69, arg227_1, arg228_1, arg103_1, arg104_1, 476672, grid=grid(476672), stream=stream0)
        del arg103_1
        del arg104_1
        del arg227_1
        del arg228_1
        # Source Nodes: [l__mod___features_5_conv4_1, l__mod___features_5_conv4_2, l__mod___features_5_conv5_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf70 = extern_kernels.convolution(buf69, arg105_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (8, 152, 14, 14), (29792, 196, 14, 1))
        del arg105_1
        del buf69
        buf71 = empty((8, 912, 14, 14), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_6, l__mod___features_5_conv6_0], Original ATen: [aten.cat, aten.convolution]
        triton_poi_fused_cat_convolution_14.run(buf63, buf67, buf70, arg230_1, arg231_1, arg106_1, arg107_1, buf61, buf71, 1430016, grid=grid(1430016), stream=stream0)
        del arg106_1
        del arg107_1
        del arg230_1
        del arg231_1
        del buf61
        del buf63
        del buf67
        del buf70
        # Source Nodes: [cat_6, l__mod___features_5_conv6_0], Original ATen: [aten.cat, aten.convolution]
        buf72 = extern_kernels.convolution(buf71, arg108_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (8, 480, 14, 14), (94080, 196, 14, 1))
        del arg108_1
        del buf71
        buf73 = buf72; del buf72  # reuse
        # Source Nodes: [l__mod___features_5_conv6_1, l__mod___features_5_conv6_2, l__mod___head_0_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf73, arg233_1, arg234_1, arg109_1, arg110_1, 752640, grid=grid(752640), stream=stream0)
        del arg109_1
        del arg110_1
        del arg233_1
        del arg234_1
        # Source Nodes: [l__mod___features_5_conv6_1, l__mod___features_5_conv6_2, l__mod___head_0_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf74 = extern_kernels.convolution(buf73, arg111_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 960, 7, 7), (47040, 49, 7, 1))
        del arg111_1
        del buf73
        buf75 = buf74; del buf74  # reuse
        # Source Nodes: [l__mod___head_0_1, l__mod___head_0_2, l__mod___head_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_16.run(buf75, arg236_1, arg237_1, arg112_1, arg113_1, 376320, grid=grid(376320), stream=stream0)
        del arg112_1
        del arg113_1
        del arg236_1
        del arg237_1
        # Source Nodes: [l__mod___head_0_1, l__mod___head_0_2, l__mod___head_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf76 = extern_kernels.convolution(buf75, arg114_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 1024, 7, 7), (50176, 49, 7, 1))
        del arg114_1
        del buf75
        buf77 = buf76; del buf76  # reuse
        # Source Nodes: [l__mod___head_1_1, l__mod___head_1_2, l__mod___head_2_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17.run(buf77, arg239_1, arg240_1, arg115_1, arg116_1, 401408, grid=grid(401408), stream=stream0)
        del arg115_1
        del arg116_1
        del arg239_1
        del arg240_1
        # Source Nodes: [l__mod___head_1_1, l__mod___head_1_2, l__mod___head_2_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf78 = extern_kernels.convolution(buf77, arg117_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (8, 1280, 4, 4), (20480, 16, 4, 1))
        del arg117_1
        del buf77
        buf79 = buf78; del buf78  # reuse
        # Source Nodes: [l__mod___head_2_1, l__mod___head_2_2, l__mod___head_3_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_18.run(buf79, arg242_1, arg243_1, arg118_1, arg119_1, 163840, grid=grid(163840), stream=stream0)
        del arg118_1
        del arg119_1
        del arg242_1
        del arg243_1
        # Source Nodes: [l__mod___head_2_1, l__mod___head_2_2, l__mod___head_3_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf80 = extern_kernels.convolution(buf79, arg120_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (8, 1024, 4, 4), (16384, 16, 4, 1))
        del arg120_1
        del buf79
        buf81 = empty_strided((8, 1024, 1, 1), (1024, 1, 8192, 8192), device='cuda', dtype=torch.float32)
        buf82 = reinterpret_tensor(buf81, (8, 1024, 1, 1), (1024, 1, 1, 1), 0); del buf81  # reuse
        # Source Nodes: [l__mod___head_3_1, x_2, x_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_19.run(buf82, buf80, arg245_1, arg246_1, arg121_1, arg122_1, 8192, 16, grid=grid(8192), stream=stream0)
        del arg121_1
        del arg122_1
        del arg245_1
        del arg246_1
        del buf80
        buf83 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_7], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg124_1, reinterpret_tensor(buf82, (8, 1024), (1024, 1), 0), reinterpret_tensor(arg123_1, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf83)
        del arg123_1
        del arg124_1
        return (buf83, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((64, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((32, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((128, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((144, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((144, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((72, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((144, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((72, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((144, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((144, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((144, 144, 1, 1), (144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((72, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((144, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((72, 144, 3, 3), (1296, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((288, 432, 1, 1), (432, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((304, 288, 3, 3), (2592, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((304, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((152, 304, 3, 3), (2736, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((304, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((152, 304, 3, 3), (2736, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((304, 608, 1, 1), (608, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((304, 304, 3, 3), (2736, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((304, 304, 1, 1), (304, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((152, 304, 3, 3), (2736, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((304, 152, 1, 1), (152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((152, 304, 3, 3), (2736, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((480, 912, 1, 1), (912, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((960, 480, 3, 3), (4320, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((1024, 960, 3, 3), (8640, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1280, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1024, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg128_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg131_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg134_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg137_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg140_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg143_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg146_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg149_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg152_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg155_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg158_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg161_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg164_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg167_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg170_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg173_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg176_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg179_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg182_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg185_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg188_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg191_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((144, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg194_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg197_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((288, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg200_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg203_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg206_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg209_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg212_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg215_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg218_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg221_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg224_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg227_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((304, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg230_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg233_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg236_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg239_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg242_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg245_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg248_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('selecsls42b', benchmark_compiled_module)
