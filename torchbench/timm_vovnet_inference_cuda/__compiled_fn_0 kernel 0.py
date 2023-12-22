
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


# kernel path: /tmp/torchinductor_youkaichao/pj/cpjtytuqgqenxxqp3i2lk7cu7zw476xwdbdsk7opg64jv5qaytoo.py
# Source Nodes: [x_1, x_4, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_1 => add_1, mul_1, mul_2, sub
# x_4 => relu
# x_5 => convolution_1
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


# kernel path: /tmp/torchinductor_youkaichao/mv/cmvyyvsxl5s2yig5fswaerxom26onpv5mj6bzqaj4k2kgghfqzfn.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_1', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 128
    x2 = (xindex // 401408)
    x4 = xindex % 401408
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
    tl.store(out_ptr0 + (x4 + (2408448*x2)), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/6k/c6knplshufuapv3c3az7mit5friwu24bgm2otkbczrzdxdzkw4pv.py
# Source Nodes: [x_37, x_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_37 => add_15, mul_22, mul_23, sub_7
# x_40 => relu_7
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
    tl.store(out_ptr0 + (x4 + (2408448*x2)), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sa/csaymbybztpjukxcrk2snuumkky6mdh4x3qyd3ltf45ep7nbn4z6.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_3', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/ts/ctsh6tksrjrefhinhqhn3h5te5nhoj5ofumdaoqhmk2hebhzxyq7.py
# Source Nodes: [cat_10, x_44, x_49, x_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.max_pool2d_with_indices, aten.relu]
# cat_10 => cat_1
# x_44 => add_17, mul_25, mul_26, sub_8
# x_49 => relu_8
# x_50 => max_pool2d_with_indices
triton_poi_fused__native_batch_norm_legit_no_training_cat_max_pool2d_with_indices_relu_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_max_pool2d_with_indices_relu_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x4), tmp69, None)
    tl.store(out_ptr1 + (x5 + (827904*x6)), tmp69, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/up/cup6rcq6wiz27nxxdygymquqooudpkctldackglfgybdhcles3dm.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 501760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 160
    x2 = (xindex // 125440)
    x4 = xindex % 125440
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
    tl.store(out_ptr0 + (x4 + (827904*x2)), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/q4/cq4ogpcyv56sipjxt5q52omp256bdt3t5ppdoqa2nib6ivktedck.py
# Source Nodes: [x_72, x_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_72 => add_27, mul_40, mul_41, sub_13
# x_75 => relu_13
triton_poi_fused__native_batch_norm_legit_no_training_relu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x4 + (827904*x2)), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ow/cow74hzadcywcj6yhdzj4gth33jycsfz4o5hqzi3sqzqwbl7sqgb.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_7', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/a5/ca5ddavubbozqf5vezka3jhtdrsbajb2vl57amphe43q7g7drbqo.py
# Source Nodes: [cat_9, x_79, x_84, x_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.max_pool2d_with_indices, aten.relu]
# cat_9 => cat_2
# x_79 => add_29, mul_43, mul_44, sub_14
# x_84 => relu_14
# x_85 => max_pool2d_with_indices_1
triton_poi_fused__native_batch_norm_legit_no_training_cat_max_pool2d_with_indices_relu_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_max_pool2d_with_indices_relu_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x4), tmp69, None)
    tl.store(out_ptr1 + (x5 + (288512*x6)), tmp69, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7n/c7npjbpabdth443lvwv2lx7arx24z6qela7ocypqmme3eyvxn56t.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 192
    x2 = (xindex // 37632)
    x4 = xindex % 37632
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
    tl.store(out_ptr0 + (x4 + (288512*x2)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lc/clcm2u2w3olcwy5j54leddpqcx6vb7w4c2bnnkspyp3gc4gyw6sm.py
# Source Nodes: [x_107, x_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_107 => add_39, mul_58, mul_59, sub_19
# x_110 => relu_19
triton_poi_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x4 + (288512*x2)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j5/cj5gj7mz7csouoac4zb23nyduu4xkrarty2vrlck3gp7g7wjrktt.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 602112
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 768
    x2 = (xindex // 150528)
    x4 = xindex % 150528
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
    tl.store(out_ptr0 + (x4 + (338688*x2)), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/25/c256wuei3fhc4c6is6rgcngspjtf54dsq2fer2nfpus3n5lruvf3.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 150528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 192
    x2 = (xindex // 37632)
    x4 = xindex % 37632
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
    tl.store(out_ptr0 + (x4 + (338688*x2)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j3/cj3agyx3bm677giv4yz4p5e3z3ohvj2lt5ippnezdswinqemb227.py
# Source Nodes: [x_140, x_143], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_140 => add_51, mul_76, mul_77, sub_25
# x_143 => relu_25
triton_poi_fused__native_batch_norm_legit_no_training_relu_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x4 + (338688*x2)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/od/codbj52dwep4w26w6hfnho25msvzvsb75djcmtog2n727pyn2jg6.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/32/c32judgrm3kdgvjwr5lz63mgxexwjgikrtw436vvvqwmtklitm3s.py
# Source Nodes: [cat_7, x_147, x_152, x_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.max_pool2d_with_indices, aten.relu]
# cat_7 => cat_4
# x_147 => add_53, mul_79, mul_80, sub_26
# x_152 => relu_26
# x_153 => max_pool2d_with_indices_2
triton_poi_fused__native_batch_norm_legit_no_training_cat_max_pool2d_with_indices_relu_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_max_pool2d_with_indices_relu_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x4), tmp69, xmask)
    tl.store(out_ptr1 + (x5 + (92512*x6)), tmp69, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ky/ckyv2tyrjk6sw6l3xyew5dehfmlkq3z4kbzcidcqr4p23ktqnqzb.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 224
    x2 = (xindex // 10976)
    x4 = xindex % 10976
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
    tl.store(out_ptr0 + (x4 + (92512*x2)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/df/cdfqmvrc6ercg2jr4isc23ta52wwrod5g25vbwl46sg7weqsavdo.py
# Source Nodes: [x_175, x_178], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_175 => add_63, mul_94, mul_95, sub_31
# x_178 => relu_31
triton_poi_fused__native_batch_norm_legit_no_training_relu_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x4 + (92512*x2)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qi/cqiylti57i6nvwg744qdz5z7eoxlsc76eetdcivdkoigif7f2e73.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 200704
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 1024
    x2 = (xindex // 50176)
    x4 = xindex % 50176
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
    tl.store(out_ptr0 + (x4 + (105056*x2)), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hv/chvkfhltlsqretvjovuj5dyphjljfdzwf7auag22fqeoeg4en6sh.py
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43904
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 49) % 224
    x2 = (xindex // 10976)
    x4 = xindex % 10976
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
    tl.store(out_ptr0 + (x4 + (105056*x2)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l6/cl6p4ipuxfv5wuvcurni5ei4b5xddgdjpha63kc4wbh2jjp5ljog.py
# Source Nodes: [x_208, x_211], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_208 => add_75, mul_112, mul_113, sub_37
# x_211 => relu_37
triton_poi_fused__native_batch_norm_legit_no_training_relu_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x4 + (105056*x2)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fo/cfoapr7jt56lcnj72lfoirhmdkls4xyz3ocmwgqd2up7yvgf2jeo.py
# Source Nodes: [x_215, x_221, x_222], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
# x_215 => add_77, mul_115, mul_116, sub_38
# x_221 => relu_38
# x_222 => mean
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_21', 'mutated_arg_names': ['in_out_ptr0']}
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, ), (1, ))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (128, ), (1, ))
    assert_size_stride(arg5_1, (128, ), (1, ))
    assert_size_stride(arg6_1, (128, ), (1, ))
    assert_size_stride(arg7_1, (128, ), (1, ))
    assert_size_stride(arg8_1, (128, ), (1, ))
    assert_size_stride(arg9_1, (128, ), (1, ))
    assert_size_stride(arg10_1, (128, ), (1, ))
    assert_size_stride(arg11_1, (128, ), (1, ))
    assert_size_stride(arg12_1, (128, ), (1, ))
    assert_size_stride(arg13_1, (128, ), (1, ))
    assert_size_stride(arg14_1, (128, ), (1, ))
    assert_size_stride(arg15_1, (128, ), (1, ))
    assert_size_stride(arg16_1, (256, ), (1, ))
    assert_size_stride(arg17_1, (256, ), (1, ))
    assert_size_stride(arg18_1, (160, ), (1, ))
    assert_size_stride(arg19_1, (160, ), (1, ))
    assert_size_stride(arg20_1, (160, ), (1, ))
    assert_size_stride(arg21_1, (160, ), (1, ))
    assert_size_stride(arg22_1, (160, ), (1, ))
    assert_size_stride(arg23_1, (160, ), (1, ))
    assert_size_stride(arg24_1, (160, ), (1, ))
    assert_size_stride(arg25_1, (160, ), (1, ))
    assert_size_stride(arg26_1, (160, ), (1, ))
    assert_size_stride(arg27_1, (160, ), (1, ))
    assert_size_stride(arg28_1, (512, ), (1, ))
    assert_size_stride(arg29_1, (512, ), (1, ))
    assert_size_stride(arg30_1, (192, ), (1, ))
    assert_size_stride(arg31_1, (192, ), (1, ))
    assert_size_stride(arg32_1, (192, ), (1, ))
    assert_size_stride(arg33_1, (192, ), (1, ))
    assert_size_stride(arg34_1, (192, ), (1, ))
    assert_size_stride(arg35_1, (192, ), (1, ))
    assert_size_stride(arg36_1, (192, ), (1, ))
    assert_size_stride(arg37_1, (192, ), (1, ))
    assert_size_stride(arg38_1, (192, ), (1, ))
    assert_size_stride(arg39_1, (192, ), (1, ))
    assert_size_stride(arg40_1, (768, ), (1, ))
    assert_size_stride(arg41_1, (768, ), (1, ))
    assert_size_stride(arg42_1, (192, ), (1, ))
    assert_size_stride(arg43_1, (192, ), (1, ))
    assert_size_stride(arg44_1, (192, ), (1, ))
    assert_size_stride(arg45_1, (192, ), (1, ))
    assert_size_stride(arg46_1, (192, ), (1, ))
    assert_size_stride(arg47_1, (192, ), (1, ))
    assert_size_stride(arg48_1, (192, ), (1, ))
    assert_size_stride(arg49_1, (192, ), (1, ))
    assert_size_stride(arg50_1, (192, ), (1, ))
    assert_size_stride(arg51_1, (192, ), (1, ))
    assert_size_stride(arg52_1, (768, ), (1, ))
    assert_size_stride(arg53_1, (768, ), (1, ))
    assert_size_stride(arg54_1, (224, ), (1, ))
    assert_size_stride(arg55_1, (224, ), (1, ))
    assert_size_stride(arg56_1, (224, ), (1, ))
    assert_size_stride(arg57_1, (224, ), (1, ))
    assert_size_stride(arg58_1, (224, ), (1, ))
    assert_size_stride(arg59_1, (224, ), (1, ))
    assert_size_stride(arg60_1, (224, ), (1, ))
    assert_size_stride(arg61_1, (224, ), (1, ))
    assert_size_stride(arg62_1, (224, ), (1, ))
    assert_size_stride(arg63_1, (224, ), (1, ))
    assert_size_stride(arg64_1, (1024, ), (1, ))
    assert_size_stride(arg65_1, (1024, ), (1, ))
    assert_size_stride(arg66_1, (224, ), (1, ))
    assert_size_stride(arg67_1, (224, ), (1, ))
    assert_size_stride(arg68_1, (224, ), (1, ))
    assert_size_stride(arg69_1, (224, ), (1, ))
    assert_size_stride(arg70_1, (224, ), (1, ))
    assert_size_stride(arg71_1, (224, ), (1, ))
    assert_size_stride(arg72_1, (224, ), (1, ))
    assert_size_stride(arg73_1, (224, ), (1, ))
    assert_size_stride(arg74_1, (224, ), (1, ))
    assert_size_stride(arg75_1, (224, ), (1, ))
    assert_size_stride(arg76_1, (1024, ), (1, ))
    assert_size_stride(arg77_1, (1024, ), (1, ))
    assert_size_stride(arg78_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg79_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg80_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg81_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg82_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg83_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg84_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg85_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg86_1, (256, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg87_1, (160, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg88_1, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(arg89_1, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(arg90_1, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(arg91_1, (160, 160, 3, 3), (1440, 9, 3, 1))
    assert_size_stride(arg92_1, (512, 1056, 1, 1), (1056, 1, 1, 1))
    assert_size_stride(arg93_1, (192, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg94_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg95_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg96_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg97_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg98_1, (768, 1472, 1, 1), (1472, 1, 1, 1))
    assert_size_stride(arg99_1, (192, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(arg100_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg101_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg102_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg103_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg104_1, (768, 1728, 1, 1), (1728, 1, 1, 1))
    assert_size_stride(arg105_1, (224, 768, 3, 3), (6912, 9, 3, 1))
    assert_size_stride(arg106_1, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(arg107_1, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(arg108_1, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(arg109_1, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(arg110_1, (1024, 1888, 1, 1), (1888, 1, 1, 1))
    assert_size_stride(arg111_1, (224, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(arg112_1, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(arg113_1, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(arg114_1, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(arg115_1, (224, 224, 3, 3), (2016, 9, 3, 1))
    assert_size_stride(arg116_1, (1024, 2144, 1, 1), (2144, 1, 1, 1))
    assert_size_stride(arg117_1, (1000, 1024), (1024, 1))
    assert_size_stride(arg118_1, (1000, ), (1, ))
    assert_size_stride(arg119_1, (64, ), (1, ))
    assert_size_stride(arg120_1, (64, ), (1, ))
    assert_size_stride(arg121_1, (64, ), (1, ))
    assert_size_stride(arg122_1, (64, ), (1, ))
    assert_size_stride(arg123_1, (128, ), (1, ))
    assert_size_stride(arg124_1, (128, ), (1, ))
    assert_size_stride(arg125_1, (128, ), (1, ))
    assert_size_stride(arg126_1, (128, ), (1, ))
    assert_size_stride(arg127_1, (128, ), (1, ))
    assert_size_stride(arg128_1, (128, ), (1, ))
    assert_size_stride(arg129_1, (128, ), (1, ))
    assert_size_stride(arg130_1, (128, ), (1, ))
    assert_size_stride(arg131_1, (128, ), (1, ))
    assert_size_stride(arg132_1, (128, ), (1, ))
    assert_size_stride(arg133_1, (128, ), (1, ))
    assert_size_stride(arg134_1, (128, ), (1, ))
    assert_size_stride(arg135_1, (256, ), (1, ))
    assert_size_stride(arg136_1, (256, ), (1, ))
    assert_size_stride(arg137_1, (160, ), (1, ))
    assert_size_stride(arg138_1, (160, ), (1, ))
    assert_size_stride(arg139_1, (160, ), (1, ))
    assert_size_stride(arg140_1, (160, ), (1, ))
    assert_size_stride(arg141_1, (160, ), (1, ))
    assert_size_stride(arg142_1, (160, ), (1, ))
    assert_size_stride(arg143_1, (160, ), (1, ))
    assert_size_stride(arg144_1, (160, ), (1, ))
    assert_size_stride(arg145_1, (160, ), (1, ))
    assert_size_stride(arg146_1, (160, ), (1, ))
    assert_size_stride(arg147_1, (512, ), (1, ))
    assert_size_stride(arg148_1, (512, ), (1, ))
    assert_size_stride(arg149_1, (192, ), (1, ))
    assert_size_stride(arg150_1, (192, ), (1, ))
    assert_size_stride(arg151_1, (192, ), (1, ))
    assert_size_stride(arg152_1, (192, ), (1, ))
    assert_size_stride(arg153_1, (192, ), (1, ))
    assert_size_stride(arg154_1, (192, ), (1, ))
    assert_size_stride(arg155_1, (192, ), (1, ))
    assert_size_stride(arg156_1, (192, ), (1, ))
    assert_size_stride(arg157_1, (192, ), (1, ))
    assert_size_stride(arg158_1, (192, ), (1, ))
    assert_size_stride(arg159_1, (768, ), (1, ))
    assert_size_stride(arg160_1, (768, ), (1, ))
    assert_size_stride(arg161_1, (192, ), (1, ))
    assert_size_stride(arg162_1, (192, ), (1, ))
    assert_size_stride(arg163_1, (192, ), (1, ))
    assert_size_stride(arg164_1, (192, ), (1, ))
    assert_size_stride(arg165_1, (192, ), (1, ))
    assert_size_stride(arg166_1, (192, ), (1, ))
    assert_size_stride(arg167_1, (192, ), (1, ))
    assert_size_stride(arg168_1, (192, ), (1, ))
    assert_size_stride(arg169_1, (192, ), (1, ))
    assert_size_stride(arg170_1, (192, ), (1, ))
    assert_size_stride(arg171_1, (768, ), (1, ))
    assert_size_stride(arg172_1, (768, ), (1, ))
    assert_size_stride(arg173_1, (224, ), (1, ))
    assert_size_stride(arg174_1, (224, ), (1, ))
    assert_size_stride(arg175_1, (224, ), (1, ))
    assert_size_stride(arg176_1, (224, ), (1, ))
    assert_size_stride(arg177_1, (224, ), (1, ))
    assert_size_stride(arg178_1, (224, ), (1, ))
    assert_size_stride(arg179_1, (224, ), (1, ))
    assert_size_stride(arg180_1, (224, ), (1, ))
    assert_size_stride(arg181_1, (224, ), (1, ))
    assert_size_stride(arg182_1, (224, ), (1, ))
    assert_size_stride(arg183_1, (1024, ), (1, ))
    assert_size_stride(arg184_1, (1024, ), (1, ))
    assert_size_stride(arg185_1, (224, ), (1, ))
    assert_size_stride(arg186_1, (224, ), (1, ))
    assert_size_stride(arg187_1, (224, ), (1, ))
    assert_size_stride(arg188_1, (224, ), (1, ))
    assert_size_stride(arg189_1, (224, ), (1, ))
    assert_size_stride(arg190_1, (224, ), (1, ))
    assert_size_stride(arg191_1, (224, ), (1, ))
    assert_size_stride(arg192_1, (224, ), (1, ))
    assert_size_stride(arg193_1, (224, ), (1, ))
    assert_size_stride(arg194_1, (224, ), (1, ))
    assert_size_stride(arg195_1, (1024, ), (1, ))
    assert_size_stride(arg196_1, (1024, ), (1, ))
    assert_size_stride(arg197_1, (4, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg197_1, arg78_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (4, 64, 112, 112), (802816, 12544, 112, 1))
        del arg197_1
        del arg78_1
        buf1 = buf0; del buf0  # reuse
        # Source Nodes: [x_1, x_4, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf1, arg119_1, arg120_1, arg0_1, arg1_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg0_1
        del arg119_1
        del arg120_1
        del arg1_1
        # Source Nodes: [x_1, x_4, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf2 = extern_kernels.convolution(buf1, arg79_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 64, 112, 112), (802816, 12544, 112, 1))
        del arg79_1
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Source Nodes: [x_10, x_6, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf3, arg121_1, arg122_1, arg2_1, arg3_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg121_1
        del arg122_1
        del arg2_1
        del arg3_1
        # Source Nodes: [x_10, x_6, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf4 = extern_kernels.convolution(buf3, arg80_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf4, (4, 128, 56, 56), (401408, 3136, 56, 1))
        del arg80_1
        del buf3
        buf5 = buf4; del buf4  # reuse
        buf21 = empty((4, 768, 56, 56), device='cuda', dtype=torch.float32)
        buf15 = reinterpret_tensor(buf21, (4, 128, 56, 56), (2408448, 3136, 56, 1), 0)  # alias
        # Source Nodes: [cat_11, x_11, x_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_1.run(buf5, arg123_1, arg124_1, arg4_1, arg5_1, buf15, 1605632, grid=grid(1605632), stream=stream0)
        del arg123_1
        del arg124_1
        del arg4_1
        del arg5_1
        # Source Nodes: [x_16], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, arg81_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 128, 56, 56), (401408, 3136, 56, 1))
        del arg81_1
        del buf5
        buf7 = buf6; del buf6  # reuse
        buf16 = reinterpret_tensor(buf21, (4, 128, 56, 56), (2408448, 3136, 56, 1), 401408)  # alias
        # Source Nodes: [cat_11, x_17, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_1.run(buf7, arg125_1, arg126_1, arg6_1, arg7_1, buf16, 1605632, grid=grid(1605632), stream=stream0)
        del arg125_1
        del arg126_1
        del arg6_1
        del arg7_1
        # Source Nodes: [x_21], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf7, arg82_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 128, 56, 56), (401408, 3136, 56, 1))
        del arg82_1
        del buf7
        buf9 = buf8; del buf8  # reuse
        buf17 = reinterpret_tensor(buf21, (4, 128, 56, 56), (2408448, 3136, 56, 1), 802816)  # alias
        # Source Nodes: [cat_11, x_22, x_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_1.run(buf9, arg127_1, arg128_1, arg8_1, arg9_1, buf17, 1605632, grid=grid(1605632), stream=stream0)
        del arg127_1
        del arg128_1
        del arg8_1
        del arg9_1
        # Source Nodes: [x_26], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, arg83_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 128, 56, 56), (401408, 3136, 56, 1))
        del arg83_1
        del buf9
        buf11 = buf10; del buf10  # reuse
        buf18 = reinterpret_tensor(buf21, (4, 128, 56, 56), (2408448, 3136, 56, 1), 1204224)  # alias
        # Source Nodes: [cat_11, x_27, x_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_1.run(buf11, arg129_1, arg130_1, arg10_1, arg11_1, buf18, 1605632, grid=grid(1605632), stream=stream0)
        del arg10_1
        del arg11_1
        del arg129_1
        del arg130_1
        # Source Nodes: [x_31], Original ATen: [aten.convolution]
        buf12 = extern_kernels.convolution(buf11, arg84_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 128, 56, 56), (401408, 3136, 56, 1))
        del arg84_1
        del buf11
        buf13 = buf12; del buf12  # reuse
        buf19 = reinterpret_tensor(buf21, (4, 128, 56, 56), (2408448, 3136, 56, 1), 1605632)  # alias
        # Source Nodes: [cat_11, x_32, x_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_1.run(buf13, arg131_1, arg132_1, arg12_1, arg13_1, buf19, 1605632, grid=grid(1605632), stream=stream0)
        del arg12_1
        del arg131_1
        del arg132_1
        del arg13_1
        # Source Nodes: [x_36], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, arg85_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 128, 56, 56), (401408, 3136, 56, 1))
        del arg85_1
        del buf13
        buf20 = reinterpret_tensor(buf21, (4, 128, 56, 56), (2408448, 3136, 56, 1), 2007040)  # alias
        # Source Nodes: [x_37, x_40], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf14, arg133_1, arg134_1, arg14_1, arg15_1, buf20, 1605632, grid=grid(1605632), stream=stream0)
        del arg133_1
        del arg134_1
        del arg14_1
        del arg15_1
        del buf14
        del buf15
        del buf16
        del buf17
        del buf18
        del buf19
        del buf20
        # Source Nodes: [x_43], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf21, arg86_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 256, 56, 56), (802816, 3136, 56, 1))
        del arg86_1
        del buf21
        buf23 = buf22; del buf22  # reuse
        # Source Nodes: [x_44, x_49], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_3.run(buf23, arg135_1, arg136_1, arg16_1, arg17_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg135_1
        del arg136_1
        del arg16_1
        del arg17_1
        buf24 = empty((4, 256, 28, 28), device='cuda', dtype=torch.float32)
        buf40 = empty((4, 1056, 28, 28), device='cuda', dtype=torch.float32)
        buf34 = reinterpret_tensor(buf40, (4, 256, 28, 28), (827904, 784, 28, 1), 0)  # alias
        # Source Nodes: [cat_10, x_44, x_49, x_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_max_pool2d_with_indices_relu_4.run(buf23, buf24, buf34, 802816, grid=grid(802816), stream=stream0)
        del buf23
        # Source Nodes: [x_51], Original ATen: [aten.convolution]
        buf25 = extern_kernels.convolution(buf24, arg87_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (4, 160, 28, 28), (125440, 784, 28, 1))
        del arg87_1
        del buf24
        buf26 = buf25; del buf25  # reuse
        buf35 = reinterpret_tensor(buf40, (4, 160, 28, 28), (827904, 784, 28, 1), 200704)  # alias
        # Source Nodes: [cat_10, x_52, x_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5.run(buf26, arg137_1, arg138_1, arg18_1, arg19_1, buf35, 501760, grid=grid(501760), stream=stream0)
        del arg137_1
        del arg138_1
        del arg18_1
        del arg19_1
        # Source Nodes: [x_56], Original ATen: [aten.convolution]
        buf27 = extern_kernels.convolution(buf26, arg88_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 160, 28, 28), (125440, 784, 28, 1))
        del arg88_1
        del buf26
        buf28 = buf27; del buf27  # reuse
        buf36 = reinterpret_tensor(buf40, (4, 160, 28, 28), (827904, 784, 28, 1), 326144)  # alias
        # Source Nodes: [cat_10, x_57, x_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5.run(buf28, arg139_1, arg140_1, arg20_1, arg21_1, buf36, 501760, grid=grid(501760), stream=stream0)
        del arg139_1
        del arg140_1
        del arg20_1
        del arg21_1
        # Source Nodes: [x_61], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf28, arg89_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 160, 28, 28), (125440, 784, 28, 1))
        del arg89_1
        del buf28
        buf30 = buf29; del buf29  # reuse
        buf37 = reinterpret_tensor(buf40, (4, 160, 28, 28), (827904, 784, 28, 1), 451584)  # alias
        # Source Nodes: [cat_10, x_62, x_65], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5.run(buf30, arg141_1, arg142_1, arg22_1, arg23_1, buf37, 501760, grid=grid(501760), stream=stream0)
        del arg141_1
        del arg142_1
        del arg22_1
        del arg23_1
        # Source Nodes: [x_66], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, arg90_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 160, 28, 28), (125440, 784, 28, 1))
        del arg90_1
        del buf30
        buf32 = buf31; del buf31  # reuse
        buf38 = reinterpret_tensor(buf40, (4, 160, 28, 28), (827904, 784, 28, 1), 577024)  # alias
        # Source Nodes: [cat_10, x_67, x_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_5.run(buf32, arg143_1, arg144_1, arg24_1, arg25_1, buf38, 501760, grid=grid(501760), stream=stream0)
        del arg143_1
        del arg144_1
        del arg24_1
        del arg25_1
        # Source Nodes: [x_71], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, arg91_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 160, 28, 28), (125440, 784, 28, 1))
        del arg91_1
        del buf32
        buf39 = reinterpret_tensor(buf40, (4, 160, 28, 28), (827904, 784, 28, 1), 702464)  # alias
        # Source Nodes: [x_72, x_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf33, arg145_1, arg146_1, arg26_1, arg27_1, buf39, 501760, grid=grid(501760), stream=stream0)
        del arg145_1
        del arg146_1
        del arg26_1
        del arg27_1
        del buf33
        del buf34
        del buf35
        del buf36
        del buf37
        del buf38
        del buf39
        # Source Nodes: [x_78], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, arg92_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 512, 28, 28), (401408, 784, 28, 1))
        del arg92_1
        del buf40
        buf42 = buf41; del buf41  # reuse
        # Source Nodes: [x_79, x_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf42, arg147_1, arg148_1, arg28_1, arg29_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg147_1
        del arg148_1
        del arg28_1
        del arg29_1
        buf43 = empty((4, 512, 14, 14), device='cuda', dtype=torch.float32)
        buf59 = empty((4, 1472, 14, 14), device='cuda', dtype=torch.float32)
        buf53 = reinterpret_tensor(buf59, (4, 512, 14, 14), (288512, 196, 14, 1), 0)  # alias
        # Source Nodes: [cat_9, x_79, x_84, x_85], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_max_pool2d_with_indices_relu_8.run(buf42, buf43, buf53, 401408, grid=grid(401408), stream=stream0)
        del buf42
        # Source Nodes: [x_86], Original ATen: [aten.convolution]
        buf44 = extern_kernels.convolution(buf43, arg93_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 192, 14, 14), (37632, 196, 14, 1))
        del arg93_1
        del buf43
        buf45 = buf44; del buf44  # reuse
        buf54 = reinterpret_tensor(buf59, (4, 192, 14, 14), (288512, 196, 14, 1), 100352)  # alias
        # Source Nodes: [cat_9, x_87, x_90], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9.run(buf45, arg149_1, arg150_1, arg30_1, arg31_1, buf54, 150528, grid=grid(150528), stream=stream0)
        del arg149_1
        del arg150_1
        del arg30_1
        del arg31_1
        # Source Nodes: [x_91], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf45, arg94_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 192, 14, 14), (37632, 196, 14, 1))
        del arg94_1
        del buf45
        buf47 = buf46; del buf46  # reuse
        buf55 = reinterpret_tensor(buf59, (4, 192, 14, 14), (288512, 196, 14, 1), 137984)  # alias
        # Source Nodes: [cat_9, x_92, x_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9.run(buf47, arg151_1, arg152_1, arg32_1, arg33_1, buf55, 150528, grid=grid(150528), stream=stream0)
        del arg151_1
        del arg152_1
        del arg32_1
        del arg33_1
        # Source Nodes: [x_96], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, arg95_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 192, 14, 14), (37632, 196, 14, 1))
        del arg95_1
        del buf47
        buf49 = buf48; del buf48  # reuse
        buf56 = reinterpret_tensor(buf59, (4, 192, 14, 14), (288512, 196, 14, 1), 175616)  # alias
        # Source Nodes: [cat_9, x_100, x_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9.run(buf49, arg153_1, arg154_1, arg34_1, arg35_1, buf56, 150528, grid=grid(150528), stream=stream0)
        del arg153_1
        del arg154_1
        del arg34_1
        del arg35_1
        # Source Nodes: [x_101], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, arg96_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 192, 14, 14), (37632, 196, 14, 1))
        del arg96_1
        del buf49
        buf51 = buf50; del buf50  # reuse
        buf57 = reinterpret_tensor(buf59, (4, 192, 14, 14), (288512, 196, 14, 1), 213248)  # alias
        # Source Nodes: [cat_9, x_102, x_105], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_9.run(buf51, arg155_1, arg156_1, arg36_1, arg37_1, buf57, 150528, grid=grid(150528), stream=stream0)
        del arg155_1
        del arg156_1
        del arg36_1
        del arg37_1
        # Source Nodes: [x_106], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf51, arg97_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (4, 192, 14, 14), (37632, 196, 14, 1))
        del arg97_1
        del buf51
        buf58 = reinterpret_tensor(buf59, (4, 192, 14, 14), (288512, 196, 14, 1), 250880)  # alias
        # Source Nodes: [x_107, x_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf52, arg157_1, arg158_1, arg38_1, arg39_1, buf58, 150528, grid=grid(150528), stream=stream0)
        del arg157_1
        del arg158_1
        del arg38_1
        del arg39_1
        del buf52
        del buf53
        del buf54
        del buf55
        del buf56
        del buf57
        del buf58
        # Source Nodes: [x_113], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, arg98_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 768, 14, 14), (150528, 196, 14, 1))
        del arg98_1
        del buf59
        buf61 = buf60; del buf60  # reuse
        buf77 = empty((4, 1728, 14, 14), device='cuda', dtype=torch.float32)
        buf71 = reinterpret_tensor(buf77, (4, 768, 14, 14), (338688, 196, 14, 1), 0)  # alias
        # Source Nodes: [cat_8, x_114, x_118], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_11.run(buf61, arg159_1, arg160_1, arg40_1, arg41_1, buf71, 602112, grid=grid(602112), stream=stream0)
        del arg159_1
        del arg160_1
        del arg40_1
        del arg41_1
        # Source Nodes: [x_119], Original ATen: [aten.convolution]
        buf62 = extern_kernels.convolution(buf61, arg99_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 192, 14, 14), (37632, 196, 14, 1))
        del arg99_1
        del buf61
        buf63 = buf62; del buf62  # reuse
        buf72 = reinterpret_tensor(buf77, (4, 192, 14, 14), (338688, 196, 14, 1), 150528)  # alias
        # Source Nodes: [cat_8, x_120, x_123], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_12.run(buf63, arg161_1, arg162_1, arg42_1, arg43_1, buf72, 150528, grid=grid(150528), stream=stream0)
        del arg161_1
        del arg162_1
        del arg42_1
        del arg43_1
        # Source Nodes: [x_124], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf63, arg100_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (4, 192, 14, 14), (37632, 196, 14, 1))
        del arg100_1
        del buf63
        buf65 = buf64; del buf64  # reuse
        buf73 = reinterpret_tensor(buf77, (4, 192, 14, 14), (338688, 196, 14, 1), 188160)  # alias
        # Source Nodes: [cat_8, x_125, x_128], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_12.run(buf65, arg163_1, arg164_1, arg44_1, arg45_1, buf73, 150528, grid=grid(150528), stream=stream0)
        del arg163_1
        del arg164_1
        del arg44_1
        del arg45_1
        # Source Nodes: [x_129], Original ATen: [aten.convolution]
        buf66 = extern_kernels.convolution(buf65, arg101_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 192, 14, 14), (37632, 196, 14, 1))
        del arg101_1
        del buf65
        buf67 = buf66; del buf66  # reuse
        buf74 = reinterpret_tensor(buf77, (4, 192, 14, 14), (338688, 196, 14, 1), 225792)  # alias
        # Source Nodes: [cat_8, x_130, x_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_12.run(buf67, arg165_1, arg166_1, arg46_1, arg47_1, buf74, 150528, grid=grid(150528), stream=stream0)
        del arg165_1
        del arg166_1
        del arg46_1
        del arg47_1
        # Source Nodes: [x_134], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, arg102_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 192, 14, 14), (37632, 196, 14, 1))
        del arg102_1
        del buf67
        buf69 = buf68; del buf68  # reuse
        buf75 = reinterpret_tensor(buf77, (4, 192, 14, 14), (338688, 196, 14, 1), 263424)  # alias
        # Source Nodes: [cat_8, x_135, x_138], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_12.run(buf69, arg167_1, arg168_1, arg48_1, arg49_1, buf75, 150528, grid=grid(150528), stream=stream0)
        del arg167_1
        del arg168_1
        del arg48_1
        del arg49_1
        # Source Nodes: [x_139], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, arg103_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (4, 192, 14, 14), (37632, 196, 14, 1))
        del arg103_1
        del buf69
        buf76 = reinterpret_tensor(buf77, (4, 192, 14, 14), (338688, 196, 14, 1), 301056)  # alias
        # Source Nodes: [x_140, x_143], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf70, arg169_1, arg170_1, arg50_1, arg51_1, buf76, 150528, grid=grid(150528), stream=stream0)
        del arg169_1
        del arg170_1
        del arg50_1
        del arg51_1
        del buf71
        del buf72
        del buf73
        del buf74
        del buf75
        del buf76
        # Source Nodes: [x_146], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, arg104_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 768, 14, 14), (150528, 196, 14, 1))
        del arg104_1
        del buf77
        buf79 = buf78; del buf78  # reuse
        # Source Nodes: [x_147, x_152], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf79, arg171_1, arg172_1, arg52_1, arg53_1, 602112, grid=grid(602112), stream=stream0)
        del arg171_1
        del arg172_1
        del arg52_1
        del arg53_1
        buf80 = reinterpret_tensor(buf70, (4, 768, 7, 7), (37632, 49, 7, 1), 0); del buf70  # reuse
        buf96 = empty((4, 1888, 7, 7), device='cuda', dtype=torch.float32)
        buf90 = reinterpret_tensor(buf96, (4, 768, 7, 7), (92512, 49, 7, 1), 0)  # alias
        # Source Nodes: [cat_7, x_147, x_152, x_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_max_pool2d_with_indices_relu_15.run(buf79, buf80, buf90, 150528, grid=grid(150528), stream=stream0)
        del buf79
        # Source Nodes: [x_154], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, arg105_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (4, 224, 7, 7), (10976, 49, 7, 1))
        del arg105_1
        del buf80
        buf82 = buf81; del buf81  # reuse
        buf91 = reinterpret_tensor(buf96, (4, 224, 7, 7), (92512, 49, 7, 1), 37632)  # alias
        # Source Nodes: [cat_7, x_155, x_158], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16.run(buf82, arg173_1, arg174_1, arg54_1, arg55_1, buf91, 43904, grid=grid(43904), stream=stream0)
        del arg173_1
        del arg174_1
        del arg54_1
        del arg55_1
        # Source Nodes: [x_159], Original ATen: [aten.convolution]
        buf83 = extern_kernels.convolution(buf82, arg106_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf83, (4, 224, 7, 7), (10976, 49, 7, 1))
        del arg106_1
        del buf82
        buf84 = buf83; del buf83  # reuse
        buf92 = reinterpret_tensor(buf96, (4, 224, 7, 7), (92512, 49, 7, 1), 48608)  # alias
        # Source Nodes: [cat_7, x_160, x_163], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16.run(buf84, arg175_1, arg176_1, arg56_1, arg57_1, buf92, 43904, grid=grid(43904), stream=stream0)
        del arg175_1
        del arg176_1
        del arg56_1
        del arg57_1
        # Source Nodes: [x_164], Original ATen: [aten.convolution]
        buf85 = extern_kernels.convolution(buf84, arg107_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf85, (4, 224, 7, 7), (10976, 49, 7, 1))
        del arg107_1
        del buf84
        buf86 = buf85; del buf85  # reuse
        buf93 = reinterpret_tensor(buf96, (4, 224, 7, 7), (92512, 49, 7, 1), 59584)  # alias
        # Source Nodes: [cat_7, x_165, x_168], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16.run(buf86, arg177_1, arg178_1, arg58_1, arg59_1, buf93, 43904, grid=grid(43904), stream=stream0)
        del arg177_1
        del arg178_1
        del arg58_1
        del arg59_1
        # Source Nodes: [x_169], Original ATen: [aten.convolution]
        buf87 = extern_kernels.convolution(buf86, arg108_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (4, 224, 7, 7), (10976, 49, 7, 1))
        del arg108_1
        del buf86
        buf88 = buf87; del buf87  # reuse
        buf94 = reinterpret_tensor(buf96, (4, 224, 7, 7), (92512, 49, 7, 1), 70560)  # alias
        # Source Nodes: [cat_7, x_170, x_173], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_16.run(buf88, arg179_1, arg180_1, arg60_1, arg61_1, buf94, 43904, grid=grid(43904), stream=stream0)
        del arg179_1
        del arg180_1
        del arg60_1
        del arg61_1
        # Source Nodes: [x_174], Original ATen: [aten.convolution]
        buf89 = extern_kernels.convolution(buf88, arg109_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (4, 224, 7, 7), (10976, 49, 7, 1))
        del arg109_1
        del buf88
        buf95 = reinterpret_tensor(buf96, (4, 224, 7, 7), (92512, 49, 7, 1), 81536)  # alias
        # Source Nodes: [x_175, x_178], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_17.run(buf89, arg181_1, arg182_1, arg62_1, arg63_1, buf95, 43904, grid=grid(43904), stream=stream0)
        del arg181_1
        del arg182_1
        del arg62_1
        del arg63_1
        del buf89
        del buf90
        del buf91
        del buf92
        del buf93
        del buf94
        del buf95
        # Source Nodes: [x_181], Original ATen: [aten.convolution]
        buf97 = extern_kernels.convolution(buf96, arg110_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf97, (4, 1024, 7, 7), (50176, 49, 7, 1))
        del arg110_1
        del buf96
        buf98 = buf97; del buf97  # reuse
        buf114 = empty((4, 2144, 7, 7), device='cuda', dtype=torch.float32)
        buf108 = reinterpret_tensor(buf114, (4, 1024, 7, 7), (105056, 49, 7, 1), 0)  # alias
        # Source Nodes: [cat_6, x_182, x_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_18.run(buf98, arg183_1, arg184_1, arg64_1, arg65_1, buf108, 200704, grid=grid(200704), stream=stream0)
        del arg183_1
        del arg184_1
        del arg64_1
        del arg65_1
        # Source Nodes: [x_187], Original ATen: [aten.convolution]
        buf99 = extern_kernels.convolution(buf98, arg111_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf99, (4, 224, 7, 7), (10976, 49, 7, 1))
        del arg111_1
        del buf98
        buf100 = buf99; del buf99  # reuse
        buf109 = reinterpret_tensor(buf114, (4, 224, 7, 7), (105056, 49, 7, 1), 50176)  # alias
        # Source Nodes: [cat_6, x_188, x_191], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19.run(buf100, arg185_1, arg186_1, arg66_1, arg67_1, buf109, 43904, grid=grid(43904), stream=stream0)
        del arg185_1
        del arg186_1
        del arg66_1
        del arg67_1
        # Source Nodes: [x_192], Original ATen: [aten.convolution]
        buf101 = extern_kernels.convolution(buf100, arg112_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf101, (4, 224, 7, 7), (10976, 49, 7, 1))
        del arg112_1
        del buf100
        buf102 = buf101; del buf101  # reuse
        buf110 = reinterpret_tensor(buf114, (4, 224, 7, 7), (105056, 49, 7, 1), 61152)  # alias
        # Source Nodes: [cat_6, x_193, x_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19.run(buf102, arg187_1, arg188_1, arg68_1, arg69_1, buf110, 43904, grid=grid(43904), stream=stream0)
        del arg187_1
        del arg188_1
        del arg68_1
        del arg69_1
        # Source Nodes: [x_197], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf102, arg113_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (4, 224, 7, 7), (10976, 49, 7, 1))
        del arg113_1
        del buf102
        buf104 = buf103; del buf103  # reuse
        buf111 = reinterpret_tensor(buf114, (4, 224, 7, 7), (105056, 49, 7, 1), 72128)  # alias
        # Source Nodes: [cat_6, x_198, x_201], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19.run(buf104, arg189_1, arg190_1, arg70_1, arg71_1, buf111, 43904, grid=grid(43904), stream=stream0)
        del arg189_1
        del arg190_1
        del arg70_1
        del arg71_1
        # Source Nodes: [x_202], Original ATen: [aten.convolution]
        buf105 = extern_kernels.convolution(buf104, arg114_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf105, (4, 224, 7, 7), (10976, 49, 7, 1))
        del arg114_1
        del buf104
        buf106 = buf105; del buf105  # reuse
        buf112 = reinterpret_tensor(buf114, (4, 224, 7, 7), (105056, 49, 7, 1), 83104)  # alias
        # Source Nodes: [cat_6, x_203, x_206], Original ATen: [aten._native_batch_norm_legit_no_training, aten.cat, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_cat_relu_19.run(buf106, arg191_1, arg192_1, arg72_1, arg73_1, buf112, 43904, grid=grid(43904), stream=stream0)
        del arg191_1
        del arg192_1
        del arg72_1
        del arg73_1
        # Source Nodes: [x_207], Original ATen: [aten.convolution]
        buf107 = extern_kernels.convolution(buf106, arg115_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf107, (4, 224, 7, 7), (10976, 49, 7, 1))
        del arg115_1
        del buf106
        buf113 = reinterpret_tensor(buf114, (4, 224, 7, 7), (105056, 49, 7, 1), 94080)  # alias
        # Source Nodes: [x_208, x_211], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_20.run(buf107, arg193_1, arg194_1, arg74_1, arg75_1, buf113, 43904, grid=grid(43904), stream=stream0)
        del arg193_1
        del arg194_1
        del arg74_1
        del arg75_1
        del buf107
        del buf108
        del buf109
        del buf110
        del buf111
        del buf112
        del buf113
        # Source Nodes: [x_214], Original ATen: [aten.convolution]
        buf115 = extern_kernels.convolution(buf114, arg116_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (4, 1024, 7, 7), (50176, 49, 7, 1))
        del arg116_1
        del buf114
        buf116 = empty_strided((4, 1024, 1, 1), (1024, 1, 4096, 4096), device='cuda', dtype=torch.float32)
        buf117 = reinterpret_tensor(buf116, (4, 1024, 1, 1), (1024, 1, 1, 1), 0); del buf116  # reuse
        # Source Nodes: [x_215, x_221, x_222], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_21.run(buf117, buf115, arg195_1, arg196_1, arg76_1, arg77_1, 4096, 49, grid=grid(4096), stream=stream0)
        del arg195_1
        del arg196_1
        del arg76_1
        del arg77_1
        del buf115
        buf118 = empty((4, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_226], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg118_1, reinterpret_tensor(buf117, (4, 1024), (1024, 1), 0), reinterpret_tensor(arg117_1, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf118)
        del arg117_1
        del arg118_1
        return (buf118, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((256, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((160, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((160, 160, 3, 3), (1440, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((512, 1056, 1, 1), (1056, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((192, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((768, 1472, 1, 1), (1472, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((192, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((768, 1728, 1, 1), (1728, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((224, 768, 3, 3), (6912, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((1024, 1888, 1, 1), (1888, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((224, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((224, 224, 3, 3), (2016, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((1024, 2144, 1, 1), (2144, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((768, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((224, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('timm_vovnet', benchmark_compiled_module)
