
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


# kernel path: /tmp/torchinductor_youkaichao/yd/cydoucfgai66ilqenovdlf5e3llgdrp4cjnxnmxewpt5ndw7m6fr.py
# Source Nodes: [l__mod___inc_double_conv_0, l__mod___inc_double_conv_1, l__mod___inc_double_conv_2, l__mod___inc_double_conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___inc_double_conv_0 => convolution
# l__mod___inc_double_conv_1 => add_1, mul_1, mul_2, sub
# l__mod___inc_double_conv_2 => relu
# l__mod___inc_double_conv_3 => convolution_1
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[134217728], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 78561280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 613760) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt(tmp7)
    tmp9 = 1 / tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/zs/czshwgou2bfmeznolb4gugwxuuxfnht4q4vq3gpn2ppn7zmbmxal.py
# Source Nodes: [l__mod___inc_double_conv_0, l__mod___inc_double_conv_1, l__mod___inc_double_conv_2, l__mod___inc_double_conv_3, l__mod___inc_double_conv_4, x1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___inc_double_conv_0 => convolution
# l__mod___inc_double_conv_1 => add_1, mul_1, mul_2, sub
# l__mod___inc_double_conv_2 => relu
# l__mod___inc_double_conv_3 => convolution_1
# l__mod___inc_double_conv_4 => add_3, mul_4, mul_5, sub_1
# x1 => relu_1
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[134217728], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 78561280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 613760) % 64
    x2 = (xindex // 39280640)
    x4 = xindex % 39280640
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt(tmp7)
    tmp9 = 1 / tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x4 + (78561280*x2)), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/sv/csvr7lm4yujh2tl7sqvuc27ugyv45fpwbmr5rghtsdi56ugaqh6p.py
# Source Nodes: [getattr_l__mod___down1_maxpool_conv___1___double_conv_0, l__mod___down1_maxpool_conv_0], Original ATen: [aten.convolution, aten.max_pool2d_with_indices]
# getattr_l__mod___down1_maxpool_conv___1___double_conv_0 => convolution_2
# l__mod___down1_maxpool_conv_0 => max_pool2d_with_indices
triton_poi_fused_convolution_max_pool2d_with_indices_2 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19619840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 479
    x1 = (xindex // 479) % 20480
    x2 = (xindex // 9809920)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (1918*x1) + (78561280*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (1918*x1) + (78561280*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (959 + (2*x0) + (1918*x1) + (78561280*x2)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (960 + (2*x0) + (1918*x1) + (78561280*x2)), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tw/ctwlt63zbqmz3ayvvzovypyg3kbphnguoafcv4attuhrjqgdumj5.py
# Source Nodes: [getattr_l__mod___down1_maxpool_conv___1___double_conv_0, getattr_l__mod___down1_maxpool_conv___1___double_conv_1, getattr_l__mod___down1_maxpool_conv___1___double_conv_2, getattr_l__mod___down1_maxpool_conv___1___double_conv_3, l__mod___down1_maxpool_conv_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# getattr_l__mod___down1_maxpool_conv___1___double_conv_0 => convolution_2
# getattr_l__mod___down1_maxpool_conv___1___double_conv_1 => add_5, mul_7, mul_8, sub_2
# getattr_l__mod___down1_maxpool_conv___1___double_conv_2 => relu_2
# getattr_l__mod___down1_maxpool_conv___1___double_conv_3 => convolution_3
# l__mod___down1_maxpool_conv_0 => max_pool2d_with_indices
triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_3', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 39239680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 153280) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt(tmp7)
    tmp9 = 1 / tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gp/cgp3t3peydnry2s7xkelh4nmx6w7flwwpufhznaqhbb23dbcyinn.py
# Source Nodes: [getattr_l__mod___down1_maxpool_conv___1___double_conv_0, getattr_l__mod___down1_maxpool_conv___1___double_conv_1, getattr_l__mod___down1_maxpool_conv___1___double_conv_2, getattr_l__mod___down1_maxpool_conv___1___double_conv_3, getattr_l__mod___down1_maxpool_conv___1___double_conv_4, l__mod___down1_maxpool_conv_0, x2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# getattr_l__mod___down1_maxpool_conv___1___double_conv_0 => convolution_2
# getattr_l__mod___down1_maxpool_conv___1___double_conv_1 => add_5, mul_7, mul_8, sub_2
# getattr_l__mod___down1_maxpool_conv___1___double_conv_2 => relu_2
# getattr_l__mod___down1_maxpool_conv___1___double_conv_3 => convolution_3
# getattr_l__mod___down1_maxpool_conv___1___double_conv_4 => add_7, mul_10, mul_11, sub_3
# l__mod___down1_maxpool_conv_0 => max_pool2d_with_indices
# x2 => relu_3
triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 39239680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 153280) % 128
    x2 = (xindex // 19619840)
    x4 = xindex % 19619840
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt(tmp7)
    tmp9 = 1 / tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x4 + (39239680*x2)), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ff/cff3hzewbihljznrieu7xcl2q2abnjkwa7fb6it3iphnol32owdd.py
# Source Nodes: [getattr_l__mod___down2_maxpool_conv___1___double_conv_0, l__mod___down2_maxpool_conv_0], Original ATen: [aten.convolution, aten.max_pool2d_with_indices]
# getattr_l__mod___down2_maxpool_conv___1___double_conv_0 => convolution_4
# l__mod___down2_maxpool_conv_0 => max_pool2d_with_indices_1
triton_poi_fused_convolution_max_pool2d_with_indices_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9789440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 239
    x1 = (xindex // 239) % 20480
    x2 = (xindex // 4894720)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (958*x1) + (39239680*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (958*x1) + (39239680*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (479 + (2*x0) + (958*x1) + (39239680*x2)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (480 + (2*x0) + (958*x1) + (39239680*x2)), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xv/cxvlww7vhxrhqjz2liyfvijtkazeg2aftm6gmb4oc4r7racc7wlx.py
# Source Nodes: [getattr_l__mod___down2_maxpool_conv___1___double_conv_0, getattr_l__mod___down2_maxpool_conv___1___double_conv_1, getattr_l__mod___down2_maxpool_conv___1___double_conv_2, getattr_l__mod___down2_maxpool_conv___1___double_conv_3, l__mod___down2_maxpool_conv_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# getattr_l__mod___down2_maxpool_conv___1___double_conv_0 => convolution_4
# getattr_l__mod___down2_maxpool_conv___1___double_conv_1 => add_9, mul_13, mul_14, sub_4
# getattr_l__mod___down2_maxpool_conv___1___double_conv_2 => relu_4
# getattr_l__mod___down2_maxpool_conv___1___double_conv_3 => convolution_5
# l__mod___down2_maxpool_conv_0 => max_pool2d_with_indices_1
triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19578880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 38240) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt(tmp7)
    tmp9 = 1 / tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wl/cwlqn3oks4jr6hoojqjuqi4u2324nvvl3lskxcziztiixgzvi4fk.py
# Source Nodes: [getattr_l__mod___down2_maxpool_conv___1___double_conv_0, getattr_l__mod___down2_maxpool_conv___1___double_conv_1, getattr_l__mod___down2_maxpool_conv___1___double_conv_2, getattr_l__mod___down2_maxpool_conv___1___double_conv_3, getattr_l__mod___down2_maxpool_conv___1___double_conv_4, l__mod___down2_maxpool_conv_0, x3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# getattr_l__mod___down2_maxpool_conv___1___double_conv_0 => convolution_4
# getattr_l__mod___down2_maxpool_conv___1___double_conv_1 => add_9, mul_13, mul_14, sub_4
# getattr_l__mod___down2_maxpool_conv___1___double_conv_2 => relu_4
# getattr_l__mod___down2_maxpool_conv___1___double_conv_3 => convolution_5
# getattr_l__mod___down2_maxpool_conv___1___double_conv_4 => add_11, mul_16, mul_17, sub_5
# l__mod___down2_maxpool_conv_0 => max_pool2d_with_indices_1
# x3 => relu_5
triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19578880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 38240) % 256
    x2 = (xindex // 9789440)
    x4 = xindex % 9789440
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt(tmp7)
    tmp9 = 1 / tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x4 + (19578880*x2)), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ce/cceyekwwj4k62nfqcr66fzhwx5z3k63v4rdm3mkj4mneww37cbyv.py
# Source Nodes: [getattr_l__mod___down3_maxpool_conv___1___double_conv_0, l__mod___down3_maxpool_conv_0], Original ATen: [aten.convolution, aten.max_pool2d_with_indices]
# getattr_l__mod___down3_maxpool_conv___1___double_conv_0 => convolution_6
# l__mod___down3_maxpool_conv_0 => max_pool2d_with_indices_2
triton_poi_fused_convolution_max_pool2d_with_indices_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4874240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 119
    x1 = (xindex // 119) % 20480
    x2 = (xindex // 2437120)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (478*x1) + (19578880*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (478*x1) + (19578880*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (239 + (2*x0) + (478*x1) + (19578880*x2)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (240 + (2*x0) + (478*x1) + (19578880*x2)), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/on/conrira4gva56yzjjwmj2zkdvtq7zly5k5tjvapzkhmifkzzziaw.py
# Source Nodes: [getattr_l__mod___down3_maxpool_conv___1___double_conv_0, getattr_l__mod___down3_maxpool_conv___1___double_conv_1, getattr_l__mod___down3_maxpool_conv___1___double_conv_2, getattr_l__mod___down3_maxpool_conv___1___double_conv_3, l__mod___down3_maxpool_conv_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# getattr_l__mod___down3_maxpool_conv___1___double_conv_0 => convolution_6
# getattr_l__mod___down3_maxpool_conv___1___double_conv_1 => add_13, mul_19, mul_20, sub_6
# getattr_l__mod___down3_maxpool_conv___1___double_conv_2 => relu_6
# getattr_l__mod___down3_maxpool_conv___1___double_conv_3 => convolution_7
# l__mod___down3_maxpool_conv_0 => max_pool2d_with_indices_2
triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9748480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 9520) % 512
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt(tmp7)
    tmp9 = 1 / tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5x/c5xlhu5zqbdzdejn4sff4oiqjozpwjsy63xuhqh7nl3slhrahgkb.py
# Source Nodes: [getattr_l__mod___down3_maxpool_conv___1___double_conv_0, getattr_l__mod___down3_maxpool_conv___1___double_conv_1, getattr_l__mod___down3_maxpool_conv___1___double_conv_2, getattr_l__mod___down3_maxpool_conv___1___double_conv_3, getattr_l__mod___down3_maxpool_conv___1___double_conv_4, l__mod___down3_maxpool_conv_0, x4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# getattr_l__mod___down3_maxpool_conv___1___double_conv_0 => convolution_6
# getattr_l__mod___down3_maxpool_conv___1___double_conv_1 => add_13, mul_19, mul_20, sub_6
# getattr_l__mod___down3_maxpool_conv___1___double_conv_2 => relu_6
# getattr_l__mod___down3_maxpool_conv___1___double_conv_3 => convolution_7
# getattr_l__mod___down3_maxpool_conv___1___double_conv_4 => add_15, mul_22, mul_23, sub_7
# l__mod___down3_maxpool_conv_0 => max_pool2d_with_indices_2
# x4 => relu_7
triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9748480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 9520) % 512
    x2 = (xindex // 4874240)
    x4 = xindex % 4874240
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt(tmp7)
    tmp9 = 1 / tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(out_ptr0 + (x4 + (9748480*x2)), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/mi/cmizyyqax3zevrf62awu6x5b6bk6ehjd4gy66prvjle6i464k4mn.py
# Source Nodes: [getattr_l__mod___down4_maxpool_conv___1___double_conv_0, l__mod___down4_maxpool_conv_0], Original ATen: [aten.convolution, aten.max_pool2d_with_indices]
# getattr_l__mod___down4_maxpool_conv___1___double_conv_0 => convolution_8
# l__mod___down4_maxpool_conv_0 => max_pool2d_with_indices_3
triton_poi_fused_convolution_max_pool2d_with_indices_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2416640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 59
    x1 = (xindex // 59) % 20480
    x2 = (xindex // 1208320)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (238*x1) + (9748480*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (238*x1) + (9748480*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (119 + (2*x0) + (238*x1) + (9748480*x2)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (120 + (2*x0) + (238*x1) + (9748480*x2)), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (x3), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ni/cniwondakj5o5z62ckckqdw2teryjrhrzldzol2plwzitmmhwlvd.py
# Source Nodes: [getattr_l__mod___down4_maxpool_conv___1___double_conv_0, getattr_l__mod___down4_maxpool_conv___1___double_conv_1, getattr_l__mod___down4_maxpool_conv___1___double_conv_2, getattr_l__mod___down4_maxpool_conv___1___double_conv_3, l__mod___down4_maxpool_conv_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# getattr_l__mod___down4_maxpool_conv___1___double_conv_0 => convolution_8
# getattr_l__mod___down4_maxpool_conv___1___double_conv_1 => add_17, mul_25, mul_26, sub_8
# getattr_l__mod___down4_maxpool_conv___1___double_conv_2 => relu_8
# getattr_l__mod___down4_maxpool_conv___1___double_conv_3 => convolution_9
# l__mod___down4_maxpool_conv_0 => max_pool2d_with_indices_3
triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2416640
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 2360) % 512
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt(tmp7)
    tmp9 = 1 / tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/n2/cn2x5a4bsrpuntpiv635dv433ptzbrxk5xu323h7c6gnzsszza6p.py
# Source Nodes: [x1_1], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.mul, aten.rsub, aten.sub]
# x1_1 => _unsafe_index, _unsafe_index_1, _unsafe_index_2, _unsafe_index_3, add_21, add_22, add_23, add_24, convert_element_type_21, convert_element_type_24, iota_1, mul_31, mul_33, mul_34, mul_35, mul_36, mul_37, mul_38, mul_39, sub_10, sub_11, sub_12, sub_13
triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_13', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9666560
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 118) % 80
    x0 = xindex % 118
    x2 = (xindex // 9440)
    x4 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.4936708860759494
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.int32)
    tmp9 = x0
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp2
    tmp12 = tmp11 + tmp4
    tmp13 = 0.49572649572649574
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14.to(tl.int32)
    tmp16 = tl.load(in_ptr0 + (tmp15 + (59*tmp8) + (2360*x2)), None, eviction_policy='evict_last')
    tmp17 = tmp8.to(tl.float32)
    tmp18 = tmp7 - tmp17
    tmp19 = tmp2 - tmp18
    tmp20 = tmp16 * tmp19
    tmp21 = tl.math.ceil(tmp7)
    tmp22 = 39.0
    tmp23 = triton_helpers.minimum(tmp21, tmp22)
    tmp24 = tmp23.to(tl.int32)
    tmp25 = tl.load(in_ptr0 + (tmp15 + (59*tmp24) + (2360*x2)), None, eviction_policy='evict_last')
    tmp26 = tmp25 * tmp18
    tmp27 = tmp20 + tmp26
    tmp28 = tl.math.ceil(tmp14)
    tmp29 = 58.0
    tmp30 = triton_helpers.minimum(tmp28, tmp29)
    tmp31 = tmp30.to(tl.int32)
    tmp32 = tl.load(in_ptr0 + (tmp31 + (59*tmp8) + (2360*x2)), None, eviction_policy='evict_last')
    tmp33 = tmp32 * tmp19
    tmp34 = tl.load(in_ptr0 + (tmp31 + (59*tmp24) + (2360*x2)), None, eviction_policy='evict_last')
    tmp35 = tmp34 * tmp18
    tmp36 = tmp15.to(tl.float32)
    tmp37 = tmp14 - tmp36
    tmp38 = tmp2 - tmp37
    tmp39 = tmp27 * tmp38
    tmp40 = tmp33 + tmp35
    tmp41 = tmp40 * tmp37
    tmp42 = tmp39 + tmp41
    tl.store(in_out_ptr0 + (x4), tmp42, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/7l/c7lxela5cy2on6zxja6soenv2spfnbckiz4uzishnn7dhjj3oabd.py
# Source Nodes: [x1_2], Original ATen: [aten.constant_pad_nd]
# x1_2 => constant_pad_nd
triton_poi_fused_constant_pad_nd_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9748480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 119
    x3 = (xindex // 119)
    x2 = (xindex // 4874240)
    x4 = xindex % 4874240
    tmp0 = x0
    tmp1 = tl.full([1], 118, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + (118*x3)), tmp2, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x4 + (9748480*x2)), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/kl/ckllxkjda3shx5je3ifnmoydc6ljexla4ufo5g3tdbffbdo6qqjr.py
# Source Nodes: [l__mod___up1_conv_double_conv_0, l__mod___up1_conv_double_conv_1, l__mod___up1_conv_double_conv_2, l__mod___up1_conv_double_conv_3, l__mod___up1_conv_double_conv_4, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___up1_conv_double_conv_0 => convolution_10
# l__mod___up1_conv_double_conv_1 => add_26, mul_41, mul_42, sub_14
# l__mod___up1_conv_double_conv_2 => relu_10
# l__mod___up1_conv_double_conv_3 => convolution_11
# l__mod___up1_conv_double_conv_4 => add_28, mul_44, mul_45, sub_15
# x_1 => relu_11
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4874240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 9520) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt(tmp7)
    tmp9 = 1 / tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/li/clilc7m6p5upgikhoyxomjpvtr7myainfkwsfinmsuv2che2crtu.py
# Source Nodes: [x1_3], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.mul, aten.rsub, aten.sub]
# x1_3 => _unsafe_index_4, _unsafe_index_5, _unsafe_index_6, _unsafe_index_7, add_30, add_31, add_32, add_33, convert_element_type_31, convert_element_type_34, iota_3, mul_47, mul_49, mul_50, mul_51, mul_52, mul_53, mul_54, mul_55, sub_16, sub_17, sub_18, sub_19
triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19496960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 238) % 160
    x0 = xindex % 238
    x2 = (xindex // 38080)
    x4 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.4968553459119497
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.int32)
    tmp9 = x0
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp2
    tmp12 = tmp11 + tmp4
    tmp13 = 0.4978902953586498
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14.to(tl.int32)
    tmp16 = tl.load(in_ptr0 + (tmp15 + (119*tmp8) + (9520*x2)), None, eviction_policy='evict_last')
    tmp17 = tmp8.to(tl.float32)
    tmp18 = tmp7 - tmp17
    tmp19 = tmp2 - tmp18
    tmp20 = tmp16 * tmp19
    tmp21 = tl.math.ceil(tmp7)
    tmp22 = 79.0
    tmp23 = triton_helpers.minimum(tmp21, tmp22)
    tmp24 = tmp23.to(tl.int32)
    tmp25 = tl.load(in_ptr0 + (tmp15 + (119*tmp24) + (9520*x2)), None, eviction_policy='evict_last')
    tmp26 = tmp25 * tmp18
    tmp27 = tmp20 + tmp26
    tmp28 = tl.math.ceil(tmp14)
    tmp29 = 118.0
    tmp30 = triton_helpers.minimum(tmp28, tmp29)
    tmp31 = tmp30.to(tl.int32)
    tmp32 = tl.load(in_ptr0 + (tmp31 + (119*tmp8) + (9520*x2)), None, eviction_policy='evict_last')
    tmp33 = tmp32 * tmp19
    tmp34 = tl.load(in_ptr0 + (tmp31 + (119*tmp24) + (9520*x2)), None, eviction_policy='evict_last')
    tmp35 = tmp34 * tmp18
    tmp36 = tmp15.to(tl.float32)
    tmp37 = tmp14 - tmp36
    tmp38 = tmp2 - tmp37
    tmp39 = tmp27 * tmp38
    tmp40 = tmp33 + tmp35
    tmp41 = tmp40 * tmp37
    tmp42 = tmp39 + tmp41
    tl.store(in_out_ptr0 + (x4), tmp42, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/3p/c3poihclyjbp6fyzjf424fglcolq3ma72zkfx6zs6gqgovx3pqjv.py
# Source Nodes: [x1_4], Original ATen: [aten.constant_pad_nd]
# x1_4 => constant_pad_nd_1
triton_poi_fused_constant_pad_nd_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19578880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 239
    x3 = (xindex // 239)
    x2 = (xindex // 9789440)
    x4 = xindex % 9789440
    tmp0 = x0
    tmp1 = tl.full([1], 238, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + (238*x3)), tmp2, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x4 + (19578880*x2)), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/f2/cf2dsyjapnfbibndhks6sf6ocmoqxrrszdj52qfvl4fdfutifxrt.py
# Source Nodes: [l__mod___up2_conv_double_conv_0, l__mod___up2_conv_double_conv_1, l__mod___up2_conv_double_conv_2, l__mod___up2_conv_double_conv_3, l__mod___up2_conv_double_conv_4, x_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___up2_conv_double_conv_0 => convolution_12
# l__mod___up2_conv_double_conv_1 => add_35, mul_57, mul_58, sub_20
# l__mod___up2_conv_double_conv_2 => relu_12
# l__mod___up2_conv_double_conv_3 => convolution_13
# l__mod___up2_conv_double_conv_4 => add_37, mul_60, mul_61, sub_21
# x_3 => relu_13
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 9789440
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 38240) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt(tmp7)
    tmp9 = 1 / tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ha/cha7r5kzhxwlolp325qzbxy7yppwgu6ouwaxmhdxsuvn62as24sr.py
# Source Nodes: [x1_5], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.mul, aten.rsub, aten.sub]
# x1_5 => _unsafe_index_10, _unsafe_index_11, _unsafe_index_8, _unsafe_index_9, add_39, add_40, add_41, add_42, convert_element_type_41, convert_element_type_44, iota_5, mul_63, mul_65, mul_66, mul_67, mul_68, mul_69, mul_70, mul_71, sub_22, sub_23, sub_24, sub_25
triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 39157760
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 478) % 320
    x0 = xindex % 478
    x2 = (xindex // 152960)
    x4 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.49843260188087773
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.int32)
    tmp9 = x0
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp2
    tmp12 = tmp11 + tmp4
    tmp13 = 0.4989517819706499
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14.to(tl.int32)
    tmp16 = tl.load(in_ptr0 + (tmp15 + (239*tmp8) + (38240*x2)), None, eviction_policy='evict_last')
    tmp17 = tmp8.to(tl.float32)
    tmp18 = tmp7 - tmp17
    tmp19 = tmp2 - tmp18
    tmp20 = tmp16 * tmp19
    tmp21 = tl.math.ceil(tmp7)
    tmp22 = 159.0
    tmp23 = triton_helpers.minimum(tmp21, tmp22)
    tmp24 = tmp23.to(tl.int32)
    tmp25 = tl.load(in_ptr0 + (tmp15 + (239*tmp24) + (38240*x2)), None, eviction_policy='evict_last')
    tmp26 = tmp25 * tmp18
    tmp27 = tmp20 + tmp26
    tmp28 = tl.math.ceil(tmp14)
    tmp29 = 238.0
    tmp30 = triton_helpers.minimum(tmp28, tmp29)
    tmp31 = tmp30.to(tl.int32)
    tmp32 = tl.load(in_ptr0 + (tmp31 + (239*tmp8) + (38240*x2)), None, eviction_policy='evict_last')
    tmp33 = tmp32 * tmp19
    tmp34 = tl.load(in_ptr0 + (tmp31 + (239*tmp24) + (38240*x2)), None, eviction_policy='evict_last')
    tmp35 = tmp34 * tmp18
    tmp36 = tmp15.to(tl.float32)
    tmp37 = tmp14 - tmp36
    tmp38 = tmp2 - tmp37
    tmp39 = tmp27 * tmp38
    tmp40 = tmp33 + tmp35
    tmp41 = tmp40 * tmp37
    tmp42 = tmp39 + tmp41
    tl.store(in_out_ptr0 + (x4), tmp42, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/hg/chg4zf5jwfsoffiofv2f2ivbnrlyoivxd4pzf2n5cjwsnrzayav4.py
# Source Nodes: [x1_6], Original ATen: [aten.constant_pad_nd]
# x1_6 => constant_pad_nd_2
triton_poi_fused_constant_pad_nd_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[67108864], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 39239680
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 479
    x3 = (xindex // 479)
    x2 = (xindex // 19619840)
    x4 = xindex % 19619840
    tmp0 = x0
    tmp1 = tl.full([1], 478, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + (478*x3)), tmp2, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x4 + (39239680*x2)), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bw/cbwf37nvt6p5qj7re2qmyqugexdxhum5urnmjhx4vppocpzl33uh.py
# Source Nodes: [l__mod___up3_conv_double_conv_0, l__mod___up3_conv_double_conv_1, l__mod___up3_conv_double_conv_2, l__mod___up3_conv_double_conv_3, l__mod___up3_conv_double_conv_4, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___up3_conv_double_conv_0 => convolution_14
# l__mod___up3_conv_double_conv_1 => add_44, mul_73, mul_74, sub_26
# l__mod___up3_conv_double_conv_2 => relu_14
# l__mod___up3_conv_double_conv_3 => convolution_15
# l__mod___up3_conv_double_conv_4 => add_46, mul_76, mul_77, sub_27
# x_5 => relu_15
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, XBLOCK : tl.constexpr):
    xnumel = 19619840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 153280) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 - tmp3
    tmp6 = 1e-05
    tmp7 = tmp5 + tmp6
    tmp8 = tl.sqrt(tmp7)
    tmp9 = 1 / tmp8
    tmp10 = 1.0
    tmp11 = tmp9 * tmp10
    tmp12 = tmp4 * tmp11
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tl.store(in_out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/o6/co6ghfcetvyqt23a32g24ryo2mfc7l2tl35m3byquu6ogkt5kbyq.py
# Source Nodes: [x1_7], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.mul, aten.rsub, aten.sub]
# x1_7 => _unsafe_index_12, _unsafe_index_13, _unsafe_index_14, _unsafe_index_15, add_48, add_49, add_50, add_51, convert_element_type_51, convert_element_type_54, iota_7, mul_79, mul_81, mul_82, mul_83, mul_84, mul_85, mul_86, mul_87, sub_28, sub_29, sub_30, sub_31
triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[134217728], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 78479360
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 958) % 640
    x0 = xindex % 958
    x2 = (xindex // 613120)
    x4 = xindex
    tmp0 = x1
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 1.0
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.49921752738654146
    tmp7 = tmp5 * tmp6
    tmp8 = tmp7.to(tl.int32)
    tmp9 = x0
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp2
    tmp12 = tmp11 + tmp4
    tmp13 = 0.4994775339602926
    tmp14 = tmp12 * tmp13
    tmp15 = tmp14.to(tl.int32)
    tmp16 = tl.load(in_ptr0 + (tmp15 + (479*tmp8) + (153280*x2)), None, eviction_policy='evict_last')
    tmp17 = tmp8.to(tl.float32)
    tmp18 = tmp7 - tmp17
    tmp19 = tmp2 - tmp18
    tmp20 = tmp16 * tmp19
    tmp21 = tl.math.ceil(tmp7)
    tmp22 = 319.0
    tmp23 = triton_helpers.minimum(tmp21, tmp22)
    tmp24 = tmp23.to(tl.int32)
    tmp25 = tl.load(in_ptr0 + (tmp15 + (479*tmp24) + (153280*x2)), None, eviction_policy='evict_last')
    tmp26 = tmp25 * tmp18
    tmp27 = tmp20 + tmp26
    tmp28 = tl.math.ceil(tmp14)
    tmp29 = 478.0
    tmp30 = triton_helpers.minimum(tmp28, tmp29)
    tmp31 = tmp30.to(tl.int32)
    tmp32 = tl.load(in_ptr0 + (tmp31 + (479*tmp8) + (153280*x2)), None, eviction_policy='evict_last')
    tmp33 = tmp32 * tmp19
    tmp34 = tl.load(in_ptr0 + (tmp31 + (479*tmp24) + (153280*x2)), None, eviction_policy='evict_last')
    tmp35 = tmp34 * tmp18
    tmp36 = tmp15.to(tl.float32)
    tmp37 = tmp14 - tmp36
    tmp38 = tmp2 - tmp37
    tmp39 = tmp27 * tmp38
    tmp40 = tmp33 + tmp35
    tmp41 = tmp40 * tmp37
    tmp42 = tmp39 + tmp41
    tl.store(in_out_ptr0 + (x4), tmp42, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/gj/cgjeul54tde6cmeaqgjsv7ps5qgtrgepmsuadijsjwohjxjjpduu.py
# Source Nodes: [x1_8], Original ATen: [aten.constant_pad_nd]
# x1_8 => constant_pad_nd_3
triton_poi_fused_constant_pad_nd_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[134217728], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_constant_pad_nd_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 78561280
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 959
    x3 = (xindex // 959)
    x2 = (xindex // 39280640)
    x4 = xindex % 39280640
    tmp0 = x0
    tmp1 = tl.full([1], 958, tl.int64)
    tmp2 = tmp0 < tmp1
    tmp3 = tl.load(in_ptr0 + (x0 + (958*x3)), tmp2, other=0.0)
    tmp4 = tl.full(tmp3.shape, 0.0, tmp3.dtype)
    tmp5 = tl.where(tmp2, tmp3, tmp4)
    tl.store(out_ptr0 + (x4 + (78561280*x2)), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/y6/cy6gk6ekzb6g5hba3vi3twe4qlwbsdtdrdo3olirqz5ho7udf2m7.py
# Source Nodes: [l__mod___up4_conv_double_conv_0, l__mod___up4_conv_double_conv_1, l__mod___up4_conv_double_conv_2, l__mod___up4_conv_double_conv_3, l__mod___up4_conv_double_conv_4, logits, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# l__mod___up4_conv_double_conv_0 => convolution_16
# l__mod___up4_conv_double_conv_1 => add_53, mul_89, mul_90, sub_32
# l__mod___up4_conv_double_conv_2 => relu_16
# l__mod___up4_conv_double_conv_3 => convolution_17
# l__mod___up4_conv_double_conv_4 => add_55, mul_92, mul_93, sub_33
# logits => convolution_18
# x_7 => relu_17
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2455040
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 613760) % 2
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tl.store(in_out_ptr0 + (x3), tmp2, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (64, ), (1, ))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (64, ), (1, ))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg9_1, (128, ), (1, ))
    assert_size_stride(arg10_1, (128, ), (1, ))
    assert_size_stride(arg11_1, (128, ), (1, ))
    assert_size_stride(arg12_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg13_1, (128, ), (1, ))
    assert_size_stride(arg14_1, (128, ), (1, ))
    assert_size_stride(arg15_1, (128, ), (1, ))
    assert_size_stride(arg16_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg17_1, (256, ), (1, ))
    assert_size_stride(arg18_1, (256, ), (1, ))
    assert_size_stride(arg19_1, (256, ), (1, ))
    assert_size_stride(arg20_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg21_1, (256, ), (1, ))
    assert_size_stride(arg22_1, (256, ), (1, ))
    assert_size_stride(arg23_1, (256, ), (1, ))
    assert_size_stride(arg24_1, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg25_1, (512, ), (1, ))
    assert_size_stride(arg26_1, (512, ), (1, ))
    assert_size_stride(arg27_1, (512, ), (1, ))
    assert_size_stride(arg28_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg29_1, (512, ), (1, ))
    assert_size_stride(arg30_1, (512, ), (1, ))
    assert_size_stride(arg31_1, (512, ), (1, ))
    assert_size_stride(arg32_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg33_1, (512, ), (1, ))
    assert_size_stride(arg34_1, (512, ), (1, ))
    assert_size_stride(arg35_1, (512, ), (1, ))
    assert_size_stride(arg36_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg37_1, (512, ), (1, ))
    assert_size_stride(arg38_1, (512, ), (1, ))
    assert_size_stride(arg39_1, (512, ), (1, ))
    assert_size_stride(arg40_1, (512, 1024, 3, 3), (9216, 9, 3, 1))
    assert_size_stride(arg41_1, (512, ), (1, ))
    assert_size_stride(arg42_1, (512, ), (1, ))
    assert_size_stride(arg43_1, (512, ), (1, ))
    assert_size_stride(arg44_1, (256, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg45_1, (256, ), (1, ))
    assert_size_stride(arg46_1, (256, ), (1, ))
    assert_size_stride(arg47_1, (256, ), (1, ))
    assert_size_stride(arg48_1, (256, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg49_1, (256, ), (1, ))
    assert_size_stride(arg50_1, (256, ), (1, ))
    assert_size_stride(arg51_1, (256, ), (1, ))
    assert_size_stride(arg52_1, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg53_1, (128, ), (1, ))
    assert_size_stride(arg54_1, (128, ), (1, ))
    assert_size_stride(arg55_1, (128, ), (1, ))
    assert_size_stride(arg56_1, (128, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg57_1, (128, ), (1, ))
    assert_size_stride(arg58_1, (128, ), (1, ))
    assert_size_stride(arg59_1, (128, ), (1, ))
    assert_size_stride(arg60_1, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg61_1, (64, ), (1, ))
    assert_size_stride(arg62_1, (64, ), (1, ))
    assert_size_stride(arg63_1, (64, ), (1, ))
    assert_size_stride(arg64_1, (64, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg65_1, (64, ), (1, ))
    assert_size_stride(arg66_1, (64, ), (1, ))
    assert_size_stride(arg67_1, (64, ), (1, ))
    assert_size_stride(arg68_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg69_1, (64, ), (1, ))
    assert_size_stride(arg70_1, (64, ), (1, ))
    assert_size_stride(arg71_1, (64, ), (1, ))
    assert_size_stride(arg72_1, (2, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg73_1, (2, ), (1, ))
    assert_size_stride(arg74_1, (64, ), (1, ))
    assert_size_stride(arg75_1, (64, ), (1, ))
    assert_size_stride(arg76_1, (), ())
    assert_size_stride(arg77_1, (64, ), (1, ))
    assert_size_stride(arg78_1, (64, ), (1, ))
    assert_size_stride(arg79_1, (), ())
    assert_size_stride(arg80_1, (128, ), (1, ))
    assert_size_stride(arg81_1, (128, ), (1, ))
    assert_size_stride(arg82_1, (), ())
    assert_size_stride(arg83_1, (128, ), (1, ))
    assert_size_stride(arg84_1, (128, ), (1, ))
    assert_size_stride(arg85_1, (), ())
    assert_size_stride(arg86_1, (256, ), (1, ))
    assert_size_stride(arg87_1, (256, ), (1, ))
    assert_size_stride(arg88_1, (), ())
    assert_size_stride(arg89_1, (256, ), (1, ))
    assert_size_stride(arg90_1, (256, ), (1, ))
    assert_size_stride(arg91_1, (), ())
    assert_size_stride(arg92_1, (512, ), (1, ))
    assert_size_stride(arg93_1, (512, ), (1, ))
    assert_size_stride(arg94_1, (), ())
    assert_size_stride(arg95_1, (512, ), (1, ))
    assert_size_stride(arg96_1, (512, ), (1, ))
    assert_size_stride(arg97_1, (), ())
    assert_size_stride(arg98_1, (512, ), (1, ))
    assert_size_stride(arg99_1, (512, ), (1, ))
    assert_size_stride(arg100_1, (), ())
    assert_size_stride(arg101_1, (512, ), (1, ))
    assert_size_stride(arg102_1, (512, ), (1, ))
    assert_size_stride(arg103_1, (), ())
    assert_size_stride(arg104_1, (512, ), (1, ))
    assert_size_stride(arg105_1, (512, ), (1, ))
    assert_size_stride(arg106_1, (), ())
    assert_size_stride(arg107_1, (256, ), (1, ))
    assert_size_stride(arg108_1, (256, ), (1, ))
    assert_size_stride(arg109_1, (), ())
    assert_size_stride(arg110_1, (256, ), (1, ))
    assert_size_stride(arg111_1, (256, ), (1, ))
    assert_size_stride(arg112_1, (), ())
    assert_size_stride(arg113_1, (128, ), (1, ))
    assert_size_stride(arg114_1, (128, ), (1, ))
    assert_size_stride(arg115_1, (), ())
    assert_size_stride(arg116_1, (128, ), (1, ))
    assert_size_stride(arg117_1, (128, ), (1, ))
    assert_size_stride(arg118_1, (), ())
    assert_size_stride(arg119_1, (64, ), (1, ))
    assert_size_stride(arg120_1, (64, ), (1, ))
    assert_size_stride(arg121_1, (), ())
    assert_size_stride(arg122_1, (64, ), (1, ))
    assert_size_stride(arg123_1, (64, ), (1, ))
    assert_size_stride(arg124_1, (), ())
    assert_size_stride(arg125_1, (64, ), (1, ))
    assert_size_stride(arg126_1, (64, ), (1, ))
    assert_size_stride(arg127_1, (), ())
    assert_size_stride(arg128_1, (2, 3, 640, 959), (1841280, 613760, 959, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        # Source Nodes: [l__mod___inc_double_conv_0], Original ATen: [aten.convolution]
        buf0 = extern_kernels.convolution(arg128_1, arg0_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf0, (2, 64, 640, 959), (39280640, 613760, 959, 1))
        del arg0_1
        del arg128_1
        buf1 = buf0; del buf0  # reuse
        # Source Nodes: [l__mod___inc_double_conv_0, l__mod___inc_double_conv_1, l__mod___inc_double_conv_2, l__mod___inc_double_conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        stream0 = get_cuda_stream(0)
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf1, arg1_1, arg74_1, arg75_1, arg2_1, arg3_1, 78561280, grid=grid(78561280), stream=stream0)
        del arg1_1
        del arg2_1
        del arg3_1
        del arg74_1
        del arg75_1
        # Source Nodes: [l__mod___inc_double_conv_0, l__mod___inc_double_conv_1, l__mod___inc_double_conv_2, l__mod___inc_double_conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf2 = extern_kernels.convolution(buf1, arg4_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (2, 64, 640, 959), (39280640, 613760, 959, 1))
        del arg4_1
        del buf1
        buf59 = empty((2, 128, 640, 959), device='cuda', dtype=torch.float32)
        buf3 = reinterpret_tensor(buf59, (2, 64, 640, 959), (78561280, 613760, 959, 1), 0)  # alias
        # Source Nodes: [l__mod___inc_double_conv_0, l__mod___inc_double_conv_1, l__mod___inc_double_conv_2, l__mod___inc_double_conv_3, l__mod___inc_double_conv_4, x1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_1.run(buf2, arg5_1, arg77_1, arg78_1, arg6_1, arg7_1, buf3, 78561280, grid=grid(78561280), stream=stream0)
        del arg5_1
        del arg6_1
        del arg77_1
        del arg78_1
        del arg7_1
        del buf2
        buf4 = empty((2, 64, 320, 479), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___down1_maxpool_conv___1___double_conv_0, l__mod___down1_maxpool_conv_0], Original ATen: [aten.convolution, aten.max_pool2d_with_indices]
        triton_poi_fused_convolution_max_pool2d_with_indices_2.run(buf3, buf4, 19619840, grid=grid(19619840), stream=stream0)
        # Source Nodes: [getattr_l__mod___down1_maxpool_conv___1___double_conv_0, l__mod___down1_maxpool_conv_0], Original ATen: [aten.convolution, aten.max_pool2d_with_indices]
        buf5 = extern_kernels.convolution(buf4, arg8_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (2, 128, 320, 479), (19619840, 153280, 479, 1))
        del arg8_1
        del buf4
        buf6 = buf5; del buf5  # reuse
        # Source Nodes: [getattr_l__mod___down1_maxpool_conv___1___double_conv_0, getattr_l__mod___down1_maxpool_conv___1___double_conv_1, getattr_l__mod___down1_maxpool_conv___1___double_conv_2, getattr_l__mod___down1_maxpool_conv___1___double_conv_3, l__mod___down1_maxpool_conv_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_3.run(buf6, arg9_1, arg80_1, arg81_1, arg10_1, arg11_1, 39239680, grid=grid(39239680), stream=stream0)
        del arg10_1
        del arg11_1
        del arg80_1
        del arg81_1
        del arg9_1
        # Source Nodes: [getattr_l__mod___down1_maxpool_conv___1___double_conv_0, getattr_l__mod___down1_maxpool_conv___1___double_conv_1, getattr_l__mod___down1_maxpool_conv___1___double_conv_2, getattr_l__mod___down1_maxpool_conv___1___double_conv_3, l__mod___down1_maxpool_conv_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        buf7 = extern_kernels.convolution(buf6, arg12_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (2, 128, 320, 479), (19619840, 153280, 479, 1))
        del arg12_1
        del buf6
        buf49 = empty((2, 256, 320, 479), device='cuda', dtype=torch.float32)
        buf8 = reinterpret_tensor(buf49, (2, 128, 320, 479), (39239680, 153280, 479, 1), 0)  # alias
        # Source Nodes: [getattr_l__mod___down1_maxpool_conv___1___double_conv_0, getattr_l__mod___down1_maxpool_conv___1___double_conv_1, getattr_l__mod___down1_maxpool_conv___1___double_conv_2, getattr_l__mod___down1_maxpool_conv___1___double_conv_3, getattr_l__mod___down1_maxpool_conv___1___double_conv_4, l__mod___down1_maxpool_conv_0, x2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_4.run(buf7, arg13_1, arg83_1, arg84_1, arg14_1, arg15_1, buf8, 39239680, grid=grid(39239680), stream=stream0)
        del arg13_1
        del arg14_1
        del arg15_1
        del arg83_1
        del arg84_1
        del buf7
        buf9 = empty((2, 128, 160, 239), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___down2_maxpool_conv___1___double_conv_0, l__mod___down2_maxpool_conv_0], Original ATen: [aten.convolution, aten.max_pool2d_with_indices]
        triton_poi_fused_convolution_max_pool2d_with_indices_5.run(buf8, buf9, 9789440, grid=grid(9789440), stream=stream0)
        # Source Nodes: [getattr_l__mod___down2_maxpool_conv___1___double_conv_0, l__mod___down2_maxpool_conv_0], Original ATen: [aten.convolution, aten.max_pool2d_with_indices]
        buf10 = extern_kernels.convolution(buf9, arg16_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (2, 256, 160, 239), (9789440, 38240, 239, 1))
        del arg16_1
        del buf9
        buf11 = buf10; del buf10  # reuse
        # Source Nodes: [getattr_l__mod___down2_maxpool_conv___1___double_conv_0, getattr_l__mod___down2_maxpool_conv___1___double_conv_1, getattr_l__mod___down2_maxpool_conv___1___double_conv_2, getattr_l__mod___down2_maxpool_conv___1___double_conv_3, l__mod___down2_maxpool_conv_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_6.run(buf11, arg17_1, arg86_1, arg87_1, arg18_1, arg19_1, 19578880, grid=grid(19578880), stream=stream0)
        del arg17_1
        del arg18_1
        del arg19_1
        del arg86_1
        del arg87_1
        # Source Nodes: [getattr_l__mod___down2_maxpool_conv___1___double_conv_0, getattr_l__mod___down2_maxpool_conv___1___double_conv_1, getattr_l__mod___down2_maxpool_conv___1___double_conv_2, getattr_l__mod___down2_maxpool_conv___1___double_conv_3, l__mod___down2_maxpool_conv_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        buf12 = extern_kernels.convolution(buf11, arg20_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (2, 256, 160, 239), (9789440, 38240, 239, 1))
        del arg20_1
        del buf11
        buf39 = empty((2, 512, 160, 239), device='cuda', dtype=torch.float32)
        buf13 = reinterpret_tensor(buf39, (2, 256, 160, 239), (19578880, 38240, 239, 1), 0)  # alias
        # Source Nodes: [getattr_l__mod___down2_maxpool_conv___1___double_conv_0, getattr_l__mod___down2_maxpool_conv___1___double_conv_1, getattr_l__mod___down2_maxpool_conv___1___double_conv_2, getattr_l__mod___down2_maxpool_conv___1___double_conv_3, getattr_l__mod___down2_maxpool_conv___1___double_conv_4, l__mod___down2_maxpool_conv_0, x3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_7.run(buf12, arg21_1, arg89_1, arg90_1, arg22_1, arg23_1, buf13, 19578880, grid=grid(19578880), stream=stream0)
        del arg21_1
        del arg22_1
        del arg23_1
        del arg89_1
        del arg90_1
        del buf12
        buf14 = empty((2, 256, 80, 119), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___down3_maxpool_conv___1___double_conv_0, l__mod___down3_maxpool_conv_0], Original ATen: [aten.convolution, aten.max_pool2d_with_indices]
        triton_poi_fused_convolution_max_pool2d_with_indices_8.run(buf13, buf14, 4874240, grid=grid(4874240), stream=stream0)
        # Source Nodes: [getattr_l__mod___down3_maxpool_conv___1___double_conv_0, l__mod___down3_maxpool_conv_0], Original ATen: [aten.convolution, aten.max_pool2d_with_indices]
        buf15 = extern_kernels.convolution(buf14, arg24_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (2, 512, 80, 119), (4874240, 9520, 119, 1))
        del arg24_1
        del buf14
        buf16 = buf15; del buf15  # reuse
        # Source Nodes: [getattr_l__mod___down3_maxpool_conv___1___double_conv_0, getattr_l__mod___down3_maxpool_conv___1___double_conv_1, getattr_l__mod___down3_maxpool_conv___1___double_conv_2, getattr_l__mod___down3_maxpool_conv___1___double_conv_3, l__mod___down3_maxpool_conv_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_9.run(buf16, arg25_1, arg92_1, arg93_1, arg26_1, arg27_1, 9748480, grid=grid(9748480), stream=stream0)
        del arg25_1
        del arg26_1
        del arg27_1
        del arg92_1
        del arg93_1
        # Source Nodes: [getattr_l__mod___down3_maxpool_conv___1___double_conv_0, getattr_l__mod___down3_maxpool_conv___1___double_conv_1, getattr_l__mod___down3_maxpool_conv___1___double_conv_2, getattr_l__mod___down3_maxpool_conv___1___double_conv_3, l__mod___down3_maxpool_conv_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        buf17 = extern_kernels.convolution(buf16, arg28_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (2, 512, 80, 119), (4874240, 9520, 119, 1))
        del arg28_1
        del buf16
        buf29 = empty((2, 1024, 80, 119), device='cuda', dtype=torch.float32)
        buf18 = reinterpret_tensor(buf29, (2, 512, 80, 119), (9748480, 9520, 119, 1), 0)  # alias
        # Source Nodes: [getattr_l__mod___down3_maxpool_conv___1___double_conv_0, getattr_l__mod___down3_maxpool_conv___1___double_conv_1, getattr_l__mod___down3_maxpool_conv___1___double_conv_2, getattr_l__mod___down3_maxpool_conv___1___double_conv_3, getattr_l__mod___down3_maxpool_conv___1___double_conv_4, l__mod___down3_maxpool_conv_0, x4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_10.run(buf17, arg29_1, arg95_1, arg96_1, arg30_1, arg31_1, buf18, 9748480, grid=grid(9748480), stream=stream0)
        del arg29_1
        del arg30_1
        del arg31_1
        del arg95_1
        del arg96_1
        del buf17
        buf19 = empty((2, 512, 40, 59), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___down4_maxpool_conv___1___double_conv_0, l__mod___down4_maxpool_conv_0], Original ATen: [aten.convolution, aten.max_pool2d_with_indices]
        triton_poi_fused_convolution_max_pool2d_with_indices_11.run(buf18, buf19, 2416640, grid=grid(2416640), stream=stream0)
        # Source Nodes: [getattr_l__mod___down4_maxpool_conv___1___double_conv_0, l__mod___down4_maxpool_conv_0], Original ATen: [aten.convolution, aten.max_pool2d_with_indices]
        buf20 = extern_kernels.convolution(buf19, arg32_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (2, 512, 40, 59), (1208320, 2360, 59, 1))
        del arg32_1
        del buf19
        buf21 = buf20; del buf20  # reuse
        # Source Nodes: [getattr_l__mod___down4_maxpool_conv___1___double_conv_0, getattr_l__mod___down4_maxpool_conv___1___double_conv_1, getattr_l__mod___down4_maxpool_conv___1___double_conv_2, getattr_l__mod___down4_maxpool_conv___1___double_conv_3, l__mod___down4_maxpool_conv_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_12.run(buf21, arg33_1, arg98_1, arg99_1, arg34_1, arg35_1, 2416640, grid=grid(2416640), stream=stream0)
        del arg33_1
        del arg34_1
        del arg35_1
        del arg98_1
        del arg99_1
        # Source Nodes: [getattr_l__mod___down4_maxpool_conv___1___double_conv_0, getattr_l__mod___down4_maxpool_conv___1___double_conv_1, getattr_l__mod___down4_maxpool_conv___1___double_conv_2, getattr_l__mod___down4_maxpool_conv___1___double_conv_3, l__mod___down4_maxpool_conv_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        buf22 = extern_kernels.convolution(buf21, arg36_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (2, 512, 40, 59), (1208320, 2360, 59, 1))
        del arg36_1
        del buf21
        buf23 = buf22; del buf22  # reuse
        # Source Nodes: [getattr_l__mod___down4_maxpool_conv___1___double_conv_0, getattr_l__mod___down4_maxpool_conv___1___double_conv_1, getattr_l__mod___down4_maxpool_conv___1___double_conv_2, getattr_l__mod___down4_maxpool_conv___1___double_conv_3, getattr_l__mod___down4_maxpool_conv___1___double_conv_4, l__mod___down4_maxpool_conv_0, x5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_12.run(buf23, arg37_1, arg101_1, arg102_1, arg38_1, arg39_1, 2416640, grid=grid(2416640), stream=stream0)
        del arg101_1
        del arg102_1
        del arg37_1
        del arg38_1
        del arg39_1
        buf24 = empty((2, 512, 80, 118), device='cuda', dtype=torch.float32)
        buf27 = buf24; del buf24  # reuse
        # Source Nodes: [x1_1], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.mul, aten.rsub, aten.sub]
        triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_13.run(buf27, buf23, 9666560, grid=grid(9666560), stream=stream0)
        del buf23
        buf28 = reinterpret_tensor(buf29, (2, 512, 80, 119), (9748480, 9520, 119, 1), 4874240)  # alias
        # Source Nodes: [x1_2], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_14.run(buf27, buf28, 9748480, grid=grid(9748480), stream=stream0)
        del buf27
        del buf18
        del buf28
        # Source Nodes: [l__mod___up1_conv_double_conv_0], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, arg40_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (2, 512, 80, 119), (4874240, 9520, 119, 1))
        del arg40_1
        buf31 = buf30; del buf30  # reuse
        # Source Nodes: [l__mod___up1_conv_double_conv_0, l__mod___up1_conv_double_conv_1, l__mod___up1_conv_double_conv_2, l__mod___up1_conv_double_conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_9.run(buf31, arg41_1, arg104_1, arg105_1, arg42_1, arg43_1, 9748480, grid=grid(9748480), stream=stream0)
        del arg104_1
        del arg105_1
        del arg41_1
        del arg42_1
        del arg43_1
        # Source Nodes: [l__mod___up1_conv_double_conv_0, l__mod___up1_conv_double_conv_1, l__mod___up1_conv_double_conv_2, l__mod___up1_conv_double_conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf32 = extern_kernels.convolution(buf31, arg44_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (2, 256, 80, 119), (2437120, 9520, 119, 1))
        del arg44_1
        del buf31
        buf33 = buf32; del buf32  # reuse
        # Source Nodes: [l__mod___up1_conv_double_conv_0, l__mod___up1_conv_double_conv_1, l__mod___up1_conv_double_conv_2, l__mod___up1_conv_double_conv_3, l__mod___up1_conv_double_conv_4, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(buf33, arg45_1, arg107_1, arg108_1, arg46_1, arg47_1, 4874240, grid=grid(4874240), stream=stream0)
        del arg107_1
        del arg108_1
        del arg45_1
        del arg46_1
        del arg47_1
        buf34 = reinterpret_tensor(buf29, (2, 256, 160, 238), (9748480, 38080, 238, 1), 0); del buf29  # reuse
        buf37 = buf34; del buf34  # reuse
        # Source Nodes: [x1_3], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.mul, aten.rsub, aten.sub]
        triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_16.run(buf37, buf33, 19496960, grid=grid(19496960), stream=stream0)
        del buf33
        buf38 = reinterpret_tensor(buf39, (2, 256, 160, 239), (19578880, 38240, 239, 1), 9789440)  # alias
        # Source Nodes: [x1_4], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_17.run(buf37, buf38, 19578880, grid=grid(19578880), stream=stream0)
        del buf37
        del buf13
        del buf38
        # Source Nodes: [l__mod___up2_conv_double_conv_0], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, arg48_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (2, 256, 160, 239), (9789440, 38240, 239, 1))
        del arg48_1
        buf41 = buf40; del buf40  # reuse
        # Source Nodes: [l__mod___up2_conv_double_conv_0, l__mod___up2_conv_double_conv_1, l__mod___up2_conv_double_conv_2, l__mod___up2_conv_double_conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_6.run(buf41, arg49_1, arg110_1, arg111_1, arg50_1, arg51_1, 19578880, grid=grid(19578880), stream=stream0)
        del arg110_1
        del arg111_1
        del arg49_1
        del arg50_1
        del arg51_1
        # Source Nodes: [l__mod___up2_conv_double_conv_0, l__mod___up2_conv_double_conv_1, l__mod___up2_conv_double_conv_2, l__mod___up2_conv_double_conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf42 = extern_kernels.convolution(buf41, arg52_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (2, 128, 160, 239), (4894720, 38240, 239, 1))
        del arg52_1
        del buf41
        buf43 = buf42; del buf42  # reuse
        # Source Nodes: [l__mod___up2_conv_double_conv_0, l__mod___up2_conv_double_conv_1, l__mod___up2_conv_double_conv_2, l__mod___up2_conv_double_conv_3, l__mod___up2_conv_double_conv_4, x_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_18.run(buf43, arg53_1, arg113_1, arg114_1, arg54_1, arg55_1, 9789440, grid=grid(9789440), stream=stream0)
        del arg113_1
        del arg114_1
        del arg53_1
        del arg54_1
        del arg55_1
        buf44 = reinterpret_tensor(buf39, (2, 128, 320, 478), (19578880, 152960, 478, 1), 0); del buf39  # reuse
        buf47 = buf44; del buf44  # reuse
        # Source Nodes: [x1_5], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.mul, aten.rsub, aten.sub]
        triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_19.run(buf47, buf43, 39157760, grid=grid(39157760), stream=stream0)
        del buf43
        buf48 = reinterpret_tensor(buf49, (2, 128, 320, 479), (39239680, 153280, 479, 1), 19619840)  # alias
        # Source Nodes: [x1_6], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_20.run(buf47, buf48, 39239680, grid=grid(39239680), stream=stream0)
        del buf47
        del buf48
        del buf8
        # Source Nodes: [l__mod___up3_conv_double_conv_0], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, arg56_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (2, 128, 320, 479), (19619840, 153280, 479, 1))
        del arg56_1
        buf51 = buf50; del buf50  # reuse
        # Source Nodes: [l__mod___up3_conv_double_conv_0, l__mod___up3_conv_double_conv_1, l__mod___up3_conv_double_conv_2, l__mod___up3_conv_double_conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_max_pool2d_with_indices_relu_3.run(buf51, arg57_1, arg116_1, arg117_1, arg58_1, arg59_1, 39239680, grid=grid(39239680), stream=stream0)
        del arg116_1
        del arg117_1
        del arg57_1
        del arg58_1
        del arg59_1
        # Source Nodes: [l__mod___up3_conv_double_conv_0, l__mod___up3_conv_double_conv_1, l__mod___up3_conv_double_conv_2, l__mod___up3_conv_double_conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf52 = extern_kernels.convolution(buf51, arg60_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (2, 64, 320, 479), (9809920, 153280, 479, 1))
        del arg60_1
        del buf51
        buf53 = buf52; del buf52  # reuse
        # Source Nodes: [l__mod___up3_conv_double_conv_0, l__mod___up3_conv_double_conv_1, l__mod___up3_conv_double_conv_2, l__mod___up3_conv_double_conv_3, l__mod___up3_conv_double_conv_4, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_21.run(buf53, arg61_1, arg119_1, arg120_1, arg62_1, arg63_1, 19619840, grid=grid(19619840), stream=stream0)
        del arg119_1
        del arg120_1
        del arg61_1
        del arg62_1
        del arg63_1
        buf54 = reinterpret_tensor(buf49, (2, 64, 640, 958), (39239680, 613120, 958, 1), 0); del buf49  # reuse
        buf57 = buf54; del buf54  # reuse
        # Source Nodes: [x1_7], Original ATen: [aten._to_copy, aten._unsafe_index, aten.add, aten.arange, aten.mul, aten.rsub, aten.sub]
        triton_poi_fused__to_copy__unsafe_index_add_arange_mul_rsub_sub_22.run(buf57, buf53, 78479360, grid=grid(78479360), stream=stream0)
        del buf53
        buf58 = reinterpret_tensor(buf59, (2, 64, 640, 959), (78561280, 613760, 959, 1), 39280640)  # alias
        # Source Nodes: [x1_8], Original ATen: [aten.constant_pad_nd]
        triton_poi_fused_constant_pad_nd_23.run(buf57, buf58, 78561280, grid=grid(78561280), stream=stream0)
        del buf57
        del buf3
        del buf58
        # Source Nodes: [l__mod___up4_conv_double_conv_0], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, arg64_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (2, 64, 640, 959), (39280640, 613760, 959, 1))
        del arg64_1
        del buf59
        buf61 = buf60; del buf60  # reuse
        # Source Nodes: [l__mod___up4_conv_double_conv_0, l__mod___up4_conv_double_conv_1, l__mod___up4_conv_double_conv_2, l__mod___up4_conv_double_conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf61, arg65_1, arg122_1, arg123_1, arg66_1, arg67_1, 78561280, grid=grid(78561280), stream=stream0)
        del arg122_1
        del arg123_1
        del arg65_1
        del arg66_1
        del arg67_1
        # Source Nodes: [l__mod___up4_conv_double_conv_0, l__mod___up4_conv_double_conv_1, l__mod___up4_conv_double_conv_2, l__mod___up4_conv_double_conv_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf62 = extern_kernels.convolution(buf61, arg68_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (2, 64, 640, 959), (39280640, 613760, 959, 1))
        del arg68_1
        del buf61
        buf63 = buf62; del buf62  # reuse
        # Source Nodes: [l__mod___up4_conv_double_conv_0, l__mod___up4_conv_double_conv_1, l__mod___up4_conv_double_conv_2, l__mod___up4_conv_double_conv_3, l__mod___up4_conv_double_conv_4, logits, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_0.run(buf63, arg69_1, arg125_1, arg126_1, arg70_1, arg71_1, 78561280, grid=grid(78561280), stream=stream0)
        del arg125_1
        del arg126_1
        del arg69_1
        del arg70_1
        del arg71_1
        # Source Nodes: [l__mod___up4_conv_double_conv_0, l__mod___up4_conv_double_conv_1, l__mod___up4_conv_double_conv_2, l__mod___up4_conv_double_conv_3, l__mod___up4_conv_double_conv_4, logits, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf64 = extern_kernels.convolution(buf63, arg72_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (2, 2, 640, 959), (1227520, 613760, 959, 1))
        del arg72_1
        del buf63
        buf65 = buf64; del buf64  # reuse
        # Source Nodes: [l__mod___up4_conv_double_conv_0, l__mod___up4_conv_double_conv_1, l__mod___up4_conv_double_conv_2, l__mod___up4_conv_double_conv_3, l__mod___up4_conv_double_conv_4, logits, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_24.run(buf65, arg73_1, 2455040, grid=grid(2455040), stream=stream0)
        del arg73_1
        return (buf65, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((512, 1024, 3, 3), (9216, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((256, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((128, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((64, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((2, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg77_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg80_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg83_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg86_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg89_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg92_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg95_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg98_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg101_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg104_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg107_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg110_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg113_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg116_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg119_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg122_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg125_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg128_1 = rand_strided((2, 3, 640, 959), (1841280, 613760, 959, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('pytorch_unet', benchmark_compiled_module)
