
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


# kernel path: /tmp/torchinductor_youkaichao/7e/c7edajh3b7r7vld34sx5nslzqlorfl5flgti2jgo4emdfdd6vcyy.py
# Source Nodes: [l__mod___conv1_0], Original ATen: [aten.convolution]
# l__mod___conv1_0 => convolution
triton_poi_fused_convolution_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16, 65536], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12
    xnumel = 50176
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (50176*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (150528*y1)), tmp0, xmask & ymask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/rg/crgkwpletxoczwxntyy3ztnvadsvyq2czque4m2hp7ckch375fpa.py
# Source Nodes: [l__mod___conv1_0], Original ATen: [aten.convolution]
# l__mod___conv1_0 => convolution
triton_poi_fused_convolution_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 72
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 3
    y1 = (yindex // 3)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (27*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yr/cyrrln6i2jkaxbnh67nsv6mygyl52pfz5lp5tvha4hrbau3jravu.py
# Source Nodes: [l__mod___conv1_1, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___conv1_1 => add_1, mul_1, mul_2, sub
# x => relu
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1204224
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 24
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


# kernel path: /tmp/torchinductor_youkaichao/qy/cqy2b6jqcbp2yeiv5iksvug2cauctsqrhthgsdoiwadurhinmt4d.py
# Source Nodes: [l__mod___conv1_1, x, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.max_pool2d_with_indices, aten.relu]
# l__mod___conv1_1 => add_1, mul_1, mul_2, sub
# x => relu
# x_1 => max_pool2d_with_indices
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x3 = (xindex // 56)
    x2 = xindex % 56
    y4 = yindex
    x5 = xindex
    y0 = yindex % 24
    y1 = (yindex // 24)
    tmp0 = (-1) + (2*x3)
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 112, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + (2*x2)
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-113) + (2*x2) + (224*x3) + (12544*y4)), tmp10 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp12 = tl.full(tmp11.shape, float("-inf"), tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = 2*x2
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-112) + (2*x2) + (224*x3) + (12544*y4)), tmp18 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp20 = tl.full(tmp19.shape, float("-inf"), tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = triton_helpers.maximum(tmp21, tmp13)
    tmp23 = 1 + (2*x2)
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-111) + (2*x2) + (224*x3) + (12544*y4)), tmp27 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tl.full(tmp28.shape, float("-inf"), tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = triton_helpers.maximum(tmp30, tmp22)
    tmp32 = 2*x3
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-1) + (2*x2) + (224*x3) + (12544*y4)), tmp36 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp38 = tl.full(tmp37.shape, float("-inf"), tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = triton_helpers.maximum(tmp39, tmp31)
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + ((2*x2) + (224*x3) + (12544*y4)), tmp41 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp43 = tl.full(tmp42.shape, float("-inf"), tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = triton_helpers.maximum(tmp44, tmp40)
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (1 + (2*x2) + (224*x3) + (12544*y4)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.full(tmp47.shape, float("-inf"), tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = triton_helpers.maximum(tmp49, tmp45)
    tmp51 = 1 + (2*x3)
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (111 + (2*x2) + (224*x3) + (12544*y4)), tmp55 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tl.full(tmp56.shape, float("-inf"), tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = triton_helpers.maximum(tmp58, tmp50)
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (112 + (2*x2) + (224*x3) + (12544*y4)), tmp60 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp62 = tl.full(tmp61.shape, float("-inf"), tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = triton_helpers.maximum(tmp63, tmp59)
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (113 + (2*x2) + (224*x3) + (12544*y4)), tmp65 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.full(tmp66.shape, float("-inf"), tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = triton_helpers.maximum(tmp68, tmp64)
    tl.store(out_ptr0 + (y0 + (24*x5) + (75264*y1)), tmp69, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/33/c337sz226oymrdbv73q6yeo6ytx5jruzkcg7aa37gpd3ntztct25.py
# Source Nodes: [getattr_l__mod___stage2___0___branch1_1], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_l__mod___stage2___0___branch1_1 => add_3, mul_4, mul_5, sub_1
triton_poi_fused__native_batch_norm_legit_no_training_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 24
    y1 = (yindex // 24)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (24*x2) + (18816*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/37/c37tflm5sfgkalsmay3wahotnhzudriujqiur6dtgqdktjxutjya.py
# Source Nodes: [getattr_l__mod___stage2___0___branch2_1, getattr_l__mod___stage2___0___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_l__mod___stage2___0___branch2_1 => add_7, mul_10, mul_11, sub_3
# getattr_l__mod___stage2___0___branch2_2 => relu_2
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 232
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 58
    y1 = (yindex // 58)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (58*x2) + (181888*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/m5/cm5g7fxhppod5jttmfj3wy2gws24aock5lnx7a2a6lkabgjj4p6q.py
# Source Nodes: [getattr_l__mod___stage2___0___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_l__mod___stage2___0___branch2_4 => add_9, mul_13, mul_14, sub_4
triton_poi_fused__native_batch_norm_legit_no_training_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 232
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 58
    y1 = (yindex // 58)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (58*x2) + (45472*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/na/cnanqducoujl32ovu4xe5fpdfdwupqcmpu7er7mjqdfaq2k7ojaf.py
# Source Nodes: [cat_31], Original ATen: [aten.cat]
# cat_31 => cat
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 363776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 784) % 116
    x2 = (xindex // 90944)
    x3 = xindex % 90944
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 58, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (45472*x2)), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = tl.sqrt(tmp10)
    tmp12 = 1 / tmp11
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp7 * tmp14
    tmp16 = tl.load(in_ptr3 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 * tmp16
    tmp18 = tl.load(in_ptr4 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = triton_helpers.maximum(0, tmp19)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp4, tmp20, tmp21)
    tmp23 = tmp0 >= tmp3
    tmp24 = tl.full([1], 116, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tl.load(in_ptr5 + ((-45472) + x3 + (45472*x2)), tmp23 & xmask, other=0.0)
    tmp27 = tl.load(in_ptr6 + ((-58) + x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp26 - tmp27
    tmp29 = tl.load(in_ptr7 + ((-58) + x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp29 + tmp9
    tmp31 = tl.sqrt(tmp30)
    tmp32 = 1 / tmp31
    tmp33 = tmp32 * tmp13
    tmp34 = tmp28 * tmp33
    tmp35 = tl.load(in_ptr8 + ((-58) + x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp34 * tmp35
    tmp37 = tl.load(in_ptr9 + ((-58) + x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp36 + tmp37
    tmp39 = triton_helpers.maximum(0, tmp38)
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp23, tmp39, tmp40)
    tmp42 = tl.where(tmp4, tmp22, tmp41)
    tl.store(out_ptr0 + (x4), tmp42, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/os/coszc5xehcqqwmqvzz76gzf2hnwtfztbfoiulihyzilhmcfr2lwf.py
# Source Nodes: [x_3], Original ATen: [aten.clone]
# x_3 => clone
triton_poi_fused_clone_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 363776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 784
    x1 = (xindex // 784) % 2
    x2 = (xindex // 1568) % 58
    x3 = (xindex // 90944)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (784*x2) + (45472*x1) + (90944*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/42/c42qhyagybf6yetrajnsrhn6z3wvfxbumghm4pzr5bbmeobl5rj3.py
# Source Nodes: [getattr_l__mod___stage2___1___branch2_0], Original ATen: [aten.convolution]
# getattr_l__mod___stage2___1___branch2_0 => convolution_6
triton_poi_fused_convolution_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 232
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 58
    y1 = (yindex // 58)
    tmp0 = tl.load(in_ptr0 + (45472 + x2 + (784*y0) + (90944*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (58*x2) + (45472*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ih/ciheffk3la5pzmatl5ct4hcajwovq5w25khtmnfkqhinpbnhxkep.py
# Source Nodes: [getattr_l__mod___stage2___1___branch2_1, getattr_l__mod___stage2___1___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_l__mod___stage2___1___branch2_1 => add_13, mul_19, mul_20, sub_6
# getattr_l__mod___stage2___1___branch2_2 => relu_4
triton_poi_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 232
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 58
    y1 = (yindex // 58)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (58*x2) + (45472*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vy/cvydyr7lh4b2l555nl35ss6lqcvpou72upqjmhw5y2ge5seqgbby.py
# Source Nodes: [x_6], Original ATen: [aten.clone]
# x_6 => clone_1
triton_poi_fused_clone_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 363776
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = (xindex // 784) % 116
    x5 = xindex
    x3 = (xindex // 90944)
    x6 = xindex % 90944
    x0 = xindex % 784
    x1 = (xindex // 784) % 58
    x2 = (xindex // 45472) % 2
    tmp0 = x4
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 58, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x5), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 116, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-45472) + x6 + (45472*x3)), tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + ((-58) + x4), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + ((-58) + x4), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + ((-58) + x4), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + ((-58) + x4), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = triton_helpers.maximum(0, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp8, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp7, tmp28)
    tl.store(out_ptr0 + (x0 + (784*x2) + (1568*x1) + (90944*x3)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tx/ctximkli6bcvesps5munkw26ylac2jzdlaymfzoslgpbbbk6dpny.py
# Source Nodes: [getattr_l__mod___stage3___0___branch1_0, getattr_l__mod___stage3___0___branch2_0], Original ATen: [aten.convolution]
# getattr_l__mod___stage3___0___branch1_0 => convolution_15
# getattr_l__mod___stage3___0___branch2_0 => convolution_17
triton_poi_fused_convolution_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 464
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 116
    y1 = (yindex // 116)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (116*x2) + (90944*y1)), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (116*x2) + (90944*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/og/cogadbbmcycopuguwzfuzh4ylt7mguvgio4m5ayzp3mby23hnfad.py
# Source Nodes: [getattr_l__mod___stage3___0___branch1_1], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_l__mod___stage3___0___branch1_1 => add_31, mul_46, mul_47, sub_15
triton_poi_fused__native_batch_norm_legit_no_training_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 464
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 116
    y1 = (yindex // 116)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (116*x2) + (22736*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i3/ci3nrgh4uzjfureewmhmcrwe7mhqfiwdu5lim6vjgakdrxptka4w.py
# Source Nodes: [getattr_l__mod___stage3___0___branch2_1, getattr_l__mod___stage3___0___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_l__mod___stage3___0___branch2_1 => add_35, mul_52, mul_53, sub_17
# getattr_l__mod___stage3___0___branch2_2 => relu_11
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 464
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 116
    y1 = (yindex // 116)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (116*x2) + (90944*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g6/cg632ga3caxod2hpxim5brcvjrzvn5vmepgnquim6w66ckk4iwfx.py
# Source Nodes: [cat_27], Original ATen: [aten.cat]
# cat_27 => cat_4
triton_poi_fused_cat_15 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 181888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 196) % 232
    x2 = (xindex // 45472)
    x3 = xindex % 45472
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 116, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (22736*x2)), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = tl.sqrt(tmp10)
    tmp12 = 1 / tmp11
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp7 * tmp14
    tmp16 = tl.load(in_ptr3 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 * tmp16
    tmp18 = tl.load(in_ptr4 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = triton_helpers.maximum(0, tmp19)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp4, tmp20, tmp21)
    tmp23 = tmp0 >= tmp3
    tmp24 = tl.full([1], 232, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tl.load(in_ptr5 + ((-22736) + x3 + (22736*x2)), tmp23 & xmask, other=0.0)
    tmp27 = tl.load(in_ptr6 + ((-116) + x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp26 - tmp27
    tmp29 = tl.load(in_ptr7 + ((-116) + x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp29 + tmp9
    tmp31 = tl.sqrt(tmp30)
    tmp32 = 1 / tmp31
    tmp33 = tmp32 * tmp13
    tmp34 = tmp28 * tmp33
    tmp35 = tl.load(in_ptr8 + ((-116) + x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp34 * tmp35
    tmp37 = tl.load(in_ptr9 + ((-116) + x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp36 + tmp37
    tmp39 = triton_helpers.maximum(0, tmp38)
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp23, tmp39, tmp40)
    tmp42 = tl.where(tmp4, tmp22, tmp41)
    tl.store(out_ptr0 + (x4), tmp42, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v7/cv7jqknkx3yb5ezac6hfjfuq5bsom7edkahs5rblnmmzd4okgacm.py
# Source Nodes: [x_16], Original ATen: [aten.clone]
# x_16 => clone_4
triton_poi_fused_clone_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 181888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 196
    x1 = (xindex // 196) % 2
    x2 = (xindex // 392) % 116
    x3 = (xindex // 45472)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (196*x2) + (22736*x1) + (45472*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/l7/cl73rhrdnwjnpzm3bf2rj3qt6cm2grkvfytlg5midfgvus5lfi7w.py
# Source Nodes: [getattr_l__mod___stage3___1___branch2_0], Original ATen: [aten.convolution]
# getattr_l__mod___stage3___1___branch2_0 => convolution_20
triton_poi_fused_convolution_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 464
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 116
    y1 = (yindex // 116)
    tmp0 = tl.load(in_ptr0 + (22736 + x2 + (196*y0) + (45472*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (116*x2) + (22736*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q5/cq5upqaclgstxqyn522ghupp3ioilirsfhalqd5b3fdi7pqj5ry2.py
# Source Nodes: [getattr_l__mod___stage3___1___branch2_1, getattr_l__mod___stage3___1___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_l__mod___stage3___1___branch2_1 => add_41, mul_61, mul_62, sub_20
# getattr_l__mod___stage3___1___branch2_2 => relu_13
triton_poi_fused__native_batch_norm_legit_no_training_relu_18 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 464
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 116
    y1 = (yindex // 116)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (116*x2) + (22736*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s6/cs6ao77kzlel3zoo5gjtpfyvtialwsa6hsg2ptklll3am36aweap.py
# Source Nodes: [x_19], Original ATen: [aten.clone]
# x_19 => clone_5
triton_poi_fused_clone_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 181888
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = (xindex // 196) % 232
    x5 = xindex
    x3 = (xindex // 45472)
    x6 = xindex % 45472
    x0 = xindex % 196
    x1 = (xindex // 196) % 116
    x2 = (xindex // 22736) % 2
    tmp0 = x4
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 116, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x5), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 232, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-22736) + x6 + (22736*x3)), tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + ((-116) + x4), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + ((-116) + x4), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + ((-116) + x4), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + ((-116) + x4), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = triton_helpers.maximum(0, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp8, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp7, tmp28)
    tl.store(out_ptr0 + (x0 + (196*x2) + (392*x1) + (45472*x3)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/va/cvamm66y3k3ew3ulrppvjtte3xz5abjj4ph4kuw6u3gvhrnxuetg.py
# Source Nodes: [getattr_l__mod___stage4___0___branch1_0, getattr_l__mod___stage4___0___branch2_0], Original ATen: [aten.convolution]
# getattr_l__mod___stage4___0___branch1_0 => convolution_41
# getattr_l__mod___stage4___0___branch2_0 => convolution_43
triton_poi_fused_convolution_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 928
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 232
    y1 = (yindex // 232)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (232*x2) + (45472*y1)), tmp0, xmask & ymask)
    tl.store(out_ptr1 + (y0 + (232*x2) + (45472*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xe/cxem67j7p5s7i4nuxns4reu7nkt3pcjxt6kxhnutopp5oapaxxao.py
# Source Nodes: [getattr_l__mod___stage4___0___branch1_1], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_l__mod___stage4___0___branch1_1 => add_83, mul_124, mul_125, sub_41
triton_poi_fused__native_batch_norm_legit_no_training_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 928
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 232
    y1 = (yindex // 232)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (232*x2) + (11368*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3r/c3rwzluwf3gdymlyndjtfpnhkidya2xtbgirokn576mjs2rgr4vn.py
# Source Nodes: [getattr_l__mod___stage4___0___branch2_1, getattr_l__mod___stage4___0___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_l__mod___stage4___0___branch2_1 => add_87, mul_130, mul_131, sub_43
# getattr_l__mod___stage4___0___branch2_2 => relu_28
triton_poi_fused__native_batch_norm_legit_no_training_relu_22 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 928
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 232
    y1 = (yindex // 232)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (232*x2) + (45472*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hv/chvkw4dlwreiv2c5gu3s64lzxu5pcrl2e46dejoo5zfxppxzf5xy.py
# Source Nodes: [cat_19], Original ATen: [aten.cat]
# cat_19 => cat_12
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 90944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 49) % 464
    x2 = (xindex // 22736)
    x3 = xindex % 22736
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 232, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (11368*x2)), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 1e-05
    tmp10 = tmp8 + tmp9
    tmp11 = tl.sqrt(tmp10)
    tmp12 = 1 / tmp11
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp7 * tmp14
    tmp16 = tl.load(in_ptr3 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 * tmp16
    tmp18 = tl.load(in_ptr4 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = triton_helpers.maximum(0, tmp19)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp4, tmp20, tmp21)
    tmp23 = tmp0 >= tmp3
    tmp24 = tl.full([1], 464, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tl.load(in_ptr5 + ((-11368) + x3 + (11368*x2)), tmp23 & xmask, other=0.0)
    tmp27 = tl.load(in_ptr6 + ((-232) + x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp26 - tmp27
    tmp29 = tl.load(in_ptr7 + ((-232) + x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp29 + tmp9
    tmp31 = tl.sqrt(tmp30)
    tmp32 = 1 / tmp31
    tmp33 = tmp32 * tmp13
    tmp34 = tmp28 * tmp33
    tmp35 = tl.load(in_ptr8 + ((-232) + x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp34 * tmp35
    tmp37 = tl.load(in_ptr9 + ((-232) + x1), tmp23 & xmask, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp36 + tmp37
    tmp39 = triton_helpers.maximum(0, tmp38)
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp23, tmp39, tmp40)
    tmp42 = tl.where(tmp4, tmp22, tmp41)
    tl.store(out_ptr0 + (x4), tmp42, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dn/cdnbzkmbsqm7lfsbbrhygm5o745qpvqrx2wmwuqqvrqpnccssv5m.py
# Source Nodes: [x_41], Original ATen: [aten.clone]
# x_41 => clone_12
triton_poi_fused_clone_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 90944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 49
    x1 = (xindex // 49) % 2
    x2 = (xindex // 98) % 232
    x3 = (xindex // 22736)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (49*x2) + (11368*x1) + (22736*x3)), xmask)
    tl.store(out_ptr0 + (x4), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jp/cjpekwwrrneayu3gbtmz2daehcjgmxijb6pmt3hbe5ysdmdewdkk.py
# Source Nodes: [getattr_l__mod___stage4___1___branch2_0], Original ATen: [aten.convolution]
# getattr_l__mod___stage4___1___branch2_0 => convolution_46
triton_poi_fused_convolution_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 928
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 232
    y1 = (yindex // 232)
    tmp0 = tl.load(in_ptr0 + (11368 + x2 + (49*y0) + (22736*y1)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (232*x2) + (11368*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tw/ctwvy4ltmmckalln7tvlggi4xcqogmk4gmz22ah5zjgikuvxyvsj.py
# Source Nodes: [getattr_l__mod___stage4___1___branch2_1, getattr_l__mod___stage4___1___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_l__mod___stage4___1___branch2_1 => add_93, mul_139, mul_140, sub_46
# getattr_l__mod___stage4___1___branch2_2 => relu_30
triton_poi_fused__native_batch_norm_legit_no_training_relu_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 928
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 232
    y1 = (yindex // 232)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (232*x2) + (11368*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u6/cu6c5hy7tegeittwjgf4soxsbldvrjzkhlnjxnflt6p3toieosbr.py
# Source Nodes: [x_44], Original ATen: [aten.clone]
# x_44 => clone_13
triton_poi_fused_clone_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 90944
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x4 = (xindex // 49) % 464
    x5 = xindex
    x3 = (xindex // 22736)
    x6 = xindex % 22736
    x0 = xindex % 49
    x1 = (xindex // 49) % 232
    x2 = (xindex // 11368) % 2
    tmp0 = x4
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 232, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x5), tmp4 & xmask, other=0.0)
    tmp6 = tl.full(tmp5.shape, 0.0, tmp5.dtype)
    tmp7 = tl.where(tmp4, tmp5, tmp6)
    tmp8 = tmp0 >= tmp3
    tmp9 = tl.full([1], 464, tl.int64)
    tmp10 = tmp0 < tmp9
    tmp11 = tl.load(in_ptr1 + ((-11368) + x6 + (11368*x3)), tmp8 & xmask, other=0.0)
    tmp12 = tl.load(in_ptr2 + ((-232) + x4), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tmp11 - tmp12
    tmp14 = tl.load(in_ptr3 + ((-232) + x4), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.sqrt(tmp16)
    tmp18 = 1 / tmp17
    tmp19 = 1.0
    tmp20 = tmp18 * tmp19
    tmp21 = tmp13 * tmp20
    tmp22 = tl.load(in_ptr4 + ((-232) + x4), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp23 = tmp21 * tmp22
    tmp24 = tl.load(in_ptr5 + ((-232) + x4), tmp8 & xmask, eviction_policy='evict_last', other=0.0)
    tmp25 = tmp23 + tmp24
    tmp26 = triton_helpers.maximum(0, tmp25)
    tmp27 = tl.full(tmp26.shape, 0.0, tmp26.dtype)
    tmp28 = tl.where(tmp8, tmp26, tmp27)
    tmp29 = tl.where(tmp4, tmp7, tmp28)
    tl.store(out_ptr0 + (x0 + (49*x2) + (98*x1) + (22736*x3)), tmp29, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2y/c2ymh5hmuipoia3jv7lwh5yvawfg4xu3kmpmcynw46x7pdc43oup.py
# Source Nodes: [l__mod___conv5_0], Original ATen: [aten.convolution]
# l__mod___conv5_0 => convolution_55
triton_poi_fused_convolution_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1856
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 464
    y1 = (yindex // 464)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (464*x2) + (22736*y1)), tmp0, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mx/cmxgmrj2bakbgurw4rcvnrmo33smkvbfbihbjvcfhwnicfrn2o4c.py
# Source Nodes: [l__mod___conv5_1, x_53, x_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
# l__mod___conv5_1 => add_111, mul_166, mul_167, sub_55
# x_53 => relu_36
# x_54 => mean
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_29', 'mutated_arg_names': ['in_out_ptr0']}
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1 = args
    args.clear()
    assert_size_stride(arg0_1, (24, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (24, ), (1, ))
    assert_size_stride(arg2_1, (24, ), (1, ))
    assert_size_stride(arg3_1, (24, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg4_1, (24, ), (1, ))
    assert_size_stride(arg5_1, (24, ), (1, ))
    assert_size_stride(arg6_1, (58, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg7_1, (58, ), (1, ))
    assert_size_stride(arg8_1, (58, ), (1, ))
    assert_size_stride(arg9_1, (58, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg10_1, (58, ), (1, ))
    assert_size_stride(arg11_1, (58, ), (1, ))
    assert_size_stride(arg12_1, (58, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg13_1, (58, ), (1, ))
    assert_size_stride(arg14_1, (58, ), (1, ))
    assert_size_stride(arg15_1, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(arg16_1, (58, ), (1, ))
    assert_size_stride(arg17_1, (58, ), (1, ))
    assert_size_stride(arg18_1, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(arg19_1, (58, ), (1, ))
    assert_size_stride(arg20_1, (58, ), (1, ))
    assert_size_stride(arg21_1, (58, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg22_1, (58, ), (1, ))
    assert_size_stride(arg23_1, (58, ), (1, ))
    assert_size_stride(arg24_1, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(arg25_1, (58, ), (1, ))
    assert_size_stride(arg26_1, (58, ), (1, ))
    assert_size_stride(arg27_1, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(arg28_1, (58, ), (1, ))
    assert_size_stride(arg29_1, (58, ), (1, ))
    assert_size_stride(arg30_1, (58, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg31_1, (58, ), (1, ))
    assert_size_stride(arg32_1, (58, ), (1, ))
    assert_size_stride(arg33_1, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(arg34_1, (58, ), (1, ))
    assert_size_stride(arg35_1, (58, ), (1, ))
    assert_size_stride(arg36_1, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(arg37_1, (58, ), (1, ))
    assert_size_stride(arg38_1, (58, ), (1, ))
    assert_size_stride(arg39_1, (58, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg40_1, (58, ), (1, ))
    assert_size_stride(arg41_1, (58, ), (1, ))
    assert_size_stride(arg42_1, (58, 58, 1, 1), (58, 1, 1, 1))
    assert_size_stride(arg43_1, (58, ), (1, ))
    assert_size_stride(arg44_1, (58, ), (1, ))
    assert_size_stride(arg45_1, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg46_1, (116, ), (1, ))
    assert_size_stride(arg47_1, (116, ), (1, ))
    assert_size_stride(arg48_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg49_1, (116, ), (1, ))
    assert_size_stride(arg50_1, (116, ), (1, ))
    assert_size_stride(arg51_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg52_1, (116, ), (1, ))
    assert_size_stride(arg53_1, (116, ), (1, ))
    assert_size_stride(arg54_1, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg55_1, (116, ), (1, ))
    assert_size_stride(arg56_1, (116, ), (1, ))
    assert_size_stride(arg57_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg58_1, (116, ), (1, ))
    assert_size_stride(arg59_1, (116, ), (1, ))
    assert_size_stride(arg60_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg61_1, (116, ), (1, ))
    assert_size_stride(arg62_1, (116, ), (1, ))
    assert_size_stride(arg63_1, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg64_1, (116, ), (1, ))
    assert_size_stride(arg65_1, (116, ), (1, ))
    assert_size_stride(arg66_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg67_1, (116, ), (1, ))
    assert_size_stride(arg68_1, (116, ), (1, ))
    assert_size_stride(arg69_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg70_1, (116, ), (1, ))
    assert_size_stride(arg71_1, (116, ), (1, ))
    assert_size_stride(arg72_1, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg73_1, (116, ), (1, ))
    assert_size_stride(arg74_1, (116, ), (1, ))
    assert_size_stride(arg75_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg76_1, (116, ), (1, ))
    assert_size_stride(arg77_1, (116, ), (1, ))
    assert_size_stride(arg78_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg79_1, (116, ), (1, ))
    assert_size_stride(arg80_1, (116, ), (1, ))
    assert_size_stride(arg81_1, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg82_1, (116, ), (1, ))
    assert_size_stride(arg83_1, (116, ), (1, ))
    assert_size_stride(arg84_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg85_1, (116, ), (1, ))
    assert_size_stride(arg86_1, (116, ), (1, ))
    assert_size_stride(arg87_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg88_1, (116, ), (1, ))
    assert_size_stride(arg89_1, (116, ), (1, ))
    assert_size_stride(arg90_1, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg91_1, (116, ), (1, ))
    assert_size_stride(arg92_1, (116, ), (1, ))
    assert_size_stride(arg93_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg94_1, (116, ), (1, ))
    assert_size_stride(arg95_1, (116, ), (1, ))
    assert_size_stride(arg96_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg97_1, (116, ), (1, ))
    assert_size_stride(arg98_1, (116, ), (1, ))
    assert_size_stride(arg99_1, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg100_1, (116, ), (1, ))
    assert_size_stride(arg101_1, (116, ), (1, ))
    assert_size_stride(arg102_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg103_1, (116, ), (1, ))
    assert_size_stride(arg104_1, (116, ), (1, ))
    assert_size_stride(arg105_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg106_1, (116, ), (1, ))
    assert_size_stride(arg107_1, (116, ), (1, ))
    assert_size_stride(arg108_1, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg109_1, (116, ), (1, ))
    assert_size_stride(arg110_1, (116, ), (1, ))
    assert_size_stride(arg111_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg112_1, (116, ), (1, ))
    assert_size_stride(arg113_1, (116, ), (1, ))
    assert_size_stride(arg114_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg115_1, (116, ), (1, ))
    assert_size_stride(arg116_1, (116, ), (1, ))
    assert_size_stride(arg117_1, (116, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg118_1, (116, ), (1, ))
    assert_size_stride(arg119_1, (116, ), (1, ))
    assert_size_stride(arg120_1, (116, 116, 1, 1), (116, 1, 1, 1))
    assert_size_stride(arg121_1, (116, ), (1, ))
    assert_size_stride(arg122_1, (116, ), (1, ))
    assert_size_stride(arg123_1, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg124_1, (232, ), (1, ))
    assert_size_stride(arg125_1, (232, ), (1, ))
    assert_size_stride(arg126_1, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(arg127_1, (232, ), (1, ))
    assert_size_stride(arg128_1, (232, ), (1, ))
    assert_size_stride(arg129_1, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(arg130_1, (232, ), (1, ))
    assert_size_stride(arg131_1, (232, ), (1, ))
    assert_size_stride(arg132_1, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg133_1, (232, ), (1, ))
    assert_size_stride(arg134_1, (232, ), (1, ))
    assert_size_stride(arg135_1, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(arg136_1, (232, ), (1, ))
    assert_size_stride(arg137_1, (232, ), (1, ))
    assert_size_stride(arg138_1, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(arg139_1, (232, ), (1, ))
    assert_size_stride(arg140_1, (232, ), (1, ))
    assert_size_stride(arg141_1, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg142_1, (232, ), (1, ))
    assert_size_stride(arg143_1, (232, ), (1, ))
    assert_size_stride(arg144_1, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(arg145_1, (232, ), (1, ))
    assert_size_stride(arg146_1, (232, ), (1, ))
    assert_size_stride(arg147_1, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(arg148_1, (232, ), (1, ))
    assert_size_stride(arg149_1, (232, ), (1, ))
    assert_size_stride(arg150_1, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg151_1, (232, ), (1, ))
    assert_size_stride(arg152_1, (232, ), (1, ))
    assert_size_stride(arg153_1, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(arg154_1, (232, ), (1, ))
    assert_size_stride(arg155_1, (232, ), (1, ))
    assert_size_stride(arg156_1, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(arg157_1, (232, ), (1, ))
    assert_size_stride(arg158_1, (232, ), (1, ))
    assert_size_stride(arg159_1, (232, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg160_1, (232, ), (1, ))
    assert_size_stride(arg161_1, (232, ), (1, ))
    assert_size_stride(arg162_1, (232, 232, 1, 1), (232, 1, 1, 1))
    assert_size_stride(arg163_1, (232, ), (1, ))
    assert_size_stride(arg164_1, (232, ), (1, ))
    assert_size_stride(arg165_1, (1024, 464, 1, 1), (464, 1, 1, 1))
    assert_size_stride(arg166_1, (1024, ), (1, ))
    assert_size_stride(arg167_1, (1024, ), (1, ))
    assert_size_stride(arg168_1, (1000, 1024), (1024, 1))
    assert_size_stride(arg169_1, (1000, ), (1, ))
    assert_size_stride(arg170_1, (24, ), (1, ))
    assert_size_stride(arg171_1, (24, ), (1, ))
    assert_size_stride(arg172_1, (), ())
    assert_size_stride(arg173_1, (24, ), (1, ))
    assert_size_stride(arg174_1, (24, ), (1, ))
    assert_size_stride(arg175_1, (), ())
    assert_size_stride(arg176_1, (58, ), (1, ))
    assert_size_stride(arg177_1, (58, ), (1, ))
    assert_size_stride(arg178_1, (), ())
    assert_size_stride(arg179_1, (58, ), (1, ))
    assert_size_stride(arg180_1, (58, ), (1, ))
    assert_size_stride(arg181_1, (), ())
    assert_size_stride(arg182_1, (58, ), (1, ))
    assert_size_stride(arg183_1, (58, ), (1, ))
    assert_size_stride(arg184_1, (), ())
    assert_size_stride(arg185_1, (58, ), (1, ))
    assert_size_stride(arg186_1, (58, ), (1, ))
    assert_size_stride(arg187_1, (), ())
    assert_size_stride(arg188_1, (58, ), (1, ))
    assert_size_stride(arg189_1, (58, ), (1, ))
    assert_size_stride(arg190_1, (), ())
    assert_size_stride(arg191_1, (58, ), (1, ))
    assert_size_stride(arg192_1, (58, ), (1, ))
    assert_size_stride(arg193_1, (), ())
    assert_size_stride(arg194_1, (58, ), (1, ))
    assert_size_stride(arg195_1, (58, ), (1, ))
    assert_size_stride(arg196_1, (), ())
    assert_size_stride(arg197_1, (58, ), (1, ))
    assert_size_stride(arg198_1, (58, ), (1, ))
    assert_size_stride(arg199_1, (), ())
    assert_size_stride(arg200_1, (58, ), (1, ))
    assert_size_stride(arg201_1, (58, ), (1, ))
    assert_size_stride(arg202_1, (), ())
    assert_size_stride(arg203_1, (58, ), (1, ))
    assert_size_stride(arg204_1, (58, ), (1, ))
    assert_size_stride(arg205_1, (), ())
    assert_size_stride(arg206_1, (58, ), (1, ))
    assert_size_stride(arg207_1, (58, ), (1, ))
    assert_size_stride(arg208_1, (), ())
    assert_size_stride(arg209_1, (58, ), (1, ))
    assert_size_stride(arg210_1, (58, ), (1, ))
    assert_size_stride(arg211_1, (), ())
    assert_size_stride(arg212_1, (58, ), (1, ))
    assert_size_stride(arg213_1, (58, ), (1, ))
    assert_size_stride(arg214_1, (), ())
    assert_size_stride(arg215_1, (116, ), (1, ))
    assert_size_stride(arg216_1, (116, ), (1, ))
    assert_size_stride(arg217_1, (), ())
    assert_size_stride(arg218_1, (116, ), (1, ))
    assert_size_stride(arg219_1, (116, ), (1, ))
    assert_size_stride(arg220_1, (), ())
    assert_size_stride(arg221_1, (116, ), (1, ))
    assert_size_stride(arg222_1, (116, ), (1, ))
    assert_size_stride(arg223_1, (), ())
    assert_size_stride(arg224_1, (116, ), (1, ))
    assert_size_stride(arg225_1, (116, ), (1, ))
    assert_size_stride(arg226_1, (), ())
    assert_size_stride(arg227_1, (116, ), (1, ))
    assert_size_stride(arg228_1, (116, ), (1, ))
    assert_size_stride(arg229_1, (), ())
    assert_size_stride(arg230_1, (116, ), (1, ))
    assert_size_stride(arg231_1, (116, ), (1, ))
    assert_size_stride(arg232_1, (), ())
    assert_size_stride(arg233_1, (116, ), (1, ))
    assert_size_stride(arg234_1, (116, ), (1, ))
    assert_size_stride(arg235_1, (), ())
    assert_size_stride(arg236_1, (116, ), (1, ))
    assert_size_stride(arg237_1, (116, ), (1, ))
    assert_size_stride(arg238_1, (), ())
    assert_size_stride(arg239_1, (116, ), (1, ))
    assert_size_stride(arg240_1, (116, ), (1, ))
    assert_size_stride(arg241_1, (), ())
    assert_size_stride(arg242_1, (116, ), (1, ))
    assert_size_stride(arg243_1, (116, ), (1, ))
    assert_size_stride(arg244_1, (), ())
    assert_size_stride(arg245_1, (116, ), (1, ))
    assert_size_stride(arg246_1, (116, ), (1, ))
    assert_size_stride(arg247_1, (), ())
    assert_size_stride(arg248_1, (116, ), (1, ))
    assert_size_stride(arg249_1, (116, ), (1, ))
    assert_size_stride(arg250_1, (), ())
    assert_size_stride(arg251_1, (116, ), (1, ))
    assert_size_stride(arg252_1, (116, ), (1, ))
    assert_size_stride(arg253_1, (), ())
    assert_size_stride(arg254_1, (116, ), (1, ))
    assert_size_stride(arg255_1, (116, ), (1, ))
    assert_size_stride(arg256_1, (), ())
    assert_size_stride(arg257_1, (116, ), (1, ))
    assert_size_stride(arg258_1, (116, ), (1, ))
    assert_size_stride(arg259_1, (), ())
    assert_size_stride(arg260_1, (116, ), (1, ))
    assert_size_stride(arg261_1, (116, ), (1, ))
    assert_size_stride(arg262_1, (), ())
    assert_size_stride(arg263_1, (116, ), (1, ))
    assert_size_stride(arg264_1, (116, ), (1, ))
    assert_size_stride(arg265_1, (), ())
    assert_size_stride(arg266_1, (116, ), (1, ))
    assert_size_stride(arg267_1, (116, ), (1, ))
    assert_size_stride(arg268_1, (), ())
    assert_size_stride(arg269_1, (116, ), (1, ))
    assert_size_stride(arg270_1, (116, ), (1, ))
    assert_size_stride(arg271_1, (), ())
    assert_size_stride(arg272_1, (116, ), (1, ))
    assert_size_stride(arg273_1, (116, ), (1, ))
    assert_size_stride(arg274_1, (), ())
    assert_size_stride(arg275_1, (116, ), (1, ))
    assert_size_stride(arg276_1, (116, ), (1, ))
    assert_size_stride(arg277_1, (), ())
    assert_size_stride(arg278_1, (116, ), (1, ))
    assert_size_stride(arg279_1, (116, ), (1, ))
    assert_size_stride(arg280_1, (), ())
    assert_size_stride(arg281_1, (116, ), (1, ))
    assert_size_stride(arg282_1, (116, ), (1, ))
    assert_size_stride(arg283_1, (), ())
    assert_size_stride(arg284_1, (116, ), (1, ))
    assert_size_stride(arg285_1, (116, ), (1, ))
    assert_size_stride(arg286_1, (), ())
    assert_size_stride(arg287_1, (116, ), (1, ))
    assert_size_stride(arg288_1, (116, ), (1, ))
    assert_size_stride(arg289_1, (), ())
    assert_size_stride(arg290_1, (116, ), (1, ))
    assert_size_stride(arg291_1, (116, ), (1, ))
    assert_size_stride(arg292_1, (), ())
    assert_size_stride(arg293_1, (232, ), (1, ))
    assert_size_stride(arg294_1, (232, ), (1, ))
    assert_size_stride(arg295_1, (), ())
    assert_size_stride(arg296_1, (232, ), (1, ))
    assert_size_stride(arg297_1, (232, ), (1, ))
    assert_size_stride(arg298_1, (), ())
    assert_size_stride(arg299_1, (232, ), (1, ))
    assert_size_stride(arg300_1, (232, ), (1, ))
    assert_size_stride(arg301_1, (), ())
    assert_size_stride(arg302_1, (232, ), (1, ))
    assert_size_stride(arg303_1, (232, ), (1, ))
    assert_size_stride(arg304_1, (), ())
    assert_size_stride(arg305_1, (232, ), (1, ))
    assert_size_stride(arg306_1, (232, ), (1, ))
    assert_size_stride(arg307_1, (), ())
    assert_size_stride(arg308_1, (232, ), (1, ))
    assert_size_stride(arg309_1, (232, ), (1, ))
    assert_size_stride(arg310_1, (), ())
    assert_size_stride(arg311_1, (232, ), (1, ))
    assert_size_stride(arg312_1, (232, ), (1, ))
    assert_size_stride(arg313_1, (), ())
    assert_size_stride(arg314_1, (232, ), (1, ))
    assert_size_stride(arg315_1, (232, ), (1, ))
    assert_size_stride(arg316_1, (), ())
    assert_size_stride(arg317_1, (232, ), (1, ))
    assert_size_stride(arg318_1, (232, ), (1, ))
    assert_size_stride(arg319_1, (), ())
    assert_size_stride(arg320_1, (232, ), (1, ))
    assert_size_stride(arg321_1, (232, ), (1, ))
    assert_size_stride(arg322_1, (), ())
    assert_size_stride(arg323_1, (232, ), (1, ))
    assert_size_stride(arg324_1, (232, ), (1, ))
    assert_size_stride(arg325_1, (), ())
    assert_size_stride(arg326_1, (232, ), (1, ))
    assert_size_stride(arg327_1, (232, ), (1, ))
    assert_size_stride(arg328_1, (), ())
    assert_size_stride(arg329_1, (232, ), (1, ))
    assert_size_stride(arg330_1, (232, ), (1, ))
    assert_size_stride(arg331_1, (), ())
    assert_size_stride(arg332_1, (232, ), (1, ))
    assert_size_stride(arg333_1, (232, ), (1, ))
    assert_size_stride(arg334_1, (), ())
    assert_size_stride(arg335_1, (1024, ), (1, ))
    assert_size_stride(arg336_1, (1024, ), (1, ))
    assert_size_stride(arg337_1, (), ())
    assert_size_stride(arg338_1, (4, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___conv1_0], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg338_1, buf0, 12, 50176, grid=grid(12, 50176), stream=stream0)
        del arg338_1
        buf1 = empty_strided((24, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___conv1_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 72, 9, grid=grid(72, 9), stream=stream0)
        del arg0_1
        # Source Nodes: [l__mod___conv1_0], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 24, 112, 112), (301056, 12544, 112, 1))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Source Nodes: [l__mod___conv1_1, x], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf3, arg170_1, arg171_1, arg1_1, arg2_1, 1204224, grid=grid(1204224), stream=stream0)
        del arg170_1
        del arg171_1
        del arg1_1
        del arg2_1
        buf4 = empty_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___conv1_1, x, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_3.run(buf3, buf4, 96, 3136, grid=grid(96, 3136), stream=stream0)
        del buf3
        # Source Nodes: [getattr_l__mod___stage2___0___branch1_0], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, arg3_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=24, bias=None)
        assert_size_stride(buf5, (4, 24, 28, 28), (18816, 784, 28, 1))
        del arg3_1
        buf6 = empty_strided((4, 24, 28, 28), (18816, 1, 672, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage2___0___branch1_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_4.run(buf5, arg173_1, arg174_1, arg4_1, arg5_1, buf6, 96, 784, grid=grid(96, 784), stream=stream0)
        del arg173_1
        del arg174_1
        del arg4_1
        del arg5_1
        del buf5
        # Source Nodes: [getattr_l__mod___stage2___0___branch1_1, getattr_l__mod___stage2___0___branch1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf7 = extern_kernels.convolution(buf6, arg6_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 58, 28, 28), (45472, 784, 28, 1))
        del arg6_1
        del buf6
        # Source Nodes: [getattr_l__mod___stage2___0___branch2_0], Original ATen: [aten.convolution]
        buf8 = extern_kernels.convolution(buf4, arg9_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 58, 56, 56), (181888, 3136, 56, 1))
        del arg9_1
        del buf4
        buf9 = empty_strided((4, 58, 56, 56), (181888, 1, 3248, 58), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage2___0___branch2_1, getattr_l__mod___stage2___0___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf8, arg179_1, arg180_1, arg10_1, arg11_1, buf9, 232, 3136, grid=grid(232, 3136), stream=stream0)
        del arg10_1
        del arg11_1
        del arg179_1
        del arg180_1
        del buf8
        # Source Nodes: [getattr_l__mod___stage2___0___branch2_1, getattr_l__mod___stage2___0___branch2_2, getattr_l__mod___stage2___0___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf10 = extern_kernels.convolution(buf9, arg12_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=58, bias=None)
        assert_size_stride(buf10, (4, 58, 28, 28), (45472, 784, 28, 1))
        del arg12_1
        del buf9
        buf11 = empty_strided((4, 58, 28, 28), (45472, 1, 1624, 58), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage2___0___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_6.run(buf10, arg182_1, arg183_1, arg13_1, arg14_1, buf11, 232, 784, grid=grid(232, 784), stream=stream0)
        del arg13_1
        del arg14_1
        del arg182_1
        del arg183_1
        del buf10
        # Source Nodes: [getattr_l__mod___stage2___0___branch2_4, getattr_l__mod___stage2___0___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf12 = extern_kernels.convolution(buf11, arg15_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 58, 28, 28), (45472, 784, 28, 1))
        del arg15_1
        del buf11
        buf13 = empty((4, 116, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_31], Original ATen: [aten.cat]
        triton_poi_fused_cat_7.run(buf7, arg176_1, arg177_1, arg7_1, arg8_1, buf12, arg185_1, arg186_1, arg16_1, arg17_1, buf13, 363776, grid=grid(363776), stream=stream0)
        del arg16_1
        del arg176_1
        del arg177_1
        del arg17_1
        del arg185_1
        del arg186_1
        del arg7_1
        del arg8_1
        del buf12
        buf14 = empty((4, 58, 2, 28, 28), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_3], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf13, buf14, 363776, grid=grid(363776), stream=stream0)
        buf15 = reinterpret_tensor(buf7, (4, 58, 28, 28), (45472, 1, 1624, 58), 0); del buf7  # reuse
        # Source Nodes: [getattr_l__mod___stage2___1___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(buf14, buf15, 232, 784, grid=grid(232, 784), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage2___1___branch2_0], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, arg18_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 58, 28, 28), (45472, 784, 28, 1))
        del arg18_1
        buf17 = buf15; del buf15  # reuse
        # Source Nodes: [getattr_l__mod___stage2___1___branch2_1, getattr_l__mod___stage2___1___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf16, arg188_1, arg189_1, arg19_1, arg20_1, buf17, 232, 784, grid=grid(232, 784), stream=stream0)
        del arg188_1
        del arg189_1
        del arg19_1
        del arg20_1
        del buf16
        # Source Nodes: [getattr_l__mod___stage2___1___branch2_1, getattr_l__mod___stage2___1___branch2_2, getattr_l__mod___stage2___1___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf18 = extern_kernels.convolution(buf17, arg21_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=58, bias=None)
        assert_size_stride(buf18, (4, 58, 28, 28), (45472, 784, 28, 1))
        del arg21_1
        buf19 = buf17; del buf17  # reuse
        # Source Nodes: [getattr_l__mod___stage2___1___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_6.run(buf18, arg191_1, arg192_1, arg22_1, arg23_1, buf19, 232, 784, grid=grid(232, 784), stream=stream0)
        del arg191_1
        del arg192_1
        del arg22_1
        del arg23_1
        del buf18
        # Source Nodes: [getattr_l__mod___stage2___1___branch2_4, getattr_l__mod___stage2___1___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf20 = extern_kernels.convolution(buf19, arg24_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 58, 28, 28), (45472, 784, 28, 1))
        del arg24_1
        del buf19
        buf21 = reinterpret_tensor(buf13, (4, 58, 2, 28, 28), (90944, 1568, 784, 28, 1), 0); del buf13  # reuse
        # Source Nodes: [x_6], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf14, buf20, arg194_1, arg195_1, arg25_1, arg26_1, buf21, 363776, grid=grid(363776), stream=stream0)
        del arg194_1
        del arg195_1
        del arg25_1
        del arg26_1
        buf22 = reinterpret_tensor(buf20, (4, 58, 28, 28), (45472, 1, 1624, 58), 0); del buf20  # reuse
        # Source Nodes: [getattr_l__mod___stage2___2___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(buf21, buf22, 232, 784, grid=grid(232, 784), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage2___2___branch2_0], Original ATen: [aten.convolution]
        buf23 = extern_kernels.convolution(buf22, arg27_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf23, (4, 58, 28, 28), (45472, 784, 28, 1))
        del arg27_1
        buf24 = buf22; del buf22  # reuse
        # Source Nodes: [getattr_l__mod___stage2___2___branch2_1, getattr_l__mod___stage2___2___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf23, arg197_1, arg198_1, arg28_1, arg29_1, buf24, 232, 784, grid=grid(232, 784), stream=stream0)
        del arg197_1
        del arg198_1
        del arg28_1
        del arg29_1
        del buf23
        # Source Nodes: [getattr_l__mod___stage2___2___branch2_1, getattr_l__mod___stage2___2___branch2_2, getattr_l__mod___stage2___2___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf25 = extern_kernels.convolution(buf24, arg30_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=58, bias=None)
        assert_size_stride(buf25, (4, 58, 28, 28), (45472, 784, 28, 1))
        del arg30_1
        buf26 = buf24; del buf24  # reuse
        # Source Nodes: [getattr_l__mod___stage2___2___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_6.run(buf25, arg200_1, arg201_1, arg31_1, arg32_1, buf26, 232, 784, grid=grid(232, 784), stream=stream0)
        del arg200_1
        del arg201_1
        del arg31_1
        del arg32_1
        del buf25
        # Source Nodes: [getattr_l__mod___stage2___2___branch2_4, getattr_l__mod___stage2___2___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf27 = extern_kernels.convolution(buf26, arg33_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 58, 28, 28), (45472, 784, 28, 1))
        del arg33_1
        del buf26
        buf28 = buf14; del buf14  # reuse
        # Source Nodes: [x_9], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf21, buf27, arg203_1, arg204_1, arg34_1, arg35_1, buf28, 363776, grid=grid(363776), stream=stream0)
        del arg203_1
        del arg204_1
        del arg34_1
        del arg35_1
        buf29 = reinterpret_tensor(buf27, (4, 58, 28, 28), (45472, 1, 1624, 58), 0); del buf27  # reuse
        # Source Nodes: [getattr_l__mod___stage2___3___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_9.run(buf28, buf29, 232, 784, grid=grid(232, 784), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage2___3___branch2_0], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, arg36_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 58, 28, 28), (45472, 784, 28, 1))
        del arg36_1
        buf31 = buf29; del buf29  # reuse
        # Source Nodes: [getattr_l__mod___stage2___3___branch2_1, getattr_l__mod___stage2___3___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf30, arg206_1, arg207_1, arg37_1, arg38_1, buf31, 232, 784, grid=grid(232, 784), stream=stream0)
        del arg206_1
        del arg207_1
        del arg37_1
        del arg38_1
        del buf30
        # Source Nodes: [getattr_l__mod___stage2___3___branch2_1, getattr_l__mod___stage2___3___branch2_2, getattr_l__mod___stage2___3___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf32 = extern_kernels.convolution(buf31, arg39_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=58, bias=None)
        assert_size_stride(buf32, (4, 58, 28, 28), (45472, 784, 28, 1))
        del arg39_1
        buf33 = buf31; del buf31  # reuse
        # Source Nodes: [getattr_l__mod___stage2___3___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_6.run(buf32, arg209_1, arg210_1, arg40_1, arg41_1, buf33, 232, 784, grid=grid(232, 784), stream=stream0)
        del arg209_1
        del arg210_1
        del arg40_1
        del arg41_1
        # Source Nodes: [getattr_l__mod___stage2___3___branch2_4, getattr_l__mod___stage2___3___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf34 = extern_kernels.convolution(buf33, arg42_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 58, 28, 28), (45472, 784, 28, 1))
        del arg42_1
        buf35 = buf21; del buf21  # reuse
        # Source Nodes: [x_12], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf28, buf34, arg212_1, arg213_1, arg43_1, arg44_1, buf35, 363776, grid=grid(363776), stream=stream0)
        del arg212_1
        del arg213_1
        del arg43_1
        del arg44_1
        buf36 = reinterpret_tensor(buf28, (4, 116, 28, 28), (90944, 1, 3248, 116), 0); del buf28  # reuse
        buf40 = empty_strided((4, 116, 28, 28), (90944, 1, 3248, 116), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___0___branch1_0, getattr_l__mod___stage3___0___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_12.run(buf35, buf36, buf40, 464, 784, grid=grid(464, 784), stream=stream0)
        del buf35
        # Source Nodes: [getattr_l__mod___stage3___0___branch1_0], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf36, arg45_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
        assert_size_stride(buf37, (4, 116, 14, 14), (22736, 196, 14, 1))
        del arg45_1
        del buf36
        buf38 = empty_strided((4, 116, 14, 14), (22736, 1, 1624, 116), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage3___0___branch1_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_13.run(buf37, arg215_1, arg216_1, arg46_1, arg47_1, buf38, 464, 196, grid=grid(464, 196), stream=stream0)
        del arg215_1
        del arg216_1
        del arg46_1
        del arg47_1
        del buf37
        # Source Nodes: [getattr_l__mod___stage3___0___branch1_1, getattr_l__mod___stage3___0___branch1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf39 = extern_kernels.convolution(buf38, arg48_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 116, 14, 14), (22736, 196, 14, 1))
        del arg48_1
        # Source Nodes: [getattr_l__mod___stage3___0___branch2_0], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf40, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 116, 28, 28), (90944, 784, 28, 1))
        del arg51_1
        buf42 = buf40; del buf40  # reuse
        # Source Nodes: [getattr_l__mod___stage3___0___branch2_1, getattr_l__mod___stage3___0___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf41, arg221_1, arg222_1, arg52_1, arg53_1, buf42, 464, 784, grid=grid(464, 784), stream=stream0)
        del arg221_1
        del arg222_1
        del arg52_1
        del arg53_1
        del buf41
        # Source Nodes: [getattr_l__mod___stage3___0___branch2_1, getattr_l__mod___stage3___0___branch2_2, getattr_l__mod___stage3___0___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf43 = extern_kernels.convolution(buf42, arg54_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
        assert_size_stride(buf43, (4, 116, 14, 14), (22736, 196, 14, 1))
        del arg54_1
        del buf42
        buf44 = buf38; del buf38  # reuse
        # Source Nodes: [getattr_l__mod___stage3___0___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_13.run(buf43, arg224_1, arg225_1, arg55_1, arg56_1, buf44, 464, 196, grid=grid(464, 196), stream=stream0)
        del arg224_1
        del arg225_1
        del arg55_1
        del arg56_1
        del buf43
        # Source Nodes: [getattr_l__mod___stage3___0___branch2_4, getattr_l__mod___stage3___0___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf45 = extern_kernels.convolution(buf44, arg57_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (4, 116, 14, 14), (22736, 196, 14, 1))
        del arg57_1
        del buf44
        buf46 = reinterpret_tensor(buf34, (4, 232, 14, 14), (45472, 196, 14, 1), 0); del buf34  # reuse
        # Source Nodes: [cat_27], Original ATen: [aten.cat]
        triton_poi_fused_cat_15.run(buf39, arg218_1, arg219_1, arg49_1, arg50_1, buf45, arg227_1, arg228_1, arg58_1, arg59_1, buf46, 181888, grid=grid(181888), stream=stream0)
        del arg218_1
        del arg219_1
        del arg227_1
        del arg228_1
        del arg49_1
        del arg50_1
        del arg58_1
        del arg59_1
        del buf39
        buf47 = reinterpret_tensor(buf33, (4, 116, 2, 14, 14), (45472, 392, 196, 14, 1), 0); del buf33  # reuse
        # Source Nodes: [x_16], Original ATen: [aten.clone]
        triton_poi_fused_clone_16.run(buf46, buf47, 181888, grid=grid(181888), stream=stream0)
        buf48 = reinterpret_tensor(buf45, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf45  # reuse
        # Source Nodes: [getattr_l__mod___stage3___1___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf47, buf48, 464, 196, grid=grid(464, 196), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage3___1___branch2_0], Original ATen: [aten.convolution]
        buf49 = extern_kernels.convolution(buf48, arg60_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 116, 14, 14), (22736, 196, 14, 1))
        del arg60_1
        buf50 = buf48; del buf48  # reuse
        # Source Nodes: [getattr_l__mod___stage3___1___branch2_1, getattr_l__mod___stage3___1___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf49, arg230_1, arg231_1, arg61_1, arg62_1, buf50, 464, 196, grid=grid(464, 196), stream=stream0)
        del arg230_1
        del arg231_1
        del arg61_1
        del arg62_1
        del buf49
        # Source Nodes: [getattr_l__mod___stage3___1___branch2_1, getattr_l__mod___stage3___1___branch2_2, getattr_l__mod___stage3___1___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf51 = extern_kernels.convolution(buf50, arg63_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
        assert_size_stride(buf51, (4, 116, 14, 14), (22736, 196, 14, 1))
        del arg63_1
        buf52 = buf50; del buf50  # reuse
        # Source Nodes: [getattr_l__mod___stage3___1___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_13.run(buf51, arg233_1, arg234_1, arg64_1, arg65_1, buf52, 464, 196, grid=grid(464, 196), stream=stream0)
        del arg233_1
        del arg234_1
        del arg64_1
        del arg65_1
        del buf51
        # Source Nodes: [getattr_l__mod___stage3___1___branch2_4, getattr_l__mod___stage3___1___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf53 = extern_kernels.convolution(buf52, arg66_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 116, 14, 14), (22736, 196, 14, 1))
        del arg66_1
        del buf52
        buf54 = reinterpret_tensor(buf46, (4, 116, 2, 14, 14), (45472, 392, 196, 14, 1), 0); del buf46  # reuse
        # Source Nodes: [x_19], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf47, buf53, arg236_1, arg237_1, arg67_1, arg68_1, buf54, 181888, grid=grid(181888), stream=stream0)
        del arg236_1
        del arg237_1
        del arg67_1
        del arg68_1
        buf55 = reinterpret_tensor(buf53, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf53  # reuse
        # Source Nodes: [getattr_l__mod___stage3___2___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf54, buf55, 464, 196, grid=grid(464, 196), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage3___2___branch2_0], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, arg69_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 116, 14, 14), (22736, 196, 14, 1))
        del arg69_1
        buf57 = buf55; del buf55  # reuse
        # Source Nodes: [getattr_l__mod___stage3___2___branch2_1, getattr_l__mod___stage3___2___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf56, arg239_1, arg240_1, arg70_1, arg71_1, buf57, 464, 196, grid=grid(464, 196), stream=stream0)
        del arg239_1
        del arg240_1
        del arg70_1
        del arg71_1
        del buf56
        # Source Nodes: [getattr_l__mod___stage3___2___branch2_1, getattr_l__mod___stage3___2___branch2_2, getattr_l__mod___stage3___2___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf58 = extern_kernels.convolution(buf57, arg72_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
        assert_size_stride(buf58, (4, 116, 14, 14), (22736, 196, 14, 1))
        del arg72_1
        buf59 = buf57; del buf57  # reuse
        # Source Nodes: [getattr_l__mod___stage3___2___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_13.run(buf58, arg242_1, arg243_1, arg73_1, arg74_1, buf59, 464, 196, grid=grid(464, 196), stream=stream0)
        del arg242_1
        del arg243_1
        del arg73_1
        del arg74_1
        del buf58
        # Source Nodes: [getattr_l__mod___stage3___2___branch2_4, getattr_l__mod___stage3___2___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf60 = extern_kernels.convolution(buf59, arg75_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 116, 14, 14), (22736, 196, 14, 1))
        del arg75_1
        del buf59
        buf61 = buf47; del buf47  # reuse
        # Source Nodes: [x_22], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf54, buf60, arg245_1, arg246_1, arg76_1, arg77_1, buf61, 181888, grid=grid(181888), stream=stream0)
        del arg245_1
        del arg246_1
        del arg76_1
        del arg77_1
        buf62 = reinterpret_tensor(buf60, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf60  # reuse
        # Source Nodes: [getattr_l__mod___stage3___3___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf61, buf62, 464, 196, grid=grid(464, 196), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage3___3___branch2_0], Original ATen: [aten.convolution]
        buf63 = extern_kernels.convolution(buf62, arg78_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (4, 116, 14, 14), (22736, 196, 14, 1))
        del arg78_1
        buf64 = buf62; del buf62  # reuse
        # Source Nodes: [getattr_l__mod___stage3___3___branch2_1, getattr_l__mod___stage3___3___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf63, arg248_1, arg249_1, arg79_1, arg80_1, buf64, 464, 196, grid=grid(464, 196), stream=stream0)
        del arg248_1
        del arg249_1
        del arg79_1
        del arg80_1
        del buf63
        # Source Nodes: [getattr_l__mod___stage3___3___branch2_1, getattr_l__mod___stage3___3___branch2_2, getattr_l__mod___stage3___3___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf65 = extern_kernels.convolution(buf64, arg81_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
        assert_size_stride(buf65, (4, 116, 14, 14), (22736, 196, 14, 1))
        del arg81_1
        buf66 = buf64; del buf64  # reuse
        # Source Nodes: [getattr_l__mod___stage3___3___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_13.run(buf65, arg251_1, arg252_1, arg82_1, arg83_1, buf66, 464, 196, grid=grid(464, 196), stream=stream0)
        del arg251_1
        del arg252_1
        del arg82_1
        del arg83_1
        del buf65
        # Source Nodes: [getattr_l__mod___stage3___3___branch2_4, getattr_l__mod___stage3___3___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf67 = extern_kernels.convolution(buf66, arg84_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf67, (4, 116, 14, 14), (22736, 196, 14, 1))
        del arg84_1
        del buf66
        buf68 = buf54; del buf54  # reuse
        # Source Nodes: [x_25], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf61, buf67, arg254_1, arg255_1, arg85_1, arg86_1, buf68, 181888, grid=grid(181888), stream=stream0)
        del arg254_1
        del arg255_1
        del arg85_1
        del arg86_1
        buf69 = reinterpret_tensor(buf67, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf67  # reuse
        # Source Nodes: [getattr_l__mod___stage3___4___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf68, buf69, 464, 196, grid=grid(464, 196), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage3___4___branch2_0], Original ATen: [aten.convolution]
        buf70 = extern_kernels.convolution(buf69, arg87_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf70, (4, 116, 14, 14), (22736, 196, 14, 1))
        del arg87_1
        buf71 = buf69; del buf69  # reuse
        # Source Nodes: [getattr_l__mod___stage3___4___branch2_1, getattr_l__mod___stage3___4___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf70, arg257_1, arg258_1, arg88_1, arg89_1, buf71, 464, 196, grid=grid(464, 196), stream=stream0)
        del arg257_1
        del arg258_1
        del arg88_1
        del arg89_1
        del buf70
        # Source Nodes: [getattr_l__mod___stage3___4___branch2_1, getattr_l__mod___stage3___4___branch2_2, getattr_l__mod___stage3___4___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf72 = extern_kernels.convolution(buf71, arg90_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
        assert_size_stride(buf72, (4, 116, 14, 14), (22736, 196, 14, 1))
        del arg90_1
        buf73 = buf71; del buf71  # reuse
        # Source Nodes: [getattr_l__mod___stage3___4___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_13.run(buf72, arg260_1, arg261_1, arg91_1, arg92_1, buf73, 464, 196, grid=grid(464, 196), stream=stream0)
        del arg260_1
        del arg261_1
        del arg91_1
        del arg92_1
        del buf72
        # Source Nodes: [getattr_l__mod___stage3___4___branch2_4, getattr_l__mod___stage3___4___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf74 = extern_kernels.convolution(buf73, arg93_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 116, 14, 14), (22736, 196, 14, 1))
        del arg93_1
        del buf73
        buf75 = buf61; del buf61  # reuse
        # Source Nodes: [x_28], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf68, buf74, arg263_1, arg264_1, arg94_1, arg95_1, buf75, 181888, grid=grid(181888), stream=stream0)
        del arg263_1
        del arg264_1
        del arg94_1
        del arg95_1
        buf76 = reinterpret_tensor(buf74, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf74  # reuse
        # Source Nodes: [getattr_l__mod___stage3___5___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf75, buf76, 464, 196, grid=grid(464, 196), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage3___5___branch2_0], Original ATen: [aten.convolution]
        buf77 = extern_kernels.convolution(buf76, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (4, 116, 14, 14), (22736, 196, 14, 1))
        del arg96_1
        buf78 = buf76; del buf76  # reuse
        # Source Nodes: [getattr_l__mod___stage3___5___branch2_1, getattr_l__mod___stage3___5___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf77, arg266_1, arg267_1, arg97_1, arg98_1, buf78, 464, 196, grid=grid(464, 196), stream=stream0)
        del arg266_1
        del arg267_1
        del arg97_1
        del arg98_1
        del buf77
        # Source Nodes: [getattr_l__mod___stage3___5___branch2_1, getattr_l__mod___stage3___5___branch2_2, getattr_l__mod___stage3___5___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf79 = extern_kernels.convolution(buf78, arg99_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
        assert_size_stride(buf79, (4, 116, 14, 14), (22736, 196, 14, 1))
        del arg99_1
        buf80 = buf78; del buf78  # reuse
        # Source Nodes: [getattr_l__mod___stage3___5___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_13.run(buf79, arg269_1, arg270_1, arg100_1, arg101_1, buf80, 464, 196, grid=grid(464, 196), stream=stream0)
        del arg100_1
        del arg101_1
        del arg269_1
        del arg270_1
        del buf79
        # Source Nodes: [getattr_l__mod___stage3___5___branch2_4, getattr_l__mod___stage3___5___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf81 = extern_kernels.convolution(buf80, arg102_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (4, 116, 14, 14), (22736, 196, 14, 1))
        del arg102_1
        del buf80
        buf82 = buf68; del buf68  # reuse
        # Source Nodes: [x_31], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf75, buf81, arg272_1, arg273_1, arg103_1, arg104_1, buf82, 181888, grid=grid(181888), stream=stream0)
        del arg103_1
        del arg104_1
        del arg272_1
        del arg273_1
        buf83 = reinterpret_tensor(buf81, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf81  # reuse
        # Source Nodes: [getattr_l__mod___stage3___6___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf82, buf83, 464, 196, grid=grid(464, 196), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage3___6___branch2_0], Original ATen: [aten.convolution]
        buf84 = extern_kernels.convolution(buf83, arg105_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 116, 14, 14), (22736, 196, 14, 1))
        del arg105_1
        buf85 = buf83; del buf83  # reuse
        # Source Nodes: [getattr_l__mod___stage3___6___branch2_1, getattr_l__mod___stage3___6___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf84, arg275_1, arg276_1, arg106_1, arg107_1, buf85, 464, 196, grid=grid(464, 196), stream=stream0)
        del arg106_1
        del arg107_1
        del arg275_1
        del arg276_1
        del buf84
        # Source Nodes: [getattr_l__mod___stage3___6___branch2_1, getattr_l__mod___stage3___6___branch2_2, getattr_l__mod___stage3___6___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf86 = extern_kernels.convolution(buf85, arg108_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
        assert_size_stride(buf86, (4, 116, 14, 14), (22736, 196, 14, 1))
        del arg108_1
        buf87 = buf85; del buf85  # reuse
        # Source Nodes: [getattr_l__mod___stage3___6___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_13.run(buf86, arg278_1, arg279_1, arg109_1, arg110_1, buf87, 464, 196, grid=grid(464, 196), stream=stream0)
        del arg109_1
        del arg110_1
        del arg278_1
        del arg279_1
        del buf86
        # Source Nodes: [getattr_l__mod___stage3___6___branch2_4, getattr_l__mod___stage3___6___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf88 = extern_kernels.convolution(buf87, arg111_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (4, 116, 14, 14), (22736, 196, 14, 1))
        del arg111_1
        del buf87
        buf89 = buf75; del buf75  # reuse
        # Source Nodes: [x_34], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf82, buf88, arg281_1, arg282_1, arg112_1, arg113_1, buf89, 181888, grid=grid(181888), stream=stream0)
        del arg112_1
        del arg113_1
        del arg281_1
        del arg282_1
        buf90 = reinterpret_tensor(buf88, (4, 116, 14, 14), (22736, 1, 1624, 116), 0); del buf88  # reuse
        # Source Nodes: [getattr_l__mod___stage3___7___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_17.run(buf89, buf90, 464, 196, grid=grid(464, 196), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage3___7___branch2_0], Original ATen: [aten.convolution]
        buf91 = extern_kernels.convolution(buf90, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf91, (4, 116, 14, 14), (22736, 196, 14, 1))
        del arg114_1
        buf92 = buf90; del buf90  # reuse
        # Source Nodes: [getattr_l__mod___stage3___7___branch2_1, getattr_l__mod___stage3___7___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_18.run(buf91, arg284_1, arg285_1, arg115_1, arg116_1, buf92, 464, 196, grid=grid(464, 196), stream=stream0)
        del arg115_1
        del arg116_1
        del arg284_1
        del arg285_1
        del buf91
        # Source Nodes: [getattr_l__mod___stage3___7___branch2_1, getattr_l__mod___stage3___7___branch2_2, getattr_l__mod___stage3___7___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf93 = extern_kernels.convolution(buf92, arg117_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=116, bias=None)
        assert_size_stride(buf93, (4, 116, 14, 14), (22736, 196, 14, 1))
        del arg117_1
        buf94 = buf92; del buf92  # reuse
        # Source Nodes: [getattr_l__mod___stage3___7___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_13.run(buf93, arg287_1, arg288_1, arg118_1, arg119_1, buf94, 464, 196, grid=grid(464, 196), stream=stream0)
        del arg118_1
        del arg119_1
        del arg287_1
        del arg288_1
        del buf93
        # Source Nodes: [getattr_l__mod___stage3___7___branch2_4, getattr_l__mod___stage3___7___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf95 = extern_kernels.convolution(buf94, arg120_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (4, 116, 14, 14), (22736, 196, 14, 1))
        del arg120_1
        buf96 = buf82; del buf82  # reuse
        # Source Nodes: [x_37], Original ATen: [aten.clone]
        triton_poi_fused_clone_19.run(buf89, buf95, arg290_1, arg291_1, arg121_1, arg122_1, buf96, 181888, grid=grid(181888), stream=stream0)
        del arg121_1
        del arg122_1
        del arg290_1
        del arg291_1
        buf97 = reinterpret_tensor(buf89, (4, 232, 14, 14), (45472, 1, 3248, 232), 0); del buf89  # reuse
        buf101 = reinterpret_tensor(buf32, (4, 232, 14, 14), (45472, 1, 3248, 232), 0); del buf32  # reuse
        # Source Nodes: [getattr_l__mod___stage4___0___branch1_0, getattr_l__mod___stage4___0___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_20.run(buf96, buf97, buf101, 928, 196, grid=grid(928, 196), stream=stream0)
        del buf96
        # Source Nodes: [getattr_l__mod___stage4___0___branch1_0], Original ATen: [aten.convolution]
        buf98 = extern_kernels.convolution(buf97, arg123_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=232, bias=None)
        assert_size_stride(buf98, (4, 232, 7, 7), (11368, 49, 7, 1))
        del arg123_1
        del buf97
        buf99 = empty_strided((4, 232, 7, 7), (11368, 1, 1624, 232), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___stage4___0___branch1_1], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_21.run(buf98, arg293_1, arg294_1, arg124_1, arg125_1, buf99, 928, 49, grid=grid(928, 49), stream=stream0)
        del arg124_1
        del arg125_1
        del arg293_1
        del arg294_1
        del buf98
        # Source Nodes: [getattr_l__mod___stage4___0___branch1_1, getattr_l__mod___stage4___0___branch1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf100 = extern_kernels.convolution(buf99, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (4, 232, 7, 7), (11368, 49, 7, 1))
        del arg126_1
        # Source Nodes: [getattr_l__mod___stage4___0___branch2_0], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, arg129_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (4, 232, 14, 14), (45472, 196, 14, 1))
        del arg129_1
        buf103 = buf101; del buf101  # reuse
        # Source Nodes: [getattr_l__mod___stage4___0___branch2_1, getattr_l__mod___stage4___0___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_22.run(buf102, arg299_1, arg300_1, arg130_1, arg131_1, buf103, 928, 196, grid=grid(928, 196), stream=stream0)
        del arg130_1
        del arg131_1
        del arg299_1
        del arg300_1
        del buf102
        # Source Nodes: [getattr_l__mod___stage4___0___branch2_1, getattr_l__mod___stage4___0___branch2_2, getattr_l__mod___stage4___0___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf104 = extern_kernels.convolution(buf103, arg132_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=232, bias=None)
        assert_size_stride(buf104, (4, 232, 7, 7), (11368, 49, 7, 1))
        del arg132_1
        del buf103
        buf105 = buf99; del buf99  # reuse
        # Source Nodes: [getattr_l__mod___stage4___0___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_21.run(buf104, arg302_1, arg303_1, arg133_1, arg134_1, buf105, 928, 49, grid=grid(928, 49), stream=stream0)
        del arg133_1
        del arg134_1
        del arg302_1
        del arg303_1
        del buf104
        # Source Nodes: [getattr_l__mod___stage4___0___branch2_4, getattr_l__mod___stage4___0___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf106 = extern_kernels.convolution(buf105, arg135_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (4, 232, 7, 7), (11368, 49, 7, 1))
        del arg135_1
        del buf105
        buf107 = reinterpret_tensor(buf95, (4, 464, 7, 7), (22736, 49, 7, 1), 0); del buf95  # reuse
        # Source Nodes: [cat_19], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf100, arg296_1, arg297_1, arg127_1, arg128_1, buf106, arg305_1, arg306_1, arg136_1, arg137_1, buf107, 90944, grid=grid(90944), stream=stream0)
        del arg127_1
        del arg128_1
        del arg136_1
        del arg137_1
        del arg296_1
        del arg297_1
        del arg305_1
        del arg306_1
        del buf100
        buf108 = reinterpret_tensor(buf94, (4, 232, 2, 7, 7), (22736, 98, 49, 7, 1), 0); del buf94  # reuse
        # Source Nodes: [x_41], Original ATen: [aten.clone]
        triton_poi_fused_clone_24.run(buf107, buf108, 90944, grid=grid(90944), stream=stream0)
        buf109 = reinterpret_tensor(buf106, (4, 232, 7, 7), (11368, 1, 1624, 232), 0); del buf106  # reuse
        # Source Nodes: [getattr_l__mod___stage4___1___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf108, buf109, 928, 49, grid=grid(928, 49), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage4___1___branch2_0], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf109, arg138_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (4, 232, 7, 7), (11368, 49, 7, 1))
        del arg138_1
        buf111 = buf109; del buf109  # reuse
        # Source Nodes: [getattr_l__mod___stage4___1___branch2_1, getattr_l__mod___stage4___1___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf110, arg308_1, arg309_1, arg139_1, arg140_1, buf111, 928, 49, grid=grid(928, 49), stream=stream0)
        del arg139_1
        del arg140_1
        del arg308_1
        del arg309_1
        del buf110
        # Source Nodes: [getattr_l__mod___stage4___1___branch2_1, getattr_l__mod___stage4___1___branch2_2, getattr_l__mod___stage4___1___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf112 = extern_kernels.convolution(buf111, arg141_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=232, bias=None)
        assert_size_stride(buf112, (4, 232, 7, 7), (11368, 49, 7, 1))
        del arg141_1
        buf113 = buf111; del buf111  # reuse
        # Source Nodes: [getattr_l__mod___stage4___1___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_21.run(buf112, arg311_1, arg312_1, arg142_1, arg143_1, buf113, 928, 49, grid=grid(928, 49), stream=stream0)
        del arg142_1
        del arg143_1
        del arg311_1
        del arg312_1
        del buf112
        # Source Nodes: [getattr_l__mod___stage4___1___branch2_4, getattr_l__mod___stage4___1___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf114 = extern_kernels.convolution(buf113, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf114, (4, 232, 7, 7), (11368, 49, 7, 1))
        del arg144_1
        del buf113
        buf115 = reinterpret_tensor(buf107, (4, 232, 2, 7, 7), (22736, 98, 49, 7, 1), 0); del buf107  # reuse
        # Source Nodes: [x_44], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf108, buf114, arg314_1, arg315_1, arg145_1, arg146_1, buf115, 90944, grid=grid(90944), stream=stream0)
        del arg145_1
        del arg146_1
        del arg314_1
        del arg315_1
        buf116 = reinterpret_tensor(buf114, (4, 232, 7, 7), (11368, 1, 1624, 232), 0); del buf114  # reuse
        # Source Nodes: [getattr_l__mod___stage4___2___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf115, buf116, 928, 49, grid=grid(928, 49), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage4___2___branch2_0], Original ATen: [aten.convolution]
        buf117 = extern_kernels.convolution(buf116, arg147_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf117, (4, 232, 7, 7), (11368, 49, 7, 1))
        del arg147_1
        buf118 = buf116; del buf116  # reuse
        # Source Nodes: [getattr_l__mod___stage4___2___branch2_1, getattr_l__mod___stage4___2___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf117, arg317_1, arg318_1, arg148_1, arg149_1, buf118, 928, 49, grid=grid(928, 49), stream=stream0)
        del arg148_1
        del arg149_1
        del arg317_1
        del arg318_1
        del buf117
        # Source Nodes: [getattr_l__mod___stage4___2___branch2_1, getattr_l__mod___stage4___2___branch2_2, getattr_l__mod___stage4___2___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf119 = extern_kernels.convolution(buf118, arg150_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=232, bias=None)
        assert_size_stride(buf119, (4, 232, 7, 7), (11368, 49, 7, 1))
        del arg150_1
        buf120 = buf118; del buf118  # reuse
        # Source Nodes: [getattr_l__mod___stage4___2___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_21.run(buf119, arg320_1, arg321_1, arg151_1, arg152_1, buf120, 928, 49, grid=grid(928, 49), stream=stream0)
        del arg151_1
        del arg152_1
        del arg320_1
        del arg321_1
        del buf119
        # Source Nodes: [getattr_l__mod___stage4___2___branch2_4, getattr_l__mod___stage4___2___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf121 = extern_kernels.convolution(buf120, arg153_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf121, (4, 232, 7, 7), (11368, 49, 7, 1))
        del arg153_1
        del buf120
        buf122 = buf108; del buf108  # reuse
        # Source Nodes: [x_47], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf115, buf121, arg323_1, arg324_1, arg154_1, arg155_1, buf122, 90944, grid=grid(90944), stream=stream0)
        del arg154_1
        del arg155_1
        del arg323_1
        del arg324_1
        buf123 = reinterpret_tensor(buf121, (4, 232, 7, 7), (11368, 1, 1624, 232), 0); del buf121  # reuse
        # Source Nodes: [getattr_l__mod___stage4___3___branch2_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_25.run(buf122, buf123, 928, 49, grid=grid(928, 49), stream=stream0)
        # Source Nodes: [getattr_l__mod___stage4___3___branch2_0], Original ATen: [aten.convolution]
        buf124 = extern_kernels.convolution(buf123, arg156_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (4, 232, 7, 7), (11368, 49, 7, 1))
        del arg156_1
        buf125 = buf123; del buf123  # reuse
        # Source Nodes: [getattr_l__mod___stage4___3___branch2_1, getattr_l__mod___stage4___3___branch2_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf124, arg326_1, arg327_1, arg157_1, arg158_1, buf125, 928, 49, grid=grid(928, 49), stream=stream0)
        del arg157_1
        del arg158_1
        del arg326_1
        del arg327_1
        del buf124
        # Source Nodes: [getattr_l__mod___stage4___3___branch2_1, getattr_l__mod___stage4___3___branch2_2, getattr_l__mod___stage4___3___branch2_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf126 = extern_kernels.convolution(buf125, arg159_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=232, bias=None)
        assert_size_stride(buf126, (4, 232, 7, 7), (11368, 49, 7, 1))
        del arg159_1
        buf127 = buf125; del buf125  # reuse
        # Source Nodes: [getattr_l__mod___stage4___3___branch2_4], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_21.run(buf126, arg329_1, arg330_1, arg160_1, arg161_1, buf127, 928, 49, grid=grid(928, 49), stream=stream0)
        del arg160_1
        del arg161_1
        del arg329_1
        del arg330_1
        del buf126
        # Source Nodes: [getattr_l__mod___stage4___3___branch2_4, getattr_l__mod___stage4___3___branch2_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf128 = extern_kernels.convolution(buf127, arg162_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 232, 7, 7), (11368, 49, 7, 1))
        del arg162_1
        del buf127
        buf129 = buf115; del buf115  # reuse
        # Source Nodes: [x_50], Original ATen: [aten.clone]
        triton_poi_fused_clone_27.run(buf122, buf128, arg332_1, arg333_1, arg163_1, arg164_1, buf129, 90944, grid=grid(90944), stream=stream0)
        del arg163_1
        del arg164_1
        del arg332_1
        del arg333_1
        del buf128
        buf130 = reinterpret_tensor(buf122, (4, 464, 7, 7), (22736, 1, 3248, 464), 0); del buf122  # reuse
        # Source Nodes: [l__mod___conv5_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_28.run(buf129, buf130, 1856, 49, grid=grid(1856, 49), stream=stream0)
        del buf129
        # Source Nodes: [l__mod___conv5_0], Original ATen: [aten.convolution]
        buf131 = extern_kernels.convolution(buf130, arg165_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf131, (4, 1024, 7, 7), (50176, 49, 7, 1))
        del arg165_1
        del buf130
        buf132 = empty((4, 1024), device='cuda', dtype=torch.float32)
        buf133 = buf132; del buf132  # reuse
        # Source Nodes: [l__mod___conv5_1, x_53, x_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_29.run(buf133, buf131, arg335_1, arg336_1, arg166_1, arg167_1, 4096, 49, grid=grid(4096), stream=stream0)
        del arg166_1
        del arg167_1
        del arg335_1
        del arg336_1
        del buf131
        buf134 = empty((4, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___conv5_1, x_53, x_54, x_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.addmm, aten.mean, aten.relu]
        extern_kernels.addmm(arg169_1, buf133, reinterpret_tensor(arg168_1, (1024, 1000), (1, 1024), 0), alpha=1, beta=1, out=buf134)
        del arg168_1
        del arg169_1
        return (buf134, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((24, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((24, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((58, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((58, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((58, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((58, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((58, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((58, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((58, 58, 1, 1), (58, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((116, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((116, 116, 1, 1), (116, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((232, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((232, 232, 1, 1), (232, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((1024, 464, 1, 1), (464, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((1000, 1024), (1024, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg173_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg176_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg179_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg182_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg185_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg188_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg191_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg194_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg197_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg200_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg203_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg206_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg209_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg212_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((58, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg215_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg218_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg221_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg224_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg227_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg230_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg233_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg236_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg239_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg242_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg245_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg248_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg251_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg254_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg257_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg260_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg263_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg266_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg269_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg272_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg275_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg278_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg281_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg284_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg287_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg290_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((116, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg293_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg296_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg299_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg302_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg305_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg308_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg311_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg314_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg317_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg320_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg323_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg326_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg329_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg332_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((232, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg335_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg338_1 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('shufflenet_v2_x1_0', benchmark_compiled_module)
