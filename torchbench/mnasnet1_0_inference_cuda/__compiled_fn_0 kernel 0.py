
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
# Source Nodes: [l__mod___layers_0], Original ATen: [aten.convolution]
# l__mod___layers_0 => convolution
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


# kernel path: /tmp/torchinductor_youkaichao/ww/cwwbxnrzrqop4klqko3lxrugnst7kip4kb4m4tqgyf3ww2kcgb2l.py
# Source Nodes: [l__mod___layers_0], Original ATen: [aten.convolution]
# l__mod___layers_0 => convolution
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
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


# kernel path: /tmp/torchinductor_youkaichao/aw/cawcrdgbwl5g55lhv2ttqfdhj7eiedzm6d6bmnmrak2ffd5nhrxm.py
# Source Nodes: [l__mod___layers_1, l__mod___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# l__mod___layers_1 => add_1, mul_1, mul_2, sub
# l__mod___layers_2 => relu
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (32*x2) + (401408*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rk/crkd4myctoqe7hqrvhzbbyxvvmkoykma5zaaxyyijspz4i4rdwrg.py
# Source Nodes: [l__mod___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
# l__mod___layers_7 => add_5, mul_7, mul_8, sub_2
triton_poi_fused__native_batch_norm_legit_no_training_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 16
    y1 = (yindex // 16)
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (16*x2) + (200704*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uv/cuvwrsvmqsydkzfjewyu7y5xohbxlhjfn3paap3jwcpuzwkg35ef.py
# Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_1, getattr_getattr_l__mod___layers___8_____0___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___layers___8_____0___layers_1 => add_7, mul_10, mul_11, sub_3
# getattr_getattr_l__mod___layers___8_____0___layers_2 => relu_2
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 48
    y1 = (yindex // 48)
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (48*x2) + (602112*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5a/c5ahruiu2vflekpajhtposbmy3ow4wd5x67k6d4arrsg5bwcp6lw.py
# Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_4, getattr_getattr_l__mod___layers___8_____0___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___layers___8_____0___layers_4 => add_9, mul_13, mul_14, sub_4
# getattr_getattr_l__mod___layers___8_____0___layers_5 => relu_3
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 48
    y1 = (yindex // 48)
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
    tl.store(out_ptr0 + (y0 + (48*x2) + (150528*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2f/c2feu3wc5zvnrdqzylrstleoij6fnrkgscdwolns4xfidsxq3pgz.py
# Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_getattr_l__mod___layers___8_____0___layers_7 => add_11, mul_16, mul_17, sub_5
triton_poi_fused__native_batch_norm_legit_no_training_6 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 96
    xnumel = 3136
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
    tl.store(out_ptr0 + (y0 + (24*x2) + (75264*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y2/cy2ygo5pte4it3b5ak3ykjeb5q3cnt27xhvt5kcbmpo2mcw3lodi.py
# Source Nodes: [getattr_getattr_l__mod___layers___8_____1___layers_1, getattr_getattr_l__mod___layers___8_____1___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___layers___8_____1___layers_1 => add_13, mul_19, mul_20, sub_6
# getattr_getattr_l__mod___layers___8_____1___layers_2 => relu_4
triton_poi_fused__native_batch_norm_legit_no_training_relu_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 288
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 72
    y1 = (yindex // 72)
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
    tl.store(out_ptr0 + (y0 + (72*x2) + (225792*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kh/ckh7u64d75o4mco4dvbgajw7mk4pdwv4ek7xhps5r6u5hxoqq2dd.py
# Source Nodes: [add, getattr_getattr_l__mod___layers___8_____1___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# add => add_18
# getattr_getattr_l__mod___layers___8_____1___layers_7 => add_17, mul_25, mul_26, sub_8
triton_poi_fused__native_batch_norm_legit_no_training_add_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12544
    xnumel = 24
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 3136
    y1 = (yindex // 3136)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (3136*x2) + (75264*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (24*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (24*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gu/cgu5q7g56w7foh7nos2agjfq5stbrba545ae7chjw3px7rejie6f.py
# Source Nodes: [getattr_getattr_l__mod___layers___9_____0___layers_4, getattr_getattr_l__mod___layers___9_____0___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___layers___9_____0___layers_4 => add_29, mul_40, mul_41, sub_13
# getattr_getattr_l__mod___layers___9_____0___layers_5 => relu_9
triton_poi_fused__native_batch_norm_legit_no_training_relu_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 288
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 72
    y1 = (yindex // 72)
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
    tl.store(out_ptr0 + (y0 + (72*x2) + (56448*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6d/c6dlvcx3yzvlggrweu3vdga5fqa4ilrnur453mum55l723k4o2cl.py
# Source Nodes: [getattr_getattr_l__mod___layers___9_____0___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_getattr_l__mod___layers___9_____0___layers_7 => add_31, mul_43, mul_44, sub_14
triton_poi_fused__native_batch_norm_legit_no_training_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 160
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 40
    y1 = (yindex // 40)
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
    tl.store(out_ptr0 + (y0 + (40*x2) + (31360*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bj/cbjutzsztgj6im36er3urxa3eh7asoxy5fhv4iwgcjafvaosnlrx.py
# Source Nodes: [getattr_getattr_l__mod___layers___9_____1___layers_1, getattr_getattr_l__mod___layers___9_____1___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___layers___9_____1___layers_1 => add_33, mul_46, mul_47, sub_15
# getattr_getattr_l__mod___layers___9_____1___layers_2 => relu_10
triton_poi_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 480
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 120
    y1 = (yindex // 120)
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
    tl.store(out_ptr0 + (y0 + (120*x2) + (94080*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/en/cenfom4kz2pv45zjvu2ofoimmgc45tfyfs6a3ucp5rp7qm5e7yvh.py
# Source Nodes: [add_2, getattr_getattr_l__mod___layers___9_____1___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# add_2 => add_38
# getattr_getattr_l__mod___layers___9_____1___layers_7 => add_37, mul_52, mul_53, sub_17
triton_poi_fused__native_batch_norm_legit_no_training_add_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3136
    xnumel = 40
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 784
    y1 = (yindex // 784)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (784*x2) + (31360*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (40*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (40*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/tz/ctzne23neco3c4suva6qkcxx7rlnkag2dotlufmih6jglj35e2zz.py
# Source Nodes: [getattr_getattr_l__mod___layers___10_____0___layers_1, getattr_getattr_l__mod___layers___10_____0___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___layers___10_____0___layers_1 => add_47, mul_64, mul_65, sub_21
# getattr_getattr_l__mod___layers___10_____0___layers_2 => relu_14
triton_poi_fused__native_batch_norm_legit_no_training_relu_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 240
    y1 = (yindex // 240)
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
    tl.store(out_ptr0 + (y0 + (240*x2) + (188160*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ki/ckico2yrip33d5covchswsmdhe27sqse6cgfoyvdudjmxxl4d3j5.py
# Source Nodes: [getattr_getattr_l__mod___layers___10_____0___layers_4, getattr_getattr_l__mod___layers___10_____0___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___layers___10_____0___layers_4 => add_49, mul_67, mul_68, sub_22
# getattr_getattr_l__mod___layers___10_____0___layers_5 => relu_15
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 960
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 240
    y1 = (yindex // 240)
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
    tl.store(out_ptr0 + (y0 + (240*x2) + (47040*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/k4/ck4wncboxbmu4yn7e45ay5xz2edywc5kdkyv35in4zopkaypnruw.py
# Source Nodes: [getattr_getattr_l__mod___layers___10_____0___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_getattr_l__mod___layers___10_____0___layers_7 => add_51, mul_70, mul_71, sub_23
triton_poi_fused__native_batch_norm_legit_no_training_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 320
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 80
    y1 = (yindex // 80)
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
    tl.store(out_ptr0 + (y0 + (80*x2) + (15680*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nc/cncmhc26vj33powbsom2cuzwxp247jslael4zuqn4glya5xdf7dt.py
# Source Nodes: [getattr_getattr_l__mod___layers___10_____1___layers_1, getattr_getattr_l__mod___layers___10_____1___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___layers___10_____1___layers_1 => add_53, mul_73, mul_74, sub_24
# getattr_getattr_l__mod___layers___10_____1___layers_2 => relu_16
triton_poi_fused__native_batch_norm_legit_no_training_relu_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 480
    y1 = (yindex // 480)
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
    tl.store(out_ptr0 + (y0 + (480*x2) + (94080*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2z/c2zkr2747ux32pwvsgqk4tsppi6jdugmsdssahcg6qmowjhtoh4p.py
# Source Nodes: [add_4, getattr_getattr_l__mod___layers___10_____1___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# add_4 => add_58
# getattr_getattr_l__mod___layers___10_____1___layers_7 => add_57, mul_79, mul_80, sub_26
triton_poi_fused__native_batch_norm_legit_no_training_add_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_17', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
    xnumel = 80
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (15680*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (80*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (80*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ih/cih3sqeyovqhsl4cqrtqe65ofjj6i7vzl4bgck6haxij534ojul2.py
# Source Nodes: [getattr_getattr_l__mod___layers___11_____0___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_getattr_l__mod___layers___11_____0___layers_7 => add_71, mul_97, mul_98, sub_32
triton_poi_fused__native_batch_norm_legit_no_training_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 96
    y1 = (yindex // 96)
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
    tl.store(out_ptr0 + (y0 + (96*x2) + (18816*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jq/cjq2bxmk4b7qnxm44se2dfijtadr6r5stisqddmzblbkbcewfivf.py
# Source Nodes: [getattr_getattr_l__mod___layers___11_____1___layers_1, getattr_getattr_l__mod___layers___11_____1___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___layers___11_____1___layers_1 => add_73, mul_100, mul_101, sub_33
# getattr_getattr_l__mod___layers___11_____1___layers_2 => relu_22
triton_poi_fused__native_batch_norm_legit_no_training_relu_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2304
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 576
    y1 = (yindex // 576)
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
    tl.store(out_ptr0 + (y0 + (576*x2) + (112896*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/or/corjplfotkqnois5mzye5o6ijw2g4xvn245taiavt5ct2cop67fs.py
# Source Nodes: [add_6, getattr_getattr_l__mod___layers___11_____1___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# add_6 => add_78
# getattr_getattr_l__mod___layers___11_____1___layers_7 => add_77, mul_106, mul_107, sub_35
triton_poi_fused__native_batch_norm_legit_no_training_add_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
    xnumel = 96
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 196
    y1 = (yindex // 196)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (18816*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (96*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (96*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xo/cxomsjy72sk2hp64lltxogwy3r7hyuetpgbdnerg3x73oojspzsf.py
# Source Nodes: [getattr_getattr_l__mod___layers___12_____0___layers_4, getattr_getattr_l__mod___layers___12_____0___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___layers___12_____0___layers_4 => add_82, mul_112, mul_113, sub_37
# getattr_getattr_l__mod___layers___12_____0___layers_5 => relu_25
triton_poi_fused__native_batch_norm_legit_no_training_relu_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2304
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 576
    y1 = (yindex // 576)
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
    tl.store(out_ptr0 + (y0 + (576*x2) + (28224*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uc/cucwnnkbn7fph6hc3irv3yoyivenxditt5ikuqmg2yfbueuu75n6.py
# Source Nodes: [getattr_getattr_l__mod___layers___12_____0___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_getattr_l__mod___layers___12_____0___layers_7 => add_84, mul_115, mul_116, sub_38
triton_poi_fused__native_batch_norm_legit_no_training_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 192
    y1 = (yindex // 192)
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
    tl.store(out_ptr0 + (y0 + (192*x2) + (9408*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ls/clsslzl2xwecqkjmxpaojbkzdgjax6qffu7ukjljqso7x3mc7mtd.py
# Source Nodes: [getattr_getattr_l__mod___layers___12_____1___layers_1, getattr_getattr_l__mod___layers___12_____1___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_getattr_l__mod___layers___12_____1___layers_1 => add_86, mul_118, mul_119, sub_39
# getattr_getattr_l__mod___layers___12_____1___layers_2 => relu_26
triton_poi_fused__native_batch_norm_legit_no_training_relu_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4608
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1152
    y1 = (yindex // 1152)
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
    tl.store(out_ptr0 + (y0 + (1152*x2) + (56448*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kv/ckvkzmw3qoi5i6nnrhjc2myc3rcgc5moplgwj2umwj2bgf2owvcu.py
# Source Nodes: [add_7, getattr_getattr_l__mod___layers___12_____1___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# add_7 => add_91
# getattr_getattr_l__mod___layers___12_____1___layers_7 => add_90, mul_124, mul_125, sub_41
triton_poi_fused__native_batch_norm_legit_no_training_add_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 196
    xnumel = 192
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 49
    y1 = (yindex // 49)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (9408*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (192*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (192*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/i7/ci7vne3lzzyrm3i4i2aa2ffht4bqjoutoztizlgwq52czqvoxaoq.py
# Source Nodes: [getattr_getattr_l__mod___layers___13_____0___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
# getattr_getattr_l__mod___layers___13_____0___layers_7 => add_111, mul_151, mul_152, sub_50
triton_poi_fused__native_batch_norm_legit_no_training_25 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 320
    y1 = (yindex // 320)
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
    tl.store(out_ptr0 + (y0 + (320*x2) + (15680*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/z7/cz7cth2uo32pz6lkcd4ccnzr75hktgltw3yf5pi7iecblm4kl4n7.py
# Source Nodes: [l__mod___layers_15, x, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
# l__mod___layers_15 => add_113, mul_154, mul_155, sub_51
# x => relu_34
# x_1 => mean
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_26 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_26', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5120
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1280
    tmp0 = tl.load(in_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x0), xmask, eviction_policy='evict_last')
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
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = tl.sum(tmp18, 1)[:, None]
    tmp20 = 49.0
    tmp21 = tmp19 / tmp20
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp21, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (32, ), (1, ))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg4_1, (32, ), (1, ))
    assert_size_stride(arg5_1, (32, ), (1, ))
    assert_size_stride(arg6_1, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg7_1, (16, ), (1, ))
    assert_size_stride(arg8_1, (16, ), (1, ))
    assert_size_stride(arg9_1, (48, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg10_1, (48, ), (1, ))
    assert_size_stride(arg11_1, (48, ), (1, ))
    assert_size_stride(arg12_1, (48, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg13_1, (48, ), (1, ))
    assert_size_stride(arg14_1, (48, ), (1, ))
    assert_size_stride(arg15_1, (24, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg16_1, (24, ), (1, ))
    assert_size_stride(arg17_1, (24, ), (1, ))
    assert_size_stride(arg18_1, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg19_1, (72, ), (1, ))
    assert_size_stride(arg20_1, (72, ), (1, ))
    assert_size_stride(arg21_1, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg22_1, (72, ), (1, ))
    assert_size_stride(arg23_1, (72, ), (1, ))
    assert_size_stride(arg24_1, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg25_1, (24, ), (1, ))
    assert_size_stride(arg26_1, (24, ), (1, ))
    assert_size_stride(arg27_1, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg28_1, (72, ), (1, ))
    assert_size_stride(arg29_1, (72, ), (1, ))
    assert_size_stride(arg30_1, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg31_1, (72, ), (1, ))
    assert_size_stride(arg32_1, (72, ), (1, ))
    assert_size_stride(arg33_1, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg34_1, (24, ), (1, ))
    assert_size_stride(arg35_1, (24, ), (1, ))
    assert_size_stride(arg36_1, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg37_1, (72, ), (1, ))
    assert_size_stride(arg38_1, (72, ), (1, ))
    assert_size_stride(arg39_1, (72, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg40_1, (72, ), (1, ))
    assert_size_stride(arg41_1, (72, ), (1, ))
    assert_size_stride(arg42_1, (40, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg43_1, (40, ), (1, ))
    assert_size_stride(arg44_1, (40, ), (1, ))
    assert_size_stride(arg45_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg46_1, (120, ), (1, ))
    assert_size_stride(arg47_1, (120, ), (1, ))
    assert_size_stride(arg48_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg49_1, (120, ), (1, ))
    assert_size_stride(arg50_1, (120, ), (1, ))
    assert_size_stride(arg51_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg52_1, (40, ), (1, ))
    assert_size_stride(arg53_1, (40, ), (1, ))
    assert_size_stride(arg54_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg55_1, (120, ), (1, ))
    assert_size_stride(arg56_1, (120, ), (1, ))
    assert_size_stride(arg57_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg58_1, (120, ), (1, ))
    assert_size_stride(arg59_1, (120, ), (1, ))
    assert_size_stride(arg60_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg61_1, (40, ), (1, ))
    assert_size_stride(arg62_1, (40, ), (1, ))
    assert_size_stride(arg63_1, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg64_1, (240, ), (1, ))
    assert_size_stride(arg65_1, (240, ), (1, ))
    assert_size_stride(arg66_1, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg67_1, (240, ), (1, ))
    assert_size_stride(arg68_1, (240, ), (1, ))
    assert_size_stride(arg69_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg70_1, (80, ), (1, ))
    assert_size_stride(arg71_1, (80, ), (1, ))
    assert_size_stride(arg72_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg73_1, (480, ), (1, ))
    assert_size_stride(arg74_1, (480, ), (1, ))
    assert_size_stride(arg75_1, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg76_1, (480, ), (1, ))
    assert_size_stride(arg77_1, (480, ), (1, ))
    assert_size_stride(arg78_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg79_1, (80, ), (1, ))
    assert_size_stride(arg80_1, (80, ), (1, ))
    assert_size_stride(arg81_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg82_1, (480, ), (1, ))
    assert_size_stride(arg83_1, (480, ), (1, ))
    assert_size_stride(arg84_1, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg85_1, (480, ), (1, ))
    assert_size_stride(arg86_1, (480, ), (1, ))
    assert_size_stride(arg87_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg88_1, (80, ), (1, ))
    assert_size_stride(arg89_1, (80, ), (1, ))
    assert_size_stride(arg90_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg91_1, (480, ), (1, ))
    assert_size_stride(arg92_1, (480, ), (1, ))
    assert_size_stride(arg93_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg94_1, (480, ), (1, ))
    assert_size_stride(arg95_1, (480, ), (1, ))
    assert_size_stride(arg96_1, (96, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg97_1, (96, ), (1, ))
    assert_size_stride(arg98_1, (96, ), (1, ))
    assert_size_stride(arg99_1, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg100_1, (576, ), (1, ))
    assert_size_stride(arg101_1, (576, ), (1, ))
    assert_size_stride(arg102_1, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg103_1, (576, ), (1, ))
    assert_size_stride(arg104_1, (576, ), (1, ))
    assert_size_stride(arg105_1, (96, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(arg106_1, (96, ), (1, ))
    assert_size_stride(arg107_1, (96, ), (1, ))
    assert_size_stride(arg108_1, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg109_1, (576, ), (1, ))
    assert_size_stride(arg110_1, (576, ), (1, ))
    assert_size_stride(arg111_1, (576, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg112_1, (576, ), (1, ))
    assert_size_stride(arg113_1, (576, ), (1, ))
    assert_size_stride(arg114_1, (192, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(arg115_1, (192, ), (1, ))
    assert_size_stride(arg116_1, (192, ), (1, ))
    assert_size_stride(arg117_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg118_1, (1152, ), (1, ))
    assert_size_stride(arg119_1, (1152, ), (1, ))
    assert_size_stride(arg120_1, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg121_1, (1152, ), (1, ))
    assert_size_stride(arg122_1, (1152, ), (1, ))
    assert_size_stride(arg123_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg124_1, (192, ), (1, ))
    assert_size_stride(arg125_1, (192, ), (1, ))
    assert_size_stride(arg126_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg127_1, (1152, ), (1, ))
    assert_size_stride(arg128_1, (1152, ), (1, ))
    assert_size_stride(arg129_1, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg130_1, (1152, ), (1, ))
    assert_size_stride(arg131_1, (1152, ), (1, ))
    assert_size_stride(arg132_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg133_1, (192, ), (1, ))
    assert_size_stride(arg134_1, (192, ), (1, ))
    assert_size_stride(arg135_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg136_1, (1152, ), (1, ))
    assert_size_stride(arg137_1, (1152, ), (1, ))
    assert_size_stride(arg138_1, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg139_1, (1152, ), (1, ))
    assert_size_stride(arg140_1, (1152, ), (1, ))
    assert_size_stride(arg141_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg142_1, (192, ), (1, ))
    assert_size_stride(arg143_1, (192, ), (1, ))
    assert_size_stride(arg144_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg145_1, (1152, ), (1, ))
    assert_size_stride(arg146_1, (1152, ), (1, ))
    assert_size_stride(arg147_1, (1152, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg148_1, (1152, ), (1, ))
    assert_size_stride(arg149_1, (1152, ), (1, ))
    assert_size_stride(arg150_1, (320, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg151_1, (320, ), (1, ))
    assert_size_stride(arg152_1, (320, ), (1, ))
    assert_size_stride(arg153_1, (1280, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(arg154_1, (1280, ), (1, ))
    assert_size_stride(arg155_1, (1280, ), (1, ))
    assert_size_stride(arg156_1, (1000, 1280), (1280, 1))
    assert_size_stride(arg157_1, (1000, ), (1, ))
    assert_size_stride(arg158_1, (32, ), (1, ))
    assert_size_stride(arg159_1, (32, ), (1, ))
    assert_size_stride(arg160_1, (), ())
    assert_size_stride(arg161_1, (32, ), (1, ))
    assert_size_stride(arg162_1, (32, ), (1, ))
    assert_size_stride(arg163_1, (), ())
    assert_size_stride(arg164_1, (16, ), (1, ))
    assert_size_stride(arg165_1, (16, ), (1, ))
    assert_size_stride(arg166_1, (), ())
    assert_size_stride(arg167_1, (48, ), (1, ))
    assert_size_stride(arg168_1, (48, ), (1, ))
    assert_size_stride(arg169_1, (), ())
    assert_size_stride(arg170_1, (48, ), (1, ))
    assert_size_stride(arg171_1, (48, ), (1, ))
    assert_size_stride(arg172_1, (), ())
    assert_size_stride(arg173_1, (24, ), (1, ))
    assert_size_stride(arg174_1, (24, ), (1, ))
    assert_size_stride(arg175_1, (), ())
    assert_size_stride(arg176_1, (72, ), (1, ))
    assert_size_stride(arg177_1, (72, ), (1, ))
    assert_size_stride(arg178_1, (), ())
    assert_size_stride(arg179_1, (72, ), (1, ))
    assert_size_stride(arg180_1, (72, ), (1, ))
    assert_size_stride(arg181_1, (), ())
    assert_size_stride(arg182_1, (24, ), (1, ))
    assert_size_stride(arg183_1, (24, ), (1, ))
    assert_size_stride(arg184_1, (), ())
    assert_size_stride(arg185_1, (72, ), (1, ))
    assert_size_stride(arg186_1, (72, ), (1, ))
    assert_size_stride(arg187_1, (), ())
    assert_size_stride(arg188_1, (72, ), (1, ))
    assert_size_stride(arg189_1, (72, ), (1, ))
    assert_size_stride(arg190_1, (), ())
    assert_size_stride(arg191_1, (24, ), (1, ))
    assert_size_stride(arg192_1, (24, ), (1, ))
    assert_size_stride(arg193_1, (), ())
    assert_size_stride(arg194_1, (72, ), (1, ))
    assert_size_stride(arg195_1, (72, ), (1, ))
    assert_size_stride(arg196_1, (), ())
    assert_size_stride(arg197_1, (72, ), (1, ))
    assert_size_stride(arg198_1, (72, ), (1, ))
    assert_size_stride(arg199_1, (), ())
    assert_size_stride(arg200_1, (40, ), (1, ))
    assert_size_stride(arg201_1, (40, ), (1, ))
    assert_size_stride(arg202_1, (), ())
    assert_size_stride(arg203_1, (120, ), (1, ))
    assert_size_stride(arg204_1, (120, ), (1, ))
    assert_size_stride(arg205_1, (), ())
    assert_size_stride(arg206_1, (120, ), (1, ))
    assert_size_stride(arg207_1, (120, ), (1, ))
    assert_size_stride(arg208_1, (), ())
    assert_size_stride(arg209_1, (40, ), (1, ))
    assert_size_stride(arg210_1, (40, ), (1, ))
    assert_size_stride(arg211_1, (), ())
    assert_size_stride(arg212_1, (120, ), (1, ))
    assert_size_stride(arg213_1, (120, ), (1, ))
    assert_size_stride(arg214_1, (), ())
    assert_size_stride(arg215_1, (120, ), (1, ))
    assert_size_stride(arg216_1, (120, ), (1, ))
    assert_size_stride(arg217_1, (), ())
    assert_size_stride(arg218_1, (40, ), (1, ))
    assert_size_stride(arg219_1, (40, ), (1, ))
    assert_size_stride(arg220_1, (), ())
    assert_size_stride(arg221_1, (240, ), (1, ))
    assert_size_stride(arg222_1, (240, ), (1, ))
    assert_size_stride(arg223_1, (), ())
    assert_size_stride(arg224_1, (240, ), (1, ))
    assert_size_stride(arg225_1, (240, ), (1, ))
    assert_size_stride(arg226_1, (), ())
    assert_size_stride(arg227_1, (80, ), (1, ))
    assert_size_stride(arg228_1, (80, ), (1, ))
    assert_size_stride(arg229_1, (), ())
    assert_size_stride(arg230_1, (480, ), (1, ))
    assert_size_stride(arg231_1, (480, ), (1, ))
    assert_size_stride(arg232_1, (), ())
    assert_size_stride(arg233_1, (480, ), (1, ))
    assert_size_stride(arg234_1, (480, ), (1, ))
    assert_size_stride(arg235_1, (), ())
    assert_size_stride(arg236_1, (80, ), (1, ))
    assert_size_stride(arg237_1, (80, ), (1, ))
    assert_size_stride(arg238_1, (), ())
    assert_size_stride(arg239_1, (480, ), (1, ))
    assert_size_stride(arg240_1, (480, ), (1, ))
    assert_size_stride(arg241_1, (), ())
    assert_size_stride(arg242_1, (480, ), (1, ))
    assert_size_stride(arg243_1, (480, ), (1, ))
    assert_size_stride(arg244_1, (), ())
    assert_size_stride(arg245_1, (80, ), (1, ))
    assert_size_stride(arg246_1, (80, ), (1, ))
    assert_size_stride(arg247_1, (), ())
    assert_size_stride(arg248_1, (480, ), (1, ))
    assert_size_stride(arg249_1, (480, ), (1, ))
    assert_size_stride(arg250_1, (), ())
    assert_size_stride(arg251_1, (480, ), (1, ))
    assert_size_stride(arg252_1, (480, ), (1, ))
    assert_size_stride(arg253_1, (), ())
    assert_size_stride(arg254_1, (96, ), (1, ))
    assert_size_stride(arg255_1, (96, ), (1, ))
    assert_size_stride(arg256_1, (), ())
    assert_size_stride(arg257_1, (576, ), (1, ))
    assert_size_stride(arg258_1, (576, ), (1, ))
    assert_size_stride(arg259_1, (), ())
    assert_size_stride(arg260_1, (576, ), (1, ))
    assert_size_stride(arg261_1, (576, ), (1, ))
    assert_size_stride(arg262_1, (), ())
    assert_size_stride(arg263_1, (96, ), (1, ))
    assert_size_stride(arg264_1, (96, ), (1, ))
    assert_size_stride(arg265_1, (), ())
    assert_size_stride(arg266_1, (576, ), (1, ))
    assert_size_stride(arg267_1, (576, ), (1, ))
    assert_size_stride(arg268_1, (), ())
    assert_size_stride(arg269_1, (576, ), (1, ))
    assert_size_stride(arg270_1, (576, ), (1, ))
    assert_size_stride(arg271_1, (), ())
    assert_size_stride(arg272_1, (192, ), (1, ))
    assert_size_stride(arg273_1, (192, ), (1, ))
    assert_size_stride(arg274_1, (), ())
    assert_size_stride(arg275_1, (1152, ), (1, ))
    assert_size_stride(arg276_1, (1152, ), (1, ))
    assert_size_stride(arg277_1, (), ())
    assert_size_stride(arg278_1, (1152, ), (1, ))
    assert_size_stride(arg279_1, (1152, ), (1, ))
    assert_size_stride(arg280_1, (), ())
    assert_size_stride(arg281_1, (192, ), (1, ))
    assert_size_stride(arg282_1, (192, ), (1, ))
    assert_size_stride(arg283_1, (), ())
    assert_size_stride(arg284_1, (1152, ), (1, ))
    assert_size_stride(arg285_1, (1152, ), (1, ))
    assert_size_stride(arg286_1, (), ())
    assert_size_stride(arg287_1, (1152, ), (1, ))
    assert_size_stride(arg288_1, (1152, ), (1, ))
    assert_size_stride(arg289_1, (), ())
    assert_size_stride(arg290_1, (192, ), (1, ))
    assert_size_stride(arg291_1, (192, ), (1, ))
    assert_size_stride(arg292_1, (), ())
    assert_size_stride(arg293_1, (1152, ), (1, ))
    assert_size_stride(arg294_1, (1152, ), (1, ))
    assert_size_stride(arg295_1, (), ())
    assert_size_stride(arg296_1, (1152, ), (1, ))
    assert_size_stride(arg297_1, (1152, ), (1, ))
    assert_size_stride(arg298_1, (), ())
    assert_size_stride(arg299_1, (192, ), (1, ))
    assert_size_stride(arg300_1, (192, ), (1, ))
    assert_size_stride(arg301_1, (), ())
    assert_size_stride(arg302_1, (1152, ), (1, ))
    assert_size_stride(arg303_1, (1152, ), (1, ))
    assert_size_stride(arg304_1, (), ())
    assert_size_stride(arg305_1, (1152, ), (1, ))
    assert_size_stride(arg306_1, (1152, ), (1, ))
    assert_size_stride(arg307_1, (), ())
    assert_size_stride(arg308_1, (320, ), (1, ))
    assert_size_stride(arg309_1, (320, ), (1, ))
    assert_size_stride(arg310_1, (), ())
    assert_size_stride(arg311_1, (1280, ), (1, ))
    assert_size_stride(arg312_1, (1280, ), (1, ))
    assert_size_stride(arg313_1, (), ())
    assert_size_stride(arg314_1, (4, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___layers_0], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg314_1, buf0, 12, 50176, grid=grid(12, 50176), stream=stream0)
        del arg314_1
        buf1 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___layers_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 96, 9, grid=grid(96, 9), stream=stream0)
        del arg0_1
        # Source Nodes: [l__mod___layers_0], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 32, 112, 112), (401408, 12544, 112, 1))
        del buf1
        buf3 = empty_strided((4, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___layers_1, l__mod___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf2, arg158_1, arg159_1, arg1_1, arg2_1, buf3, 128, 12544, grid=grid(128, 12544), stream=stream0)
        del arg158_1
        del arg159_1
        del arg1_1
        del arg2_1
        del buf2
        # Source Nodes: [l__mod___layers_1, l__mod___layers_2, l__mod___layers_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf4 = extern_kernels.convolution(buf3, arg3_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf4, (4, 32, 112, 112), (401408, 12544, 112, 1))
        del arg3_1
        buf5 = buf3; del buf3  # reuse
        # Source Nodes: [l__mod___layers_4, l__mod___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf4, arg161_1, arg162_1, arg4_1, arg5_1, buf5, 128, 12544, grid=grid(128, 12544), stream=stream0)
        del arg161_1
        del arg162_1
        del arg4_1
        del arg5_1
        del buf4
        # Source Nodes: [l__mod___layers_4, l__mod___layers_5, l__mod___layers_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf6 = extern_kernels.convolution(buf5, arg6_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 16, 112, 112), (200704, 12544, 112, 1))
        del arg6_1
        del buf5
        buf7 = empty_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_3.run(buf6, arg164_1, arg165_1, arg7_1, arg8_1, buf7, 64, 12544, grid=grid(64, 12544), stream=stream0)
        del arg164_1
        del arg165_1
        del arg7_1
        del arg8_1
        del buf6
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_0, l__mod___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf8 = extern_kernels.convolution(buf7, arg9_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (4, 48, 112, 112), (602112, 12544, 112, 1))
        del arg9_1
        del buf7
        buf9 = empty_strided((4, 48, 112, 112), (602112, 1, 5376, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_1, getattr_getattr_l__mod___layers___8_____0___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf8, arg167_1, arg168_1, arg10_1, arg11_1, buf9, 192, 12544, grid=grid(192, 12544), stream=stream0)
        del arg10_1
        del arg11_1
        del arg167_1
        del arg168_1
        del buf8
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_1, getattr_getattr_l__mod___layers___8_____0___layers_2, getattr_getattr_l__mod___layers___8_____0___layers_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf10 = extern_kernels.convolution(buf9, arg12_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf10, (4, 48, 56, 56), (150528, 3136, 56, 1))
        del arg12_1
        del buf9
        buf11 = reinterpret_tensor(buf0, (4, 48, 56, 56), (150528, 1, 2688, 48), 0); del buf0  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_4, getattr_getattr_l__mod___layers___8_____0___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf10, arg170_1, arg171_1, arg13_1, arg14_1, buf11, 192, 3136, grid=grid(192, 3136), stream=stream0)
        del arg13_1
        del arg14_1
        del arg170_1
        del arg171_1
        del buf10
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_4, getattr_getattr_l__mod___layers___8_____0___layers_5, getattr_getattr_l__mod___layers___8_____0___layers_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf12 = extern_kernels.convolution(buf11, arg15_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 24, 56, 56), (75264, 3136, 56, 1))
        del arg15_1
        del buf11
        buf13 = empty_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____0___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_6.run(buf12, arg173_1, arg174_1, arg16_1, arg17_1, buf13, 96, 3136, grid=grid(96, 3136), stream=stream0)
        del arg16_1
        del arg173_1
        del arg174_1
        del arg17_1
        del buf12
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____1___layers_0], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, arg18_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 72, 56, 56), (225792, 3136, 56, 1))
        del arg18_1
        buf15 = empty_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____1___layers_1, getattr_getattr_l__mod___layers___8_____1___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf14, arg176_1, arg177_1, arg19_1, arg20_1, buf15, 288, 3136, grid=grid(288, 3136), stream=stream0)
        del arg176_1
        del arg177_1
        del arg19_1
        del arg20_1
        del buf14
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____1___layers_1, getattr_getattr_l__mod___layers___8_____1___layers_2, getattr_getattr_l__mod___layers___8_____1___layers_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf16 = extern_kernels.convolution(buf15, arg21_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf16, (4, 72, 56, 56), (225792, 3136, 56, 1))
        del arg21_1
        buf17 = buf15; del buf15  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____1___layers_4, getattr_getattr_l__mod___layers___8_____1___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf16, arg179_1, arg180_1, arg22_1, arg23_1, buf17, 288, 3136, grid=grid(288, 3136), stream=stream0)
        del arg179_1
        del arg180_1
        del arg22_1
        del arg23_1
        del buf16
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____1___layers_4, getattr_getattr_l__mod___layers___8_____1___layers_5, getattr_getattr_l__mod___layers___8_____1___layers_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf18 = extern_kernels.convolution(buf17, arg24_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 24, 56, 56), (75264, 3136, 56, 1))
        del arg24_1
        buf19 = buf13; del buf13  # reuse
        # Source Nodes: [add, getattr_getattr_l__mod___layers___8_____1___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_8.run(buf19, buf18, arg182_1, arg183_1, arg25_1, arg26_1, 12544, 24, grid=grid(12544, 24), stream=stream0)
        del arg182_1
        del arg183_1
        del arg25_1
        del arg26_1
        del buf18
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____2___layers_0], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, arg27_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 72, 56, 56), (225792, 3136, 56, 1))
        del arg27_1
        buf21 = buf17; del buf17  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____2___layers_1, getattr_getattr_l__mod___layers___8_____2___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf20, arg185_1, arg186_1, arg28_1, arg29_1, buf21, 288, 3136, grid=grid(288, 3136), stream=stream0)
        del arg185_1
        del arg186_1
        del arg28_1
        del arg29_1
        del buf20
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____2___layers_1, getattr_getattr_l__mod___layers___8_____2___layers_2, getattr_getattr_l__mod___layers___8_____2___layers_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf22 = extern_kernels.convolution(buf21, arg30_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf22, (4, 72, 56, 56), (225792, 3136, 56, 1))
        del arg30_1
        buf23 = buf21; del buf21  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____2___layers_4, getattr_getattr_l__mod___layers___8_____2___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf22, arg188_1, arg189_1, arg31_1, arg32_1, buf23, 288, 3136, grid=grid(288, 3136), stream=stream0)
        del arg188_1
        del arg189_1
        del arg31_1
        del arg32_1
        del buf22
        # Source Nodes: [getattr_getattr_l__mod___layers___8_____2___layers_4, getattr_getattr_l__mod___layers___8_____2___layers_5, getattr_getattr_l__mod___layers___8_____2___layers_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf24 = extern_kernels.convolution(buf23, arg33_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 24, 56, 56), (75264, 3136, 56, 1))
        del arg33_1
        buf25 = buf19; del buf19  # reuse
        # Source Nodes: [add_1, getattr_getattr_l__mod___layers___8_____2___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_8.run(buf25, buf24, arg191_1, arg192_1, arg34_1, arg35_1, 12544, 24, grid=grid(12544, 24), stream=stream0)
        del arg191_1
        del arg192_1
        del arg34_1
        del arg35_1
        del buf24
        # Source Nodes: [add_1, getattr_getattr_l__mod___layers___8_____2___layers_7, getattr_getattr_l__mod___layers___9_____0___layers_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf26 = extern_kernels.convolution(buf25, arg36_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 72, 56, 56), (225792, 3136, 56, 1))
        del arg36_1
        del buf25
        buf27 = buf23; del buf23  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____0___layers_1, getattr_getattr_l__mod___layers___9_____0___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf26, arg194_1, arg195_1, arg37_1, arg38_1, buf27, 288, 3136, grid=grid(288, 3136), stream=stream0)
        del arg194_1
        del arg195_1
        del arg37_1
        del arg38_1
        del buf26
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____0___layers_1, getattr_getattr_l__mod___layers___9_____0___layers_2, getattr_getattr_l__mod___layers___9_____0___layers_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf28 = extern_kernels.convolution(buf27, arg39_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf28, (4, 72, 28, 28), (56448, 784, 28, 1))
        del arg39_1
        del buf27
        buf29 = empty_strided((4, 72, 28, 28), (56448, 1, 2016, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____0___layers_4, getattr_getattr_l__mod___layers___9_____0___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf28, arg197_1, arg198_1, arg40_1, arg41_1, buf29, 288, 784, grid=grid(288, 784), stream=stream0)
        del arg197_1
        del arg198_1
        del arg40_1
        del arg41_1
        del buf28
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____0___layers_4, getattr_getattr_l__mod___layers___9_____0___layers_5, getattr_getattr_l__mod___layers___9_____0___layers_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf30 = extern_kernels.convolution(buf29, arg42_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (4, 40, 28, 28), (31360, 784, 28, 1))
        del arg42_1
        buf31 = empty_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____0___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_10.run(buf30, arg200_1, arg201_1, arg43_1, arg44_1, buf31, 160, 784, grid=grid(160, 784), stream=stream0)
        del arg200_1
        del arg201_1
        del arg43_1
        del arg44_1
        del buf30
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____1___layers_0], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, arg45_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 120, 28, 28), (94080, 784, 28, 1))
        del arg45_1
        buf33 = empty_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____1___layers_1, getattr_getattr_l__mod___layers___9_____1___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf32, arg203_1, arg204_1, arg46_1, arg47_1, buf33, 480, 784, grid=grid(480, 784), stream=stream0)
        del arg203_1
        del arg204_1
        del arg46_1
        del arg47_1
        del buf32
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____1___layers_1, getattr_getattr_l__mod___layers___9_____1___layers_2, getattr_getattr_l__mod___layers___9_____1___layers_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf34 = extern_kernels.convolution(buf33, arg48_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf34, (4, 120, 28, 28), (94080, 784, 28, 1))
        del arg48_1
        buf35 = buf33; del buf33  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____1___layers_4, getattr_getattr_l__mod___layers___9_____1___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf34, arg206_1, arg207_1, arg49_1, arg50_1, buf35, 480, 784, grid=grid(480, 784), stream=stream0)
        del arg206_1
        del arg207_1
        del arg49_1
        del arg50_1
        del buf34
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____1___layers_4, getattr_getattr_l__mod___layers___9_____1___layers_5, getattr_getattr_l__mod___layers___9_____1___layers_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf36 = extern_kernels.convolution(buf35, arg51_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 40, 28, 28), (31360, 784, 28, 1))
        del arg51_1
        buf37 = buf31; del buf31  # reuse
        # Source Nodes: [add_2, getattr_getattr_l__mod___layers___9_____1___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_12.run(buf37, buf36, arg209_1, arg210_1, arg52_1, arg53_1, 3136, 40, grid=grid(3136, 40), stream=stream0)
        del arg209_1
        del arg210_1
        del arg52_1
        del arg53_1
        del buf36
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____2___layers_0], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, arg54_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 120, 28, 28), (94080, 784, 28, 1))
        del arg54_1
        buf39 = buf35; del buf35  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____2___layers_1, getattr_getattr_l__mod___layers___9_____2___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf38, arg212_1, arg213_1, arg55_1, arg56_1, buf39, 480, 784, grid=grid(480, 784), stream=stream0)
        del arg212_1
        del arg213_1
        del arg55_1
        del arg56_1
        del buf38
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____2___layers_1, getattr_getattr_l__mod___layers___9_____2___layers_2, getattr_getattr_l__mod___layers___9_____2___layers_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf40 = extern_kernels.convolution(buf39, arg57_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf40, (4, 120, 28, 28), (94080, 784, 28, 1))
        del arg57_1
        buf41 = buf39; del buf39  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____2___layers_4, getattr_getattr_l__mod___layers___9_____2___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf40, arg215_1, arg216_1, arg58_1, arg59_1, buf41, 480, 784, grid=grid(480, 784), stream=stream0)
        del arg215_1
        del arg216_1
        del arg58_1
        del arg59_1
        del buf40
        # Source Nodes: [getattr_getattr_l__mod___layers___9_____2___layers_4, getattr_getattr_l__mod___layers___9_____2___layers_5, getattr_getattr_l__mod___layers___9_____2___layers_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf42 = extern_kernels.convolution(buf41, arg60_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 40, 28, 28), (31360, 784, 28, 1))
        del arg60_1
        buf43 = buf37; del buf37  # reuse
        # Source Nodes: [add_3, getattr_getattr_l__mod___layers___9_____2___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_12.run(buf43, buf42, arg218_1, arg219_1, arg61_1, arg62_1, 3136, 40, grid=grid(3136, 40), stream=stream0)
        del arg218_1
        del arg219_1
        del arg61_1
        del arg62_1
        del buf42
        # Source Nodes: [add_3, getattr_getattr_l__mod___layers___10_____0___layers_0, getattr_getattr_l__mod___layers___9_____2___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf44 = extern_kernels.convolution(buf43, arg63_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (4, 240, 28, 28), (188160, 784, 28, 1))
        del arg63_1
        del buf43
        buf45 = empty_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____0___layers_1, getattr_getattr_l__mod___layers___10_____0___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf44, arg221_1, arg222_1, arg64_1, arg65_1, buf45, 960, 784, grid=grid(960, 784), stream=stream0)
        del arg221_1
        del arg222_1
        del arg64_1
        del arg65_1
        del buf44
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____0___layers_1, getattr_getattr_l__mod___layers___10_____0___layers_2, getattr_getattr_l__mod___layers___10_____0___layers_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf46 = extern_kernels.convolution(buf45, arg66_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf46, (4, 240, 14, 14), (47040, 196, 14, 1))
        del arg66_1
        del buf45
        buf47 = empty_strided((4, 240, 14, 14), (47040, 1, 3360, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____0___layers_4, getattr_getattr_l__mod___layers___10_____0___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf46, arg224_1, arg225_1, arg67_1, arg68_1, buf47, 960, 196, grid=grid(960, 196), stream=stream0)
        del arg224_1
        del arg225_1
        del arg67_1
        del arg68_1
        del buf46
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____0___layers_4, getattr_getattr_l__mod___layers___10_____0___layers_5, getattr_getattr_l__mod___layers___10_____0___layers_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf48 = extern_kernels.convolution(buf47, arg69_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 80, 14, 14), (15680, 196, 14, 1))
        del arg69_1
        del buf47
        buf49 = empty_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____0___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_15.run(buf48, arg227_1, arg228_1, arg70_1, arg71_1, buf49, 320, 196, grid=grid(320, 196), stream=stream0)
        del arg227_1
        del arg228_1
        del arg70_1
        del arg71_1
        del buf48
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____1___layers_0], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, arg72_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 480, 14, 14), (94080, 196, 14, 1))
        del arg72_1
        buf51 = reinterpret_tensor(buf41, (4, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf41  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____1___layers_1, getattr_getattr_l__mod___layers___10_____1___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf50, arg230_1, arg231_1, arg73_1, arg74_1, buf51, 1920, 196, grid=grid(1920, 196), stream=stream0)
        del arg230_1
        del arg231_1
        del arg73_1
        del arg74_1
        del buf50
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____1___layers_1, getattr_getattr_l__mod___layers___10_____1___layers_2, getattr_getattr_l__mod___layers___10_____1___layers_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf52 = extern_kernels.convolution(buf51, arg75_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf52, (4, 480, 14, 14), (94080, 196, 14, 1))
        del arg75_1
        buf53 = buf51; del buf51  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____1___layers_4, getattr_getattr_l__mod___layers___10_____1___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf52, arg233_1, arg234_1, arg76_1, arg77_1, buf53, 1920, 196, grid=grid(1920, 196), stream=stream0)
        del arg233_1
        del arg234_1
        del arg76_1
        del arg77_1
        del buf52
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____1___layers_4, getattr_getattr_l__mod___layers___10_____1___layers_5, getattr_getattr_l__mod___layers___10_____1___layers_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf54 = extern_kernels.convolution(buf53, arg78_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (4, 80, 14, 14), (15680, 196, 14, 1))
        del arg78_1
        buf55 = buf49; del buf49  # reuse
        # Source Nodes: [add_4, getattr_getattr_l__mod___layers___10_____1___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_17.run(buf55, buf54, arg236_1, arg237_1, arg79_1, arg80_1, 784, 80, grid=grid(784, 80), stream=stream0)
        del arg236_1
        del arg237_1
        del arg79_1
        del arg80_1
        del buf54
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____2___layers_0], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, arg81_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 480, 14, 14), (94080, 196, 14, 1))
        del arg81_1
        buf57 = buf53; del buf53  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____2___layers_1, getattr_getattr_l__mod___layers___10_____2___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf56, arg239_1, arg240_1, arg82_1, arg83_1, buf57, 1920, 196, grid=grid(1920, 196), stream=stream0)
        del arg239_1
        del arg240_1
        del arg82_1
        del arg83_1
        del buf56
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____2___layers_1, getattr_getattr_l__mod___layers___10_____2___layers_2, getattr_getattr_l__mod___layers___10_____2___layers_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf58 = extern_kernels.convolution(buf57, arg84_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf58, (4, 480, 14, 14), (94080, 196, 14, 1))
        del arg84_1
        buf59 = buf57; del buf57  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____2___layers_4, getattr_getattr_l__mod___layers___10_____2___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf58, arg242_1, arg243_1, arg85_1, arg86_1, buf59, 1920, 196, grid=grid(1920, 196), stream=stream0)
        del arg242_1
        del arg243_1
        del arg85_1
        del arg86_1
        del buf58
        # Source Nodes: [getattr_getattr_l__mod___layers___10_____2___layers_4, getattr_getattr_l__mod___layers___10_____2___layers_5, getattr_getattr_l__mod___layers___10_____2___layers_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf60 = extern_kernels.convolution(buf59, arg87_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (4, 80, 14, 14), (15680, 196, 14, 1))
        del arg87_1
        buf61 = buf55; del buf55  # reuse
        # Source Nodes: [add_5, getattr_getattr_l__mod___layers___10_____2___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_17.run(buf61, buf60, arg245_1, arg246_1, arg88_1, arg89_1, 784, 80, grid=grid(784, 80), stream=stream0)
        del arg245_1
        del arg246_1
        del arg88_1
        del arg89_1
        del buf60
        # Source Nodes: [add_5, getattr_getattr_l__mod___layers___10_____2___layers_7, getattr_getattr_l__mod___layers___11_____0___layers_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf62 = extern_kernels.convolution(buf61, arg90_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (4, 480, 14, 14), (94080, 196, 14, 1))
        del arg90_1
        buf63 = buf59; del buf59  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____0___layers_1, getattr_getattr_l__mod___layers___11_____0___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf62, arg248_1, arg249_1, arg91_1, arg92_1, buf63, 1920, 196, grid=grid(1920, 196), stream=stream0)
        del arg248_1
        del arg249_1
        del arg91_1
        del arg92_1
        del buf62
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____0___layers_1, getattr_getattr_l__mod___layers___11_____0___layers_2, getattr_getattr_l__mod___layers___11_____0___layers_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf64 = extern_kernels.convolution(buf63, arg93_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf64, (4, 480, 14, 14), (94080, 196, 14, 1))
        del arg93_1
        buf65 = buf63; del buf63  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____0___layers_4, getattr_getattr_l__mod___layers___11_____0___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf64, arg251_1, arg252_1, arg94_1, arg95_1, buf65, 1920, 196, grid=grid(1920, 196), stream=stream0)
        del arg251_1
        del arg252_1
        del arg94_1
        del arg95_1
        del buf64
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____0___layers_4, getattr_getattr_l__mod___layers___11_____0___layers_5, getattr_getattr_l__mod___layers___11_____0___layers_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf66 = extern_kernels.convolution(buf65, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (4, 96, 14, 14), (18816, 196, 14, 1))
        del arg96_1
        del buf65
        buf67 = empty_strided((4, 96, 14, 14), (18816, 1, 1344, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____0___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_18.run(buf66, arg254_1, arg255_1, arg97_1, arg98_1, buf67, 384, 196, grid=grid(384, 196), stream=stream0)
        del arg254_1
        del arg255_1
        del arg97_1
        del arg98_1
        del buf66
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____1___layers_0], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, arg99_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (4, 576, 14, 14), (112896, 196, 14, 1))
        del arg99_1
        buf69 = empty_strided((4, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____1___layers_1, getattr_getattr_l__mod___layers___11_____1___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf68, arg257_1, arg258_1, arg100_1, arg101_1, buf69, 2304, 196, grid=grid(2304, 196), stream=stream0)
        del arg100_1
        del arg101_1
        del arg257_1
        del arg258_1
        del buf68
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____1___layers_1, getattr_getattr_l__mod___layers___11_____1___layers_2, getattr_getattr_l__mod___layers___11_____1___layers_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf70 = extern_kernels.convolution(buf69, arg102_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
        assert_size_stride(buf70, (4, 576, 14, 14), (112896, 196, 14, 1))
        del arg102_1
        buf71 = buf69; del buf69  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____1___layers_4, getattr_getattr_l__mod___layers___11_____1___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf70, arg260_1, arg261_1, arg103_1, arg104_1, buf71, 2304, 196, grid=grid(2304, 196), stream=stream0)
        del arg103_1
        del arg104_1
        del arg260_1
        del arg261_1
        del buf70
        # Source Nodes: [getattr_getattr_l__mod___layers___11_____1___layers_4, getattr_getattr_l__mod___layers___11_____1___layers_5, getattr_getattr_l__mod___layers___11_____1___layers_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf72 = extern_kernels.convolution(buf71, arg105_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (4, 96, 14, 14), (18816, 196, 14, 1))
        del arg105_1
        buf73 = buf67; del buf67  # reuse
        # Source Nodes: [add_6, getattr_getattr_l__mod___layers___11_____1___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_20.run(buf73, buf72, arg263_1, arg264_1, arg106_1, arg107_1, 784, 96, grid=grid(784, 96), stream=stream0)
        del arg106_1
        del arg107_1
        del arg263_1
        del arg264_1
        del buf72
        # Source Nodes: [add_6, getattr_getattr_l__mod___layers___11_____1___layers_7, getattr_getattr_l__mod___layers___12_____0___layers_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf74 = extern_kernels.convolution(buf73, arg108_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (4, 576, 14, 14), (112896, 196, 14, 1))
        del arg108_1
        del buf73
        buf75 = buf71; del buf71  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____0___layers_1, getattr_getattr_l__mod___layers___12_____0___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf74, arg266_1, arg267_1, arg109_1, arg110_1, buf75, 2304, 196, grid=grid(2304, 196), stream=stream0)
        del arg109_1
        del arg110_1
        del arg266_1
        del arg267_1
        del buf74
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____0___layers_1, getattr_getattr_l__mod___layers___12_____0___layers_2, getattr_getattr_l__mod___layers___12_____0___layers_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf76 = extern_kernels.convolution(buf75, arg111_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
        assert_size_stride(buf76, (4, 576, 7, 7), (28224, 49, 7, 1))
        del arg111_1
        del buf75
        buf77 = empty_strided((4, 576, 7, 7), (28224, 1, 4032, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____0___layers_4, getattr_getattr_l__mod___layers___12_____0___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf76, arg269_1, arg270_1, arg112_1, arg113_1, buf77, 2304, 49, grid=grid(2304, 49), stream=stream0)
        del arg112_1
        del arg113_1
        del arg269_1
        del arg270_1
        del buf76
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____0___layers_4, getattr_getattr_l__mod___layers___12_____0___layers_5, getattr_getattr_l__mod___layers___12_____0___layers_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf78 = extern_kernels.convolution(buf77, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (4, 192, 7, 7), (9408, 49, 7, 1))
        del arg114_1
        del buf77
        buf79 = empty_strided((4, 192, 7, 7), (9408, 1, 1344, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____0___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_22.run(buf78, arg272_1, arg273_1, arg115_1, arg116_1, buf79, 768, 49, grid=grid(768, 49), stream=stream0)
        del arg115_1
        del arg116_1
        del arg272_1
        del arg273_1
        del buf78
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____1___layers_0], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, arg117_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (4, 1152, 7, 7), (56448, 49, 7, 1))
        del arg117_1
        buf81 = reinterpret_tensor(buf29, (4, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf29  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____1___layers_1, getattr_getattr_l__mod___layers___12_____1___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf80, arg275_1, arg276_1, arg118_1, arg119_1, buf81, 4608, 49, grid=grid(4608, 49), stream=stream0)
        del arg118_1
        del arg119_1
        del arg275_1
        del arg276_1
        del buf80
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____1___layers_1, getattr_getattr_l__mod___layers___12_____1___layers_2, getattr_getattr_l__mod___layers___12_____1___layers_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf82 = extern_kernels.convolution(buf81, arg120_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf82, (4, 1152, 7, 7), (56448, 49, 7, 1))
        del arg120_1
        buf83 = buf81; del buf81  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____1___layers_4, getattr_getattr_l__mod___layers___12_____1___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf82, arg278_1, arg279_1, arg121_1, arg122_1, buf83, 4608, 49, grid=grid(4608, 49), stream=stream0)
        del arg121_1
        del arg122_1
        del arg278_1
        del arg279_1
        del buf82
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____1___layers_4, getattr_getattr_l__mod___layers___12_____1___layers_5, getattr_getattr_l__mod___layers___12_____1___layers_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf84 = extern_kernels.convolution(buf83, arg123_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (4, 192, 7, 7), (9408, 49, 7, 1))
        del arg123_1
        buf85 = buf79; del buf79  # reuse
        # Source Nodes: [add_7, getattr_getattr_l__mod___layers___12_____1___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_24.run(buf85, buf84, arg281_1, arg282_1, arg124_1, arg125_1, 196, 192, grid=grid(196, 192), stream=stream0)
        del arg124_1
        del arg125_1
        del arg281_1
        del arg282_1
        del buf84
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____2___layers_0], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (4, 1152, 7, 7), (56448, 49, 7, 1))
        del arg126_1
        buf87 = buf83; del buf83  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____2___layers_1, getattr_getattr_l__mod___layers___12_____2___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf86, arg284_1, arg285_1, arg127_1, arg128_1, buf87, 4608, 49, grid=grid(4608, 49), stream=stream0)
        del arg127_1
        del arg128_1
        del arg284_1
        del arg285_1
        del buf86
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____2___layers_1, getattr_getattr_l__mod___layers___12_____2___layers_2, getattr_getattr_l__mod___layers___12_____2___layers_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf88 = extern_kernels.convolution(buf87, arg129_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf88, (4, 1152, 7, 7), (56448, 49, 7, 1))
        del arg129_1
        buf89 = buf87; del buf87  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____2___layers_4, getattr_getattr_l__mod___layers___12_____2___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf88, arg287_1, arg288_1, arg130_1, arg131_1, buf89, 4608, 49, grid=grid(4608, 49), stream=stream0)
        del arg130_1
        del arg131_1
        del arg287_1
        del arg288_1
        del buf88
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____2___layers_4, getattr_getattr_l__mod___layers___12_____2___layers_5, getattr_getattr_l__mod___layers___12_____2___layers_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf90 = extern_kernels.convolution(buf89, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (4, 192, 7, 7), (9408, 49, 7, 1))
        del arg132_1
        buf91 = buf85; del buf85  # reuse
        # Source Nodes: [add_8, getattr_getattr_l__mod___layers___12_____2___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_24.run(buf91, buf90, arg290_1, arg291_1, arg133_1, arg134_1, 196, 192, grid=grid(196, 192), stream=stream0)
        del arg133_1
        del arg134_1
        del arg290_1
        del arg291_1
        del buf90
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____3___layers_0], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, arg135_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (4, 1152, 7, 7), (56448, 49, 7, 1))
        del arg135_1
        buf93 = buf89; del buf89  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____3___layers_1, getattr_getattr_l__mod___layers___12_____3___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf92, arg293_1, arg294_1, arg136_1, arg137_1, buf93, 4608, 49, grid=grid(4608, 49), stream=stream0)
        del arg136_1
        del arg137_1
        del arg293_1
        del arg294_1
        del buf92
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____3___layers_1, getattr_getattr_l__mod___layers___12_____3___layers_2, getattr_getattr_l__mod___layers___12_____3___layers_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf94 = extern_kernels.convolution(buf93, arg138_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf94, (4, 1152, 7, 7), (56448, 49, 7, 1))
        del arg138_1
        buf95 = buf93; del buf93  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____3___layers_4, getattr_getattr_l__mod___layers___12_____3___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf94, arg296_1, arg297_1, arg139_1, arg140_1, buf95, 4608, 49, grid=grid(4608, 49), stream=stream0)
        del arg139_1
        del arg140_1
        del arg296_1
        del arg297_1
        del buf94
        # Source Nodes: [getattr_getattr_l__mod___layers___12_____3___layers_4, getattr_getattr_l__mod___layers___12_____3___layers_5, getattr_getattr_l__mod___layers___12_____3___layers_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf96 = extern_kernels.convolution(buf95, arg141_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 192, 7, 7), (9408, 49, 7, 1))
        del arg141_1
        buf97 = buf91; del buf91  # reuse
        # Source Nodes: [add_9, getattr_getattr_l__mod___layers___12_____3___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_24.run(buf97, buf96, arg299_1, arg300_1, arg142_1, arg143_1, 196, 192, grid=grid(196, 192), stream=stream0)
        del arg142_1
        del arg143_1
        del arg299_1
        del arg300_1
        del buf96
        # Source Nodes: [add_9, getattr_getattr_l__mod___layers___12_____3___layers_7, getattr_getattr_l__mod___layers___13_____0___layers_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf98 = extern_kernels.convolution(buf97, arg144_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 1152, 7, 7), (56448, 49, 7, 1))
        del arg144_1
        del buf97
        buf99 = buf95; del buf95  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___13_____0___layers_1, getattr_getattr_l__mod___layers___13_____0___layers_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf98, arg302_1, arg303_1, arg145_1, arg146_1, buf99, 4608, 49, grid=grid(4608, 49), stream=stream0)
        del arg145_1
        del arg146_1
        del arg302_1
        del arg303_1
        del buf98
        # Source Nodes: [getattr_getattr_l__mod___layers___13_____0___layers_1, getattr_getattr_l__mod___layers___13_____0___layers_2, getattr_getattr_l__mod___layers___13_____0___layers_3], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf100 = extern_kernels.convolution(buf99, arg147_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf100, (4, 1152, 7, 7), (56448, 49, 7, 1))
        del arg147_1
        buf101 = buf99; del buf99  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___13_____0___layers_4, getattr_getattr_l__mod___layers___13_____0___layers_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf100, arg305_1, arg306_1, arg148_1, arg149_1, buf101, 4608, 49, grid=grid(4608, 49), stream=stream0)
        del arg148_1
        del arg149_1
        del arg305_1
        del arg306_1
        del buf100
        # Source Nodes: [getattr_getattr_l__mod___layers___13_____0___layers_4, getattr_getattr_l__mod___layers___13_____0___layers_5, getattr_getattr_l__mod___layers___13_____0___layers_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf102 = extern_kernels.convolution(buf101, arg150_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (4, 320, 7, 7), (15680, 49, 7, 1))
        del arg150_1
        del buf101
        buf103 = reinterpret_tensor(buf61, (4, 320, 7, 7), (15680, 1, 2240, 320), 0); del buf61  # reuse
        # Source Nodes: [getattr_getattr_l__mod___layers___13_____0___layers_7], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf102, arg308_1, arg309_1, arg151_1, arg152_1, buf103, 1280, 49, grid=grid(1280, 49), stream=stream0)
        del arg151_1
        del arg152_1
        del arg308_1
        del arg309_1
        del buf102
        # Source Nodes: [getattr_getattr_l__mod___layers___13_____0___layers_7, l__mod___layers_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf104 = extern_kernels.convolution(buf103, arg153_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (4, 1280, 7, 7), (62720, 49, 7, 1))
        del arg153_1
        del buf103
        buf105 = empty((4, 1280), device='cuda', dtype=torch.float32)
        buf106 = buf105; del buf105  # reuse
        # Source Nodes: [l__mod___layers_15, x, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_26.run(buf106, buf104, arg311_1, arg312_1, arg154_1, arg155_1, 5120, 49, grid=grid(5120), stream=stream0)
        del arg154_1
        del arg155_1
        del arg311_1
        del arg312_1
        del buf104
        buf107 = empty((4, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___classifier_1, l__mod___layers_15, x, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.addmm, aten.mean, aten.relu]
        extern_kernels.addmm(arg157_1, buf106, reinterpret_tensor(arg156_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf107)
        del arg156_1
        del arg157_1
        return (buf107, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((48, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((48, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((24, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((72, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((40, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((96, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((96, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((576, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((192, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((1152, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((320, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg161_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg164_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg167_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg170_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg173_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg176_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg179_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg182_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg185_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg188_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg191_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg194_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg197_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg200_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg203_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg206_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg209_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg212_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg215_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg218_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg221_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg224_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg227_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg230_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg233_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg236_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg239_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg242_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg245_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg248_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg251_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg254_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg257_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg260_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg263_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg266_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg269_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg272_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg275_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg278_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg281_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg284_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg287_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg290_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg293_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg296_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg299_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg302_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg305_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg308_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg311_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg314_1 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mnasnet1_0', benchmark_compiled_module)
