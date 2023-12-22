
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
# Source Nodes: [l__mod___features_0_0], Original ATen: [aten.convolution]
# l__mod___features_0_0 => convolution
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


# kernel path: /tmp/torchinductor_youkaichao/zy/czyirduix6cyyuec7dfvcmlnzfoyrtbb4zm5cpjn54ggbwkfp337.py
# Source Nodes: [l__mod___features_0_0], Original ATen: [aten.convolution]
# l__mod___features_0_0 => convolution
triton_poi_fused_convolution_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 48
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


# kernel path: /tmp/torchinductor_youkaichao/ha/cha3jempagd2ehul3kq6v43ccswtjovht5lr3blzcqtspej66vaf.py
# Source Nodes: [l__mod___features_0_1, l__mod___features_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# l__mod___features_0_1 => add_1, mul_1, mul_2, sub
# l__mod___features_0_2 => add_2, clamp_max, clamp_min, div, mul_3
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (y0 + (16*x2) + (200704*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/25/c25u3ojwk4ljw6hbdhxefipcdsmv2wtt7knaudeelxmewlt2ex3h.py
# Source Nodes: [getattr_l__mod___features___1___block_0_1, getattr_l__mod___features___1___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_l__mod___features___1___block_0_1 => add_4, mul_5, mul_6, sub_1
# getattr_l__mod___features___1___block_0_2 => relu
triton_poi_fused__native_batch_norm_legit_no_training_relu_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_3', 'mutated_arg_names': []},
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
    tl.store(out_ptr0 + (y0 + (16*x2) + (200704*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mh/cmhuj7u3k3doha5viqfpen4kf6bh6tfvibemjjusmqda3tgb4kpx.py
# Source Nodes: [result, result_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# result => add_6, mul_8, mul_9, sub_2
# result_1 => add_7
triton_poi_fused__native_batch_norm_legit_no_training_add_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 50176
    xnumel = 16
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 12544
    y1 = (yindex // 12544)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (12544*x2) + (200704*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (16*y3)), xmask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (16*y3)), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h6/ch6jlmnhogmq6nc76cr2pwrarx2frohzsuqfltqgq6lh6rqxdkc3.py
# Source Nodes: [getattr_l__mod___features___2___block_0_1, getattr_l__mod___features___2___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_l__mod___features___2___block_0_1 => add_9, mul_11, mul_12, sub_3
# getattr_l__mod___features___2___block_0_2 => relu_1
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (64*x2) + (802816*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b4/cb4b5vbsg7xnjakk6ilxvqpg54hbwpoopf7g3q2dxi2qrof6u7ir.py
# Source Nodes: [getattr_l__mod___features___2___block_1_1, getattr_l__mod___features___2___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_l__mod___features___2___block_1_1 => add_11, mul_14, mul_15, sub_4
# getattr_l__mod___features___2___block_1_2 => relu_2
triton_poi_fused__native_batch_norm_legit_no_training_relu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (64*x2) + (200704*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ob/cob46t7txy7u75gtpf24by55gmzywti6wu5cxkr5pbg3onv7g6yi.py
# Source Nodes: [result_2], Original ATen: [aten._native_batch_norm_legit_no_training]
# result_2 => add_13, mul_17, mul_18, sub_5
triton_poi_fused__native_batch_norm_legit_no_training_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_7', 'mutated_arg_names': []},
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/bi/cbiuom2eb7pgy5obayuhr5exzhfye2zpayibkso73ekhvb2mxvy5.py
# Source Nodes: [getattr_l__mod___features___3___block_0_1, getattr_l__mod___features___3___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_l__mod___features___3___block_0_1 => add_15, mul_20, mul_21, sub_6
# getattr_l__mod___features___3___block_0_2 => relu_3
triton_poi_fused__native_batch_norm_legit_no_training_relu_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_8', 'mutated_arg_names': []},
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
    tl.store(out_ptr0 + (y0 + (72*x2) + (225792*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3f/c3f3bcajstvdcqrkzvaekuimlxrrp5r5eqggpo4elrgvlvabxm5h.py
# Source Nodes: [result_3, result_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# result_3 => add_19, mul_26, mul_27, sub_8
# result_4 => add_20
triton_poi_fused__native_batch_norm_legit_no_training_add_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_9', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/vz/cvzwnmvurf6duyzpybvfeabun5nv23ras24qm5ualect7v3bw2wp.py
# Source Nodes: [getattr_l__mod___features___4___block_1_1, getattr_l__mod___features___4___block_1_2, scale], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
# getattr_l__mod___features___4___block_1_1 => add_24, mul_32, mul_33, sub_10
# getattr_l__mod___features___4___block_1_2 => relu_6
# scale => mean
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_10', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 288
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 72
    tmp0 = tl.load(in_out_ptr0 + (r2 + (784*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = 784.0
    tmp21 = tmp19 / tmp20
    tl.store(in_out_ptr0 + (r2 + (784*x3)), tmp15, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zy/czyc5xnb526spr3mdlbor73cd2hiwki22mbkiibwxm53t5stpkfn.py
# Source Nodes: [scale, scale_1, scale_2], Original ATen: [aten.convolution, aten.mean, aten.relu]
# scale => mean
# scale_1 => convolution_11
# scale_2 => relu_7
triton_poi_fused_convolution_mean_relu_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 96
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 24
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vq/cvqm76keyf3ub3mc6azypq4u62s3znvgbqg5bomyz7kjmqgxctid.py
# Source Nodes: [mul, scale, scale_1, scale_2, scale_3, scale_4], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
# mul => mul_34
# scale => mean
# scale_1 => convolution_11
# scale_2 => relu_7
# scale_3 => convolution_12
# scale_4 => add_25, clamp_max_1, clamp_min_1, div_1
triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 288
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y3 = yindex
    y0 = yindex % 72
    x2 = xindex
    y1 = (yindex // 72)
    tmp0 = tl.load(in_ptr0 + (y3), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 / tmp7
    tmp11 = tmp9 * tmp10
    tl.store(out_ptr0 + (y0 + (72*x2) + (56448*y1)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vx/cvxyijaje2nnrmkr3tkjde5icjbetmawx37gniosyisvzow3dwn4.py
# Source Nodes: [result_5], Original ATen: [aten._native_batch_norm_legit_no_training]
# result_5 => add_27, mul_36, mul_37, sub_11
triton_poi_fused__native_batch_norm_legit_no_training_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_13', 'mutated_arg_names': []},
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/tr/ctrs64or2rlnwemirpgfbbkguax3ojyciuhulukqkg2ujxor7fzc.py
# Source Nodes: [getattr_l__mod___features___5___block_0_1, getattr_l__mod___features___5___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# getattr_l__mod___features___5___block_0_1 => add_29, mul_39, mul_40, sub_12
# getattr_l__mod___features___5___block_0_2 => relu_8
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
    tl.store(out_ptr0 + (y0 + (120*x2) + (94080*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ym/cymvfetjuljstpnyilmc733ioln7jo5kpqjbujoam32lnsnvfe7r.py
# Source Nodes: [getattr_l__mod___features___5___block_1_1, getattr_l__mod___features___5___block_1_2, scale_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
# getattr_l__mod___features___5___block_1_1 => add_31, mul_42, mul_43, sub_13
# getattr_l__mod___features___5___block_1_2 => relu_9
# scale_5 => mean_1
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[512, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_15', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 480
    XBLOCK: tl.constexpr = 1
    rnumel = 784
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 120
    tmp0 = tl.load(in_out_ptr0 + (r2 + (784*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = 784.0
    tmp21 = tmp19 / tmp20
    tl.store(in_out_ptr0 + (r2 + (784*x3)), tmp15, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/s7/cs7jdwhf4rntt7ek6gc7khlmwwdvjjqay6j2ny2uin4bjy5w7uni.py
# Source Nodes: [scale_5, scale_6, scale_7], Original ATen: [aten.convolution, aten.mean, aten.relu]
# scale_5 => mean_1
# scale_6 => convolution_16
# scale_7 => relu_10
triton_poi_fused_convolution_mean_relu_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[128], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 32
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7v/c7vzbq4iodziaruxqnz3lhtyt2tis3iwpn5lqd4la2k6777tlobu.py
# Source Nodes: [mul_1, scale_5, scale_6, scale_7, scale_8, scale_9], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
# mul_1 => mul_44
# scale_5 => mean_1
# scale_6 => convolution_16
# scale_7 => relu_10
# scale_8 => convolution_17
# scale_9 => add_32, clamp_max_2, clamp_min_2, div_2
triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 480
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y3 = yindex
    y0 = yindex % 120
    x2 = xindex
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (y3), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 / tmp7
    tmp11 = tmp9 * tmp10
    tl.store(out_ptr0 + (y0 + (120*x2) + (94080*y1)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ki/ckicycubh2gpi5d6hqohkqev4afj6ojfdln3egu3te42l4xjc62i.py
# Source Nodes: [result_6, result_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# result_6 => add_34, mul_46, mul_47, sub_14
# result_7 => add_35
triton_poi_fused__native_batch_norm_legit_no_training_add_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_18', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/mn/cmnfrspuk3j6vrnk7tco4atomb6toz7iwdwq3lx4rhf3mrsml452.py
# Source Nodes: [getattr_l__mod___features___7___block_0_1, getattr_l__mod___features___7___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# getattr_l__mod___features___7___block_0_1 => add_45, mul_59, mul_60, sub_18
# getattr_l__mod___features___7___block_0_2 => add_46, clamp_max_4, clamp_min_4, div_4, mul_61
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (y0 + (240*x2) + (188160*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5k/c5kjazfkyumtncvxqic4diynwqeoiii54llwpm6zany32vpmdml7.py
# Source Nodes: [getattr_l__mod___features___7___block_1_1, getattr_l__mod___features___7___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# getattr_l__mod___features___7___block_1_1 => add_48, mul_63, mul_64, sub_19
# getattr_l__mod___features___7___block_1_2 => add_49, clamp_max_5, clamp_min_5, div_5, mul_65
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (y0 + (240*x2) + (47040*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yc/cycqj5oawdih2kqepdwdfa4iwwhr3djg7z5ztqofex6htcrlvhev.py
# Source Nodes: [result_10], Original ATen: [aten._native_batch_norm_legit_no_training]
# result_10 => add_51, mul_67, mul_68, sub_20
triton_poi_fused__native_batch_norm_legit_no_training_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_21', 'mutated_arg_names': []},
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/fb/cfbgzzh4im7mevdtfrahf2i6ikaqlidzwsb377sjsv4db73ilkdu.py
# Source Nodes: [getattr_l__mod___features___8___block_0_1, getattr_l__mod___features___8___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# getattr_l__mod___features___8___block_0_1 => add_53, mul_70, mul_71, sub_21
# getattr_l__mod___features___8___block_0_2 => add_54, clamp_max_6, clamp_min_6, div_6, mul_72
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 800
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 200
    y1 = (yindex // 200)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (y0 + (200*x2) + (39200*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pi/cpisr6tqyo2sf2sqhn2w656l5tlbfn42uyywkxn2ugtvi5z4ic7j.py
# Source Nodes: [result_11, result_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# result_11 => add_59, mul_78, mul_79, sub_23
# result_12 => add_60
triton_poi_fused__native_batch_norm_legit_no_training_add_23 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_23', 'mutated_arg_names': ['in_out_ptr0']},
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
    tmp4 = 0.001
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


# kernel path: /tmp/torchinductor_youkaichao/s7/cs7vyf6lytigz4tr6j2gr7lvp3xyebhxi3uplwgx7clj4go3bt42.py
# Source Nodes: [getattr_l__mod___features___9___block_0_1, getattr_l__mod___features___9___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# getattr_l__mod___features___9___block_0_1 => add_62, mul_81, mul_82, sub_24
# getattr_l__mod___features___9___block_0_2 => add_63, clamp_max_8, clamp_min_8, div_8, mul_83
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 736
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 184
    y1 = (yindex // 184)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (y0 + (184*x2) + (36064*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v6/cv6lmjd2oaxgmfjjhdmpet226v4qy55brk2ub4fvnfruie4blzdb.py
# Source Nodes: [getattr_l__mod___features___11___block_0_1, getattr_l__mod___features___11___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# getattr_l__mod___features___11___block_0_1 => add_80, mul_103, mul_104, sub_30
# getattr_l__mod___features___11___block_0_2 => add_81, clamp_max_12, clamp_min_12, div_12, mul_105
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (y0 + (480*x2) + (94080*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jv/cjv7t3rwwy6sit4idlkqpv5v37hsrmutubiancsjusfmq2nfjjxy.py
# Source Nodes: [getattr_l__mod___features___11___block_1_1, getattr_l__mod___features___11___block_1_2, scale_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
# getattr_l__mod___features___11___block_1_1 => add_83, mul_107, mul_108, sub_31
# getattr_l__mod___features___11___block_1_2 => add_84, clamp_max_13, clamp_min_13, div_13, mul_109
# scale_15 => mean_3
triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_26', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 480
    tmp0 = tl.load(in_out_ptr0 + (r2 + (196*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = 196.0
    tmp28 = tmp26 / tmp27
    tl.store(in_out_ptr0 + (r2 + (196*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ch/cchjhxvvujksqtla7rucbusjulxlxtmmfoba2evvvprohpg5orpn.py
# Source Nodes: [getattr_l__mod___features___11___block_1_2, scale_15, scale_16, scale_17], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
# getattr_l__mod___features___11___block_1_2 => add_84, clamp_max_13, clamp_min_13, div_13, mul_109
# scale_15 => mean_3
# scale_16 => convolution_38
# scale_17 => relu_14
triton_poi_fused_convolution_hardswish_mean_relu_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_relu_27', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 480
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 120
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/77/c77nzsiks5zk3v563uq6wgo4vvta45qepfinkdwtvc5nacrxf5ah.py
# Source Nodes: [getattr_l__mod___features___11___block_1_2, mul_3, scale_15, scale_16, scale_17, scale_18, scale_19], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
# getattr_l__mod___features___11___block_1_2 => add_84, clamp_max_13, clamp_min_13, div_13, mul_109
# mul_3 => mul_110
# scale_15 => mean_3
# scale_16 => convolution_38
# scale_17 => relu_14
# scale_18 => convolution_39
# scale_19 => add_85, clamp_max_14, clamp_min_14, div_14
triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_28 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y3 = yindex
    y0 = yindex % 480
    x2 = xindex
    y1 = (yindex // 480)
    tmp0 = tl.load(in_ptr0 + (y3), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 / tmp7
    tmp11 = tmp10 + tmp3
    tmp12 = triton_helpers.maximum(tmp11, tmp5)
    tmp13 = triton_helpers.minimum(tmp12, tmp7)
    tmp14 = tmp10 * tmp13
    tmp15 = tmp14 / tmp7
    tmp16 = tmp9 * tmp15
    tl.store(out_ptr0 + (y0 + (480*x2) + (94080*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bn/cbndaxxhowlzolslyhhmx4g5nx6r46x2ru3scdunga3ijnlfgdgk.py
# Source Nodes: [result_17], Original ATen: [aten._native_batch_norm_legit_no_training]
# result_17 => add_87, mul_112, mul_113, sub_32
triton_poi_fused__native_batch_norm_legit_no_training_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 448
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 112
    y1 = (yindex // 112)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (112*x2) + (21952*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bu/cbuso7ga3kbzj435tcfqbwokdghd7jp6hkrz57s5u4z4zufsz23k.py
# Source Nodes: [getattr_l__mod___features___12___block_0_1, getattr_l__mod___features___12___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# getattr_l__mod___features___12___block_0_1 => add_89, mul_115, mul_116, sub_33
# getattr_l__mod___features___12___block_0_2 => add_90, clamp_max_15, clamp_min_15, div_15, mul_117
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_30 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_30', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2688
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 672
    y1 = (yindex // 672)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (y0 + (672*x2) + (131712*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nb/cnbepb7vn33ptnwhely7p43gsecug4wh7hiiuxbjlmxvoqp7cxoo.py
# Source Nodes: [getattr_l__mod___features___12___block_1_1, getattr_l__mod___features___12___block_1_2, scale_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
# getattr_l__mod___features___12___block_1_1 => add_92, mul_119, mul_120, sub_34
# getattr_l__mod___features___12___block_1_2 => add_93, clamp_max_16, clamp_min_16, div_16, mul_121
# scale_20 => mean_4
triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[4096, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_31', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2688
    rnumel = 196
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 672
    tmp0 = tl.load(in_out_ptr0 + (r2 + (196*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = 196.0
    tmp28 = tmp26 / tmp27
    tl.store(in_out_ptr0 + (r2 + (196*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/74/c74pzhbb7eri2piivppurxadvgybl6rjs6r2lzjqkrpcgcyizojo.py
# Source Nodes: [getattr_l__mod___features___12___block_1_2, scale_20, scale_21, scale_22], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
# getattr_l__mod___features___12___block_1_2 => add_93, clamp_max_16, clamp_min_16, div_16, mul_121
# scale_20 => mean_4
# scale_21 => convolution_43
# scale_22 => relu_15
triton_poi_fused_convolution_hardswish_mean_relu_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_relu_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 672
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 168
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7h/c7hs6i4dbfztc7t6ymamse3o2aiu7zt4yspakrlwx7lihkl2ysga.py
# Source Nodes: [getattr_l__mod___features___12___block_1_2, mul_4, scale_20, scale_21, scale_22, scale_23, scale_24], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
# getattr_l__mod___features___12___block_1_2 => add_93, clamp_max_16, clamp_min_16, div_16, mul_121
# mul_4 => mul_122
# scale_20 => mean_4
# scale_21 => convolution_43
# scale_22 => relu_15
# scale_23 => convolution_44
# scale_24 => add_94, clamp_max_17, clamp_min_17, div_17
triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_33 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2688
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y3 = yindex
    y0 = yindex % 672
    x2 = xindex
    y1 = (yindex // 672)
    tmp0 = tl.load(in_ptr0 + (y3), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 / tmp7
    tmp11 = tmp10 + tmp3
    tmp12 = triton_helpers.maximum(tmp11, tmp5)
    tmp13 = triton_helpers.minimum(tmp12, tmp7)
    tmp14 = tmp10 * tmp13
    tmp15 = tmp14 / tmp7
    tmp16 = tmp9 * tmp15
    tl.store(out_ptr0 + (y0 + (672*x2) + (131712*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/25/c2532osh34sh7l7xemhi3lv254davrdfcewdotkqfktjnhiwkdrk.py
# Source Nodes: [result_18, result_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# result_18 => add_96, mul_124, mul_125, sub_35
# result_19 => add_97
triton_poi_fused__native_batch_norm_legit_no_training_add_34 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_34', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 784
    xnumel = 112
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
    tmp0 = tl.load(in_ptr0 + (y0 + (196*x2) + (21952*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (112*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (112*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jf/cjf3tmulvn33455rhlhe3xj3nvlqk2zd2vrauf75vb4bomqnzmhx.py
# Source Nodes: [getattr_l__mod___features___13___block_1_1, getattr_l__mod___features___13___block_1_2, scale_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
# getattr_l__mod___features___13___block_1_1 => add_102, mul_131, mul_132, sub_37
# getattr_l__mod___features___13___block_1_2 => add_103, clamp_max_19, clamp_min_19, div_19, mul_133
# scale_25 => mean_5
triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_35', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2688
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 672
    tmp0 = tl.load(in_out_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = 49.0
    tmp28 = tmp26 / tmp27
    tl.store(in_out_ptr0 + (r2 + (49*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mz/cmz3emgp7clbyhctkjnudmmfiko4iuam4qh77sandbozmab7lwrt.py
# Source Nodes: [getattr_l__mod___features___13___block_1_2, mul_5, scale_25, scale_26, scale_27, scale_28, scale_29], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
# getattr_l__mod___features___13___block_1_2 => add_103, clamp_max_19, clamp_min_19, div_19, mul_133
# mul_5 => mul_134
# scale_25 => mean_5
# scale_26 => convolution_48
# scale_27 => relu_16
# scale_28 => convolution_49
# scale_29 => add_104, clamp_max_20, clamp_min_20, div_20
triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2688
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y3 = yindex
    y0 = yindex % 672
    x2 = xindex
    y1 = (yindex // 672)
    tmp0 = tl.load(in_ptr0 + (y3), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 / tmp7
    tmp11 = tmp10 + tmp3
    tmp12 = triton_helpers.maximum(tmp11, tmp5)
    tmp13 = triton_helpers.minimum(tmp12, tmp7)
    tmp14 = tmp10 * tmp13
    tmp15 = tmp14 / tmp7
    tmp16 = tmp9 * tmp15
    tl.store(out_ptr0 + (y0 + (672*x2) + (32928*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u4/cu4klfn72i63el3wb7ybqpfvq43wj4nbuuxnpn4imsigefz4lhd7.py
# Source Nodes: [result_20], Original ATen: [aten._native_batch_norm_legit_no_training]
# result_20 => add_106, mul_136, mul_137, sub_38
triton_poi_fused__native_batch_norm_legit_no_training_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 640
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 160
    y1 = (yindex // 160)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (160*x2) + (7840*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3w/c3wxjxfkkbgyjrethpxr23nc72xeejaxpcqebd47lxfegi35z6kn.py
# Source Nodes: [getattr_l__mod___features___14___block_0_1, getattr_l__mod___features___14___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# getattr_l__mod___features___14___block_0_1 => add_108, mul_139, mul_140, sub_39
# getattr_l__mod___features___14___block_0_2 => add_109, clamp_max_21, clamp_min_21, div_21, mul_141
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3840
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 960
    y1 = (yindex // 960)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tl.store(out_ptr0 + (y0 + (960*x2) + (47040*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4o/c4oad7llh43llrvr3xbse6uhccs5rjtzwifi7yln4hk2citp736x.py
# Source Nodes: [getattr_l__mod___features___14___block_1_1, getattr_l__mod___features___14___block_1_2, scale_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
# getattr_l__mod___features___14___block_1_1 => add_111, mul_143, mul_144, sub_40
# getattr_l__mod___features___14___block_1_2 => add_112, clamp_max_22, clamp_min_22, div_22, mul_145
# scale_30 => mean_6
triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_39', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 960
    tmp0 = tl.load(in_out_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = 49.0
    tmp28 = tmp26 / tmp27
    tl.store(in_out_ptr0 + (r2 + (49*x3)), tmp14, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3d/c3dtuehhnqtox4jc3sam275g374bc4q7ikabtllddg6mhbnlv77y.py
# Source Nodes: [getattr_l__mod___features___14___block_1_2, scale_30, scale_31, scale_32], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
# getattr_l__mod___features___14___block_1_2 => add_112, clamp_max_22, clamp_min_22, div_22, mul_145
# scale_30 => mean_6
# scale_31 => convolution_53
# scale_32 => relu_17
triton_poi_fused_convolution_hardswish_mean_relu_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_relu_40', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 960
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 240
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ja/cjaigwfqvoipd2q6wbxkjbj2jrmhzv4h575z3v4unchma4ak2mdg.py
# Source Nodes: [getattr_l__mod___features___14___block_1_2, mul_6, scale_30, scale_31, scale_32, scale_33, scale_34], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
# getattr_l__mod___features___14___block_1_2 => add_112, clamp_max_22, clamp_min_22, div_22, mul_145
# mul_6 => mul_146
# scale_30 => mean_6
# scale_31 => convolution_53
# scale_32 => relu_17
# scale_33 => convolution_54
# scale_34 => add_113, clamp_max_23, clamp_min_23, div_23
triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3840
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y3 = yindex
    y0 = yindex % 960
    x2 = xindex
    y1 = (yindex // 960)
    tmp0 = tl.load(in_ptr0 + (y3), ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp8 / tmp7
    tmp11 = tmp10 + tmp3
    tmp12 = triton_helpers.maximum(tmp11, tmp5)
    tmp13 = triton_helpers.minimum(tmp12, tmp7)
    tmp14 = tmp10 * tmp13
    tmp15 = tmp14 / tmp7
    tmp16 = tmp9 * tmp15
    tl.store(out_ptr0 + (y0 + (960*x2) + (47040*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/be/cbehqa6p4kjjjrd6j2amzwxr2u2jechqrworzzf6nynucnv44lci.py
# Source Nodes: [result_21, result_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# result_21 => add_115, mul_148, mul_149, sub_41
# result_22 => add_116
triton_poi_fused__native_batch_norm_legit_no_training_add_42 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_42', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 196
    xnumel = 160
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
    tmp0 = tl.load(in_ptr0 + (y0 + (49*x2) + (7840*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (x2), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x2), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x2), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x2), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_out_ptr0 + (x2 + (160*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tmp16 = tmp14 + tmp15
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x2 + (160*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lf/clfsuumbcokforvytiltdrgejuunt6a6mkarpo4dhx4cgnpbuvnv.py
# Source Nodes: [l__mod___features_16_1, x, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
# l__mod___features_16_1 => add_128, mul_163, mul_164, sub_45
# x => add_129, clamp_max_27, clamp_min_27, div_27, mul_165
# x_1 => mean_8
triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_43', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 960
    tmp0 = tl.load(in_out_ptr0 + (r2 + (49*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), xmask, eviction_policy='evict_last')
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
    tmp15 = 3.0
    tmp16 = tmp14 + tmp15
    tmp17 = 0.0
    tmp18 = triton_helpers.maximum(tmp16, tmp17)
    tmp19 = 6.0
    tmp20 = triton_helpers.minimum(tmp18, tmp19)
    tmp21 = tmp14 * tmp20
    tmp22 = tmp21 / tmp19
    tmp23 = tl.broadcast_to(tmp22, [XBLOCK, RBLOCK])
    tmp25 = tl.where(rmask & xmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = 49.0
    tmp28 = tmp26 / tmp27
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vb/cvbynm62udnauifn76nsdkajvsmm2apsq4cs4xrzwnrwp2ko64jc.py
# Source Nodes: [l__mod___classifier_1], Original ATen: [aten.hardswish]
# l__mod___classifier_1 => add_130, clamp_max_28, clamp_min_28, div_28, mul_166
triton_poi_fused_hardswish_44 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_hardswish_44', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 5120
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1280
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp2 * tmp8
    tmp10 = tmp9 / tmp7
    tl.store(in_out_ptr0 + (x2), tmp10, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (16, ), (1, ))
    assert_size_stride(arg2_1, (16, ), (1, ))
    assert_size_stride(arg3_1, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg4_1, (16, ), (1, ))
    assert_size_stride(arg5_1, (16, ), (1, ))
    assert_size_stride(arg6_1, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg7_1, (16, ), (1, ))
    assert_size_stride(arg8_1, (16, ), (1, ))
    assert_size_stride(arg9_1, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (64, ), (1, ))
    assert_size_stride(arg12_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg13_1, (64, ), (1, ))
    assert_size_stride(arg14_1, (64, ), (1, ))
    assert_size_stride(arg15_1, (24, 64, 1, 1), (64, 1, 1, 1))
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
    assert_size_stride(arg30_1, (72, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg31_1, (72, ), (1, ))
    assert_size_stride(arg32_1, (72, ), (1, ))
    assert_size_stride(arg33_1, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg34_1, (24, ), (1, ))
    assert_size_stride(arg35_1, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg36_1, (72, ), (1, ))
    assert_size_stride(arg37_1, (40, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg38_1, (40, ), (1, ))
    assert_size_stride(arg39_1, (40, ), (1, ))
    assert_size_stride(arg40_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg41_1, (120, ), (1, ))
    assert_size_stride(arg42_1, (120, ), (1, ))
    assert_size_stride(arg43_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg44_1, (120, ), (1, ))
    assert_size_stride(arg45_1, (120, ), (1, ))
    assert_size_stride(arg46_1, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg47_1, (32, ), (1, ))
    assert_size_stride(arg48_1, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg49_1, (120, ), (1, ))
    assert_size_stride(arg50_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg51_1, (40, ), (1, ))
    assert_size_stride(arg52_1, (40, ), (1, ))
    assert_size_stride(arg53_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg54_1, (120, ), (1, ))
    assert_size_stride(arg55_1, (120, ), (1, ))
    assert_size_stride(arg56_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg57_1, (120, ), (1, ))
    assert_size_stride(arg58_1, (120, ), (1, ))
    assert_size_stride(arg59_1, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg60_1, (32, ), (1, ))
    assert_size_stride(arg61_1, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg62_1, (120, ), (1, ))
    assert_size_stride(arg63_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg64_1, (40, ), (1, ))
    assert_size_stride(arg65_1, (40, ), (1, ))
    assert_size_stride(arg66_1, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg67_1, (240, ), (1, ))
    assert_size_stride(arg68_1, (240, ), (1, ))
    assert_size_stride(arg69_1, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg70_1, (240, ), (1, ))
    assert_size_stride(arg71_1, (240, ), (1, ))
    assert_size_stride(arg72_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg73_1, (80, ), (1, ))
    assert_size_stride(arg74_1, (80, ), (1, ))
    assert_size_stride(arg75_1, (200, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg76_1, (200, ), (1, ))
    assert_size_stride(arg77_1, (200, ), (1, ))
    assert_size_stride(arg78_1, (200, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg79_1, (200, ), (1, ))
    assert_size_stride(arg80_1, (200, ), (1, ))
    assert_size_stride(arg81_1, (80, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(arg82_1, (80, ), (1, ))
    assert_size_stride(arg83_1, (80, ), (1, ))
    assert_size_stride(arg84_1, (184, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg85_1, (184, ), (1, ))
    assert_size_stride(arg86_1, (184, ), (1, ))
    assert_size_stride(arg87_1, (184, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg88_1, (184, ), (1, ))
    assert_size_stride(arg89_1, (184, ), (1, ))
    assert_size_stride(arg90_1, (80, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg91_1, (80, ), (1, ))
    assert_size_stride(arg92_1, (80, ), (1, ))
    assert_size_stride(arg93_1, (184, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg94_1, (184, ), (1, ))
    assert_size_stride(arg95_1, (184, ), (1, ))
    assert_size_stride(arg96_1, (184, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg97_1, (184, ), (1, ))
    assert_size_stride(arg98_1, (184, ), (1, ))
    assert_size_stride(arg99_1, (80, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg100_1, (80, ), (1, ))
    assert_size_stride(arg101_1, (80, ), (1, ))
    assert_size_stride(arg102_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg103_1, (480, ), (1, ))
    assert_size_stride(arg104_1, (480, ), (1, ))
    assert_size_stride(arg105_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg106_1, (480, ), (1, ))
    assert_size_stride(arg107_1, (480, ), (1, ))
    assert_size_stride(arg108_1, (120, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg109_1, (120, ), (1, ))
    assert_size_stride(arg110_1, (480, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg111_1, (480, ), (1, ))
    assert_size_stride(arg112_1, (112, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg113_1, (112, ), (1, ))
    assert_size_stride(arg114_1, (112, ), (1, ))
    assert_size_stride(arg115_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg116_1, (672, ), (1, ))
    assert_size_stride(arg117_1, (672, ), (1, ))
    assert_size_stride(arg118_1, (672, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg119_1, (672, ), (1, ))
    assert_size_stride(arg120_1, (672, ), (1, ))
    assert_size_stride(arg121_1, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg122_1, (168, ), (1, ))
    assert_size_stride(arg123_1, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg124_1, (672, ), (1, ))
    assert_size_stride(arg125_1, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg126_1, (112, ), (1, ))
    assert_size_stride(arg127_1, (112, ), (1, ))
    assert_size_stride(arg128_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg129_1, (672, ), (1, ))
    assert_size_stride(arg130_1, (672, ), (1, ))
    assert_size_stride(arg131_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg132_1, (672, ), (1, ))
    assert_size_stride(arg133_1, (672, ), (1, ))
    assert_size_stride(arg134_1, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg135_1, (168, ), (1, ))
    assert_size_stride(arg136_1, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg137_1, (672, ), (1, ))
    assert_size_stride(arg138_1, (160, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg139_1, (160, ), (1, ))
    assert_size_stride(arg140_1, (160, ), (1, ))
    assert_size_stride(arg141_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg142_1, (960, ), (1, ))
    assert_size_stride(arg143_1, (960, ), (1, ))
    assert_size_stride(arg144_1, (960, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg145_1, (960, ), (1, ))
    assert_size_stride(arg146_1, (960, ), (1, ))
    assert_size_stride(arg147_1, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg148_1, (240, ), (1, ))
    assert_size_stride(arg149_1, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg150_1, (960, ), (1, ))
    assert_size_stride(arg151_1, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg152_1, (160, ), (1, ))
    assert_size_stride(arg153_1, (160, ), (1, ))
    assert_size_stride(arg154_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg155_1, (960, ), (1, ))
    assert_size_stride(arg156_1, (960, ), (1, ))
    assert_size_stride(arg157_1, (960, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg158_1, (960, ), (1, ))
    assert_size_stride(arg159_1, (960, ), (1, ))
    assert_size_stride(arg160_1, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg161_1, (240, ), (1, ))
    assert_size_stride(arg162_1, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg163_1, (960, ), (1, ))
    assert_size_stride(arg164_1, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg165_1, (160, ), (1, ))
    assert_size_stride(arg166_1, (160, ), (1, ))
    assert_size_stride(arg167_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg168_1, (960, ), (1, ))
    assert_size_stride(arg169_1, (960, ), (1, ))
    assert_size_stride(arg170_1, (1280, 960), (960, 1))
    assert_size_stride(arg171_1, (1280, ), (1, ))
    assert_size_stride(arg172_1, (1000, 1280), (1280, 1))
    assert_size_stride(arg173_1, (1000, ), (1, ))
    assert_size_stride(arg174_1, (16, ), (1, ))
    assert_size_stride(arg175_1, (16, ), (1, ))
    assert_size_stride(arg176_1, (), ())
    assert_size_stride(arg177_1, (16, ), (1, ))
    assert_size_stride(arg178_1, (16, ), (1, ))
    assert_size_stride(arg179_1, (), ())
    assert_size_stride(arg180_1, (16, ), (1, ))
    assert_size_stride(arg181_1, (16, ), (1, ))
    assert_size_stride(arg182_1, (), ())
    assert_size_stride(arg183_1, (64, ), (1, ))
    assert_size_stride(arg184_1, (64, ), (1, ))
    assert_size_stride(arg185_1, (), ())
    assert_size_stride(arg186_1, (64, ), (1, ))
    assert_size_stride(arg187_1, (64, ), (1, ))
    assert_size_stride(arg188_1, (), ())
    assert_size_stride(arg189_1, (24, ), (1, ))
    assert_size_stride(arg190_1, (24, ), (1, ))
    assert_size_stride(arg191_1, (), ())
    assert_size_stride(arg192_1, (72, ), (1, ))
    assert_size_stride(arg193_1, (72, ), (1, ))
    assert_size_stride(arg194_1, (), ())
    assert_size_stride(arg195_1, (72, ), (1, ))
    assert_size_stride(arg196_1, (72, ), (1, ))
    assert_size_stride(arg197_1, (), ())
    assert_size_stride(arg198_1, (24, ), (1, ))
    assert_size_stride(arg199_1, (24, ), (1, ))
    assert_size_stride(arg200_1, (), ())
    assert_size_stride(arg201_1, (72, ), (1, ))
    assert_size_stride(arg202_1, (72, ), (1, ))
    assert_size_stride(arg203_1, (), ())
    assert_size_stride(arg204_1, (72, ), (1, ))
    assert_size_stride(arg205_1, (72, ), (1, ))
    assert_size_stride(arg206_1, (), ())
    assert_size_stride(arg207_1, (40, ), (1, ))
    assert_size_stride(arg208_1, (40, ), (1, ))
    assert_size_stride(arg209_1, (), ())
    assert_size_stride(arg210_1, (120, ), (1, ))
    assert_size_stride(arg211_1, (120, ), (1, ))
    assert_size_stride(arg212_1, (), ())
    assert_size_stride(arg213_1, (120, ), (1, ))
    assert_size_stride(arg214_1, (120, ), (1, ))
    assert_size_stride(arg215_1, (), ())
    assert_size_stride(arg216_1, (40, ), (1, ))
    assert_size_stride(arg217_1, (40, ), (1, ))
    assert_size_stride(arg218_1, (), ())
    assert_size_stride(arg219_1, (120, ), (1, ))
    assert_size_stride(arg220_1, (120, ), (1, ))
    assert_size_stride(arg221_1, (), ())
    assert_size_stride(arg222_1, (120, ), (1, ))
    assert_size_stride(arg223_1, (120, ), (1, ))
    assert_size_stride(arg224_1, (), ())
    assert_size_stride(arg225_1, (40, ), (1, ))
    assert_size_stride(arg226_1, (40, ), (1, ))
    assert_size_stride(arg227_1, (), ())
    assert_size_stride(arg228_1, (240, ), (1, ))
    assert_size_stride(arg229_1, (240, ), (1, ))
    assert_size_stride(arg230_1, (), ())
    assert_size_stride(arg231_1, (240, ), (1, ))
    assert_size_stride(arg232_1, (240, ), (1, ))
    assert_size_stride(arg233_1, (), ())
    assert_size_stride(arg234_1, (80, ), (1, ))
    assert_size_stride(arg235_1, (80, ), (1, ))
    assert_size_stride(arg236_1, (), ())
    assert_size_stride(arg237_1, (200, ), (1, ))
    assert_size_stride(arg238_1, (200, ), (1, ))
    assert_size_stride(arg239_1, (), ())
    assert_size_stride(arg240_1, (200, ), (1, ))
    assert_size_stride(arg241_1, (200, ), (1, ))
    assert_size_stride(arg242_1, (), ())
    assert_size_stride(arg243_1, (80, ), (1, ))
    assert_size_stride(arg244_1, (80, ), (1, ))
    assert_size_stride(arg245_1, (), ())
    assert_size_stride(arg246_1, (184, ), (1, ))
    assert_size_stride(arg247_1, (184, ), (1, ))
    assert_size_stride(arg248_1, (), ())
    assert_size_stride(arg249_1, (184, ), (1, ))
    assert_size_stride(arg250_1, (184, ), (1, ))
    assert_size_stride(arg251_1, (), ())
    assert_size_stride(arg252_1, (80, ), (1, ))
    assert_size_stride(arg253_1, (80, ), (1, ))
    assert_size_stride(arg254_1, (), ())
    assert_size_stride(arg255_1, (184, ), (1, ))
    assert_size_stride(arg256_1, (184, ), (1, ))
    assert_size_stride(arg257_1, (), ())
    assert_size_stride(arg258_1, (184, ), (1, ))
    assert_size_stride(arg259_1, (184, ), (1, ))
    assert_size_stride(arg260_1, (), ())
    assert_size_stride(arg261_1, (80, ), (1, ))
    assert_size_stride(arg262_1, (80, ), (1, ))
    assert_size_stride(arg263_1, (), ())
    assert_size_stride(arg264_1, (480, ), (1, ))
    assert_size_stride(arg265_1, (480, ), (1, ))
    assert_size_stride(arg266_1, (), ())
    assert_size_stride(arg267_1, (480, ), (1, ))
    assert_size_stride(arg268_1, (480, ), (1, ))
    assert_size_stride(arg269_1, (), ())
    assert_size_stride(arg270_1, (112, ), (1, ))
    assert_size_stride(arg271_1, (112, ), (1, ))
    assert_size_stride(arg272_1, (), ())
    assert_size_stride(arg273_1, (672, ), (1, ))
    assert_size_stride(arg274_1, (672, ), (1, ))
    assert_size_stride(arg275_1, (), ())
    assert_size_stride(arg276_1, (672, ), (1, ))
    assert_size_stride(arg277_1, (672, ), (1, ))
    assert_size_stride(arg278_1, (), ())
    assert_size_stride(arg279_1, (112, ), (1, ))
    assert_size_stride(arg280_1, (112, ), (1, ))
    assert_size_stride(arg281_1, (), ())
    assert_size_stride(arg282_1, (672, ), (1, ))
    assert_size_stride(arg283_1, (672, ), (1, ))
    assert_size_stride(arg284_1, (), ())
    assert_size_stride(arg285_1, (672, ), (1, ))
    assert_size_stride(arg286_1, (672, ), (1, ))
    assert_size_stride(arg287_1, (), ())
    assert_size_stride(arg288_1, (160, ), (1, ))
    assert_size_stride(arg289_1, (160, ), (1, ))
    assert_size_stride(arg290_1, (), ())
    assert_size_stride(arg291_1, (960, ), (1, ))
    assert_size_stride(arg292_1, (960, ), (1, ))
    assert_size_stride(arg293_1, (), ())
    assert_size_stride(arg294_1, (960, ), (1, ))
    assert_size_stride(arg295_1, (960, ), (1, ))
    assert_size_stride(arg296_1, (), ())
    assert_size_stride(arg297_1, (160, ), (1, ))
    assert_size_stride(arg298_1, (160, ), (1, ))
    assert_size_stride(arg299_1, (), ())
    assert_size_stride(arg300_1, (960, ), (1, ))
    assert_size_stride(arg301_1, (960, ), (1, ))
    assert_size_stride(arg302_1, (), ())
    assert_size_stride(arg303_1, (960, ), (1, ))
    assert_size_stride(arg304_1, (960, ), (1, ))
    assert_size_stride(arg305_1, (), ())
    assert_size_stride(arg306_1, (160, ), (1, ))
    assert_size_stride(arg307_1, (160, ), (1, ))
    assert_size_stride(arg308_1, (), ())
    assert_size_stride(arg309_1, (960, ), (1, ))
    assert_size_stride(arg310_1, (960, ), (1, ))
    assert_size_stride(arg311_1, (), ())
    assert_size_stride(arg312_1, (4, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0_0], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg312_1, buf0, 12, 50176, grid=grid(12, 50176), stream=stream0)
        del arg312_1
        buf1 = empty_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 48, 9, grid=grid(48, 9), stream=stream0)
        del arg0_1
        # Source Nodes: [l__mod___features_0_0], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 16, 112, 112), (200704, 12544, 112, 1))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        buf4 = empty_strided((4, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0_1, l__mod___features_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2.run(buf3, arg174_1, arg175_1, arg1_1, arg2_1, buf4, 64, 12544, grid=grid(64, 12544), stream=stream0)
        del arg174_1
        del arg175_1
        del arg1_1
        del arg2_1
        # Source Nodes: [getattr_l__mod___features___1___block_0_0, l__mod___features_0_2], Original ATen: [aten.convolution, aten.hardswish]
        buf5 = extern_kernels.convolution(buf4, arg3_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf5, (4, 16, 112, 112), (200704, 12544, 112, 1))
        del arg3_1
        buf6 = reinterpret_tensor(buf3, (4, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf3  # reuse
        # Source Nodes: [getattr_l__mod___features___1___block_0_1, getattr_l__mod___features___1___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_3.run(buf5, arg177_1, arg178_1, arg4_1, arg5_1, buf6, 64, 12544, grid=grid(64, 12544), stream=stream0)
        del arg177_1
        del arg178_1
        del arg4_1
        del arg5_1
        del buf5
        # Source Nodes: [getattr_l__mod___features___1___block_0_1, getattr_l__mod___features___1___block_0_2, getattr_l__mod___features___1___block_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf7 = extern_kernels.convolution(buf6, arg6_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 16, 112, 112), (200704, 12544, 112, 1))
        del arg6_1
        del buf6
        buf8 = buf4; del buf4  # reuse
        # Source Nodes: [result, result_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_4.run(buf8, buf7, arg180_1, arg181_1, arg7_1, arg8_1, 50176, 16, grid=grid(50176, 16), stream=stream0)
        del arg180_1
        del arg181_1
        del arg7_1
        del arg8_1
        del buf7
        # Source Nodes: [getattr_l__mod___features___2___block_0_0, result, result_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf9 = extern_kernels.convolution(buf8, arg9_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 64, 112, 112), (802816, 12544, 112, 1))
        del arg9_1
        buf10 = empty_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___2___block_0_1, getattr_l__mod___features___2___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf9, arg183_1, arg184_1, arg10_1, arg11_1, buf10, 256, 12544, grid=grid(256, 12544), stream=stream0)
        del arg10_1
        del arg11_1
        del arg183_1
        del arg184_1
        del buf9
        # Source Nodes: [getattr_l__mod___features___2___block_0_1, getattr_l__mod___features___2___block_0_2, getattr_l__mod___features___2___block_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf11 = extern_kernels.convolution(buf10, arg12_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf11, (4, 64, 56, 56), (200704, 3136, 56, 1))
        del arg12_1
        del buf10
        buf12 = reinterpret_tensor(buf8, (4, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf8  # reuse
        # Source Nodes: [getattr_l__mod___features___2___block_1_1, getattr_l__mod___features___2___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf11, arg186_1, arg187_1, arg13_1, arg14_1, buf12, 256, 3136, grid=grid(256, 3136), stream=stream0)
        del arg13_1
        del arg14_1
        del arg186_1
        del arg187_1
        del buf11
        # Source Nodes: [getattr_l__mod___features___2___block_1_1, getattr_l__mod___features___2___block_1_2, getattr_l__mod___features___2___block_2_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf13 = extern_kernels.convolution(buf12, arg15_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 24, 56, 56), (75264, 3136, 56, 1))
        del arg15_1
        del buf12
        buf14 = empty_strided((4, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [result_2], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_7.run(buf13, arg189_1, arg190_1, arg16_1, arg17_1, buf14, 96, 3136, grid=grid(96, 3136), stream=stream0)
        del arg16_1
        del arg17_1
        del arg189_1
        del arg190_1
        del buf13
        # Source Nodes: [getattr_l__mod___features___3___block_0_0], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, arg18_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 72, 56, 56), (225792, 3136, 56, 1))
        del arg18_1
        buf16 = empty_strided((4, 72, 56, 56), (225792, 1, 4032, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___3___block_0_1, getattr_l__mod___features___3___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf15, arg192_1, arg193_1, arg19_1, arg20_1, buf16, 288, 3136, grid=grid(288, 3136), stream=stream0)
        del arg192_1
        del arg193_1
        del arg19_1
        del arg20_1
        del buf15
        # Source Nodes: [getattr_l__mod___features___3___block_0_1, getattr_l__mod___features___3___block_0_2, getattr_l__mod___features___3___block_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf17 = extern_kernels.convolution(buf16, arg21_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf17, (4, 72, 56, 56), (225792, 3136, 56, 1))
        del arg21_1
        buf18 = buf16; del buf16  # reuse
        # Source Nodes: [getattr_l__mod___features___3___block_1_1, getattr_l__mod___features___3___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf17, arg195_1, arg196_1, arg22_1, arg23_1, buf18, 288, 3136, grid=grid(288, 3136), stream=stream0)
        del arg195_1
        del arg196_1
        del arg22_1
        del arg23_1
        del buf17
        # Source Nodes: [getattr_l__mod___features___3___block_1_1, getattr_l__mod___features___3___block_1_2, getattr_l__mod___features___3___block_2_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf19 = extern_kernels.convolution(buf18, arg24_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 24, 56, 56), (75264, 3136, 56, 1))
        del arg24_1
        buf20 = buf14; del buf14  # reuse
        # Source Nodes: [result_3, result_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_9.run(buf20, buf19, arg198_1, arg199_1, arg25_1, arg26_1, 12544, 24, grid=grid(12544, 24), stream=stream0)
        del arg198_1
        del arg199_1
        del arg25_1
        del arg26_1
        del buf19
        # Source Nodes: [getattr_l__mod___features___4___block_0_0, result_3, result_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf21 = extern_kernels.convolution(buf20, arg27_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 72, 56, 56), (225792, 3136, 56, 1))
        del arg27_1
        del buf20
        buf22 = buf18; del buf18  # reuse
        # Source Nodes: [getattr_l__mod___features___4___block_0_1, getattr_l__mod___features___4___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf21, arg201_1, arg202_1, arg28_1, arg29_1, buf22, 288, 3136, grid=grid(288, 3136), stream=stream0)
        del arg201_1
        del arg202_1
        del arg28_1
        del arg29_1
        del buf21
        # Source Nodes: [getattr_l__mod___features___4___block_0_1, getattr_l__mod___features___4___block_0_2, getattr_l__mod___features___4___block_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf23 = extern_kernels.convolution(buf22, arg30_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf23, (4, 72, 28, 28), (56448, 784, 28, 1))
        del arg30_1
        del buf22
        buf24 = buf23; del buf23  # reuse
        buf25 = empty_strided((4, 72, 1, 1), (72, 1, 288, 288), device='cuda', dtype=torch.float32)
        buf26 = reinterpret_tensor(buf25, (4, 72, 1, 1), (72, 1, 72, 72), 0); del buf25  # reuse
        # Source Nodes: [getattr_l__mod___features___4___block_1_1, getattr_l__mod___features___4___block_1_2, scale], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_10.run(buf24, buf26, arg204_1, arg205_1, arg31_1, arg32_1, 288, 784, grid=grid(288), stream=stream0)
        del arg204_1
        del arg205_1
        del arg31_1
        del arg32_1
        # Source Nodes: [scale, scale_1], Original ATen: [aten.convolution, aten.mean]
        buf27 = extern_kernels.convolution(buf26, arg33_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (4, 24, 1, 1), (24, 1, 1, 1))
        del arg33_1
        del buf26
        buf28 = reinterpret_tensor(buf27, (4, 24, 1, 1), (24, 1, 24, 24), 0); del buf27  # reuse
        # Source Nodes: [scale, scale_1, scale_2], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_11.run(buf28, arg34_1, 96, grid=grid(96), stream=stream0)
        del arg34_1
        # Source Nodes: [scale, scale_1, scale_2, scale_3], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf29 = extern_kernels.convolution(buf28, arg35_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 72, 1, 1), (72, 1, 1, 1))
        del arg35_1
        del buf28
        buf30 = empty_strided((4, 72, 28, 28), (56448, 1, 2016, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [mul, scale, scale_1, scale_2, scale_3, scale_4], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_12.run(buf29, arg36_1, buf24, buf30, 288, 784, grid=grid(288, 784), stream=stream0)
        del arg36_1
        del buf24
        del buf29
        # Source Nodes: [getattr_l__mod___features___4___block_3_0, mul, scale, scale_1, scale_2, scale_3, scale_4], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
        buf31 = extern_kernels.convolution(buf30, arg37_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 40, 28, 28), (31360, 784, 28, 1))
        del arg37_1
        del buf30
        buf32 = empty_strided((4, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [result_5], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_13.run(buf31, arg207_1, arg208_1, arg38_1, arg39_1, buf32, 160, 784, grid=grid(160, 784), stream=stream0)
        del arg207_1
        del arg208_1
        del arg38_1
        del arg39_1
        del buf31
        # Source Nodes: [getattr_l__mod___features___5___block_0_0], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, arg40_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 120, 28, 28), (94080, 784, 28, 1))
        del arg40_1
        buf34 = empty_strided((4, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___5___block_0_1, getattr_l__mod___features___5___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf33, arg210_1, arg211_1, arg41_1, arg42_1, buf34, 480, 784, grid=grid(480, 784), stream=stream0)
        del arg210_1
        del arg211_1
        del arg41_1
        del arg42_1
        del buf33
        # Source Nodes: [getattr_l__mod___features___5___block_0_1, getattr_l__mod___features___5___block_0_2, getattr_l__mod___features___5___block_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf35 = extern_kernels.convolution(buf34, arg43_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf35, (4, 120, 28, 28), (94080, 784, 28, 1))
        del arg43_1
        buf36 = buf35; del buf35  # reuse
        buf37 = empty_strided((4, 120, 1, 1), (120, 1, 480, 480), device='cuda', dtype=torch.float32)
        buf38 = reinterpret_tensor(buf37, (4, 120, 1, 1), (120, 1, 120, 120), 0); del buf37  # reuse
        # Source Nodes: [getattr_l__mod___features___5___block_1_1, getattr_l__mod___features___5___block_1_2, scale_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_15.run(buf36, buf38, arg213_1, arg214_1, arg44_1, arg45_1, 480, 784, grid=grid(480), stream=stream0)
        del arg213_1
        del arg214_1
        del arg44_1
        del arg45_1
        # Source Nodes: [scale_5, scale_6], Original ATen: [aten.convolution, aten.mean]
        buf39 = extern_kernels.convolution(buf38, arg46_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 32, 1, 1), (32, 1, 1, 1))
        del arg46_1
        del buf38
        buf40 = reinterpret_tensor(buf39, (4, 32, 1, 1), (32, 1, 32, 32), 0); del buf39  # reuse
        # Source Nodes: [scale_5, scale_6, scale_7], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_16.run(buf40, arg47_1, 128, grid=grid(128), stream=stream0)
        del arg47_1
        # Source Nodes: [scale_5, scale_6, scale_7, scale_8], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf41 = extern_kernels.convolution(buf40, arg48_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 120, 1, 1), (120, 1, 1, 1))
        del arg48_1
        del buf40
        buf42 = buf34; del buf34  # reuse
        # Source Nodes: [mul_1, scale_5, scale_6, scale_7, scale_8, scale_9], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_17.run(buf41, arg49_1, buf36, buf42, 480, 784, grid=grid(480, 784), stream=stream0)
        del arg49_1
        del buf36
        # Source Nodes: [getattr_l__mod___features___5___block_3_0, mul_1, scale_5, scale_6, scale_7, scale_8, scale_9], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
        buf43 = extern_kernels.convolution(buf42, arg50_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 40, 28, 28), (31360, 784, 28, 1))
        del arg50_1
        buf44 = buf32; del buf32  # reuse
        # Source Nodes: [result_6, result_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_18.run(buf44, buf43, arg216_1, arg217_1, arg51_1, arg52_1, 3136, 40, grid=grid(3136, 40), stream=stream0)
        del arg216_1
        del arg217_1
        del arg51_1
        del arg52_1
        del buf43
        # Source Nodes: [getattr_l__mod___features___6___block_0_0], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, arg53_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (4, 120, 28, 28), (94080, 784, 28, 1))
        del arg53_1
        buf46 = buf42; del buf42  # reuse
        # Source Nodes: [getattr_l__mod___features___6___block_0_1, getattr_l__mod___features___6___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf45, arg219_1, arg220_1, arg54_1, arg55_1, buf46, 480, 784, grid=grid(480, 784), stream=stream0)
        del arg219_1
        del arg220_1
        del arg54_1
        del arg55_1
        del buf45
        # Source Nodes: [getattr_l__mod___features___6___block_0_1, getattr_l__mod___features___6___block_0_2, getattr_l__mod___features___6___block_1_0], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf47 = extern_kernels.convolution(buf46, arg56_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf47, (4, 120, 28, 28), (94080, 784, 28, 1))
        del arg56_1
        buf48 = buf47; del buf47  # reuse
        buf49 = reinterpret_tensor(buf41, (4, 120, 1, 1), (120, 1, 480, 480), 0); del buf41  # reuse
        buf50 = reinterpret_tensor(buf49, (4, 120, 1, 1), (120, 1, 120, 120), 0); del buf49  # reuse
        # Source Nodes: [getattr_l__mod___features___6___block_1_1, getattr_l__mod___features___6___block_1_2, scale_10], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_15.run(buf48, buf50, arg222_1, arg223_1, arg57_1, arg58_1, 480, 784, grid=grid(480), stream=stream0)
        del arg222_1
        del arg223_1
        del arg57_1
        del arg58_1
        # Source Nodes: [scale_10, scale_11], Original ATen: [aten.convolution, aten.mean]
        buf51 = extern_kernels.convolution(buf50, arg59_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 32, 1, 1), (32, 1, 1, 1))
        del arg59_1
        del buf50
        buf52 = reinterpret_tensor(buf51, (4, 32, 1, 1), (32, 1, 32, 32), 0); del buf51  # reuse
        # Source Nodes: [scale_10, scale_11, scale_12], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_16.run(buf52, arg60_1, 128, grid=grid(128), stream=stream0)
        del arg60_1
        # Source Nodes: [scale_10, scale_11, scale_12, scale_13], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf53 = extern_kernels.convolution(buf52, arg61_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 120, 1, 1), (120, 1, 1, 1))
        del arg61_1
        del buf52
        buf54 = buf46; del buf46  # reuse
        # Source Nodes: [mul_2, scale_10, scale_11, scale_12, scale_13, scale_14], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_17.run(buf53, arg62_1, buf48, buf54, 480, 784, grid=grid(480, 784), stream=stream0)
        del arg62_1
        del buf48
        del buf53
        # Source Nodes: [getattr_l__mod___features___6___block_3_0, mul_2, scale_10, scale_11, scale_12, scale_13, scale_14], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
        buf55 = extern_kernels.convolution(buf54, arg63_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 40, 28, 28), (31360, 784, 28, 1))
        del arg63_1
        buf56 = buf44; del buf44  # reuse
        # Source Nodes: [result_8, result_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_18.run(buf56, buf55, arg225_1, arg226_1, arg64_1, arg65_1, 3136, 40, grid=grid(3136, 40), stream=stream0)
        del arg225_1
        del arg226_1
        del arg64_1
        del arg65_1
        del buf55
        # Source Nodes: [getattr_l__mod___features___7___block_0_0, result_8, result_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf57 = extern_kernels.convolution(buf56, arg66_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (4, 240, 28, 28), (188160, 784, 28, 1))
        del arg66_1
        del buf56
        buf58 = buf57; del buf57  # reuse
        buf59 = empty_strided((4, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___7___block_0_1, getattr_l__mod___features___7___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_19.run(buf58, arg228_1, arg229_1, arg67_1, arg68_1, buf59, 960, 784, grid=grid(960, 784), stream=stream0)
        del arg228_1
        del arg229_1
        del arg67_1
        del arg68_1
        del buf58
        # Source Nodes: [getattr_l__mod___features___7___block_0_2, getattr_l__mod___features___7___block_1_0], Original ATen: [aten.convolution, aten.hardswish]
        buf60 = extern_kernels.convolution(buf59, arg69_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf60, (4, 240, 14, 14), (47040, 196, 14, 1))
        del arg69_1
        del buf59
        buf61 = buf60; del buf60  # reuse
        buf62 = empty_strided((4, 240, 14, 14), (47040, 1, 3360, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___7___block_1_1, getattr_l__mod___features___7___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_20.run(buf61, arg231_1, arg232_1, arg70_1, arg71_1, buf62, 960, 196, grid=grid(960, 196), stream=stream0)
        del arg231_1
        del arg232_1
        del arg70_1
        del arg71_1
        del buf61
        # Source Nodes: [getattr_l__mod___features___7___block_1_2, getattr_l__mod___features___7___block_2_0], Original ATen: [aten.convolution, aten.hardswish]
        buf63 = extern_kernels.convolution(buf62, arg72_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (4, 80, 14, 14), (15680, 196, 14, 1))
        del arg72_1
        buf64 = empty_strided((4, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [result_10], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_21.run(buf63, arg234_1, arg235_1, arg73_1, arg74_1, buf64, 320, 196, grid=grid(320, 196), stream=stream0)
        del arg234_1
        del arg235_1
        del arg73_1
        del arg74_1
        del buf63
        # Source Nodes: [getattr_l__mod___features___8___block_0_0], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, arg75_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (4, 200, 14, 14), (39200, 196, 14, 1))
        del arg75_1
        buf66 = buf65; del buf65  # reuse
        buf67 = empty_strided((4, 200, 14, 14), (39200, 1, 2800, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___8___block_0_1, getattr_l__mod___features___8___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22.run(buf66, arg237_1, arg238_1, arg76_1, arg77_1, buf67, 800, 196, grid=grid(800, 196), stream=stream0)
        del arg237_1
        del arg238_1
        del arg76_1
        del arg77_1
        del buf66
        # Source Nodes: [getattr_l__mod___features___8___block_0_2, getattr_l__mod___features___8___block_1_0], Original ATen: [aten.convolution, aten.hardswish]
        buf68 = extern_kernels.convolution(buf67, arg78_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=200, bias=None)
        assert_size_stride(buf68, (4, 200, 14, 14), (39200, 196, 14, 1))
        del arg78_1
        buf69 = buf68; del buf68  # reuse
        buf70 = buf67; del buf67  # reuse
        # Source Nodes: [getattr_l__mod___features___8___block_1_1, getattr_l__mod___features___8___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22.run(buf69, arg240_1, arg241_1, arg79_1, arg80_1, buf70, 800, 196, grid=grid(800, 196), stream=stream0)
        del arg240_1
        del arg241_1
        del arg79_1
        del arg80_1
        del buf69
        # Source Nodes: [getattr_l__mod___features___8___block_1_2, getattr_l__mod___features___8___block_2_0], Original ATen: [aten.convolution, aten.hardswish]
        buf71 = extern_kernels.convolution(buf70, arg81_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (4, 80, 14, 14), (15680, 196, 14, 1))
        del arg81_1
        del buf70
        buf72 = buf64; del buf64  # reuse
        # Source Nodes: [result_11, result_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_23.run(buf72, buf71, arg243_1, arg244_1, arg82_1, arg83_1, 784, 80, grid=grid(784, 80), stream=stream0)
        del arg243_1
        del arg244_1
        del arg82_1
        del arg83_1
        del buf71
        # Source Nodes: [getattr_l__mod___features___9___block_0_0], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, arg84_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (4, 184, 14, 14), (36064, 196, 14, 1))
        del arg84_1
        buf74 = buf73; del buf73  # reuse
        buf75 = empty_strided((4, 184, 14, 14), (36064, 1, 2576, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___9___block_0_1, getattr_l__mod___features___9___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_24.run(buf74, arg246_1, arg247_1, arg85_1, arg86_1, buf75, 736, 196, grid=grid(736, 196), stream=stream0)
        del arg246_1
        del arg247_1
        del arg85_1
        del arg86_1
        del buf74
        # Source Nodes: [getattr_l__mod___features___9___block_0_2, getattr_l__mod___features___9___block_1_0], Original ATen: [aten.convolution, aten.hardswish]
        buf76 = extern_kernels.convolution(buf75, arg87_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=184, bias=None)
        assert_size_stride(buf76, (4, 184, 14, 14), (36064, 196, 14, 1))
        del arg87_1
        buf77 = buf76; del buf76  # reuse
        buf78 = buf75; del buf75  # reuse
        # Source Nodes: [getattr_l__mod___features___9___block_1_1, getattr_l__mod___features___9___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_24.run(buf77, arg249_1, arg250_1, arg88_1, arg89_1, buf78, 736, 196, grid=grid(736, 196), stream=stream0)
        del arg249_1
        del arg250_1
        del arg88_1
        del arg89_1
        del buf77
        # Source Nodes: [getattr_l__mod___features___9___block_1_2, getattr_l__mod___features___9___block_2_0], Original ATen: [aten.convolution, aten.hardswish]
        buf79 = extern_kernels.convolution(buf78, arg90_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (4, 80, 14, 14), (15680, 196, 14, 1))
        del arg90_1
        buf80 = buf72; del buf72  # reuse
        # Source Nodes: [result_13, result_14], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_23.run(buf80, buf79, arg252_1, arg253_1, arg91_1, arg92_1, 784, 80, grid=grid(784, 80), stream=stream0)
        del arg252_1
        del arg253_1
        del arg91_1
        del arg92_1
        del buf79
        # Source Nodes: [getattr_l__mod___features___10___block_0_0], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, arg93_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (4, 184, 14, 14), (36064, 196, 14, 1))
        del arg93_1
        buf82 = buf81; del buf81  # reuse
        buf83 = buf78; del buf78  # reuse
        # Source Nodes: [getattr_l__mod___features___10___block_0_1, getattr_l__mod___features___10___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_24.run(buf82, arg255_1, arg256_1, arg94_1, arg95_1, buf83, 736, 196, grid=grid(736, 196), stream=stream0)
        del arg255_1
        del arg256_1
        del arg94_1
        del arg95_1
        del buf82
        # Source Nodes: [getattr_l__mod___features___10___block_0_2, getattr_l__mod___features___10___block_1_0], Original ATen: [aten.convolution, aten.hardswish]
        buf84 = extern_kernels.convolution(buf83, arg96_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=184, bias=None)
        assert_size_stride(buf84, (4, 184, 14, 14), (36064, 196, 14, 1))
        del arg96_1
        buf85 = buf84; del buf84  # reuse
        buf86 = buf83; del buf83  # reuse
        # Source Nodes: [getattr_l__mod___features___10___block_1_1, getattr_l__mod___features___10___block_1_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_24.run(buf85, arg258_1, arg259_1, arg97_1, arg98_1, buf86, 736, 196, grid=grid(736, 196), stream=stream0)
        del arg258_1
        del arg259_1
        del arg97_1
        del arg98_1
        del buf85
        # Source Nodes: [getattr_l__mod___features___10___block_1_2, getattr_l__mod___features___10___block_2_0], Original ATen: [aten.convolution, aten.hardswish]
        buf87 = extern_kernels.convolution(buf86, arg99_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (4, 80, 14, 14), (15680, 196, 14, 1))
        del arg99_1
        del buf86
        buf88 = buf80; del buf80  # reuse
        # Source Nodes: [result_15, result_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_23.run(buf88, buf87, arg261_1, arg262_1, arg100_1, arg101_1, 784, 80, grid=grid(784, 80), stream=stream0)
        del arg100_1
        del arg101_1
        del arg261_1
        del arg262_1
        del buf87
        # Source Nodes: [getattr_l__mod___features___11___block_0_0, result_15, result_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf89 = extern_kernels.convolution(buf88, arg102_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (4, 480, 14, 14), (94080, 196, 14, 1))
        del arg102_1
        del buf88
        buf90 = buf89; del buf89  # reuse
        buf91 = reinterpret_tensor(buf54, (4, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf54  # reuse
        # Source Nodes: [getattr_l__mod___features___11___block_0_1, getattr_l__mod___features___11___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_25.run(buf90, arg264_1, arg265_1, arg103_1, arg104_1, buf91, 1920, 196, grid=grid(1920, 196), stream=stream0)
        del arg103_1
        del arg104_1
        del arg264_1
        del arg265_1
        del buf90
        # Source Nodes: [getattr_l__mod___features___11___block_0_2, getattr_l__mod___features___11___block_1_0], Original ATen: [aten.convolution, aten.hardswish]
        buf92 = extern_kernels.convolution(buf91, arg105_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf92, (4, 480, 14, 14), (94080, 196, 14, 1))
        del arg105_1
        buf93 = buf92; del buf92  # reuse
        buf94 = empty_strided((4, 480, 1, 1), (480, 1, 1920, 1920), device='cuda', dtype=torch.float32)
        buf95 = reinterpret_tensor(buf94, (4, 480, 1, 1), (480, 1, 480, 480), 0); del buf94  # reuse
        # Source Nodes: [getattr_l__mod___features___11___block_1_1, getattr_l__mod___features___11___block_1_2, scale_15], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_26.run(buf93, buf95, arg267_1, arg268_1, arg106_1, arg107_1, 1920, 196, grid=grid(1920), stream=stream0)
        del arg106_1
        del arg107_1
        del arg267_1
        del arg268_1
        # Source Nodes: [getattr_l__mod___features___11___block_1_2, scale_15, scale_16], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf96 = extern_kernels.convolution(buf95, arg108_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (4, 120, 1, 1), (120, 1, 1, 1))
        del arg108_1
        del buf95
        buf97 = reinterpret_tensor(buf96, (4, 120, 1, 1), (120, 1, 120, 120), 0); del buf96  # reuse
        # Source Nodes: [getattr_l__mod___features___11___block_1_2, scale_15, scale_16, scale_17], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
        triton_poi_fused_convolution_hardswish_mean_relu_27.run(buf97, arg109_1, 480, grid=grid(480), stream=stream0)
        del arg109_1
        # Source Nodes: [getattr_l__mod___features___11___block_1_2, scale_15, scale_16, scale_17, scale_18], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
        buf98 = extern_kernels.convolution(buf97, arg110_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (4, 480, 1, 1), (480, 1, 1, 1))
        del arg110_1
        del buf97
        buf99 = buf91; del buf91  # reuse
        # Source Nodes: [getattr_l__mod___features___11___block_1_2, mul_3, scale_15, scale_16, scale_17, scale_18, scale_19], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_28.run(buf98, arg111_1, buf93, buf99, 1920, 196, grid=grid(1920, 196), stream=stream0)
        del arg111_1
        del buf93
        del buf98
        # Source Nodes: [getattr_l__mod___features___11___block_1_2, getattr_l__mod___features___11___block_3_0, mul_3, scale_15, scale_16, scale_17, scale_18, scale_19], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
        buf100 = extern_kernels.convolution(buf99, arg112_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (4, 112, 14, 14), (21952, 196, 14, 1))
        del arg112_1
        del buf99
        buf101 = empty_strided((4, 112, 14, 14), (21952, 1, 1568, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [result_17], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_29.run(buf100, arg270_1, arg271_1, arg113_1, arg114_1, buf101, 448, 196, grid=grid(448, 196), stream=stream0)
        del arg113_1
        del arg114_1
        del arg270_1
        del arg271_1
        del buf100
        # Source Nodes: [getattr_l__mod___features___12___block_0_0], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, arg115_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (4, 672, 14, 14), (131712, 196, 14, 1))
        del arg115_1
        buf103 = buf102; del buf102  # reuse
        buf104 = empty_strided((4, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___12___block_0_1, getattr_l__mod___features___12___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_30.run(buf103, arg273_1, arg274_1, arg116_1, arg117_1, buf104, 2688, 196, grid=grid(2688, 196), stream=stream0)
        del arg116_1
        del arg117_1
        del arg273_1
        del arg274_1
        del buf103
        # Source Nodes: [getattr_l__mod___features___12___block_0_2, getattr_l__mod___features___12___block_1_0], Original ATen: [aten.convolution, aten.hardswish]
        buf105 = extern_kernels.convolution(buf104, arg118_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf105, (4, 672, 14, 14), (131712, 196, 14, 1))
        del arg118_1
        buf106 = buf105; del buf105  # reuse
        buf107 = empty_strided((4, 672, 1, 1), (672, 1, 2688, 2688), device='cuda', dtype=torch.float32)
        buf108 = reinterpret_tensor(buf107, (4, 672, 1, 1), (672, 1, 672, 672), 0); del buf107  # reuse
        # Source Nodes: [getattr_l__mod___features___12___block_1_1, getattr_l__mod___features___12___block_1_2, scale_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_31.run(buf106, buf108, arg276_1, arg277_1, arg119_1, arg120_1, 2688, 196, grid=grid(2688), stream=stream0)
        del arg119_1
        del arg120_1
        del arg276_1
        del arg277_1
        # Source Nodes: [getattr_l__mod___features___12___block_1_2, scale_20, scale_21], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf109 = extern_kernels.convolution(buf108, arg121_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (4, 168, 1, 1), (168, 1, 1, 1))
        del arg121_1
        del buf108
        buf110 = reinterpret_tensor(buf109, (4, 168, 1, 1), (168, 1, 168, 168), 0); del buf109  # reuse
        # Source Nodes: [getattr_l__mod___features___12___block_1_2, scale_20, scale_21, scale_22], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
        triton_poi_fused_convolution_hardswish_mean_relu_32.run(buf110, arg122_1, 672, grid=grid(672), stream=stream0)
        del arg122_1
        # Source Nodes: [getattr_l__mod___features___12___block_1_2, scale_20, scale_21, scale_22, scale_23], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
        buf111 = extern_kernels.convolution(buf110, arg123_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (4, 672, 1, 1), (672, 1, 1, 1))
        del arg123_1
        del buf110
        buf112 = buf104; del buf104  # reuse
        # Source Nodes: [getattr_l__mod___features___12___block_1_2, mul_4, scale_20, scale_21, scale_22, scale_23, scale_24], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_33.run(buf111, arg124_1, buf106, buf112, 2688, 196, grid=grid(2688, 196), stream=stream0)
        del arg124_1
        del buf106
        # Source Nodes: [getattr_l__mod___features___12___block_1_2, getattr_l__mod___features___12___block_3_0, mul_4, scale_20, scale_21, scale_22, scale_23, scale_24], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
        buf113 = extern_kernels.convolution(buf112, arg125_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (4, 112, 14, 14), (21952, 196, 14, 1))
        del arg125_1
        buf114 = buf101; del buf101  # reuse
        # Source Nodes: [result_18, result_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_34.run(buf114, buf113, arg279_1, arg280_1, arg126_1, arg127_1, 784, 112, grid=grid(784, 112), stream=stream0)
        del arg126_1
        del arg127_1
        del arg279_1
        del arg280_1
        del buf113
        # Source Nodes: [getattr_l__mod___features___13___block_0_0, result_18, result_19], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf115 = extern_kernels.convolution(buf114, arg128_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (4, 672, 14, 14), (131712, 196, 14, 1))
        del arg128_1
        del buf114
        buf116 = buf115; del buf115  # reuse
        buf117 = buf112; del buf112  # reuse
        # Source Nodes: [getattr_l__mod___features___13___block_0_1, getattr_l__mod___features___13___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_30.run(buf116, arg282_1, arg283_1, arg129_1, arg130_1, buf117, 2688, 196, grid=grid(2688, 196), stream=stream0)
        del arg129_1
        del arg130_1
        del arg282_1
        del arg283_1
        del buf116
        # Source Nodes: [getattr_l__mod___features___13___block_0_2, getattr_l__mod___features___13___block_1_0], Original ATen: [aten.convolution, aten.hardswish]
        buf118 = extern_kernels.convolution(buf117, arg131_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf118, (4, 672, 7, 7), (32928, 49, 7, 1))
        del arg131_1
        del buf117
        buf119 = buf118; del buf118  # reuse
        buf120 = reinterpret_tensor(buf111, (4, 672, 1, 1), (672, 1, 2688, 2688), 0); del buf111  # reuse
        buf121 = reinterpret_tensor(buf120, (4, 672, 1, 1), (672, 1, 672, 672), 0); del buf120  # reuse
        # Source Nodes: [getattr_l__mod___features___13___block_1_1, getattr_l__mod___features___13___block_1_2, scale_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_35.run(buf119, buf121, arg285_1, arg286_1, arg132_1, arg133_1, 2688, 49, grid=grid(2688), stream=stream0)
        del arg132_1
        del arg133_1
        del arg285_1
        del arg286_1
        # Source Nodes: [getattr_l__mod___features___13___block_1_2, scale_25, scale_26], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf122 = extern_kernels.convolution(buf121, arg134_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (4, 168, 1, 1), (168, 1, 1, 1))
        del arg134_1
        del buf121
        buf123 = reinterpret_tensor(buf122, (4, 168, 1, 1), (168, 1, 168, 168), 0); del buf122  # reuse
        # Source Nodes: [getattr_l__mod___features___13___block_1_2, scale_25, scale_26, scale_27], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
        triton_poi_fused_convolution_hardswish_mean_relu_32.run(buf123, arg135_1, 672, grid=grid(672), stream=stream0)
        del arg135_1
        # Source Nodes: [getattr_l__mod___features___13___block_1_2, scale_25, scale_26, scale_27, scale_28], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
        buf124 = extern_kernels.convolution(buf123, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (4, 672, 1, 1), (672, 1, 1, 1))
        del arg136_1
        del buf123
        buf125 = empty_strided((4, 672, 7, 7), (32928, 1, 4704, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___13___block_1_2, mul_5, scale_25, scale_26, scale_27, scale_28, scale_29], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_36.run(buf124, arg137_1, buf119, buf125, 2688, 49, grid=grid(2688, 49), stream=stream0)
        del arg137_1
        del buf119
        del buf124
        # Source Nodes: [getattr_l__mod___features___13___block_1_2, getattr_l__mod___features___13___block_3_0, mul_5, scale_25, scale_26, scale_27, scale_28, scale_29], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
        buf126 = extern_kernels.convolution(buf125, arg138_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (4, 160, 7, 7), (7840, 49, 7, 1))
        del arg138_1
        del buf125
        buf127 = empty_strided((4, 160, 7, 7), (7840, 1, 1120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [result_20], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_37.run(buf126, arg288_1, arg289_1, arg139_1, arg140_1, buf127, 640, 49, grid=grid(640, 49), stream=stream0)
        del arg139_1
        del arg140_1
        del arg288_1
        del arg289_1
        del buf126
        # Source Nodes: [getattr_l__mod___features___14___block_0_0], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, arg141_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (4, 960, 7, 7), (47040, 49, 7, 1))
        del arg141_1
        buf129 = buf128; del buf128  # reuse
        buf130 = reinterpret_tensor(buf62, (4, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf62  # reuse
        # Source Nodes: [getattr_l__mod___features___14___block_0_1, getattr_l__mod___features___14___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38.run(buf129, arg291_1, arg292_1, arg142_1, arg143_1, buf130, 3840, 49, grid=grid(3840, 49), stream=stream0)
        del arg142_1
        del arg143_1
        del arg291_1
        del arg292_1
        del buf129
        # Source Nodes: [getattr_l__mod___features___14___block_0_2, getattr_l__mod___features___14___block_1_0], Original ATen: [aten.convolution, aten.hardswish]
        buf131 = extern_kernels.convolution(buf130, arg144_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf131, (4, 960, 7, 7), (47040, 49, 7, 1))
        del arg144_1
        buf132 = buf131; del buf131  # reuse
        buf133 = empty_strided((4, 960, 1, 1), (960, 1, 3840, 3840), device='cuda', dtype=torch.float32)
        buf134 = reinterpret_tensor(buf133, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf133  # reuse
        # Source Nodes: [getattr_l__mod___features___14___block_1_1, getattr_l__mod___features___14___block_1_2, scale_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_39.run(buf132, buf134, arg294_1, arg295_1, arg145_1, arg146_1, 3840, 49, grid=grid(3840), stream=stream0)
        del arg145_1
        del arg146_1
        del arg294_1
        del arg295_1
        # Source Nodes: [getattr_l__mod___features___14___block_1_2, scale_30, scale_31], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf135 = extern_kernels.convolution(buf134, arg147_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (4, 240, 1, 1), (240, 1, 1, 1))
        del arg147_1
        del buf134
        buf136 = reinterpret_tensor(buf135, (4, 240, 1, 1), (240, 1, 240, 240), 0); del buf135  # reuse
        # Source Nodes: [getattr_l__mod___features___14___block_1_2, scale_30, scale_31, scale_32], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
        triton_poi_fused_convolution_hardswish_mean_relu_40.run(buf136, arg148_1, 960, grid=grid(960), stream=stream0)
        del arg148_1
        # Source Nodes: [getattr_l__mod___features___14___block_1_2, scale_30, scale_31, scale_32, scale_33], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
        buf137 = extern_kernels.convolution(buf136, arg149_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (4, 960, 1, 1), (960, 1, 1, 1))
        del arg149_1
        del buf136
        buf138 = buf130; del buf130  # reuse
        # Source Nodes: [getattr_l__mod___features___14___block_1_2, mul_6, scale_30, scale_31, scale_32, scale_33, scale_34], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_41.run(buf137, arg150_1, buf132, buf138, 3840, 49, grid=grid(3840, 49), stream=stream0)
        del arg150_1
        del buf132
        # Source Nodes: [getattr_l__mod___features___14___block_1_2, getattr_l__mod___features___14___block_3_0, mul_6, scale_30, scale_31, scale_32, scale_33, scale_34], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
        buf139 = extern_kernels.convolution(buf138, arg151_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf139, (4, 160, 7, 7), (7840, 49, 7, 1))
        del arg151_1
        buf140 = buf127; del buf127  # reuse
        # Source Nodes: [result_21, result_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_42.run(buf140, buf139, arg297_1, arg298_1, arg152_1, arg153_1, 196, 160, grid=grid(196, 160), stream=stream0)
        del arg152_1
        del arg153_1
        del arg297_1
        del arg298_1
        del buf139
        # Source Nodes: [getattr_l__mod___features___15___block_0_0], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, arg154_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (4, 960, 7, 7), (47040, 49, 7, 1))
        del arg154_1
        buf142 = buf141; del buf141  # reuse
        buf143 = buf138; del buf138  # reuse
        # Source Nodes: [getattr_l__mod___features___15___block_0_1, getattr_l__mod___features___15___block_0_2], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38.run(buf142, arg300_1, arg301_1, arg155_1, arg156_1, buf143, 3840, 49, grid=grid(3840, 49), stream=stream0)
        del arg155_1
        del arg156_1
        del arg300_1
        del arg301_1
        del buf142
        # Source Nodes: [getattr_l__mod___features___15___block_0_2, getattr_l__mod___features___15___block_1_0], Original ATen: [aten.convolution, aten.hardswish]
        buf144 = extern_kernels.convolution(buf143, arg157_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf144, (4, 960, 7, 7), (47040, 49, 7, 1))
        del arg157_1
        buf145 = buf144; del buf144  # reuse
        buf146 = reinterpret_tensor(buf137, (4, 960, 1, 1), (960, 1, 3840, 3840), 0); del buf137  # reuse
        buf147 = reinterpret_tensor(buf146, (4, 960, 1, 1), (960, 1, 960, 960), 0); del buf146  # reuse
        # Source Nodes: [getattr_l__mod___features___15___block_1_1, getattr_l__mod___features___15___block_1_2, scale_35], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_39.run(buf145, buf147, arg303_1, arg304_1, arg158_1, arg159_1, 3840, 49, grid=grid(3840), stream=stream0)
        del arg158_1
        del arg159_1
        del arg303_1
        del arg304_1
        # Source Nodes: [getattr_l__mod___features___15___block_1_2, scale_35, scale_36], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf148 = extern_kernels.convolution(buf147, arg160_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (4, 240, 1, 1), (240, 1, 1, 1))
        del arg160_1
        del buf147
        buf149 = reinterpret_tensor(buf148, (4, 240, 1, 1), (240, 1, 240, 240), 0); del buf148  # reuse
        # Source Nodes: [getattr_l__mod___features___15___block_1_2, scale_35, scale_36, scale_37], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
        triton_poi_fused_convolution_hardswish_mean_relu_40.run(buf149, arg161_1, 960, grid=grid(960), stream=stream0)
        del arg161_1
        # Source Nodes: [getattr_l__mod___features___15___block_1_2, scale_35, scale_36, scale_37, scale_38], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
        buf150 = extern_kernels.convolution(buf149, arg162_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (4, 960, 1, 1), (960, 1, 1, 1))
        del arg162_1
        del buf149
        buf151 = buf143; del buf143  # reuse
        # Source Nodes: [getattr_l__mod___features___15___block_1_2, mul_7, scale_35, scale_36, scale_37, scale_38, scale_39], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_41.run(buf150, arg163_1, buf145, buf151, 3840, 49, grid=grid(3840, 49), stream=stream0)
        del arg163_1
        del buf145
        # Source Nodes: [getattr_l__mod___features___15___block_1_2, getattr_l__mod___features___15___block_3_0, mul_7, scale_35, scale_36, scale_37, scale_38, scale_39], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
        buf152 = extern_kernels.convolution(buf151, arg164_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (4, 160, 7, 7), (7840, 49, 7, 1))
        del arg164_1
        del buf151
        buf153 = buf140; del buf140  # reuse
        # Source Nodes: [result_23, result_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_42.run(buf153, buf152, arg306_1, arg307_1, arg165_1, arg166_1, 196, 160, grid=grid(196, 160), stream=stream0)
        del arg165_1
        del arg166_1
        del arg306_1
        del arg307_1
        del buf152
        # Source Nodes: [l__mod___features_16_0, result_23, result_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf154 = extern_kernels.convolution(buf153, arg167_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (4, 960, 7, 7), (47040, 49, 7, 1))
        del arg167_1
        del buf153
        buf155 = buf154; del buf154  # reuse
        buf156 = reinterpret_tensor(buf150, (4, 960, 1, 1), (960, 1, 3840, 3840), 0); del buf150  # reuse
        buf157 = reinterpret_tensor(buf156, (4, 960, 1, 1), (960, 1, 1, 1), 0); del buf156  # reuse
        # Source Nodes: [l__mod___features_16_1, x, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_43.run(buf155, buf157, arg309_1, arg310_1, arg168_1, arg169_1, 3840, 49, grid=grid(3840), stream=stream0)
        del arg168_1
        del arg169_1
        del arg309_1
        del arg310_1
        del buf155
        buf158 = empty((4, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf157, (4, 960), (960, 1), 0), reinterpret_tensor(arg170_1, (960, 1280), (1, 960), 0), out=buf158)
        del arg170_1
        del buf157
        buf159 = buf158; del buf158  # reuse
        # Source Nodes: [l__mod___classifier_1], Original ATen: [aten.hardswish]
        triton_poi_fused_hardswish_44.run(buf159, arg171_1, 5120, grid=grid(5120), stream=stream0)
        del arg171_1
        buf160 = empty((4, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___classifier_1, x_3], Original ATen: [aten.addmm, aten.hardswish]
        extern_kernels.addmm(arg173_1, buf159, reinterpret_tensor(arg172_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf160)
        del arg172_1
        del arg173_1
        return (buf160, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((24, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
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
    arg30_1 = rand_strided((72, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((40, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((200, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((200, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((80, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((184, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((184, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((80, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((184, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((184, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((80, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((120, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((480, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((672, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((160, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((960, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((960, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((1280, 960), (960, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg177_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg180_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg183_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg186_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg189_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg192_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg195_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg198_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg201_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg204_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg207_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg210_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg213_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg216_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg219_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg222_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg225_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg228_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg231_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg234_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg237_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg240_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg243_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg246_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg249_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg252_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg255_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg258_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg261_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg264_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg267_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg270_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg273_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg276_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg279_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg282_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg285_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg288_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg291_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg294_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg297_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg300_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg303_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg306_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg309_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((), (), device='cuda:0', dtype=torch.int64)
    arg312_1 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mobilenet_v3_large', benchmark_compiled_module)
