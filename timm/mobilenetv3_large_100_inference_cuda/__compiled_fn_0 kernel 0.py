
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


# kernel path: /tmp/torchinductor_youkaichao/k3/ck3mdeneejappqqe5fk6zuhypf6spx7mpv4obazedrrc25iia6ca.py
# Source Nodes: [x], Original ATen: [aten.convolution]
# x => convolution
triton_poi_fused_convolution_0 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32, 65536], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
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
# Source Nodes: [x], Original ATen: [aten.convolution]
# x => convolution
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


# kernel path: /tmp/torchinductor_youkaichao/ws/cwstsvzkgofnrlpmcr2adeqsfiqwrcfo33cnr2grdy5vqy2dw2mt.py
# Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# shortcut => add_2, clamp_max, clamp_min, div, mul_3
# x_1 => add_1, mul_1, mul_2, sub
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 16
    y1 = (yindex // 16)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/2p/c2p4vgjomopnejwbuhkztqfjvyzzti466xjjhhghhmguvjqf4bed.py
# Source Nodes: [x_6, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_6 => add_4, mul_5, mul_6, sub_1
# x_9 => relu
triton_poi_fused__native_batch_norm_legit_no_training_relu_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_3', 'mutated_arg_names': []},
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
    tmp15 = triton_helpers.maximum(0, tmp14)
    tl.store(out_ptr0 + (y0 + (16*x2) + (200704*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bv/cbv65uf6isabdj3pfcniwtelbyes36xkw4kkieafghg42le2zsh4.py
# Source Nodes: [shortcut_1, x_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_1 => add_7
# x_12 => add_6, mul_8, mul_9, sub_2
triton_poi_fused__native_batch_norm_legit_no_training_add_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072, 16], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 100352
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
    tl.store(in_out_ptr0 + (x2 + (16*y3)), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uw/cuwfrbu2c2dw6de45liv6bh3hs742cz3otpsfy2djgjzgg725off.py
# Source Nodes: [x_18, x_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_18 => add_9, mul_11, mul_12, sub_3
# x_21 => relu_1
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 16384], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
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
    tl.store(out_ptr0 + (y0 + (64*x2) + (802816*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qz/cqzwtyjek5oqiygnphgbdfvgymsxoj2m44evyvzwvy4ziet3onkb.py
# Source Nodes: [x_23, x_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_23 => add_11, mul_14, mul_15, sub_4
# x_26 => relu_2
triton_poi_fused__native_batch_norm_legit_no_training_relu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
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
    tl.store(out_ptr0 + (y0 + (64*x2) + (200704*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3j/c3jpv7c72pnyhr7pie5fjd4wzhiddh7p27rmidadt33oxzawr6ub.py
# Source Nodes: [x_29], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_29 => add_13, mul_17, mul_18, sub_5
triton_poi_fused__native_batch_norm_legit_no_training_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_7', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/46/c46z5y4pcxggnv4vtgeyyz2bcvteqddydkzgim76yvjemrz2f62d.py
# Source Nodes: [x_34, x_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_34 => add_15, mul_20, mul_21, sub_6
# x_37 => relu_3
triton_poi_fused__native_batch_norm_legit_no_training_relu_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
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


# kernel path: /tmp/torchinductor_youkaichao/e5/ce5ys5vdtrzlcsth2maimxwgjkg5fwai4hsbfmsqmrb7wiuhhuww.py
# Source Nodes: [shortcut_3, x_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_3 => add_20
# x_45 => add_19, mul_26, mul_27, sub_8
triton_poi_fused__native_batch_norm_legit_no_training_add_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 32], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25088
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


# kernel path: /tmp/torchinductor_youkaichao/m4/cm4dmxuwfouz3wg4szftj4ppa4zmyyayhrguagfrgdcdij6qkajz.py
# Source Nodes: [x_56, x_59, x_se], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
# x_56 => add_24, mul_32, mul_33, sub_10
# x_59 => relu_6
# x_se => mean
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_10', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 576
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
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = 784.0
    tmp21 = tmp19 / tmp20
    tl.store(in_out_ptr0 + (r2 + (784*x3)), tmp15, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xh/cxhsi76cxfernzkfo6dqvqg4z3rk6knsq6qevm46ao62nmvyzgav.py
# Source Nodes: [x_se, x_se_1, x_se_2], Original ATen: [aten.convolution, aten.mean, aten.relu]
# x_se => mean
# x_se_1 => convolution_11
# x_se_2 => relu_7
triton_poi_fused_convolution_mean_relu_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_11', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 192
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


# kernel path: /tmp/torchinductor_youkaichao/e3/ce3qjpeziezmrtpfh4pgoleoeyiuqvvv7mzurrpbf7idbsu3lfu6.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_60, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
# getattr_getattr_l__mod___blocks___2_____0___se_gate => add_25, clamp_max_1, clamp_min_1, div_1
# x_60 => mul_34
# x_se => mean
# x_se_1 => convolution_11
# x_se_2 => relu_7
# x_se_3 => convolution_12
triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 576
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
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 3.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tmp10 = tmp9 / tmp8
    tmp11 = tmp0 * tmp10
    tl.store(out_ptr0 + (y0 + (72*x2) + (56448*y1)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xw/cxwwxbkmpvmottlofl6nrdzhdgr2nbwmw23l4nrhwus5aces2ycx.py
# Source Nodes: [x_62], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_62 => add_27, mul_36, mul_37, sub_11
triton_poi_fused__native_batch_norm_legit_no_training_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 320
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


# kernel path: /tmp/torchinductor_youkaichao/nn/cnnkhz3nk4uw5qg2blmihmemj2jtqpv2fupez5jjteb3po4c42bh.py
# Source Nodes: [x_67, x_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_67 => add_29, mul_39, mul_40, sub_12
# x_70 => relu_8
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/yt/cyti3jr4jpp4g6ha6neqjgzf4gsbu7ftoqal7c5rw2d227axxqjb.py
# Source Nodes: [x_72, x_75, x_se_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
# x_72 => add_31, mul_42, mul_43, sub_13
# x_75 => relu_9
# x_se_4 => mean_1
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_15', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel):
    xnumel = 960
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
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp18 = tl.where(rmask & xmask, tmp16, 0)
    tmp19 = triton_helpers.promote_to_tensor(tl.sum(tmp18, 0))
    tmp20 = 784.0
    tmp21 = tmp19 / tmp20
    tl.store(in_out_ptr0 + (r2 + (784*x3)), tmp15, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp21, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yh/cyh2oztgegaxml72rup6blia2gzydy6besdj44wzlgva5dgwknaj.py
# Source Nodes: [x_se_4, x_se_5, x_se_6], Original ATen: [aten.convolution, aten.mean, aten.relu]
# x_se_4 => mean_1
# x_se_5 => convolution_16
# x_se_6 => relu_10
triton_poi_fused_convolution_mean_relu_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_mean_relu_16', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 256
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


# kernel path: /tmp/torchinductor_youkaichao/2s/c2srtmdzusunksscwkga5bcm6llpaqtz42afhedxmcith3grii6c.py
# Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_76, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
# getattr_getattr_l__mod___blocks___2_____1___se_gate => add_32, clamp_max_2, clamp_min_2, div_2
# x_76 => mul_44
# x_se_4 => mean_1
# x_se_5 => convolution_16
# x_se_6 => relu_10
# x_se_7 => convolution_17
triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 120
    y1 = (yindex // 120)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp2 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tmp1 + tmp2
    tmp4 = 3.0
    tmp5 = tmp3 + tmp4
    tmp6 = 0.0
    tmp7 = triton_helpers.maximum(tmp5, tmp6)
    tmp8 = 6.0
    tmp9 = triton_helpers.minimum(tmp7, tmp8)
    tmp10 = tmp9 / tmp8
    tmp11 = tmp0 * tmp10
    tl.store(out_ptr0 + (y0 + (120*x2) + (94080*y1)), tmp11, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/hn/chnmrqi3url6o52fqvimhscxb42lzxdys2g35t6ypouhghdowmfk.py
# Source Nodes: [shortcut_5, x_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_5 => add_35
# x_78 => add_34, mul_46, mul_47, sub_14
triton_poi_fused__native_batch_norm_legit_no_training_add_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_18', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6272
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


# kernel path: /tmp/torchinductor_youkaichao/a3/ca3fjffqfj4lilpvsauoxu4fvfgifsionwdkrudsvumkdbojbodj.py
# Source Nodes: [x_101, x_104], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# x_101 => add_45, mul_59, mul_60, sub_18
# x_104 => add_46, clamp_max_4, clamp_min_4, div_4, mul_61
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 1024], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1920
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
    tmp4 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/tg/ctg2wqzsilueh53xaws7hv3x5kylhk4l4d7oqvw6a4l333itt2sq.py
# Source Nodes: [x_106, x_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# x_106 => add_48, mul_63, mul_64, sub_19
# x_109 => add_49, clamp_max_5, clamp_min_5, div_5, mul_65
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_20', 'mutated_arg_names': ['in_out_ptr0']},
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
    y0 = yindex % 240
    y1 = (yindex // 240)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), ymask, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_youkaichao/l4/cl46klulwv3sauimxjlmujixrzkqxbhacyalg2hfmxwg5jnqfd6b.py
# Source Nodes: [x_112], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_112 => add_51, mul_67, mul_68, sub_20
triton_poi_fused__native_batch_norm_legit_no_training_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 640
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


# kernel path: /tmp/torchinductor_youkaichao/4a/c4a4qt7ald35e5lysp6oqvn5pdsgt7a4flptmoc5gbswfr26jg2w.py
# Source Nodes: [x_117, x_120], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# x_117 => add_53, mul_70, mul_71, sub_21
# x_120 => add_54, clamp_max_6, clamp_min_6, div_6, mul_72
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1600
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
    tmp4 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/xi/cxi63wy47qlkohfrsserzkz6jc3z7wjyvrgyffi5m36ckeef6xcv.py
# Source Nodes: [shortcut_8, x_128], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_8 => add_60
# x_128 => add_59, mul_78, mul_79, sub_23
triton_poi_fused__native_batch_norm_legit_no_training_add_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
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


# kernel path: /tmp/torchinductor_youkaichao/vf/cvffg4x34b7jdydpppfigcdlugc65q2pobxdwaotw2zutphcecom.py
# Source Nodes: [x_134, x_137], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# x_134 => add_62, mul_81, mul_82, sub_24
# x_137 => add_63, clamp_max_8, clamp_min_8, div_8, mul_83
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1472
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
    tmp4 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/yc/cycj7y2vksfkoxgapybn4ikv3yfefrmj6gqlrmpdzpdagc2zhw7s.py
# Source Nodes: [x_168, x_171], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# x_168 => add_80, mul_103, mul_104, sub_30
# x_171 => add_81, clamp_max_12, clamp_min_12, div_12, mul_105
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_25', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3840
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
    tmp4 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/2e/c2edfqiadav2rg66d6bmv3hqvm75lpwgjtlep3f5bzfk22iprn4k.py
# Source Nodes: [x_173, x_176, x_se_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
# x_173 => add_83, mul_107, mul_108, sub_31
# x_176 => add_84, clamp_max_13, clamp_min_13, div_13, mul_109
# x_se_12 => mean_3
triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_26', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 3840
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
    tmp4 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/ld/cldkdtijacsoctofotshsjgmufjvy766m7xpg3omnndqf42j2vvd.py
# Source Nodes: [x_176, x_se_12, x_se_13, x_se_14], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
# x_176 => add_84, clamp_max_13, clamp_min_13, div_13, mul_109
# x_se_12 => mean_3
# x_se_13 => convolution_38
# x_se_14 => relu_14
triton_poi_fused_convolution_hardswish_mean_relu_27 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_relu_27', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 960
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


# kernel path: /tmp/torchinductor_youkaichao/kq/ckqztwxykb2f4yndtukn6csjq7qdohylkomze4ea2u4syowlcajw.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_176, x_177, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
# getattr_getattr_l__mod___blocks___4_____0___se_gate => add_85, clamp_max_14, clamp_min_14, div_14
# x_176 => add_84, clamp_max_13, clamp_min_13, div_13, mul_109
# x_177 => mul_110
# x_se_12 => mean_3
# x_se_13 => convolution_38
# x_se_14 => relu_14
# x_se_15 => convolution_39
triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_28 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3840
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
    tmp9 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tmp11 = tmp9 + tmp10
    tmp12 = tmp11 + tmp1
    tmp13 = triton_helpers.maximum(tmp12, tmp3)
    tmp14 = triton_helpers.minimum(tmp13, tmp5)
    tmp15 = tmp14 / tmp5
    tmp16 = tmp8 * tmp15
    tl.store(out_ptr0 + (y0 + (480*x2) + (94080*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fp/cfpncjsuogxlmwkhfmyrocj2f643i6sx5rkrcbtwivpezv5de2fn.py
# Source Nodes: [x_179], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_179 => add_87, mul_112, mul_113, sub_32
triton_poi_fused__native_batch_norm_legit_no_training_29 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 896
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
    tmp4 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/du/cdu7wsrvwy4viztfn7kfi3ghanlswlcgeyekalljmqec4olxtuzf.py
# Source Nodes: [x_184, x_187], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# x_184 => add_89, mul_115, mul_116, sub_33
# x_187 => add_90, clamp_max_15, clamp_min_15, div_15, mul_117
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_30', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5376
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
    tmp4 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/ea/ceatwschehye2ahabbt6iwscbv2qkktytxdhgoti3hsi6c2jmh6o.py
# Source Nodes: [x_189, x_192, x_se_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
# x_189 => add_92, mul_119, mul_120, sub_34
# x_192 => add_93, clamp_max_16, clamp_min_16, div_16, mul_121
# x_se_16 => mean_4
triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 256],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_31', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
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
    tmp4 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/43/c43dj35mqmbvm26736uwjka2667xh6vznil4ybpjwicayk3bd7y5.py
# Source Nodes: [x_192, x_se_16, x_se_17, x_se_18], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
# x_192 => add_93, clamp_max_16, clamp_min_16, div_16, mul_121
# x_se_16 => mean_4
# x_se_17 => convolution_43
# x_se_18 => relu_15
triton_poi_fused_convolution_hardswish_mean_relu_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_relu_32', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1344
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


# kernel path: /tmp/torchinductor_youkaichao/om/comorkorxdfplqkbednpysiet3ey6xy46lklj5x4lugvmd2fnynd.py
# Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_192, x_193, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
# getattr_getattr_l__mod___blocks___4_____1___se_gate => add_94, clamp_max_17, clamp_min_17, div_17
# x_192 => add_93, clamp_max_16, clamp_min_16, div_16, mul_121
# x_193 => mul_122
# x_se_16 => mean_4
# x_se_17 => convolution_43
# x_se_18 => relu_15
# x_se_19 => convolution_44
triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 256], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5376
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
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tmp11 = tmp9 + tmp10
    tmp12 = tmp11 + tmp1
    tmp13 = triton_helpers.maximum(tmp12, tmp3)
    tmp14 = triton_helpers.minimum(tmp13, tmp5)
    tmp15 = tmp14 / tmp5
    tmp16 = tmp8 * tmp15
    tl.store(out_ptr0 + (y0 + (672*x2) + (131712*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/b5/cb5wvbe6ipkcu2wgnbfrir6fthgjabfcofd6ahwawhb5id3k76q3.py
# Source Nodes: [shortcut_12, x_195], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_12 => add_97
# x_195 => add_96, mul_124, mul_125, sub_35
triton_poi_fused__native_batch_norm_legit_no_training_add_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 128], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_34', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
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
    tl.store(in_out_ptr0 + (x2 + (112*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yc/cycaxyktl4ommw5kf6tcea72gighhbawda2v3a5nkaetucdhoqqx.py
# Source Nodes: [x_206, x_209, x_se_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
# x_206 => add_102, mul_131, mul_132, sub_37
# x_209 => add_103, clamp_max_19, clamp_min_19, div_19, mul_133
# x_se_20 => mean_5
triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_35 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_35', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 5376
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
    tmp4 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/a6/ca6vaehafyfswsrmg2mj77lqwnwoslzuygdxk3jbcpbnve4tfefa.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_209, x_210, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => add_104, clamp_max_20, clamp_min_20, div_20
# x_209 => add_103, clamp_max_19, clamp_min_19, div_19, mul_133
# x_210 => mul_134
# x_se_20 => mean_5
# x_se_21 => convolution_48
# x_se_22 => relu_16
# x_se_23 => convolution_49
triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_36 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 5376
    xnumel = 49
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tmp11 = tmp9 + tmp10
    tmp12 = tmp11 + tmp1
    tmp13 = triton_helpers.maximum(tmp12, tmp3)
    tmp14 = triton_helpers.minimum(tmp13, tmp5)
    tmp15 = tmp14 / tmp5
    tmp16 = tmp8 * tmp15
    tl.store(out_ptr0 + (y0 + (672*x2) + (32928*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/bb/cbbl6jwf3bihv7f5pxrojz72nzuu4tcjtf3omexfdrl45frtxwou.py
# Source Nodes: [x_212], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_212 => add_106, mul_136, mul_137, sub_38
triton_poi_fused__native_batch_norm_legit_no_training_37 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_37', 'mutated_arg_names': []},
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
    y0 = yindex % 160
    y1 = (yindex // 160)
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
    tl.store(out_ptr0 + (y0 + (160*x2) + (7840*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6w/c6wwsu4jbjdwlfd5imjxnr3xcydk6hbfiff7posuicnfjtcitlfd.py
# Source Nodes: [x_217, x_220], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# x_217 => add_108, mul_139, mul_140, sub_39
# x_220 => add_109, clamp_max_21, clamp_min_21, div_21, mul_141
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7680
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
    tmp4 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/ok/cokplmej5id2qf6kvwamjojnhwwos4cvt3m265uj4ktj56hpknqy.py
# Source Nodes: [x_222, x_225, x_se_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
# x_222 => add_111, mul_143, mul_144, sub_40
# x_225 => add_112, clamp_max_22, clamp_min_22, div_22, mul_145
# x_se_24 => mean_6
triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_39 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_39', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
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
    tmp4 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/2z/c2zbz23wz5bn3e3aaudyviowrmfvctfgjatbjw4nxfurtx7keq4x.py
# Source Nodes: [x_225, x_se_24, x_se_25, x_se_26], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
# x_225 => add_112, clamp_max_22, clamp_min_22, div_22, mul_145
# x_se_24 => mean_6
# x_se_25 => convolution_53
# x_se_26 => relu_17
triton_poi_fused_convolution_hardswish_mean_relu_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_relu_40', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1920
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


# kernel path: /tmp/torchinductor_youkaichao/do/cdopsfpgsaypyxn6tppw4iwtww2fmwjioatuxy7775xrdvxhfq7n.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_225, x_226, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
# getattr_getattr_l__mod___blocks___5_____1___se_gate => add_113, clamp_max_23, clamp_min_23, div_23
# x_225 => add_112, clamp_max_22, clamp_min_22, div_22, mul_145
# x_226 => mul_146
# x_se_24 => mean_6
# x_se_25 => convolution_53
# x_se_26 => relu_17
# x_se_27 => convolution_54
triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_41 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 7680
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (y3), ymask, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (y0), ymask, eviction_policy='evict_last')
    tmp1 = 3.0
    tmp2 = tmp0 + tmp1
    tmp3 = 0.0
    tmp4 = triton_helpers.maximum(tmp2, tmp3)
    tmp5 = 6.0
    tmp6 = triton_helpers.minimum(tmp4, tmp5)
    tmp7 = tmp0 * tmp6
    tmp8 = tmp7 / tmp5
    tmp11 = tmp9 + tmp10
    tmp12 = tmp11 + tmp1
    tmp13 = triton_helpers.maximum(tmp12, tmp3)
    tmp14 = triton_helpers.minimum(tmp13, tmp5)
    tmp15 = tmp14 / tmp5
    tmp16 = tmp8 * tmp15
    tl.store(out_ptr0 + (y0 + (960*x2) + (47040*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mo/cmoswqh6larm5uhck52s5c6rfjlmizp43ydwgm2pisdxfbhyhotj.py
# Source Nodes: [shortcut_14, x_228], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_14 => add_116
# x_228 => add_115, mul_148, mul_149, sub_41
triton_poi_fused__native_batch_norm_legit_no_training_add_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_42', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
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
    tl.store(in_out_ptr0 + (x2 + (160*y3)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wy/cwy3yj7cvxfwyjzo2vo6sgqfxjqw5bldgrgrxwgygxh4sjy537gu.py
# Source Nodes: [x_251, x_256, x_257], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
# x_251 => add_128, mul_163, mul_164, sub_45
# x_256 => add_129, clamp_max_27, clamp_min_27, div_27, mul_165
# x_257 => mean_8
triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_43 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_43', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 7680
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
    tmp4 = 1e-05
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


# kernel path: /tmp/torchinductor_youkaichao/rf/crfxo5zjmchcz2eifmt62qljac5mvhi6piufzrs4qi4ombjbc5d7.py
# Source Nodes: [x_256, x_257, x_260, x_261], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
# x_256 => add_129, clamp_max_27, clamp_min_27, div_27, mul_165
# x_257 => mean_8
# x_260 => convolution_62
# x_261 => add_130, clamp_max_28, clamp_min_28, div_28, mul_166
triton_poi_fused_convolution_hardswish_mean_44 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_44', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 1280
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = 3.0
    tmp4 = tmp2 + tmp3
    tmp5 = 0.0
    tmp6 = triton_helpers.maximum(tmp4, tmp5)
    tmp7 = 6.0
    tmp8 = triton_helpers.minimum(tmp6, tmp7)
    tmp9 = tmp2 * tmp8
    tmp10 = tmp9 / tmp7
    tl.store(in_out_ptr0 + (x2), tmp10, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1 = args
    args.clear()
    assert_size_stride(arg0_1, (16, ), (1, ))
    assert_size_stride(arg1_1, (16, ), (1, ))
    assert_size_stride(arg2_1, (16, ), (1, ))
    assert_size_stride(arg3_1, (16, ), (1, ))
    assert_size_stride(arg4_1, (16, ), (1, ))
    assert_size_stride(arg5_1, (16, ), (1, ))
    assert_size_stride(arg6_1, (64, ), (1, ))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (64, ), (1, ))
    assert_size_stride(arg9_1, (64, ), (1, ))
    assert_size_stride(arg10_1, (24, ), (1, ))
    assert_size_stride(arg11_1, (24, ), (1, ))
    assert_size_stride(arg12_1, (72, ), (1, ))
    assert_size_stride(arg13_1, (72, ), (1, ))
    assert_size_stride(arg14_1, (72, ), (1, ))
    assert_size_stride(arg15_1, (72, ), (1, ))
    assert_size_stride(arg16_1, (24, ), (1, ))
    assert_size_stride(arg17_1, (24, ), (1, ))
    assert_size_stride(arg18_1, (72, ), (1, ))
    assert_size_stride(arg19_1, (72, ), (1, ))
    assert_size_stride(arg20_1, (72, ), (1, ))
    assert_size_stride(arg21_1, (72, ), (1, ))
    assert_size_stride(arg22_1, (40, ), (1, ))
    assert_size_stride(arg23_1, (40, ), (1, ))
    assert_size_stride(arg24_1, (120, ), (1, ))
    assert_size_stride(arg25_1, (120, ), (1, ))
    assert_size_stride(arg26_1, (120, ), (1, ))
    assert_size_stride(arg27_1, (120, ), (1, ))
    assert_size_stride(arg28_1, (40, ), (1, ))
    assert_size_stride(arg29_1, (40, ), (1, ))
    assert_size_stride(arg30_1, (120, ), (1, ))
    assert_size_stride(arg31_1, (120, ), (1, ))
    assert_size_stride(arg32_1, (120, ), (1, ))
    assert_size_stride(arg33_1, (120, ), (1, ))
    assert_size_stride(arg34_1, (40, ), (1, ))
    assert_size_stride(arg35_1, (40, ), (1, ))
    assert_size_stride(arg36_1, (240, ), (1, ))
    assert_size_stride(arg37_1, (240, ), (1, ))
    assert_size_stride(arg38_1, (240, ), (1, ))
    assert_size_stride(arg39_1, (240, ), (1, ))
    assert_size_stride(arg40_1, (80, ), (1, ))
    assert_size_stride(arg41_1, (80, ), (1, ))
    assert_size_stride(arg42_1, (200, ), (1, ))
    assert_size_stride(arg43_1, (200, ), (1, ))
    assert_size_stride(arg44_1, (200, ), (1, ))
    assert_size_stride(arg45_1, (200, ), (1, ))
    assert_size_stride(arg46_1, (80, ), (1, ))
    assert_size_stride(arg47_1, (80, ), (1, ))
    assert_size_stride(arg48_1, (184, ), (1, ))
    assert_size_stride(arg49_1, (184, ), (1, ))
    assert_size_stride(arg50_1, (184, ), (1, ))
    assert_size_stride(arg51_1, (184, ), (1, ))
    assert_size_stride(arg52_1, (80, ), (1, ))
    assert_size_stride(arg53_1, (80, ), (1, ))
    assert_size_stride(arg54_1, (184, ), (1, ))
    assert_size_stride(arg55_1, (184, ), (1, ))
    assert_size_stride(arg56_1, (184, ), (1, ))
    assert_size_stride(arg57_1, (184, ), (1, ))
    assert_size_stride(arg58_1, (80, ), (1, ))
    assert_size_stride(arg59_1, (80, ), (1, ))
    assert_size_stride(arg60_1, (480, ), (1, ))
    assert_size_stride(arg61_1, (480, ), (1, ))
    assert_size_stride(arg62_1, (480, ), (1, ))
    assert_size_stride(arg63_1, (480, ), (1, ))
    assert_size_stride(arg64_1, (112, ), (1, ))
    assert_size_stride(arg65_1, (112, ), (1, ))
    assert_size_stride(arg66_1, (672, ), (1, ))
    assert_size_stride(arg67_1, (672, ), (1, ))
    assert_size_stride(arg68_1, (672, ), (1, ))
    assert_size_stride(arg69_1, (672, ), (1, ))
    assert_size_stride(arg70_1, (112, ), (1, ))
    assert_size_stride(arg71_1, (112, ), (1, ))
    assert_size_stride(arg72_1, (672, ), (1, ))
    assert_size_stride(arg73_1, (672, ), (1, ))
    assert_size_stride(arg74_1, (672, ), (1, ))
    assert_size_stride(arg75_1, (672, ), (1, ))
    assert_size_stride(arg76_1, (160, ), (1, ))
    assert_size_stride(arg77_1, (160, ), (1, ))
    assert_size_stride(arg78_1, (960, ), (1, ))
    assert_size_stride(arg79_1, (960, ), (1, ))
    assert_size_stride(arg80_1, (960, ), (1, ))
    assert_size_stride(arg81_1, (960, ), (1, ))
    assert_size_stride(arg82_1, (160, ), (1, ))
    assert_size_stride(arg83_1, (160, ), (1, ))
    assert_size_stride(arg84_1, (960, ), (1, ))
    assert_size_stride(arg85_1, (960, ), (1, ))
    assert_size_stride(arg86_1, (960, ), (1, ))
    assert_size_stride(arg87_1, (960, ), (1, ))
    assert_size_stride(arg88_1, (160, ), (1, ))
    assert_size_stride(arg89_1, (160, ), (1, ))
    assert_size_stride(arg90_1, (960, ), (1, ))
    assert_size_stride(arg91_1, (960, ), (1, ))
    assert_size_stride(arg92_1, (1000, 1280), (1280, 1))
    assert_size_stride(arg93_1, (1000, ), (1, ))
    assert_size_stride(arg94_1, (16, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg95_1, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg96_1, (16, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg97_1, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg98_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg99_1, (24, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg100_1, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg101_1, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg102_1, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg103_1, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg104_1, (72, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg105_1, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg106_1, (24, ), (1, ))
    assert_size_stride(arg107_1, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg108_1, (72, ), (1, ))
    assert_size_stride(arg109_1, (40, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg110_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg111_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg112_1, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg113_1, (32, ), (1, ))
    assert_size_stride(arg114_1, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg115_1, (120, ), (1, ))
    assert_size_stride(arg116_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg117_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg118_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg119_1, (32, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg120_1, (32, ), (1, ))
    assert_size_stride(arg121_1, (120, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg122_1, (120, ), (1, ))
    assert_size_stride(arg123_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg124_1, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg125_1, (240, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg126_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg127_1, (200, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg128_1, (200, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg129_1, (80, 200, 1, 1), (200, 1, 1, 1))
    assert_size_stride(arg130_1, (184, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg131_1, (184, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg132_1, (80, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg133_1, (184, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg134_1, (184, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg135_1, (80, 184, 1, 1), (184, 1, 1, 1))
    assert_size_stride(arg136_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg137_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg138_1, (120, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg139_1, (120, ), (1, ))
    assert_size_stride(arg140_1, (480, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg141_1, (480, ), (1, ))
    assert_size_stride(arg142_1, (112, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg143_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg144_1, (672, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg145_1, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg146_1, (168, ), (1, ))
    assert_size_stride(arg147_1, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg148_1, (672, ), (1, ))
    assert_size_stride(arg149_1, (112, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg150_1, (672, 112, 1, 1), (112, 1, 1, 1))
    assert_size_stride(arg151_1, (672, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg152_1, (168, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg153_1, (168, ), (1, ))
    assert_size_stride(arg154_1, (672, 168, 1, 1), (168, 1, 1, 1))
    assert_size_stride(arg155_1, (672, ), (1, ))
    assert_size_stride(arg156_1, (160, 672, 1, 1), (672, 1, 1, 1))
    assert_size_stride(arg157_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg158_1, (960, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg159_1, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg160_1, (240, ), (1, ))
    assert_size_stride(arg161_1, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg162_1, (960, ), (1, ))
    assert_size_stride(arg163_1, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg164_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg165_1, (960, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg166_1, (240, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg167_1, (240, ), (1, ))
    assert_size_stride(arg168_1, (960, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg169_1, (960, ), (1, ))
    assert_size_stride(arg170_1, (160, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg171_1, (960, 160, 1, 1), (160, 1, 1, 1))
    assert_size_stride(arg172_1, (1280, 960, 1, 1), (960, 1, 1, 1))
    assert_size_stride(arg173_1, (1280, ), (1, ))
    assert_size_stride(arg174_1, (16, ), (1, ))
    assert_size_stride(arg175_1, (16, ), (1, ))
    assert_size_stride(arg176_1, (16, ), (1, ))
    assert_size_stride(arg177_1, (16, ), (1, ))
    assert_size_stride(arg178_1, (16, ), (1, ))
    assert_size_stride(arg179_1, (16, ), (1, ))
    assert_size_stride(arg180_1, (64, ), (1, ))
    assert_size_stride(arg181_1, (64, ), (1, ))
    assert_size_stride(arg182_1, (64, ), (1, ))
    assert_size_stride(arg183_1, (64, ), (1, ))
    assert_size_stride(arg184_1, (24, ), (1, ))
    assert_size_stride(arg185_1, (24, ), (1, ))
    assert_size_stride(arg186_1, (72, ), (1, ))
    assert_size_stride(arg187_1, (72, ), (1, ))
    assert_size_stride(arg188_1, (72, ), (1, ))
    assert_size_stride(arg189_1, (72, ), (1, ))
    assert_size_stride(arg190_1, (24, ), (1, ))
    assert_size_stride(arg191_1, (24, ), (1, ))
    assert_size_stride(arg192_1, (72, ), (1, ))
    assert_size_stride(arg193_1, (72, ), (1, ))
    assert_size_stride(arg194_1, (72, ), (1, ))
    assert_size_stride(arg195_1, (72, ), (1, ))
    assert_size_stride(arg196_1, (40, ), (1, ))
    assert_size_stride(arg197_1, (40, ), (1, ))
    assert_size_stride(arg198_1, (120, ), (1, ))
    assert_size_stride(arg199_1, (120, ), (1, ))
    assert_size_stride(arg200_1, (120, ), (1, ))
    assert_size_stride(arg201_1, (120, ), (1, ))
    assert_size_stride(arg202_1, (40, ), (1, ))
    assert_size_stride(arg203_1, (40, ), (1, ))
    assert_size_stride(arg204_1, (120, ), (1, ))
    assert_size_stride(arg205_1, (120, ), (1, ))
    assert_size_stride(arg206_1, (120, ), (1, ))
    assert_size_stride(arg207_1, (120, ), (1, ))
    assert_size_stride(arg208_1, (40, ), (1, ))
    assert_size_stride(arg209_1, (40, ), (1, ))
    assert_size_stride(arg210_1, (240, ), (1, ))
    assert_size_stride(arg211_1, (240, ), (1, ))
    assert_size_stride(arg212_1, (240, ), (1, ))
    assert_size_stride(arg213_1, (240, ), (1, ))
    assert_size_stride(arg214_1, (80, ), (1, ))
    assert_size_stride(arg215_1, (80, ), (1, ))
    assert_size_stride(arg216_1, (200, ), (1, ))
    assert_size_stride(arg217_1, (200, ), (1, ))
    assert_size_stride(arg218_1, (200, ), (1, ))
    assert_size_stride(arg219_1, (200, ), (1, ))
    assert_size_stride(arg220_1, (80, ), (1, ))
    assert_size_stride(arg221_1, (80, ), (1, ))
    assert_size_stride(arg222_1, (184, ), (1, ))
    assert_size_stride(arg223_1, (184, ), (1, ))
    assert_size_stride(arg224_1, (184, ), (1, ))
    assert_size_stride(arg225_1, (184, ), (1, ))
    assert_size_stride(arg226_1, (80, ), (1, ))
    assert_size_stride(arg227_1, (80, ), (1, ))
    assert_size_stride(arg228_1, (184, ), (1, ))
    assert_size_stride(arg229_1, (184, ), (1, ))
    assert_size_stride(arg230_1, (184, ), (1, ))
    assert_size_stride(arg231_1, (184, ), (1, ))
    assert_size_stride(arg232_1, (80, ), (1, ))
    assert_size_stride(arg233_1, (80, ), (1, ))
    assert_size_stride(arg234_1, (480, ), (1, ))
    assert_size_stride(arg235_1, (480, ), (1, ))
    assert_size_stride(arg236_1, (480, ), (1, ))
    assert_size_stride(arg237_1, (480, ), (1, ))
    assert_size_stride(arg238_1, (112, ), (1, ))
    assert_size_stride(arg239_1, (112, ), (1, ))
    assert_size_stride(arg240_1, (672, ), (1, ))
    assert_size_stride(arg241_1, (672, ), (1, ))
    assert_size_stride(arg242_1, (672, ), (1, ))
    assert_size_stride(arg243_1, (672, ), (1, ))
    assert_size_stride(arg244_1, (112, ), (1, ))
    assert_size_stride(arg245_1, (112, ), (1, ))
    assert_size_stride(arg246_1, (672, ), (1, ))
    assert_size_stride(arg247_1, (672, ), (1, ))
    assert_size_stride(arg248_1, (672, ), (1, ))
    assert_size_stride(arg249_1, (672, ), (1, ))
    assert_size_stride(arg250_1, (160, ), (1, ))
    assert_size_stride(arg251_1, (160, ), (1, ))
    assert_size_stride(arg252_1, (960, ), (1, ))
    assert_size_stride(arg253_1, (960, ), (1, ))
    assert_size_stride(arg254_1, (960, ), (1, ))
    assert_size_stride(arg255_1, (960, ), (1, ))
    assert_size_stride(arg256_1, (160, ), (1, ))
    assert_size_stride(arg257_1, (160, ), (1, ))
    assert_size_stride(arg258_1, (960, ), (1, ))
    assert_size_stride(arg259_1, (960, ), (1, ))
    assert_size_stride(arg260_1, (960, ), (1, ))
    assert_size_stride(arg261_1, (960, ), (1, ))
    assert_size_stride(arg262_1, (160, ), (1, ))
    assert_size_stride(arg263_1, (160, ), (1, ))
    assert_size_stride(arg264_1, (960, ), (1, ))
    assert_size_stride(arg265_1, (960, ), (1, ))
    assert_size_stride(arg266_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg266_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg266_1
        buf1 = empty_strided((16, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg94_1, buf1, 48, 9, grid=grid(48, 9), stream=stream0)
        del arg94_1
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 16, 112, 112), (200704, 12544, 112, 1))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        buf4 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2.run(buf3, arg174_1, arg175_1, arg0_1, arg1_1, buf4, 128, 12544, grid=grid(128, 12544), stream=stream0)
        del arg0_1
        del arg174_1
        del arg175_1
        del arg1_1
        # Source Nodes: [shortcut, x_5], Original ATen: [aten.convolution, aten.hardswish]
        buf5 = extern_kernels.convolution(buf4, arg95_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf5, (8, 16, 112, 112), (200704, 12544, 112, 1))
        del arg95_1
        buf6 = reinterpret_tensor(buf3, (8, 16, 112, 112), (200704, 1, 1792, 16), 0); del buf3  # reuse
        # Source Nodes: [x_6, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_3.run(buf5, arg176_1, arg177_1, arg2_1, arg3_1, buf6, 128, 12544, grid=grid(128, 12544), stream=stream0)
        del arg176_1
        del arg177_1
        del arg2_1
        del arg3_1
        del buf5
        # Source Nodes: [x_11, x_6, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf7 = extern_kernels.convolution(buf6, arg96_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (8, 16, 112, 112), (200704, 12544, 112, 1))
        del arg96_1
        del buf6
        buf8 = buf4; del buf4  # reuse
        # Source Nodes: [shortcut_1, x_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_4.run(buf8, buf7, arg178_1, arg179_1, arg4_1, arg5_1, 100352, 16, grid=grid(100352, 16), stream=stream0)
        del arg178_1
        del arg179_1
        del arg4_1
        del arg5_1
        del buf7
        # Source Nodes: [shortcut_1, x_12, x_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf9 = extern_kernels.convolution(buf8, arg97_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (8, 64, 112, 112), (802816, 12544, 112, 1))
        del arg97_1
        buf10 = empty_strided((8, 64, 112, 112), (802816, 1, 7168, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_18, x_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf9, arg180_1, arg181_1, arg6_1, arg7_1, buf10, 512, 12544, grid=grid(512, 12544), stream=stream0)
        del arg180_1
        del arg181_1
        del arg6_1
        del arg7_1
        del buf9
        # Source Nodes: [x_18, x_21, x_22], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf11 = extern_kernels.convolution(buf10, arg98_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf11, (8, 64, 56, 56), (200704, 3136, 56, 1))
        del arg98_1
        del buf10
        buf12 = reinterpret_tensor(buf8, (8, 64, 56, 56), (200704, 1, 3584, 64), 0); del buf8  # reuse
        # Source Nodes: [x_23, x_26], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf11, arg182_1, arg183_1, arg8_1, arg9_1, buf12, 512, 3136, grid=grid(512, 3136), stream=stream0)
        del arg182_1
        del arg183_1
        del arg8_1
        del arg9_1
        del buf11
        # Source Nodes: [x_23, x_26, x_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf13 = extern_kernels.convolution(buf12, arg99_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (8, 24, 56, 56), (75264, 3136, 56, 1))
        del arg99_1
        del buf12
        buf14 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_29], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_7.run(buf13, arg184_1, arg185_1, arg10_1, arg11_1, buf14, 192, 3136, grid=grid(192, 3136), stream=stream0)
        del arg10_1
        del arg11_1
        del arg184_1
        del arg185_1
        del buf13
        # Source Nodes: [x_33], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf14, arg100_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (8, 72, 56, 56), (225792, 3136, 56, 1))
        del arg100_1
        buf16 = empty_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_34, x_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf15, arg186_1, arg187_1, arg12_1, arg13_1, buf16, 576, 3136, grid=grid(576, 3136), stream=stream0)
        del arg12_1
        del arg13_1
        del arg186_1
        del arg187_1
        del buf15
        # Source Nodes: [x_34, x_37, x_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf17 = extern_kernels.convolution(buf16, arg101_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf17, (8, 72, 56, 56), (225792, 3136, 56, 1))
        del arg101_1
        buf18 = buf16; del buf16  # reuse
        # Source Nodes: [x_39, x_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf17, arg188_1, arg189_1, arg14_1, arg15_1, buf18, 576, 3136, grid=grid(576, 3136), stream=stream0)
        del arg14_1
        del arg15_1
        del arg188_1
        del arg189_1
        del buf17
        # Source Nodes: [x_39, x_42, x_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf19 = extern_kernels.convolution(buf18, arg102_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (8, 24, 56, 56), (75264, 3136, 56, 1))
        del arg102_1
        buf20 = buf14; del buf14  # reuse
        # Source Nodes: [shortcut_3, x_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_9.run(buf20, buf19, arg190_1, arg191_1, arg16_1, arg17_1, 25088, 24, grid=grid(25088, 24), stream=stream0)
        del arg16_1
        del arg17_1
        del arg190_1
        del arg191_1
        del buf19
        # Source Nodes: [shortcut_3, x_45, x_50], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf21 = extern_kernels.convolution(buf20, arg103_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (8, 72, 56, 56), (225792, 3136, 56, 1))
        del arg103_1
        del buf20
        buf22 = buf18; del buf18  # reuse
        # Source Nodes: [x_51, x_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf21, arg192_1, arg193_1, arg18_1, arg19_1, buf22, 576, 3136, grid=grid(576, 3136), stream=stream0)
        del arg18_1
        del arg192_1
        del arg193_1
        del arg19_1
        del buf21
        # Source Nodes: [x_51, x_54, x_55], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf23 = extern_kernels.convolution(buf22, arg104_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf23, (8, 72, 28, 28), (56448, 784, 28, 1))
        del arg104_1
        del buf22
        buf24 = buf23; del buf23  # reuse
        buf25 = empty_strided((8, 72, 1, 1), (72, 1, 576, 576), device='cuda', dtype=torch.float32)
        buf26 = reinterpret_tensor(buf25, (8, 72, 1, 1), (72, 1, 72, 72), 0); del buf25  # reuse
        # Source Nodes: [x_56, x_59, x_se], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_10.run(buf24, buf26, arg194_1, arg195_1, arg20_1, arg21_1, 576, 784, grid=grid(576), stream=stream0)
        del arg194_1
        del arg195_1
        del arg20_1
        del arg21_1
        # Source Nodes: [x_se, x_se_1], Original ATen: [aten.convolution, aten.mean]
        buf27 = extern_kernels.convolution(buf26, arg105_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf27, (8, 24, 1, 1), (24, 1, 1, 1))
        del arg105_1
        del buf26
        buf28 = reinterpret_tensor(buf27, (8, 24, 1, 1), (24, 1, 24, 24), 0); del buf27  # reuse
        # Source Nodes: [x_se, x_se_1, x_se_2], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_11.run(buf28, arg106_1, 192, grid=grid(192), stream=stream0)
        del arg106_1
        # Source Nodes: [x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf29 = extern_kernels.convolution(buf28, arg107_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (8, 72, 1, 1), (72, 1, 1, 1))
        del arg107_1
        del buf28
        buf30 = empty_strided((8, 72, 28, 28), (56448, 1, 2016, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_60, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_12.run(buf24, buf29, arg108_1, buf30, 576, 784, grid=grid(576, 784), stream=stream0)
        del arg108_1
        del buf24
        del buf29
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____0___se_gate, x_60, x_61, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
        buf31 = extern_kernels.convolution(buf30, arg109_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (8, 40, 28, 28), (31360, 784, 28, 1))
        del arg109_1
        del buf30
        buf32 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_62], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_13.run(buf31, arg196_1, arg197_1, arg22_1, arg23_1, buf32, 320, 784, grid=grid(320, 784), stream=stream0)
        del arg196_1
        del arg197_1
        del arg22_1
        del arg23_1
        del buf31
        # Source Nodes: [x_66], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, arg110_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (8, 120, 28, 28), (94080, 784, 28, 1))
        del arg110_1
        buf34 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_67, x_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf33, arg198_1, arg199_1, arg24_1, arg25_1, buf34, 960, 784, grid=grid(960, 784), stream=stream0)
        del arg198_1
        del arg199_1
        del arg24_1
        del arg25_1
        del buf33
        # Source Nodes: [x_67, x_70, x_71], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf35 = extern_kernels.convolution(buf34, arg111_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf35, (8, 120, 28, 28), (94080, 784, 28, 1))
        del arg111_1
        buf36 = buf35; del buf35  # reuse
        buf37 = empty_strided((8, 120, 1, 1), (120, 1, 960, 960), device='cuda', dtype=torch.float32)
        buf38 = reinterpret_tensor(buf37, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf37  # reuse
        # Source Nodes: [x_72, x_75, x_se_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_15.run(buf36, buf38, arg200_1, arg201_1, arg26_1, arg27_1, 960, 784, grid=grid(960), stream=stream0)
        del arg200_1
        del arg201_1
        del arg26_1
        del arg27_1
        # Source Nodes: [x_se_4, x_se_5], Original ATen: [aten.convolution, aten.mean]
        buf39 = extern_kernels.convolution(buf38, arg112_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg112_1
        del buf38
        buf40 = reinterpret_tensor(buf39, (8, 32, 1, 1), (32, 1, 32, 32), 0); del buf39  # reuse
        # Source Nodes: [x_se_4, x_se_5, x_se_6], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_16.run(buf40, arg113_1, 256, grid=grid(256), stream=stream0)
        del arg113_1
        # Source Nodes: [x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf41 = extern_kernels.convolution(buf40, arg114_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (8, 120, 1, 1), (120, 1, 1, 1))
        del arg114_1
        del buf40
        buf42 = buf34; del buf34  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_76, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_17.run(buf36, buf41, arg115_1, buf42, 960, 784, grid=grid(960, 784), stream=stream0)
        del arg115_1
        del buf36
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____1___se_gate, x_76, x_77, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
        buf43 = extern_kernels.convolution(buf42, arg116_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (8, 40, 28, 28), (31360, 784, 28, 1))
        del arg116_1
        buf44 = buf32; del buf32  # reuse
        # Source Nodes: [shortcut_5, x_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_18.run(buf44, buf43, arg202_1, arg203_1, arg28_1, arg29_1, 6272, 40, grid=grid(6272, 40), stream=stream0)
        del arg202_1
        del arg203_1
        del arg28_1
        del arg29_1
        del buf43
        # Source Nodes: [x_83], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, arg117_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (8, 120, 28, 28), (94080, 784, 28, 1))
        del arg117_1
        buf46 = buf42; del buf42  # reuse
        # Source Nodes: [x_84, x_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf45, arg204_1, arg205_1, arg30_1, arg31_1, buf46, 960, 784, grid=grid(960, 784), stream=stream0)
        del arg204_1
        del arg205_1
        del arg30_1
        del arg31_1
        del buf45
        # Source Nodes: [x_84, x_87, x_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf47 = extern_kernels.convolution(buf46, arg118_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf47, (8, 120, 28, 28), (94080, 784, 28, 1))
        del arg118_1
        buf48 = buf47; del buf47  # reuse
        buf49 = reinterpret_tensor(buf41, (8, 120, 1, 1), (120, 1, 960, 960), 0); del buf41  # reuse
        buf50 = reinterpret_tensor(buf49, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf49  # reuse
        # Source Nodes: [x_89, x_92, x_se_8], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_15.run(buf48, buf50, arg206_1, arg207_1, arg32_1, arg33_1, 960, 784, grid=grid(960), stream=stream0)
        del arg206_1
        del arg207_1
        del arg32_1
        del arg33_1
        # Source Nodes: [x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean]
        buf51 = extern_kernels.convolution(buf50, arg119_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg119_1
        del buf50
        buf52 = reinterpret_tensor(buf51, (8, 32, 1, 1), (32, 1, 32, 32), 0); del buf51  # reuse
        # Source Nodes: [x_se_10, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.relu]
        triton_poi_fused_convolution_mean_relu_16.run(buf52, arg120_1, 256, grid=grid(256), stream=stream0)
        del arg120_1
        # Source Nodes: [x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.mean, aten.relu]
        buf53 = extern_kernels.convolution(buf52, arg121_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (8, 120, 1, 1), (120, 1, 1, 1))
        del arg121_1
        del buf52
        buf54 = buf46; del buf46  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___se_gate, x_93, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_convolution_hardsigmoid_mean_mul_relu_17.run(buf48, buf53, arg122_1, buf54, 960, 784, grid=grid(960, 784), stream=stream0)
        del arg122_1
        del buf48
        del buf53
        # Source Nodes: [getattr_getattr_l__mod___blocks___2_____2___se_gate, x_93, x_94, x_se_10, x_se_11, x_se_8, x_se_9], Original ATen: [aten.convolution, aten.hardsigmoid, aten.mean, aten.mul, aten.relu]
        buf55 = extern_kernels.convolution(buf54, arg123_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (8, 40, 28, 28), (31360, 784, 28, 1))
        del arg123_1
        buf56 = buf44; del buf44  # reuse
        # Source Nodes: [shortcut_6, x_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_18.run(buf56, buf55, arg208_1, arg209_1, arg34_1, arg35_1, 6272, 40, grid=grid(6272, 40), stream=stream0)
        del arg208_1
        del arg209_1
        del arg34_1
        del arg35_1
        del buf55
        # Source Nodes: [shortcut_6, x_100, x_95], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf57 = extern_kernels.convolution(buf56, arg124_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf57, (8, 240, 28, 28), (188160, 784, 28, 1))
        del arg124_1
        del buf56
        buf58 = buf57; del buf57  # reuse
        buf59 = empty_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_101, x_104], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_19.run(buf58, arg210_1, arg211_1, arg36_1, arg37_1, buf59, 1920, 784, grid=grid(1920, 784), stream=stream0)
        del arg210_1
        del arg211_1
        del arg36_1
        del arg37_1
        del buf58
        # Source Nodes: [x_104, x_105], Original ATen: [aten.convolution, aten.hardswish]
        buf60 = extern_kernels.convolution(buf59, arg125_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf60, (8, 240, 14, 14), (47040, 196, 14, 1))
        del arg125_1
        del buf59
        buf61 = buf60; del buf60  # reuse
        buf62 = empty_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_106, x_109], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_20.run(buf61, arg212_1, arg213_1, arg38_1, arg39_1, buf62, 1920, 196, grid=grid(1920, 196), stream=stream0)
        del arg212_1
        del arg213_1
        del arg38_1
        del arg39_1
        del buf61
        # Source Nodes: [x_109, x_111], Original ATen: [aten.convolution, aten.hardswish]
        buf63 = extern_kernels.convolution(buf62, arg126_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf63, (8, 80, 14, 14), (15680, 196, 14, 1))
        del arg126_1
        buf64 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_112], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_21.run(buf63, arg214_1, arg215_1, arg40_1, arg41_1, buf64, 640, 196, grid=grid(640, 196), stream=stream0)
        del arg214_1
        del arg215_1
        del arg40_1
        del arg41_1
        del buf63
        # Source Nodes: [x_116], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf64, arg127_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (8, 200, 14, 14), (39200, 196, 14, 1))
        del arg127_1
        buf66 = buf65; del buf65  # reuse
        buf67 = empty_strided((8, 200, 14, 14), (39200, 1, 2800, 200), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_117, x_120], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22.run(buf66, arg216_1, arg217_1, arg42_1, arg43_1, buf67, 1600, 196, grid=grid(1600, 196), stream=stream0)
        del arg216_1
        del arg217_1
        del arg42_1
        del arg43_1
        del buf66
        # Source Nodes: [x_120, x_121], Original ATen: [aten.convolution, aten.hardswish]
        buf68 = extern_kernels.convolution(buf67, arg128_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=200, bias=None)
        assert_size_stride(buf68, (8, 200, 14, 14), (39200, 196, 14, 1))
        del arg128_1
        buf69 = buf68; del buf68  # reuse
        buf70 = buf67; del buf67  # reuse
        # Source Nodes: [x_122, x_125], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_22.run(buf69, arg218_1, arg219_1, arg44_1, arg45_1, buf70, 1600, 196, grid=grid(1600, 196), stream=stream0)
        del arg218_1
        del arg219_1
        del arg44_1
        del arg45_1
        del buf69
        # Source Nodes: [x_125, x_127], Original ATen: [aten.convolution, aten.hardswish]
        buf71 = extern_kernels.convolution(buf70, arg129_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (8, 80, 14, 14), (15680, 196, 14, 1))
        del arg129_1
        del buf70
        buf72 = buf64; del buf64  # reuse
        # Source Nodes: [shortcut_8, x_128], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_23.run(buf72, buf71, arg220_1, arg221_1, arg46_1, arg47_1, 1568, 80, grid=grid(1568, 80), stream=stream0)
        del arg220_1
        del arg221_1
        del arg46_1
        del arg47_1
        del buf71
        # Source Nodes: [x_133], Original ATen: [aten.convolution]
        buf73 = extern_kernels.convolution(buf72, arg130_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf73, (8, 184, 14, 14), (36064, 196, 14, 1))
        del arg130_1
        buf74 = buf73; del buf73  # reuse
        buf75 = empty_strided((8, 184, 14, 14), (36064, 1, 2576, 184), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_134, x_137], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_24.run(buf74, arg222_1, arg223_1, arg48_1, arg49_1, buf75, 1472, 196, grid=grid(1472, 196), stream=stream0)
        del arg222_1
        del arg223_1
        del arg48_1
        del arg49_1
        del buf74
        # Source Nodes: [x_137, x_138], Original ATen: [aten.convolution, aten.hardswish]
        buf76 = extern_kernels.convolution(buf75, arg131_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=184, bias=None)
        assert_size_stride(buf76, (8, 184, 14, 14), (36064, 196, 14, 1))
        del arg131_1
        buf77 = buf76; del buf76  # reuse
        buf78 = buf75; del buf75  # reuse
        # Source Nodes: [x_139, x_142], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_24.run(buf77, arg224_1, arg225_1, arg50_1, arg51_1, buf78, 1472, 196, grid=grid(1472, 196), stream=stream0)
        del arg224_1
        del arg225_1
        del arg50_1
        del arg51_1
        del buf77
        # Source Nodes: [x_142, x_144], Original ATen: [aten.convolution, aten.hardswish]
        buf79 = extern_kernels.convolution(buf78, arg132_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (8, 80, 14, 14), (15680, 196, 14, 1))
        del arg132_1
        buf80 = buf72; del buf72  # reuse
        # Source Nodes: [shortcut_9, x_145], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_23.run(buf80, buf79, arg226_1, arg227_1, arg52_1, arg53_1, 1568, 80, grid=grid(1568, 80), stream=stream0)
        del arg226_1
        del arg227_1
        del arg52_1
        del arg53_1
        del buf79
        # Source Nodes: [x_150], Original ATen: [aten.convolution]
        buf81 = extern_kernels.convolution(buf80, arg133_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (8, 184, 14, 14), (36064, 196, 14, 1))
        del arg133_1
        buf82 = buf81; del buf81  # reuse
        buf83 = buf78; del buf78  # reuse
        # Source Nodes: [x_151, x_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_24.run(buf82, arg228_1, arg229_1, arg54_1, arg55_1, buf83, 1472, 196, grid=grid(1472, 196), stream=stream0)
        del arg228_1
        del arg229_1
        del arg54_1
        del arg55_1
        del buf82
        # Source Nodes: [x_154, x_155], Original ATen: [aten.convolution, aten.hardswish]
        buf84 = extern_kernels.convolution(buf83, arg134_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=184, bias=None)
        assert_size_stride(buf84, (8, 184, 14, 14), (36064, 196, 14, 1))
        del arg134_1
        buf85 = buf84; del buf84  # reuse
        buf86 = buf83; del buf83  # reuse
        # Source Nodes: [x_156, x_159], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_24.run(buf85, arg230_1, arg231_1, arg56_1, arg57_1, buf86, 1472, 196, grid=grid(1472, 196), stream=stream0)
        del arg230_1
        del arg231_1
        del arg56_1
        del arg57_1
        del buf85
        # Source Nodes: [x_159, x_161], Original ATen: [aten.convolution, aten.hardswish]
        buf87 = extern_kernels.convolution(buf86, arg135_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf87, (8, 80, 14, 14), (15680, 196, 14, 1))
        del arg135_1
        del buf86
        buf88 = buf80; del buf80  # reuse
        # Source Nodes: [shortcut_10, x_162], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_23.run(buf88, buf87, arg232_1, arg233_1, arg58_1, arg59_1, 1568, 80, grid=grid(1568, 80), stream=stream0)
        del arg232_1
        del arg233_1
        del arg58_1
        del arg59_1
        del buf87
        # Source Nodes: [shortcut_10, x_162, x_167], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf89 = extern_kernels.convolution(buf88, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (8, 480, 14, 14), (94080, 196, 14, 1))
        del arg136_1
        del buf88
        buf90 = buf89; del buf89  # reuse
        buf91 = reinterpret_tensor(buf54, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf54  # reuse
        # Source Nodes: [x_168, x_171], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_25.run(buf90, arg234_1, arg235_1, arg60_1, arg61_1, buf91, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del arg234_1
        del arg235_1
        del arg60_1
        del arg61_1
        del buf90
        # Source Nodes: [x_171, x_172], Original ATen: [aten.convolution, aten.hardswish]
        buf92 = extern_kernels.convolution(buf91, arg137_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf92, (8, 480, 14, 14), (94080, 196, 14, 1))
        del arg137_1
        buf93 = buf92; del buf92  # reuse
        buf94 = empty_strided((8, 480, 1, 1), (480, 1, 3840, 3840), device='cuda', dtype=torch.float32)
        buf95 = reinterpret_tensor(buf94, (8, 480, 1, 1), (480, 1, 480, 480), 0); del buf94  # reuse
        # Source Nodes: [x_173, x_176, x_se_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_26.run(buf93, buf95, arg236_1, arg237_1, arg62_1, arg63_1, 3840, 196, grid=grid(3840), stream=stream0)
        del arg236_1
        del arg237_1
        del arg62_1
        del arg63_1
        # Source Nodes: [x_176, x_se_12, x_se_13], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf96 = extern_kernels.convolution(buf95, arg138_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (8, 120, 1, 1), (120, 1, 1, 1))
        del arg138_1
        del buf95
        buf97 = reinterpret_tensor(buf96, (8, 120, 1, 1), (120, 1, 120, 120), 0); del buf96  # reuse
        # Source Nodes: [x_176, x_se_12, x_se_13, x_se_14], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
        triton_poi_fused_convolution_hardswish_mean_relu_27.run(buf97, arg139_1, 960, grid=grid(960), stream=stream0)
        del arg139_1
        # Source Nodes: [x_176, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
        buf98 = extern_kernels.convolution(buf97, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (8, 480, 1, 1), (480, 1, 1, 1))
        del arg140_1
        del buf97
        buf99 = buf91; del buf91  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_176, x_177, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_28.run(buf93, buf98, arg141_1, buf99, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del arg141_1
        del buf93
        del buf98
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____0___se_gate, x_176, x_177, x_178, x_se_12, x_se_13, x_se_14, x_se_15], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
        buf100 = extern_kernels.convolution(buf99, arg142_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (8, 112, 14, 14), (21952, 196, 14, 1))
        del arg142_1
        del buf99
        buf101 = empty_strided((8, 112, 14, 14), (21952, 1, 1568, 112), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_179], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_29.run(buf100, arg238_1, arg239_1, arg64_1, arg65_1, buf101, 896, 196, grid=grid(896, 196), stream=stream0)
        del arg238_1
        del arg239_1
        del arg64_1
        del arg65_1
        del buf100
        # Source Nodes: [x_183], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, arg143_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 672, 14, 14), (131712, 196, 14, 1))
        del arg143_1
        buf103 = buf102; del buf102  # reuse
        buf104 = empty_strided((8, 672, 14, 14), (131712, 1, 9408, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_184, x_187], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_30.run(buf103, arg240_1, arg241_1, arg66_1, arg67_1, buf104, 5376, 196, grid=grid(5376, 196), stream=stream0)
        del arg240_1
        del arg241_1
        del arg66_1
        del arg67_1
        del buf103
        # Source Nodes: [x_187, x_188], Original ATen: [aten.convolution, aten.hardswish]
        buf105 = extern_kernels.convolution(buf104, arg144_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf105, (8, 672, 14, 14), (131712, 196, 14, 1))
        del arg144_1
        buf106 = buf105; del buf105  # reuse
        buf107 = empty_strided((8, 672, 1, 1), (672, 1, 5376, 5376), device='cuda', dtype=torch.float32)
        buf108 = reinterpret_tensor(buf107, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf107  # reuse
        # Source Nodes: [x_189, x_192, x_se_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_31.run(buf106, buf108, arg242_1, arg243_1, arg68_1, arg69_1, 5376, 196, grid=grid(5376), stream=stream0)
        del arg242_1
        del arg243_1
        del arg68_1
        del arg69_1
        # Source Nodes: [x_192, x_se_16, x_se_17], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf109 = extern_kernels.convolution(buf108, arg145_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (8, 168, 1, 1), (168, 1, 1, 1))
        del arg145_1
        del buf108
        buf110 = reinterpret_tensor(buf109, (8, 168, 1, 1), (168, 1, 168, 168), 0); del buf109  # reuse
        # Source Nodes: [x_192, x_se_16, x_se_17, x_se_18], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
        triton_poi_fused_convolution_hardswish_mean_relu_32.run(buf110, arg146_1, 1344, grid=grid(1344), stream=stream0)
        del arg146_1
        # Source Nodes: [x_192, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
        buf111 = extern_kernels.convolution(buf110, arg147_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf111, (8, 672, 1, 1), (672, 1, 1, 1))
        del arg147_1
        del buf110
        buf112 = buf104; del buf104  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_192, x_193, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_33.run(buf106, buf111, arg148_1, buf112, 5376, 196, grid=grid(5376, 196), stream=stream0)
        del arg148_1
        del buf106
        # Source Nodes: [getattr_getattr_l__mod___blocks___4_____1___se_gate, x_192, x_193, x_194, x_se_16, x_se_17, x_se_18, x_se_19], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
        buf113 = extern_kernels.convolution(buf112, arg149_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (8, 112, 14, 14), (21952, 196, 14, 1))
        del arg149_1
        buf114 = buf101; del buf101  # reuse
        # Source Nodes: [shortcut_12, x_195], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_34.run(buf114, buf113, arg244_1, arg245_1, arg70_1, arg71_1, 1568, 112, grid=grid(1568, 112), stream=stream0)
        del arg244_1
        del arg245_1
        del arg70_1
        del arg71_1
        del buf113
        # Source Nodes: [shortcut_12, x_195, x_200], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf115 = extern_kernels.convolution(buf114, arg150_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf115, (8, 672, 14, 14), (131712, 196, 14, 1))
        del arg150_1
        del buf114
        buf116 = buf115; del buf115  # reuse
        buf117 = buf112; del buf112  # reuse
        # Source Nodes: [x_201, x_204], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_30.run(buf116, arg246_1, arg247_1, arg72_1, arg73_1, buf117, 5376, 196, grid=grid(5376, 196), stream=stream0)
        del arg246_1
        del arg247_1
        del arg72_1
        del arg73_1
        del buf116
        # Source Nodes: [x_204, x_205], Original ATen: [aten.convolution, aten.hardswish]
        buf118 = extern_kernels.convolution(buf117, arg151_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=672, bias=None)
        assert_size_stride(buf118, (8, 672, 7, 7), (32928, 49, 7, 1))
        del arg151_1
        del buf117
        buf119 = buf118; del buf118  # reuse
        buf120 = reinterpret_tensor(buf111, (8, 672, 1, 1), (672, 1, 5376, 5376), 0); del buf111  # reuse
        buf121 = reinterpret_tensor(buf120, (8, 672, 1, 1), (672, 1, 672, 672), 0); del buf120  # reuse
        # Source Nodes: [x_206, x_209, x_se_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_35.run(buf119, buf121, arg248_1, arg249_1, arg74_1, arg75_1, 5376, 49, grid=grid(5376), stream=stream0)
        del arg248_1
        del arg249_1
        del arg74_1
        del arg75_1
        # Source Nodes: [x_209, x_se_20, x_se_21], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf122 = extern_kernels.convolution(buf121, arg152_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (8, 168, 1, 1), (168, 1, 1, 1))
        del arg152_1
        del buf121
        buf123 = reinterpret_tensor(buf122, (8, 168, 1, 1), (168, 1, 168, 168), 0); del buf122  # reuse
        # Source Nodes: [x_209, x_se_20, x_se_21, x_se_22], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
        triton_poi_fused_convolution_hardswish_mean_relu_32.run(buf123, arg153_1, 1344, grid=grid(1344), stream=stream0)
        del arg153_1
        # Source Nodes: [x_209, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
        buf124 = extern_kernels.convolution(buf123, arg154_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (8, 672, 1, 1), (672, 1, 1, 1))
        del arg154_1
        del buf123
        buf125 = empty_strided((8, 672, 7, 7), (32928, 1, 4704, 672), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_209, x_210, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_36.run(buf119, buf124, arg155_1, buf125, 5376, 49, grid=grid(5376, 49), stream=stream0)
        del arg155_1
        del buf119
        del buf124
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_209, x_210, x_211, x_se_20, x_se_21, x_se_22, x_se_23], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
        buf126 = extern_kernels.convolution(buf125, arg156_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (8, 160, 7, 7), (7840, 49, 7, 1))
        del arg156_1
        del buf125
        buf127 = empty_strided((8, 160, 7, 7), (7840, 1, 1120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_212], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_37.run(buf126, arg250_1, arg251_1, arg76_1, arg77_1, buf127, 1280, 49, grid=grid(1280, 49), stream=stream0)
        del arg250_1
        del arg251_1
        del arg76_1
        del arg77_1
        del buf126
        # Source Nodes: [x_216], Original ATen: [aten.convolution]
        buf128 = extern_kernels.convolution(buf127, arg157_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf128, (8, 960, 7, 7), (47040, 49, 7, 1))
        del arg157_1
        buf129 = buf128; del buf128  # reuse
        buf130 = reinterpret_tensor(buf62, (8, 960, 7, 7), (47040, 1, 6720, 960), 0); del buf62  # reuse
        # Source Nodes: [x_217, x_220], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38.run(buf129, arg252_1, arg253_1, arg78_1, arg79_1, buf130, 7680, 49, grid=grid(7680, 49), stream=stream0)
        del arg252_1
        del arg253_1
        del arg78_1
        del arg79_1
        del buf129
        # Source Nodes: [x_220, x_221], Original ATen: [aten.convolution, aten.hardswish]
        buf131 = extern_kernels.convolution(buf130, arg158_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf131, (8, 960, 7, 7), (47040, 49, 7, 1))
        del arg158_1
        buf132 = buf131; del buf131  # reuse
        buf133 = empty_strided((8, 960, 1, 1), (960, 1, 7680, 7680), device='cuda', dtype=torch.float32)
        buf134 = reinterpret_tensor(buf133, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf133  # reuse
        # Source Nodes: [x_222, x_225, x_se_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_39.run(buf132, buf134, arg254_1, arg255_1, arg80_1, arg81_1, 7680, 49, grid=grid(7680), stream=stream0)
        del arg254_1
        del arg255_1
        del arg80_1
        del arg81_1
        # Source Nodes: [x_225, x_se_24, x_se_25], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf135 = extern_kernels.convolution(buf134, arg159_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf135, (8, 240, 1, 1), (240, 1, 1, 1))
        del arg159_1
        del buf134
        buf136 = reinterpret_tensor(buf135, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf135  # reuse
        # Source Nodes: [x_225, x_se_24, x_se_25, x_se_26], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
        triton_poi_fused_convolution_hardswish_mean_relu_40.run(buf136, arg160_1, 1920, grid=grid(1920), stream=stream0)
        del arg160_1
        # Source Nodes: [x_225, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
        buf137 = extern_kernels.convolution(buf136, arg161_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (8, 960, 1, 1), (960, 1, 1, 1))
        del arg161_1
        del buf136
        buf138 = buf130; del buf130  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_225, x_226, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_41.run(buf132, buf137, arg162_1, buf138, 7680, 49, grid=grid(7680, 49), stream=stream0)
        del arg162_1
        del buf132
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_225, x_226, x_227, x_se_24, x_se_25, x_se_26, x_se_27], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
        buf139 = extern_kernels.convolution(buf138, arg163_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf139, (8, 160, 7, 7), (7840, 49, 7, 1))
        del arg163_1
        buf140 = buf127; del buf127  # reuse
        # Source Nodes: [shortcut_14, x_228], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_42.run(buf140, buf139, arg256_1, arg257_1, arg82_1, arg83_1, 392, 160, grid=grid(392, 160), stream=stream0)
        del arg256_1
        del arg257_1
        del arg82_1
        del arg83_1
        del buf139
        # Source Nodes: [x_233], Original ATen: [aten.convolution]
        buf141 = extern_kernels.convolution(buf140, arg164_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf141, (8, 960, 7, 7), (47040, 49, 7, 1))
        del arg164_1
        buf142 = buf141; del buf141  # reuse
        buf143 = buf138; del buf138  # reuse
        # Source Nodes: [x_234, x_237], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_38.run(buf142, arg258_1, arg259_1, arg84_1, arg85_1, buf143, 7680, 49, grid=grid(7680, 49), stream=stream0)
        del arg258_1
        del arg259_1
        del arg84_1
        del arg85_1
        del buf142
        # Source Nodes: [x_237, x_238], Original ATen: [aten.convolution, aten.hardswish]
        buf144 = extern_kernels.convolution(buf143, arg165_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=960, bias=None)
        assert_size_stride(buf144, (8, 960, 7, 7), (47040, 49, 7, 1))
        del arg165_1
        buf145 = buf144; del buf144  # reuse
        buf146 = reinterpret_tensor(buf137, (8, 960, 1, 1), (960, 1, 7680, 7680), 0); del buf137  # reuse
        buf147 = reinterpret_tensor(buf146, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf146  # reuse
        # Source Nodes: [x_239, x_242, x_se_28], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_39.run(buf145, buf147, arg260_1, arg261_1, arg86_1, arg87_1, 7680, 49, grid=grid(7680), stream=stream0)
        del arg260_1
        del arg261_1
        del arg86_1
        del arg87_1
        # Source Nodes: [x_242, x_se_28, x_se_29], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf148 = extern_kernels.convolution(buf147, arg166_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (8, 240, 1, 1), (240, 1, 1, 1))
        del arg166_1
        del buf147
        buf149 = reinterpret_tensor(buf148, (8, 240, 1, 1), (240, 1, 240, 240), 0); del buf148  # reuse
        # Source Nodes: [x_242, x_se_28, x_se_29, x_se_30], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
        triton_poi_fused_convolution_hardswish_mean_relu_40.run(buf149, arg167_1, 1920, grid=grid(1920), stream=stream0)
        del arg167_1
        # Source Nodes: [x_242, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
        buf150 = extern_kernels.convolution(buf149, arg168_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (8, 960, 1, 1), (960, 1, 1, 1))
        del arg168_1
        del buf149
        buf151 = buf143; del buf143  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate, x_242, x_243, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_41.run(buf145, buf150, arg169_1, buf151, 7680, 49, grid=grid(7680, 49), stream=stream0)
        del arg169_1
        del buf145
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____2___se_gate, x_242, x_243, x_244, x_se_28, x_se_29, x_se_30, x_se_31], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
        buf152 = extern_kernels.convolution(buf151, arg170_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf152, (8, 160, 7, 7), (7840, 49, 7, 1))
        del arg170_1
        del buf151
        buf153 = buf140; del buf140  # reuse
        # Source Nodes: [shortcut_15, x_245], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_42.run(buf153, buf152, arg262_1, arg263_1, arg88_1, arg89_1, 392, 160, grid=grid(392, 160), stream=stream0)
        del arg262_1
        del arg263_1
        del arg88_1
        del arg89_1
        del buf152
        # Source Nodes: [shortcut_15, x_245, x_250], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf154 = extern_kernels.convolution(buf153, arg171_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (8, 960, 7, 7), (47040, 49, 7, 1))
        del arg171_1
        del buf153
        buf155 = buf154; del buf154  # reuse
        buf156 = reinterpret_tensor(buf150, (8, 960, 1, 1), (960, 1, 7680, 7680), 0); del buf150  # reuse
        buf157 = reinterpret_tensor(buf156, (8, 960, 1, 1), (960, 1, 960, 960), 0); del buf156  # reuse
        # Source Nodes: [x_251, x_256, x_257], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_43.run(buf155, buf157, arg264_1, arg265_1, arg90_1, arg91_1, 7680, 49, grid=grid(7680), stream=stream0)
        del arg264_1
        del arg265_1
        del arg90_1
        del arg91_1
        del buf155
        # Source Nodes: [x_256, x_257, x_260], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf158 = extern_kernels.convolution(buf157, arg172_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (8, 1280, 1, 1), (1280, 1, 1, 1))
        del arg172_1
        del buf157
        buf159 = buf158; del buf158  # reuse
        # Source Nodes: [x_256, x_257, x_260, x_261], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        triton_poi_fused_convolution_hardswish_mean_44.run(buf159, arg173_1, 10240, grid=grid(10240), stream=stream0)
        del arg173_1
        buf160 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_263], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg93_1, reinterpret_tensor(buf159, (8, 1280), (1280, 1), 0), reinterpret_tensor(arg92_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf160)
        del arg92_1
        del arg93_1
        return (buf160, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((16, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((16, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((24, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((72, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((40, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((32, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((120, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((240, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((200, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((200, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((80, 200, 1, 1), (200, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((184, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((184, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((80, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((184, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((184, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((80, 184, 1, 1), (184, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((120, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((480, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((112, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((672, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((112, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((672, 112, 1, 1), (112, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((672, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((168, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((168, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((672, 168, 1, 1), (168, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((160, 672, 1, 1), (672, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((960, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((960, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((240, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((960, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((160, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((960, 160, 1, 1), (160, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((1280, 960, 1, 1), (960, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((200, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((184, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((112, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((672, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((960, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mobilenetv3_large_100', benchmark_compiled_module)
