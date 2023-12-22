
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


# kernel path: /tmp/torchinductor_youkaichao/tu/ctumxipyscewkktamsgnnu4ihs7cqf72uhnaceeoqgrcs6jmku23.py
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
    size_hints=[32, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
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


# kernel path: /tmp/torchinductor_youkaichao/4p/c4px6yem5pfx5jnkz7yfonrokb5yjfhcgyzxtrn2wu5i4dp62ymy.py
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
    y0 = yindex % 8
    y1 = (yindex // 8)
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
    tl.store(out_ptr0 + (y0 + (8*x2) + (100352*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cr/ccr7322rww23bsr7zcszhxvk4jrrrum2pb4kkiwo5z24u4muisv5.py
# Source Nodes: [shortcut_1, x_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# shortcut_1 => add_8, clamp_max_2, clamp_min_2, div_2, mul_11
# x_12 => add_7, mul_10, mul_9, sub_2
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_3', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/sl/csl36x3kb62s3hpgmdwofteo7jkwdtr4ounm3ihn7iidbizzmxxl.py
# Source Nodes: [x_17, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# x_17 => add_10, mul_13, mul_14, sub_3
# x_20 => add_11, clamp_max_3, clamp_min_3, div_3, mul_15
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 3136
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
    tmp0 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (16*x2) + (50176*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6f/c6fegtbt2xavpnki2u7t34tz7o4e4ctb45mooftaeda4a3f45dxd.py
# Source Nodes: [shortcut_2, x_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# shortcut_2 => add_14, clamp_max_4, clamp_min_4, div_4, mul_19
# x_23 => add_13, mul_17, mul_18, sub_4
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 32
    y1 = (yindex // 32)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (3136*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (32*x2) + (100352*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4f/c4ft53pml3fbcbiqyycx3i3ubfy3pohmpt236uf6fhpwg6zsfwaq.py
# Source Nodes: [x_39, x_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# x_39 => add_22, mul_29, mul_30, sub_7
# x_42 => add_23, clamp_max_7, clamp_min_7, div_7, mul_31
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 784
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
    tl.store(out_ptr0 + (y0 + (32*x2) + (25088*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/mc/cmcuv43ewv67brszhlerxok7suftc5pmhckp4e5ctqvchimeewal.py
# Source Nodes: [shortcut_4, x_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# shortcut_4 => add_26, clamp_max_8, clamp_min_8, div_8, mul_35
# x_45 => add_25, mul_33, mul_34, sub_8
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_7', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 784
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
    tl.store(out_ptr0 + (y0 + (64*x2) + (50176*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4c/c4czvwfvjow7ugnebnkx4szme5wxkh3himvfyeh5dsowlnnz2kcn.py
# Source Nodes: [x_61, x_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# x_61 => add_34, mul_45, mul_46, sub_11
# x_64 => add_35, clamp_max_11, clamp_min_11, div_11, mul_47
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_8', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 196
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
    tl.store(out_ptr0 + (y0 + (64*x2) + (12544*y1)), tmp22, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/pu/cpusto3qdngm3llq6sc2s5fpibnhdafvil52nlc7hvl2nck6xddd.py
# Source Nodes: [shortcut_6, x_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# shortcut_6 => add_38, clamp_max_12, clamp_min_12, div_12, mul_51
# x_67 => add_37, mul_49, mul_50, sub_12
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (128*x2) + (25088*y1)), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/za/czast6pysoiz2adpt5d5x2w2vpinwcbazz3i3raus3qt6e6w2rd2.py
# Source Nodes: [x_127, x_130, x_se], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
# x_127 => add_70, mul_93, mul_94, sub_23
# x_130 => add_71, clamp_max_23, clamp_min_23, div_23, mul_95
# x_se => mean
triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_10', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 128
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


# kernel path: /tmp/torchinductor_youkaichao/zj/czjj4dfp73regt4ro243evezl7end4prvvp4cnp4wfgpn2szbnek.py
# Source Nodes: [x_130, x_se, x_se_1, x_se_2], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
# x_130 => add_71, clamp_max_23, clamp_min_23, div_23, mul_95
# x_se => mean
# x_se_1 => convolution_24
# x_se_2 => relu
triton_poi_fused_convolution_hardswish_mean_relu_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_relu_11', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/uh/cuh73pqmcxkcbvvqrdxqbxxq7eymhjbat4athohoiu5xpnkhda5r.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_130, x_131, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
# getattr_getattr_l__mod___blocks___5_____0___se_gate => add_72, clamp_max_24, clamp_min_24, div_24
# x_130 => add_71, clamp_max_23, clamp_min_23, div_23, mul_95
# x_131 => mul_96
# x_se => mean
# x_se_1 => convolution_24
# x_se_2 => relu
# x_se_3 => convolution_25
triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (y3), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (128*x2) + (6272*y1)), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zl/czlf5suarqzvd3noqxk6l4skhs7xsht7jlbl5oubk2rmpjmiwndq.py
# Source Nodes: [shortcut_12, x_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
# shortcut_12 => add_75, clamp_max_25, clamp_min_25, div_25, mul_100
# x_133 => add_74, mul_98, mul_99, sub_24
triton_poi_fused__native_batch_norm_legit_no_training_hardswish_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_hardswish_13', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_out_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (256*x2) + (12544*y1)), tmp22, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kj/ckjjmola65mbfc2os7omtl3nq64ywgrjnptn43tesdzdufhz7cwn.py
# Source Nodes: [x_138, x_141, x_se_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
# x_138 => add_77, mul_102, mul_103, sub_25
# x_141 => add_78, clamp_max_26, clamp_min_26, div_26, mul_104
# x_se_4 => mean_1
triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_14', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (r2 + (49*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = 49.0
    tmp28 = tmp26 / tmp27
    tl.store(in_out_ptr0 + (r2 + (49*x3)), tmp14, rmask)
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/tk/ctk7rfckyzvutdphtnuga7olttgzxvojhxl7gf56ryy57gxdsixj.py
# Source Nodes: [x_141, x_se_4, x_se_5, x_se_6], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
# x_141 => add_78, clamp_max_26, clamp_min_26, div_26, mul_104
# x_se_4 => mean_1
# x_se_5 => convolution_28
# x_se_6 => relu_1
triton_poi_fused_convolution_hardswish_mean_relu_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_relu_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 512
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_out_ptr0 + (x2), xmask)
    tmp1 = tl.load(in_ptr0 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/kx/ckxahmfy5jwq4mp34fctl7svjmvi53cxtsw74r43fm7o4h22igf2.py
# Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_141, x_142, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
# getattr_getattr_l__mod___blocks___5_____1___se_gate => add_79, clamp_max_27, clamp_min_27, div_27
# x_141 => add_78, clamp_max_26, clamp_min_26, div_26, mul_104
# x_142 => mul_105
# x_se_4 => mean_1
# x_se_5 => convolution_28
# x_se_6 => relu_1
# x_se_7 => convolution_29
triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 49
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr1 + (y3), None, eviction_policy='evict_last')
    tmp10 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (256*x2) + (12544*y1)), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ts/ctsoi2nd3yivvnsaz5qni3ovrmfwcstzqythixqwe4be6l7tsuc5.py
# Source Nodes: [x_144, x_149, x_150], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
# x_144 => add_81, mul_107, mul_108, sub_26
# x_149 => add_82, clamp_max_28, clamp_min_28, div_28, mul_109
# x_150 => mean_2
triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_17', 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1']}
)
@triton.jit
def triton_(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 49
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 256
    tmp0 = tl.load(in_out_ptr0 + (r2 + (49*x3)), rmask, other=0.0)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
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
    tmp25 = tl.where(rmask, tmp23, 0)
    tmp26 = tl.sum(tmp25, 1)[:, None]
    tmp27 = 49.0
    tmp28 = tmp26 / tmp27
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x3), tmp28, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/vm/cvmppq7pmmnyep4hss56y6pivfhrt37ttqtgyfi6klu4b77ofsrv.py
# Source Nodes: [x_149, x_150, x_153, x_154], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
# x_149 => add_82, clamp_max_28, clamp_min_28, div_28, mul_109
# x_150 => mean_2
# x_153 => convolution_31
# x_154 => add_83, clamp_max_29, clamp_min_29, div_29, mul_110
triton_poi_fused_convolution_hardswish_mean_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_hardswish_mean_18', 'mutated_arg_names': ['in_out_ptr0']},
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1 = args
    args.clear()
    assert_size_stride(arg0_1, (8, ), (1, ))
    assert_size_stride(arg1_1, (8, ), (1, ))
    assert_size_stride(arg2_1, (8, ), (1, ))
    assert_size_stride(arg3_1, (8, ), (1, ))
    assert_size_stride(arg4_1, (16, ), (1, ))
    assert_size_stride(arg5_1, (16, ), (1, ))
    assert_size_stride(arg6_1, (16, ), (1, ))
    assert_size_stride(arg7_1, (16, ), (1, ))
    assert_size_stride(arg8_1, (32, ), (1, ))
    assert_size_stride(arg9_1, (32, ), (1, ))
    assert_size_stride(arg10_1, (32, ), (1, ))
    assert_size_stride(arg11_1, (32, ), (1, ))
    assert_size_stride(arg12_1, (32, ), (1, ))
    assert_size_stride(arg13_1, (32, ), (1, ))
    assert_size_stride(arg14_1, (32, ), (1, ))
    assert_size_stride(arg15_1, (32, ), (1, ))
    assert_size_stride(arg16_1, (64, ), (1, ))
    assert_size_stride(arg17_1, (64, ), (1, ))
    assert_size_stride(arg18_1, (64, ), (1, ))
    assert_size_stride(arg19_1, (64, ), (1, ))
    assert_size_stride(arg20_1, (64, ), (1, ))
    assert_size_stride(arg21_1, (64, ), (1, ))
    assert_size_stride(arg22_1, (64, ), (1, ))
    assert_size_stride(arg23_1, (64, ), (1, ))
    assert_size_stride(arg24_1, (128, ), (1, ))
    assert_size_stride(arg25_1, (128, ), (1, ))
    assert_size_stride(arg26_1, (128, ), (1, ))
    assert_size_stride(arg27_1, (128, ), (1, ))
    assert_size_stride(arg28_1, (128, ), (1, ))
    assert_size_stride(arg29_1, (128, ), (1, ))
    assert_size_stride(arg30_1, (128, ), (1, ))
    assert_size_stride(arg31_1, (128, ), (1, ))
    assert_size_stride(arg32_1, (128, ), (1, ))
    assert_size_stride(arg33_1, (128, ), (1, ))
    assert_size_stride(arg34_1, (128, ), (1, ))
    assert_size_stride(arg35_1, (128, ), (1, ))
    assert_size_stride(arg36_1, (128, ), (1, ))
    assert_size_stride(arg37_1, (128, ), (1, ))
    assert_size_stride(arg38_1, (128, ), (1, ))
    assert_size_stride(arg39_1, (128, ), (1, ))
    assert_size_stride(arg40_1, (128, ), (1, ))
    assert_size_stride(arg41_1, (128, ), (1, ))
    assert_size_stride(arg42_1, (128, ), (1, ))
    assert_size_stride(arg43_1, (128, ), (1, ))
    assert_size_stride(arg44_1, (128, ), (1, ))
    assert_size_stride(arg45_1, (128, ), (1, ))
    assert_size_stride(arg46_1, (128, ), (1, ))
    assert_size_stride(arg47_1, (128, ), (1, ))
    assert_size_stride(arg48_1, (256, ), (1, ))
    assert_size_stride(arg49_1, (256, ), (1, ))
    assert_size_stride(arg50_1, (256, ), (1, ))
    assert_size_stride(arg51_1, (256, ), (1, ))
    assert_size_stride(arg52_1, (256, ), (1, ))
    assert_size_stride(arg53_1, (256, ), (1, ))
    assert_size_stride(arg54_1, (1000, 1280), (1280, 1))
    assert_size_stride(arg55_1, (1000, ), (1, ))
    assert_size_stride(arg56_1, (8, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg57_1, (8, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg58_1, (16, 8, 1, 1), (8, 1, 1, 1))
    assert_size_stride(arg59_1, (16, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg60_1, (32, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg61_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg62_1, (32, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg63_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg64_1, (64, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg65_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg66_1, (64, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg67_1, (64, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg68_1, (128, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg69_1, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg70_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg71_1, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg72_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg73_1, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg74_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg75_1, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg76_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg77_1, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg78_1, (128, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg79_1, (128, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg80_1, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg81_1, (32, ), (1, ))
    assert_size_stride(arg82_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg83_1, (128, ), (1, ))
    assert_size_stride(arg84_1, (256, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg85_1, (256, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg86_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg87_1, (64, ), (1, ))
    assert_size_stride(arg88_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg89_1, (256, ), (1, ))
    assert_size_stride(arg90_1, (256, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg91_1, (1280, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg92_1, (1280, ), (1, ))
    assert_size_stride(arg93_1, (8, ), (1, ))
    assert_size_stride(arg94_1, (8, ), (1, ))
    assert_size_stride(arg95_1, (8, ), (1, ))
    assert_size_stride(arg96_1, (8, ), (1, ))
    assert_size_stride(arg97_1, (16, ), (1, ))
    assert_size_stride(arg98_1, (16, ), (1, ))
    assert_size_stride(arg99_1, (16, ), (1, ))
    assert_size_stride(arg100_1, (16, ), (1, ))
    assert_size_stride(arg101_1, (32, ), (1, ))
    assert_size_stride(arg102_1, (32, ), (1, ))
    assert_size_stride(arg103_1, (32, ), (1, ))
    assert_size_stride(arg104_1, (32, ), (1, ))
    assert_size_stride(arg105_1, (32, ), (1, ))
    assert_size_stride(arg106_1, (32, ), (1, ))
    assert_size_stride(arg107_1, (32, ), (1, ))
    assert_size_stride(arg108_1, (32, ), (1, ))
    assert_size_stride(arg109_1, (64, ), (1, ))
    assert_size_stride(arg110_1, (64, ), (1, ))
    assert_size_stride(arg111_1, (64, ), (1, ))
    assert_size_stride(arg112_1, (64, ), (1, ))
    assert_size_stride(arg113_1, (64, ), (1, ))
    assert_size_stride(arg114_1, (64, ), (1, ))
    assert_size_stride(arg115_1, (64, ), (1, ))
    assert_size_stride(arg116_1, (64, ), (1, ))
    assert_size_stride(arg117_1, (128, ), (1, ))
    assert_size_stride(arg118_1, (128, ), (1, ))
    assert_size_stride(arg119_1, (128, ), (1, ))
    assert_size_stride(arg120_1, (128, ), (1, ))
    assert_size_stride(arg121_1, (128, ), (1, ))
    assert_size_stride(arg122_1, (128, ), (1, ))
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
    assert_size_stride(arg135_1, (128, ), (1, ))
    assert_size_stride(arg136_1, (128, ), (1, ))
    assert_size_stride(arg137_1, (128, ), (1, ))
    assert_size_stride(arg138_1, (128, ), (1, ))
    assert_size_stride(arg139_1, (128, ), (1, ))
    assert_size_stride(arg140_1, (128, ), (1, ))
    assert_size_stride(arg141_1, (256, ), (1, ))
    assert_size_stride(arg142_1, (256, ), (1, ))
    assert_size_stride(arg143_1, (256, ), (1, ))
    assert_size_stride(arg144_1, (256, ), (1, ))
    assert_size_stride(arg145_1, (256, ), (1, ))
    assert_size_stride(arg146_1, (256, ), (1, ))
    assert_size_stride(arg147_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg147_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg147_1
        buf1 = empty_strided((8, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg56_1, buf1, 24, 9, grid=grid(24, 9), stream=stream0)
        del arg56_1
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 8, 112, 112), (100352, 12544, 112, 1))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        buf4 = empty_strided((8, 8, 112, 112), (100352, 1, 896, 8), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2.run(buf3, arg93_1, arg94_1, arg0_1, arg1_1, buf4, 64, 12544, grid=grid(64, 12544), stream=stream0)
        del arg0_1
        del arg1_1
        del arg93_1
        del arg94_1
        del buf3
        # Source Nodes: [shortcut, x_5], Original ATen: [aten.convolution, aten.hardswish]
        buf5 = extern_kernels.convolution(buf4, arg57_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=8, bias=None)
        assert_size_stride(buf5, (8, 8, 112, 112), (100352, 12544, 112, 1))
        del arg57_1
        buf6 = buf5; del buf5  # reuse
        buf7 = buf4; del buf4  # reuse
        # Source Nodes: [x_6, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_2.run(buf6, arg95_1, arg96_1, arg2_1, arg3_1, buf7, 64, 12544, grid=grid(64, 12544), stream=stream0)
        del arg2_1
        del arg3_1
        del arg95_1
        del arg96_1
        del buf6
        # Source Nodes: [x_11, x_9], Original ATen: [aten.convolution, aten.hardswish]
        buf8 = extern_kernels.convolution(buf7, arg58_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (8, 16, 112, 112), (200704, 12544, 112, 1))
        del arg58_1
        buf9 = buf8; del buf8  # reuse
        buf10 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut_1, x_12], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_3.run(buf9, arg97_1, arg98_1, arg4_1, arg5_1, buf10, 128, 12544, grid=grid(128, 12544), stream=stream0)
        del arg4_1
        del arg5_1
        del arg97_1
        del arg98_1
        del buf9
        # Source Nodes: [shortcut_1, x_16], Original ATen: [aten.convolution, aten.hardswish]
        buf11 = extern_kernels.convolution(buf10, arg59_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=16, bias=None)
        assert_size_stride(buf11, (8, 16, 56, 56), (50176, 3136, 56, 1))
        del arg59_1
        del buf10
        buf12 = buf11; del buf11  # reuse
        buf13 = empty_strided((8, 16, 56, 56), (50176, 1, 896, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_4.run(buf12, arg99_1, arg100_1, arg6_1, arg7_1, buf13, 128, 3136, grid=grid(128, 3136), stream=stream0)
        del arg100_1
        del arg6_1
        del arg7_1
        del arg99_1
        del buf12
        # Source Nodes: [x_20, x_22], Original ATen: [aten.convolution, aten.hardswish]
        buf14 = extern_kernels.convolution(buf13, arg60_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (8, 32, 56, 56), (100352, 3136, 56, 1))
        del arg60_1
        buf15 = buf14; del buf14  # reuse
        buf16 = reinterpret_tensor(buf7, (8, 32, 56, 56), (100352, 1, 1792, 32), 0); del buf7  # reuse
        # Source Nodes: [shortcut_2, x_23], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_5.run(buf15, arg101_1, arg102_1, arg8_1, arg9_1, buf16, 256, 3136, grid=grid(256, 3136), stream=stream0)
        del arg101_1
        del arg102_1
        del arg8_1
        del arg9_1
        del buf15
        # Source Nodes: [shortcut_2, x_27], Original ATen: [aten.convolution, aten.hardswish]
        buf17 = extern_kernels.convolution(buf16, arg61_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf17, (8, 32, 56, 56), (100352, 3136, 56, 1))
        del arg61_1
        buf18 = buf17; del buf17  # reuse
        buf19 = buf16; del buf16  # reuse
        # Source Nodes: [x_28, x_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_5.run(buf18, arg103_1, arg104_1, arg10_1, arg11_1, buf19, 256, 3136, grid=grid(256, 3136), stream=stream0)
        del arg103_1
        del arg104_1
        del arg10_1
        del arg11_1
        del buf18
        # Source Nodes: [x_31, x_33], Original ATen: [aten.convolution, aten.hardswish]
        buf20 = extern_kernels.convolution(buf19, arg62_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (8, 32, 56, 56), (100352, 3136, 56, 1))
        del arg62_1
        buf21 = buf20; del buf20  # reuse
        buf22 = buf19; del buf19  # reuse
        # Source Nodes: [shortcut_3, x_34], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_5.run(buf21, arg105_1, arg106_1, arg12_1, arg13_1, buf22, 256, 3136, grid=grid(256, 3136), stream=stream0)
        del arg105_1
        del arg106_1
        del arg12_1
        del arg13_1
        del buf21
        # Source Nodes: [shortcut_3, x_38], Original ATen: [aten.convolution, aten.hardswish]
        buf23 = extern_kernels.convolution(buf22, arg63_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf23, (8, 32, 28, 28), (25088, 784, 28, 1))
        del arg63_1
        del buf22
        buf24 = buf23; del buf23  # reuse
        buf25 = empty_strided((8, 32, 28, 28), (25088, 1, 896, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_39, x_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_6.run(buf24, arg107_1, arg108_1, arg14_1, arg15_1, buf25, 256, 784, grid=grid(256, 784), stream=stream0)
        del arg107_1
        del arg108_1
        del arg14_1
        del arg15_1
        del buf24
        # Source Nodes: [x_42, x_44], Original ATen: [aten.convolution, aten.hardswish]
        buf26 = extern_kernels.convolution(buf25, arg64_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 64, 28, 28), (50176, 784, 28, 1))
        del arg64_1
        buf27 = buf26; del buf26  # reuse
        buf28 = reinterpret_tensor(buf13, (8, 64, 28, 28), (50176, 1, 1792, 64), 0); del buf13  # reuse
        # Source Nodes: [shortcut_4, x_45], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_7.run(buf27, arg109_1, arg110_1, arg16_1, arg17_1, buf28, 512, 784, grid=grid(512, 784), stream=stream0)
        del arg109_1
        del arg110_1
        del arg16_1
        del arg17_1
        del buf27
        # Source Nodes: [shortcut_4, x_49], Original ATen: [aten.convolution, aten.hardswish]
        buf29 = extern_kernels.convolution(buf28, arg65_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf29, (8, 64, 28, 28), (50176, 784, 28, 1))
        del arg65_1
        buf30 = buf29; del buf29  # reuse
        buf31 = buf28; del buf28  # reuse
        # Source Nodes: [x_50, x_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_7.run(buf30, arg111_1, arg112_1, arg18_1, arg19_1, buf31, 512, 784, grid=grid(512, 784), stream=stream0)
        del arg111_1
        del arg112_1
        del arg18_1
        del arg19_1
        del buf30
        # Source Nodes: [x_53, x_55], Original ATen: [aten.convolution, aten.hardswish]
        buf32 = extern_kernels.convolution(buf31, arg66_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 64, 28, 28), (50176, 784, 28, 1))
        del arg66_1
        buf33 = buf32; del buf32  # reuse
        buf34 = buf31; del buf31  # reuse
        # Source Nodes: [shortcut_5, x_56], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_7.run(buf33, arg113_1, arg114_1, arg20_1, arg21_1, buf34, 512, 784, grid=grid(512, 784), stream=stream0)
        del arg113_1
        del arg114_1
        del arg20_1
        del arg21_1
        del buf33
        # Source Nodes: [shortcut_5, x_60], Original ATen: [aten.convolution, aten.hardswish]
        buf35 = extern_kernels.convolution(buf34, arg67_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=64, bias=None)
        assert_size_stride(buf35, (8, 64, 14, 14), (12544, 196, 14, 1))
        del arg67_1
        del buf34
        buf36 = buf35; del buf35  # reuse
        buf37 = empty_strided((8, 64, 14, 14), (12544, 1, 896, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_61, x_64], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_8.run(buf36, arg115_1, arg116_1, arg22_1, arg23_1, buf37, 512, 196, grid=grid(512, 196), stream=stream0)
        del arg115_1
        del arg116_1
        del arg22_1
        del arg23_1
        del buf36
        # Source Nodes: [x_64, x_66], Original ATen: [aten.convolution, aten.hardswish]
        buf38 = extern_kernels.convolution(buf37, arg68_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg68_1
        buf39 = buf38; del buf38  # reuse
        buf40 = reinterpret_tensor(buf25, (8, 128, 14, 14), (25088, 1, 1792, 128), 0); del buf25  # reuse
        # Source Nodes: [shortcut_6, x_67], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9.run(buf39, arg117_1, arg118_1, arg24_1, arg25_1, buf40, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg117_1
        del arg118_1
        del arg24_1
        del arg25_1
        del buf39
        # Source Nodes: [shortcut_6, x_71], Original ATen: [aten.convolution, aten.hardswish]
        buf41 = extern_kernels.convolution(buf40, arg69_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf41, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg69_1
        buf42 = buf41; del buf41  # reuse
        buf43 = buf40; del buf40  # reuse
        # Source Nodes: [x_72, x_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9.run(buf42, arg119_1, arg120_1, arg26_1, arg27_1, buf43, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg119_1
        del arg120_1
        del arg26_1
        del arg27_1
        del buf42
        # Source Nodes: [x_75, x_77], Original ATen: [aten.convolution, aten.hardswish]
        buf44 = extern_kernels.convolution(buf43, arg70_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg70_1
        buf45 = buf44; del buf44  # reuse
        buf46 = buf43; del buf43  # reuse
        # Source Nodes: [shortcut_7, x_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9.run(buf45, arg121_1, arg122_1, arg28_1, arg29_1, buf46, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg121_1
        del arg122_1
        del arg28_1
        del arg29_1
        del buf45
        # Source Nodes: [shortcut_7, x_82], Original ATen: [aten.convolution, aten.hardswish]
        buf47 = extern_kernels.convolution(buf46, arg71_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf47, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg71_1
        buf48 = buf47; del buf47  # reuse
        buf49 = buf46; del buf46  # reuse
        # Source Nodes: [x_83, x_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9.run(buf48, arg123_1, arg124_1, arg30_1, arg31_1, buf49, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg123_1
        del arg124_1
        del arg30_1
        del arg31_1
        del buf48
        # Source Nodes: [x_86, x_88], Original ATen: [aten.convolution, aten.hardswish]
        buf50 = extern_kernels.convolution(buf49, arg72_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg72_1
        buf51 = buf50; del buf50  # reuse
        buf52 = buf49; del buf49  # reuse
        # Source Nodes: [shortcut_8, x_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9.run(buf51, arg125_1, arg126_1, arg32_1, arg33_1, buf52, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg125_1
        del arg126_1
        del arg32_1
        del arg33_1
        del buf51
        # Source Nodes: [shortcut_8, x_93], Original ATen: [aten.convolution, aten.hardswish]
        buf53 = extern_kernels.convolution(buf52, arg73_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf53, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg73_1
        buf54 = buf53; del buf53  # reuse
        buf55 = buf52; del buf52  # reuse
        # Source Nodes: [x_94, x_97], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9.run(buf54, arg127_1, arg128_1, arg34_1, arg35_1, buf55, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg127_1
        del arg128_1
        del arg34_1
        del arg35_1
        del buf54
        # Source Nodes: [x_97, x_99], Original ATen: [aten.convolution, aten.hardswish]
        buf56 = extern_kernels.convolution(buf55, arg74_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg74_1
        buf57 = buf56; del buf56  # reuse
        buf58 = buf55; del buf55  # reuse
        # Source Nodes: [shortcut_9, x_100], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9.run(buf57, arg129_1, arg130_1, arg36_1, arg37_1, buf58, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg129_1
        del arg130_1
        del arg36_1
        del arg37_1
        del buf57
        # Source Nodes: [shortcut_9, x_104], Original ATen: [aten.convolution, aten.hardswish]
        buf59 = extern_kernels.convolution(buf58, arg75_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf59, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg75_1
        buf60 = buf59; del buf59  # reuse
        buf61 = buf58; del buf58  # reuse
        # Source Nodes: [x_105, x_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9.run(buf60, arg131_1, arg132_1, arg38_1, arg39_1, buf61, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg131_1
        del arg132_1
        del arg38_1
        del arg39_1
        del buf60
        # Source Nodes: [x_108, x_110], Original ATen: [aten.convolution, aten.hardswish]
        buf62 = extern_kernels.convolution(buf61, arg76_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg76_1
        buf63 = buf62; del buf62  # reuse
        buf64 = buf61; del buf61  # reuse
        # Source Nodes: [shortcut_10, x_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9.run(buf63, arg133_1, arg134_1, arg40_1, arg41_1, buf64, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg133_1
        del arg134_1
        del arg40_1
        del arg41_1
        del buf63
        # Source Nodes: [shortcut_10, x_115], Original ATen: [aten.convolution, aten.hardswish]
        buf65 = extern_kernels.convolution(buf64, arg77_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf65, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg77_1
        buf66 = buf65; del buf65  # reuse
        buf67 = buf64; del buf64  # reuse
        # Source Nodes: [x_116, x_119], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9.run(buf66, arg135_1, arg136_1, arg42_1, arg43_1, buf67, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg135_1
        del arg136_1
        del arg42_1
        del arg43_1
        del buf66
        # Source Nodes: [x_119, x_121], Original ATen: [aten.convolution, aten.hardswish]
        buf68 = extern_kernels.convolution(buf67, arg78_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 128, 14, 14), (25088, 196, 14, 1))
        del arg78_1
        buf69 = buf68; del buf68  # reuse
        buf70 = buf67; del buf67  # reuse
        # Source Nodes: [shortcut_11, x_122], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_9.run(buf69, arg137_1, arg138_1, arg44_1, arg45_1, buf70, 1024, 196, grid=grid(1024, 196), stream=stream0)
        del arg137_1
        del arg138_1
        del arg44_1
        del arg45_1
        del buf69
        # Source Nodes: [shortcut_11, x_126], Original ATen: [aten.convolution, aten.hardswish]
        buf71 = extern_kernels.convolution(buf70, arg79_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=128, bias=None)
        assert_size_stride(buf71, (8, 128, 7, 7), (6272, 49, 7, 1))
        del arg79_1
        del buf70
        buf72 = buf71; del buf71  # reuse
        buf73 = empty_strided((8, 128, 1, 1), (128, 1, 1024, 1024), device='cuda', dtype=torch.float32)
        buf74 = reinterpret_tensor(buf73, (8, 128, 1, 1), (128, 1, 128, 128), 0); del buf73  # reuse
        # Source Nodes: [x_127, x_130, x_se], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_10.run(buf72, buf74, arg139_1, arg140_1, arg46_1, arg47_1, 1024, 49, grid=grid(1024), stream=stream0)
        del arg139_1
        del arg140_1
        del arg46_1
        del arg47_1
        # Source Nodes: [x_130, x_se, x_se_1], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf75 = extern_kernels.convolution(buf74, arg80_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf75, (8, 32, 1, 1), (32, 1, 1, 1))
        del arg80_1
        del buf74
        buf76 = reinterpret_tensor(buf75, (8, 32, 1, 1), (32, 1, 32, 32), 0); del buf75  # reuse
        # Source Nodes: [x_130, x_se, x_se_1, x_se_2], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
        triton_poi_fused_convolution_hardswish_mean_relu_11.run(buf76, arg81_1, 256, grid=grid(256), stream=stream0)
        del arg81_1
        # Source Nodes: [x_130, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
        buf77 = extern_kernels.convolution(buf76, arg82_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf77, (8, 128, 1, 1), (128, 1, 1, 1))
        del arg82_1
        del buf76
        buf78 = empty_strided((8, 128, 7, 7), (6272, 1, 896, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_130, x_131, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_12.run(buf72, buf77, arg83_1, buf78, 1024, 49, grid=grid(1024, 49), stream=stream0)
        del arg83_1
        del buf72
        del buf77
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____0___se_gate, x_130, x_131, x_132, x_se, x_se_1, x_se_2, x_se_3], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
        buf79 = extern_kernels.convolution(buf78, arg84_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf79, (8, 256, 7, 7), (12544, 49, 7, 1))
        del arg84_1
        del buf78
        buf80 = buf79; del buf79  # reuse
        buf81 = reinterpret_tensor(buf37, (8, 256, 7, 7), (12544, 1, 1792, 256), 0); del buf37  # reuse
        # Source Nodes: [shortcut_12, x_133], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish]
        triton_poi_fused__native_batch_norm_legit_no_training_hardswish_13.run(buf80, arg141_1, arg142_1, arg48_1, arg49_1, buf81, 2048, 49, grid=grid(2048, 49), stream=stream0)
        del arg141_1
        del arg142_1
        del arg48_1
        del arg49_1
        del buf80
        # Source Nodes: [shortcut_12, x_137], Original ATen: [aten.convolution, aten.hardswish]
        buf82 = extern_kernels.convolution(buf81, arg85_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=256, bias=None)
        assert_size_stride(buf82, (8, 256, 7, 7), (12544, 49, 7, 1))
        del arg85_1
        buf83 = buf82; del buf82  # reuse
        buf84 = empty_strided((8, 256, 1, 1), (256, 1, 2048, 2048), device='cuda', dtype=torch.float32)
        buf85 = reinterpret_tensor(buf84, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf84  # reuse
        # Source Nodes: [x_138, x_141, x_se_4], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_14.run(buf83, buf85, arg143_1, arg144_1, arg50_1, arg51_1, 2048, 49, grid=grid(2048), stream=stream0)
        del arg143_1
        del arg144_1
        del arg50_1
        del arg51_1
        # Source Nodes: [x_141, x_se_4, x_se_5], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf86 = extern_kernels.convolution(buf85, arg86_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (8, 64, 1, 1), (64, 1, 1, 1))
        del arg86_1
        del buf85
        buf87 = reinterpret_tensor(buf86, (8, 64, 1, 1), (64, 1, 64, 64), 0); del buf86  # reuse
        # Source Nodes: [x_141, x_se_4, x_se_5, x_se_6], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
        triton_poi_fused_convolution_hardswish_mean_relu_15.run(buf87, arg87_1, 512, grid=grid(512), stream=stream0)
        del arg87_1
        # Source Nodes: [x_141, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.hardswish, aten.mean, aten.relu]
        buf88 = extern_kernels.convolution(buf87, arg88_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf88, (8, 256, 1, 1), (256, 1, 1, 1))
        del arg88_1
        del buf87
        buf89 = buf81; del buf81  # reuse
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_141, x_142, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
        triton_poi_fused_convolution_hardsigmoid_hardswish_mean_mul_relu_16.run(buf83, buf88, arg89_1, buf89, 2048, 49, grid=grid(2048, 49), stream=stream0)
        del arg89_1
        del buf83
        # Source Nodes: [getattr_getattr_l__mod___blocks___5_____1___se_gate, x_141, x_142, x_143, x_se_4, x_se_5, x_se_6, x_se_7], Original ATen: [aten.convolution, aten.hardsigmoid, aten.hardswish, aten.mean, aten.mul, aten.relu]
        buf90 = extern_kernels.convolution(buf89, arg90_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (8, 256, 7, 7), (12544, 49, 7, 1))
        del arg90_1
        del buf89
        buf91 = buf90; del buf90  # reuse
        buf92 = reinterpret_tensor(buf88, (8, 256, 1, 1), (256, 1, 2048, 2048), 0); del buf88  # reuse
        buf93 = reinterpret_tensor(buf92, (8, 256, 1, 1), (256, 1, 256, 256), 0); del buf92  # reuse
        # Source Nodes: [x_144, x_149, x_150], Original ATen: [aten._native_batch_norm_legit_no_training, aten.hardswish, aten.mean]
        triton_per_fused__native_batch_norm_legit_no_training_hardswish_mean_17.run(buf91, buf93, arg145_1, arg146_1, arg52_1, arg53_1, 2048, 49, grid=grid(2048), stream=stream0)
        del arg145_1
        del arg146_1
        del arg52_1
        del arg53_1
        del buf91
        # Source Nodes: [x_149, x_150, x_153], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        buf94 = extern_kernels.convolution(buf93, arg91_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf94, (8, 1280, 1, 1), (1280, 1, 1, 1))
        del arg91_1
        del buf93
        buf95 = buf94; del buf94  # reuse
        # Source Nodes: [x_149, x_150, x_153, x_154], Original ATen: [aten.convolution, aten.hardswish, aten.mean]
        triton_poi_fused_convolution_hardswish_mean_18.run(buf95, arg92_1, 10240, grid=grid(10240), stream=stream0)
        del arg92_1
        buf96 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_156], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg55_1, reinterpret_tensor(buf95, (8, 1280), (1280, 1), 0), reinterpret_tensor(arg54_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf96)
        del arg54_1
        del arg55_1
        return (buf96, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((8, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((8, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((16, 8, 1, 1), (8, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((16, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((32, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((32, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((64, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((64, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((64, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((128, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((128, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((128, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((256, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((256, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((256, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1280, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    arg135_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('lcnet_050', benchmark_compiled_module)
