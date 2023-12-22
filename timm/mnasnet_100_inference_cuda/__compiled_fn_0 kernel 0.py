
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


# kernel path: /tmp/torchinductor_youkaichao/ww/cwwbxnrzrqop4klqko3lxrugnst7kip4kb4m4tqgyf3ww2kcgb2l.py
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


# kernel path: /tmp/torchinductor_youkaichao/nq/cnq67vlpghzu3lwxhcrcedhypaoxhumxb4r5pvovsloacpgjr24i.py
# Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# shortcut => relu
# x_1 => add_1, mul_1, mul_2, sub
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/oe/coet2liwkmp2tlopc2neilt2wszcokbvq2qbpctud5zwhscboet7.py
# Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_12 => add_5, mul_7, mul_8, sub_2
triton_poi_fused__native_batch_norm_legit_no_training_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_3', 'mutated_arg_names': []},
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
    tl.store(out_ptr0 + (y0 + (16*x2) + (200704*y1)), tmp14, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nb/cnbyktcue3mbwhah4aqksbji6eqtoani26c634nqvpa5md2re7xq.py
# Source Nodes: [x_17, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_17 => add_7, mul_10, mul_11, sub_3
# x_20 => relu_2
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
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


# kernel path: /tmp/torchinductor_youkaichao/hk/chkd4q6cwju2pqne2jcsrz32byljej63abuals5cw7wdgzkca7ok.py
# Source Nodes: [x_22, x_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_22 => add_9, mul_13, mul_14, sub_4
# x_25 => relu_3
triton_poi_fused__native_batch_norm_legit_no_training_relu_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
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


# kernel path: /tmp/torchinductor_youkaichao/4u/c4u5hy42dcu2z2hgzwgyn2by3rra5xomi4py5rh2ygkmqp2ybfsq.py
# Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_28 => add_11, mul_16, mul_17, sub_5
triton_poi_fused__native_batch_norm_legit_no_training_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_6', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/dm/cdmjg6qbofxi4g6lcbf4qircxatiy4jzrodzd432fcdyf62owvvc.py
# Source Nodes: [x_33, x_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_33 => add_13, mul_19, mul_20, sub_6
# x_36 => relu_4
triton_poi_fused__native_batch_norm_legit_no_training_relu_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_7', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/sl/cslj64gmbik3oypemkvmodvdlwvc3sdmzrlcdx6frjfbpigkaiyf.py
# Source Nodes: [shortcut_3, x_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_3 => add_18
# x_44 => add_17, mul_25, mul_26, sub_8
triton_poi_fused__native_batch_norm_legit_no_training_add_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_8', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/4b/c4bkq4uluexwuffsel4awoa7f3ip7vvrtn7yhbtb5buonvbseafu.py
# Source Nodes: [x_72, x_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_72 => add_29, mul_40, mul_41, sub_13
# x_75 => relu_9
triton_poi_fused__native_batch_norm_legit_no_training_relu_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_youkaichao/or/cor4mh2t2rk7kxj6i3wytdjebd425gsijcrgn3tgd3k66m2vcivw.py
# Source Nodes: [x_78], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_78 => add_31, mul_43, mul_44, sub_14
triton_poi_fused__native_batch_norm_legit_no_training_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_10', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/sh/csha4b3uecdtybwr6eyfoygy5roksvxcmxc2rnob4xnqi6hy4uiz.py
# Source Nodes: [x_83, x_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_83 => add_33, mul_46, mul_47, sub_15
# x_86 => relu_10
triton_poi_fused__native_batch_norm_legit_no_training_relu_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_11', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/4p/c4pomh56ltx7bqmnsbrt247kv7pnpokz42djm2g6p7omrpgmswjm.py
# Source Nodes: [shortcut_6, x_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_6 => add_38
# x_94 => add_37, mul_52, mul_53, sub_17
triton_poi_fused__native_batch_norm_legit_no_training_add_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_12', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/f3/cf33lvpzavt33wpixn4eniegqtgh6qyuysuf2ehjlxwhteptqtub.py
# Source Nodes: [x_117, x_120], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_117 => add_47, mul_64, mul_65, sub_21
# x_120 => relu_14
triton_poi_fused__native_batch_norm_legit_no_training_relu_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_youkaichao/qr/cqrijj7pglirhdkrc5t65pir7mej2sidiph3wxitsyvjz2seg5j7.py
# Source Nodes: [x_122, x_125], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_122 => add_49, mul_67, mul_68, sub_22
# x_125 => relu_15
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/kj/ckjhmrhwigpmj5jkpfwrzut776h64xcr4gr6iwkxpsn2vwhi4poc.py
# Source Nodes: [x_128], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_128 => add_51, mul_70, mul_71, sub_23
triton_poi_fused__native_batch_norm_legit_no_training_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_15', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/nn/cnn3fjnycuhtu7o6myp7leonpfqtxygvncz26by3ad5efy6e5ths.py
# Source Nodes: [x_133, x_136], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_133 => add_53, mul_73, mul_74, sub_24
# x_136 => relu_16
triton_poi_fused__native_batch_norm_legit_no_training_relu_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_youkaichao/ey/ceyaqoyyejyltsxmjomc4k73yczca3evjynht3sblynalhsf4mnu.py
# Source Nodes: [shortcut_9, x_144], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_9 => add_58
# x_144 => add_57, mul_79, mul_80, sub_26
triton_poi_fused__native_batch_norm_legit_no_training_add_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_17', 'mutated_arg_names': ['in_out_ptr0']},
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


# kernel path: /tmp/torchinductor_youkaichao/od/codkkm3h7h5mgxeme5o54w7t46muhawqvtvh2ir3bwbuyxvfygv7.py
# Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_178 => add_71, mul_97, mul_98, sub_32
triton_poi_fused__native_batch_norm_legit_no_training_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
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


# kernel path: /tmp/torchinductor_youkaichao/ux/cux2cxxpvg2k2pzqoauiwo47xkl424dx45sjpslgehjcc3cvntk4.py
# Source Nodes: [x_183, x_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_183 => add_73, mul_100, mul_101, sub_33
# x_186 => relu_22
triton_poi_fused__native_batch_norm_legit_no_training_relu_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4608
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


# kernel path: /tmp/torchinductor_youkaichao/77/c77dybyxj4egz7ignhczrcemr4k5xfdlejpzxlhmcnpwwjnxekln.py
# Source Nodes: [shortcut_12, x_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_12 => add_78
# x_194 => add_77, mul_106, mul_107, sub_35
triton_poi_fused__native_batch_norm_legit_no_training_add_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_20', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1568
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


# kernel path: /tmp/torchinductor_youkaichao/xa/cxa6k64ftjsznvvo6yt652jvr2jc757ueoncgladad4berxnp2py.py
# Source Nodes: [x_205, x_208], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_205 => add_82, mul_112, mul_113, sub_37
# x_208 => relu_25
triton_poi_fused__native_batch_norm_legit_no_training_relu_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_21', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/hf/chf2b6krqvt4uj67itlllsu4j2c6vgrsojijo5ph3tegh3ediq4h.py
# Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_211 => add_84, mul_115, mul_116, sub_38
triton_poi_fused__native_batch_norm_legit_no_training_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
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


# kernel path: /tmp/torchinductor_youkaichao/ii/ciixn4zkezyvxmnffca3hs5bb265a7y2stvvad72mk25wldhco6c.py
# Source Nodes: [x_216, x_219], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_216 => add_86, mul_118, mul_119, sub_39
# x_219 => relu_26
triton_poi_fused__native_batch_norm_legit_no_training_relu_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 64], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9216
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
    tmp0 = tl.load(in_ptr0 + (x2 + (49*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (1152*x2) + (56448*y1)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/cp/ccpprsxdcmg56topt6zil5k6wiocfn6ehxxbhqvpdo7aw4hnakgq.py
# Source Nodes: [shortcut_14, x_227], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
# shortcut_14 => add_91
# x_227 => add_90, mul_124, mul_125, sub_41
triton_poi_fused__native_batch_norm_legit_no_training_add_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_add_24', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 392
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


# kernel path: /tmp/torchinductor_youkaichao/ct/cctj5m22nohflhdq6mi6a52lftb6idtrbd5p3dfghq7nwxbdjmp2.py
# Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_no_training]
# x_278 => add_111, mul_151, mul_152, sub_50
triton_poi_fused__native_batch_norm_legit_no_training_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2560
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


# kernel path: /tmp/torchinductor_youkaichao/bh/cbheoyh5jxsy2oqvvbr7bvonwgm5hdb2e5m6divmckuxrdjbto5a.py
# Source Nodes: [x_284, x_288, x_289], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
# x_284 => add_113, mul_154, mul_155, sub_51
# x_288 => relu_34
# x_289 => mean
triton_per_fused__native_batch_norm_legit_no_training_mean_relu_26 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16384, 64],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused__native_batch_norm_legit_no_training_mean_relu_26', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 10240
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, ), (1, ))
    assert_size_stride(arg1_1, (32, ), (1, ))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, ), (1, ))
    assert_size_stride(arg4_1, (16, ), (1, ))
    assert_size_stride(arg5_1, (16, ), (1, ))
    assert_size_stride(arg6_1, (48, ), (1, ))
    assert_size_stride(arg7_1, (48, ), (1, ))
    assert_size_stride(arg8_1, (48, ), (1, ))
    assert_size_stride(arg9_1, (48, ), (1, ))
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
    assert_size_stride(arg22_1, (24, ), (1, ))
    assert_size_stride(arg23_1, (24, ), (1, ))
    assert_size_stride(arg24_1, (72, ), (1, ))
    assert_size_stride(arg25_1, (72, ), (1, ))
    assert_size_stride(arg26_1, (72, ), (1, ))
    assert_size_stride(arg27_1, (72, ), (1, ))
    assert_size_stride(arg28_1, (40, ), (1, ))
    assert_size_stride(arg29_1, (40, ), (1, ))
    assert_size_stride(arg30_1, (120, ), (1, ))
    assert_size_stride(arg31_1, (120, ), (1, ))
    assert_size_stride(arg32_1, (120, ), (1, ))
    assert_size_stride(arg33_1, (120, ), (1, ))
    assert_size_stride(arg34_1, (40, ), (1, ))
    assert_size_stride(arg35_1, (40, ), (1, ))
    assert_size_stride(arg36_1, (120, ), (1, ))
    assert_size_stride(arg37_1, (120, ), (1, ))
    assert_size_stride(arg38_1, (120, ), (1, ))
    assert_size_stride(arg39_1, (120, ), (1, ))
    assert_size_stride(arg40_1, (40, ), (1, ))
    assert_size_stride(arg41_1, (40, ), (1, ))
    assert_size_stride(arg42_1, (240, ), (1, ))
    assert_size_stride(arg43_1, (240, ), (1, ))
    assert_size_stride(arg44_1, (240, ), (1, ))
    assert_size_stride(arg45_1, (240, ), (1, ))
    assert_size_stride(arg46_1, (80, ), (1, ))
    assert_size_stride(arg47_1, (80, ), (1, ))
    assert_size_stride(arg48_1, (480, ), (1, ))
    assert_size_stride(arg49_1, (480, ), (1, ))
    assert_size_stride(arg50_1, (480, ), (1, ))
    assert_size_stride(arg51_1, (480, ), (1, ))
    assert_size_stride(arg52_1, (80, ), (1, ))
    assert_size_stride(arg53_1, (80, ), (1, ))
    assert_size_stride(arg54_1, (480, ), (1, ))
    assert_size_stride(arg55_1, (480, ), (1, ))
    assert_size_stride(arg56_1, (480, ), (1, ))
    assert_size_stride(arg57_1, (480, ), (1, ))
    assert_size_stride(arg58_1, (80, ), (1, ))
    assert_size_stride(arg59_1, (80, ), (1, ))
    assert_size_stride(arg60_1, (480, ), (1, ))
    assert_size_stride(arg61_1, (480, ), (1, ))
    assert_size_stride(arg62_1, (480, ), (1, ))
    assert_size_stride(arg63_1, (480, ), (1, ))
    assert_size_stride(arg64_1, (96, ), (1, ))
    assert_size_stride(arg65_1, (96, ), (1, ))
    assert_size_stride(arg66_1, (576, ), (1, ))
    assert_size_stride(arg67_1, (576, ), (1, ))
    assert_size_stride(arg68_1, (576, ), (1, ))
    assert_size_stride(arg69_1, (576, ), (1, ))
    assert_size_stride(arg70_1, (96, ), (1, ))
    assert_size_stride(arg71_1, (96, ), (1, ))
    assert_size_stride(arg72_1, (576, ), (1, ))
    assert_size_stride(arg73_1, (576, ), (1, ))
    assert_size_stride(arg74_1, (576, ), (1, ))
    assert_size_stride(arg75_1, (576, ), (1, ))
    assert_size_stride(arg76_1, (192, ), (1, ))
    assert_size_stride(arg77_1, (192, ), (1, ))
    assert_size_stride(arg78_1, (1152, ), (1, ))
    assert_size_stride(arg79_1, (1152, ), (1, ))
    assert_size_stride(arg80_1, (1152, ), (1, ))
    assert_size_stride(arg81_1, (1152, ), (1, ))
    assert_size_stride(arg82_1, (192, ), (1, ))
    assert_size_stride(arg83_1, (192, ), (1, ))
    assert_size_stride(arg84_1, (1152, ), (1, ))
    assert_size_stride(arg85_1, (1152, ), (1, ))
    assert_size_stride(arg86_1, (1152, ), (1, ))
    assert_size_stride(arg87_1, (1152, ), (1, ))
    assert_size_stride(arg88_1, (192, ), (1, ))
    assert_size_stride(arg89_1, (192, ), (1, ))
    assert_size_stride(arg90_1, (1152, ), (1, ))
    assert_size_stride(arg91_1, (1152, ), (1, ))
    assert_size_stride(arg92_1, (1152, ), (1, ))
    assert_size_stride(arg93_1, (1152, ), (1, ))
    assert_size_stride(arg94_1, (192, ), (1, ))
    assert_size_stride(arg95_1, (192, ), (1, ))
    assert_size_stride(arg96_1, (1152, ), (1, ))
    assert_size_stride(arg97_1, (1152, ), (1, ))
    assert_size_stride(arg98_1, (1152, ), (1, ))
    assert_size_stride(arg99_1, (1152, ), (1, ))
    assert_size_stride(arg100_1, (320, ), (1, ))
    assert_size_stride(arg101_1, (320, ), (1, ))
    assert_size_stride(arg102_1, (1280, ), (1, ))
    assert_size_stride(arg103_1, (1280, ), (1, ))
    assert_size_stride(arg104_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg105_1, (32, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg106_1, (16, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg107_1, (48, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg108_1, (48, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg109_1, (24, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg110_1, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg111_1, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg112_1, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg113_1, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg114_1, (72, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg115_1, (24, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg116_1, (72, 24, 1, 1), (24, 1, 1, 1))
    assert_size_stride(arg117_1, (72, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg118_1, (40, 72, 1, 1), (72, 1, 1, 1))
    assert_size_stride(arg119_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg120_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg121_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg122_1, (120, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg123_1, (120, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg124_1, (40, 120, 1, 1), (120, 1, 1, 1))
    assert_size_stride(arg125_1, (240, 40, 1, 1), (40, 1, 1, 1))
    assert_size_stride(arg126_1, (240, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg127_1, (80, 240, 1, 1), (240, 1, 1, 1))
    assert_size_stride(arg128_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg129_1, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg130_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg131_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg132_1, (480, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg133_1, (80, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg134_1, (480, 80, 1, 1), (80, 1, 1, 1))
    assert_size_stride(arg135_1, (480, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg136_1, (96, 480, 1, 1), (480, 1, 1, 1))
    assert_size_stride(arg137_1, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg138_1, (576, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg139_1, (96, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(arg140_1, (576, 96, 1, 1), (96, 1, 1, 1))
    assert_size_stride(arg141_1, (576, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg142_1, (192, 576, 1, 1), (576, 1, 1, 1))
    assert_size_stride(arg143_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg144_1, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg145_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg146_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg147_1, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg148_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg149_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg150_1, (1152, 1, 5, 5), (25, 25, 5, 1))
    assert_size_stride(arg151_1, (192, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg152_1, (1152, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg153_1, (1152, 1, 3, 3), (9, 9, 3, 1))
    assert_size_stride(arg154_1, (320, 1152, 1, 1), (1152, 1, 1, 1))
    assert_size_stride(arg155_1, (1280, 320, 1, 1), (320, 1, 1, 1))
    assert_size_stride(arg156_1, (1000, 1280), (1280, 1))
    assert_size_stride(arg157_1, (1000, ), (1, ))
    assert_size_stride(arg158_1, (32, ), (1, ))
    assert_size_stride(arg159_1, (32, ), (1, ))
    assert_size_stride(arg160_1, (32, ), (1, ))
    assert_size_stride(arg161_1, (32, ), (1, ))
    assert_size_stride(arg162_1, (16, ), (1, ))
    assert_size_stride(arg163_1, (16, ), (1, ))
    assert_size_stride(arg164_1, (48, ), (1, ))
    assert_size_stride(arg165_1, (48, ), (1, ))
    assert_size_stride(arg166_1, (48, ), (1, ))
    assert_size_stride(arg167_1, (48, ), (1, ))
    assert_size_stride(arg168_1, (24, ), (1, ))
    assert_size_stride(arg169_1, (24, ), (1, ))
    assert_size_stride(arg170_1, (72, ), (1, ))
    assert_size_stride(arg171_1, (72, ), (1, ))
    assert_size_stride(arg172_1, (72, ), (1, ))
    assert_size_stride(arg173_1, (72, ), (1, ))
    assert_size_stride(arg174_1, (24, ), (1, ))
    assert_size_stride(arg175_1, (24, ), (1, ))
    assert_size_stride(arg176_1, (72, ), (1, ))
    assert_size_stride(arg177_1, (72, ), (1, ))
    assert_size_stride(arg178_1, (72, ), (1, ))
    assert_size_stride(arg179_1, (72, ), (1, ))
    assert_size_stride(arg180_1, (24, ), (1, ))
    assert_size_stride(arg181_1, (24, ), (1, ))
    assert_size_stride(arg182_1, (72, ), (1, ))
    assert_size_stride(arg183_1, (72, ), (1, ))
    assert_size_stride(arg184_1, (72, ), (1, ))
    assert_size_stride(arg185_1, (72, ), (1, ))
    assert_size_stride(arg186_1, (40, ), (1, ))
    assert_size_stride(arg187_1, (40, ), (1, ))
    assert_size_stride(arg188_1, (120, ), (1, ))
    assert_size_stride(arg189_1, (120, ), (1, ))
    assert_size_stride(arg190_1, (120, ), (1, ))
    assert_size_stride(arg191_1, (120, ), (1, ))
    assert_size_stride(arg192_1, (40, ), (1, ))
    assert_size_stride(arg193_1, (40, ), (1, ))
    assert_size_stride(arg194_1, (120, ), (1, ))
    assert_size_stride(arg195_1, (120, ), (1, ))
    assert_size_stride(arg196_1, (120, ), (1, ))
    assert_size_stride(arg197_1, (120, ), (1, ))
    assert_size_stride(arg198_1, (40, ), (1, ))
    assert_size_stride(arg199_1, (40, ), (1, ))
    assert_size_stride(arg200_1, (240, ), (1, ))
    assert_size_stride(arg201_1, (240, ), (1, ))
    assert_size_stride(arg202_1, (240, ), (1, ))
    assert_size_stride(arg203_1, (240, ), (1, ))
    assert_size_stride(arg204_1, (80, ), (1, ))
    assert_size_stride(arg205_1, (80, ), (1, ))
    assert_size_stride(arg206_1, (480, ), (1, ))
    assert_size_stride(arg207_1, (480, ), (1, ))
    assert_size_stride(arg208_1, (480, ), (1, ))
    assert_size_stride(arg209_1, (480, ), (1, ))
    assert_size_stride(arg210_1, (80, ), (1, ))
    assert_size_stride(arg211_1, (80, ), (1, ))
    assert_size_stride(arg212_1, (480, ), (1, ))
    assert_size_stride(arg213_1, (480, ), (1, ))
    assert_size_stride(arg214_1, (480, ), (1, ))
    assert_size_stride(arg215_1, (480, ), (1, ))
    assert_size_stride(arg216_1, (80, ), (1, ))
    assert_size_stride(arg217_1, (80, ), (1, ))
    assert_size_stride(arg218_1, (480, ), (1, ))
    assert_size_stride(arg219_1, (480, ), (1, ))
    assert_size_stride(arg220_1, (480, ), (1, ))
    assert_size_stride(arg221_1, (480, ), (1, ))
    assert_size_stride(arg222_1, (96, ), (1, ))
    assert_size_stride(arg223_1, (96, ), (1, ))
    assert_size_stride(arg224_1, (576, ), (1, ))
    assert_size_stride(arg225_1, (576, ), (1, ))
    assert_size_stride(arg226_1, (576, ), (1, ))
    assert_size_stride(arg227_1, (576, ), (1, ))
    assert_size_stride(arg228_1, (96, ), (1, ))
    assert_size_stride(arg229_1, (96, ), (1, ))
    assert_size_stride(arg230_1, (576, ), (1, ))
    assert_size_stride(arg231_1, (576, ), (1, ))
    assert_size_stride(arg232_1, (576, ), (1, ))
    assert_size_stride(arg233_1, (576, ), (1, ))
    assert_size_stride(arg234_1, (192, ), (1, ))
    assert_size_stride(arg235_1, (192, ), (1, ))
    assert_size_stride(arg236_1, (1152, ), (1, ))
    assert_size_stride(arg237_1, (1152, ), (1, ))
    assert_size_stride(arg238_1, (1152, ), (1, ))
    assert_size_stride(arg239_1, (1152, ), (1, ))
    assert_size_stride(arg240_1, (192, ), (1, ))
    assert_size_stride(arg241_1, (192, ), (1, ))
    assert_size_stride(arg242_1, (1152, ), (1, ))
    assert_size_stride(arg243_1, (1152, ), (1, ))
    assert_size_stride(arg244_1, (1152, ), (1, ))
    assert_size_stride(arg245_1, (1152, ), (1, ))
    assert_size_stride(arg246_1, (192, ), (1, ))
    assert_size_stride(arg247_1, (192, ), (1, ))
    assert_size_stride(arg248_1, (1152, ), (1, ))
    assert_size_stride(arg249_1, (1152, ), (1, ))
    assert_size_stride(arg250_1, (1152, ), (1, ))
    assert_size_stride(arg251_1, (1152, ), (1, ))
    assert_size_stride(arg252_1, (192, ), (1, ))
    assert_size_stride(arg253_1, (192, ), (1, ))
    assert_size_stride(arg254_1, (1152, ), (1, ))
    assert_size_stride(arg255_1, (1152, ), (1, ))
    assert_size_stride(arg256_1, (1152, ), (1, ))
    assert_size_stride(arg257_1, (1152, ), (1, ))
    assert_size_stride(arg258_1, (320, ), (1, ))
    assert_size_stride(arg259_1, (320, ), (1, ))
    assert_size_stride(arg260_1, (1280, ), (1, ))
    assert_size_stride(arg261_1, (1280, ), (1, ))
    assert_size_stride(arg262_1, (8, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg262_1, buf0, 24, 50176, grid=grid(24, 50176), stream=stream0)
        del arg262_1
        buf1 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg104_1, buf1, 96, 9, grid=grid(96, 9), stream=stream0)
        del arg104_1
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 32, 112, 112), (401408, 12544, 112, 1))
        del buf1
        buf3 = empty_strided((8, 32, 112, 112), (401408, 1, 3584, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [shortcut, x_1], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf2, arg158_1, arg159_1, arg0_1, arg1_1, buf3, 256, 12544, grid=grid(256, 12544), stream=stream0)
        del arg0_1
        del arg158_1
        del arg159_1
        del arg1_1
        del buf2
        # Source Nodes: [shortcut, x_1, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf4 = extern_kernels.convolution(buf3, arg105_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=32, bias=None)
        assert_size_stride(buf4, (8, 32, 112, 112), (401408, 12544, 112, 1))
        del arg105_1
        buf5 = buf3; del buf3  # reuse
        # Source Nodes: [x_6, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf4, arg160_1, arg161_1, arg2_1, arg3_1, buf5, 256, 12544, grid=grid(256, 12544), stream=stream0)
        del arg160_1
        del arg161_1
        del arg2_1
        del arg3_1
        del buf4
        # Source Nodes: [x_11, x_6, x_9], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf6 = extern_kernels.convolution(buf5, arg106_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (8, 16, 112, 112), (200704, 12544, 112, 1))
        del arg106_1
        del buf5
        buf7 = empty_strided((8, 16, 112, 112), (200704, 1, 1792, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_12], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_3.run(buf6, arg162_1, arg163_1, arg4_1, arg5_1, buf7, 128, 12544, grid=grid(128, 12544), stream=stream0)
        del arg162_1
        del arg163_1
        del arg4_1
        del arg5_1
        del buf6
        # Source Nodes: [x_12, x_16], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf8 = extern_kernels.convolution(buf7, arg107_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (8, 48, 112, 112), (602112, 12544, 112, 1))
        del arg107_1
        del buf7
        buf9 = empty_strided((8, 48, 112, 112), (602112, 1, 5376, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_17, x_20], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf8, arg164_1, arg165_1, arg6_1, arg7_1, buf9, 384, 12544, grid=grid(384, 12544), stream=stream0)
        del arg164_1
        del arg165_1
        del arg6_1
        del arg7_1
        del buf8
        # Source Nodes: [x_17, x_20, x_21], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf10 = extern_kernels.convolution(buf9, arg108_1, stride=(2, 2), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=48, bias=None)
        assert_size_stride(buf10, (8, 48, 56, 56), (150528, 3136, 56, 1))
        del arg108_1
        del buf9
        buf11 = reinterpret_tensor(buf0, (8, 48, 56, 56), (150528, 1, 2688, 48), 0); del buf0  # reuse
        # Source Nodes: [x_22, x_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_5.run(buf10, arg166_1, arg167_1, arg8_1, arg9_1, buf11, 384, 3136, grid=grid(384, 3136), stream=stream0)
        del arg166_1
        del arg167_1
        del arg8_1
        del arg9_1
        del buf10
        # Source Nodes: [x_22, x_25, x_27], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf12 = extern_kernels.convolution(buf11, arg109_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (8, 24, 56, 56), (75264, 3136, 56, 1))
        del arg109_1
        del buf11
        buf13 = empty_strided((8, 24, 56, 56), (75264, 1, 1344, 24), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_28], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_6.run(buf12, arg168_1, arg169_1, arg10_1, arg11_1, buf13, 192, 3136, grid=grid(192, 3136), stream=stream0)
        del arg10_1
        del arg11_1
        del arg168_1
        del arg169_1
        del buf12
        # Source Nodes: [x_32], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf13, arg110_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (8, 72, 56, 56), (225792, 3136, 56, 1))
        del arg110_1
        buf15 = empty_strided((8, 72, 56, 56), (225792, 1, 4032, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_33, x_36], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf14, arg170_1, arg171_1, arg12_1, arg13_1, buf15, 576, 3136, grid=grid(576, 3136), stream=stream0)
        del arg12_1
        del arg13_1
        del arg170_1
        del arg171_1
        del buf14
        # Source Nodes: [x_33, x_36, x_37], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf16 = extern_kernels.convolution(buf15, arg111_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf16, (8, 72, 56, 56), (225792, 3136, 56, 1))
        del arg111_1
        buf17 = buf15; del buf15  # reuse
        # Source Nodes: [x_38, x_41], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf16, arg172_1, arg173_1, arg14_1, arg15_1, buf17, 576, 3136, grid=grid(576, 3136), stream=stream0)
        del arg14_1
        del arg15_1
        del arg172_1
        del arg173_1
        del buf16
        # Source Nodes: [x_38, x_41, x_43], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf18 = extern_kernels.convolution(buf17, arg112_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (8, 24, 56, 56), (75264, 3136, 56, 1))
        del arg112_1
        buf19 = buf13; del buf13  # reuse
        # Source Nodes: [shortcut_3, x_44], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_8.run(buf19, buf18, arg174_1, arg175_1, arg16_1, arg17_1, 25088, 24, grid=grid(25088, 24), stream=stream0)
        del arg16_1
        del arg174_1
        del arg175_1
        del arg17_1
        del buf18
        # Source Nodes: [x_49], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, arg113_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (8, 72, 56, 56), (225792, 3136, 56, 1))
        del arg113_1
        buf21 = buf17; del buf17  # reuse
        # Source Nodes: [x_50, x_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf20, arg176_1, arg177_1, arg18_1, arg19_1, buf21, 576, 3136, grid=grid(576, 3136), stream=stream0)
        del arg176_1
        del arg177_1
        del arg18_1
        del arg19_1
        del buf20
        # Source Nodes: [x_50, x_53, x_54], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf22 = extern_kernels.convolution(buf21, arg114_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf22, (8, 72, 56, 56), (225792, 3136, 56, 1))
        del arg114_1
        buf23 = buf21; del buf21  # reuse
        # Source Nodes: [x_55, x_58], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf22, arg178_1, arg179_1, arg20_1, arg21_1, buf23, 576, 3136, grid=grid(576, 3136), stream=stream0)
        del arg178_1
        del arg179_1
        del arg20_1
        del arg21_1
        del buf22
        # Source Nodes: [x_55, x_58, x_60], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf24 = extern_kernels.convolution(buf23, arg115_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (8, 24, 56, 56), (75264, 3136, 56, 1))
        del arg115_1
        buf25 = buf19; del buf19  # reuse
        # Source Nodes: [shortcut_4, x_61], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_8.run(buf25, buf24, arg180_1, arg181_1, arg22_1, arg23_1, 25088, 24, grid=grid(25088, 24), stream=stream0)
        del arg180_1
        del arg181_1
        del arg22_1
        del arg23_1
        del buf24
        # Source Nodes: [shortcut_4, x_61, x_66], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf26 = extern_kernels.convolution(buf25, arg116_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (8, 72, 56, 56), (225792, 3136, 56, 1))
        del arg116_1
        del buf25
        buf27 = buf23; del buf23  # reuse
        # Source Nodes: [x_67, x_70], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_7.run(buf26, arg182_1, arg183_1, arg24_1, arg25_1, buf27, 576, 3136, grid=grid(576, 3136), stream=stream0)
        del arg182_1
        del arg183_1
        del arg24_1
        del arg25_1
        del buf26
        # Source Nodes: [x_67, x_70, x_71], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf28 = extern_kernels.convolution(buf27, arg117_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=72, bias=None)
        assert_size_stride(buf28, (8, 72, 28, 28), (56448, 784, 28, 1))
        del arg117_1
        del buf27
        buf29 = empty_strided((8, 72, 28, 28), (56448, 1, 2016, 72), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_72, x_75], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_9.run(buf28, arg184_1, arg185_1, arg26_1, arg27_1, buf29, 576, 784, grid=grid(576, 784), stream=stream0)
        del arg184_1
        del arg185_1
        del arg26_1
        del arg27_1
        del buf28
        # Source Nodes: [x_72, x_75, x_77], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf30 = extern_kernels.convolution(buf29, arg118_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 40, 28, 28), (31360, 784, 28, 1))
        del arg118_1
        buf31 = empty_strided((8, 40, 28, 28), (31360, 1, 1120, 40), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_78], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_10.run(buf30, arg186_1, arg187_1, arg28_1, arg29_1, buf31, 320, 784, grid=grid(320, 784), stream=stream0)
        del arg186_1
        del arg187_1
        del arg28_1
        del arg29_1
        del buf30
        # Source Nodes: [x_82], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, arg119_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 120, 28, 28), (94080, 784, 28, 1))
        del arg119_1
        buf33 = empty_strided((8, 120, 28, 28), (94080, 1, 3360, 120), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_83, x_86], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf32, arg188_1, arg189_1, arg30_1, arg31_1, buf33, 960, 784, grid=grid(960, 784), stream=stream0)
        del arg188_1
        del arg189_1
        del arg30_1
        del arg31_1
        del buf32
        # Source Nodes: [x_83, x_86, x_87], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf34 = extern_kernels.convolution(buf33, arg120_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf34, (8, 120, 28, 28), (94080, 784, 28, 1))
        del arg120_1
        buf35 = buf33; del buf33  # reuse
        # Source Nodes: [x_88, x_91], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf34, arg190_1, arg191_1, arg32_1, arg33_1, buf35, 960, 784, grid=grid(960, 784), stream=stream0)
        del arg190_1
        del arg191_1
        del arg32_1
        del arg33_1
        del buf34
        # Source Nodes: [x_88, x_91, x_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf36 = extern_kernels.convolution(buf35, arg121_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (8, 40, 28, 28), (31360, 784, 28, 1))
        del arg121_1
        buf37 = buf31; del buf31  # reuse
        # Source Nodes: [shortcut_6, x_94], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_12.run(buf37, buf36, arg192_1, arg193_1, arg34_1, arg35_1, 6272, 40, grid=grid(6272, 40), stream=stream0)
        del arg192_1
        del arg193_1
        del arg34_1
        del arg35_1
        del buf36
        # Source Nodes: [x_99], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf37, arg122_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (8, 120, 28, 28), (94080, 784, 28, 1))
        del arg122_1
        buf39 = buf35; del buf35  # reuse
        # Source Nodes: [x_100, x_103], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf38, arg194_1, arg195_1, arg36_1, arg37_1, buf39, 960, 784, grid=grid(960, 784), stream=stream0)
        del arg194_1
        del arg195_1
        del arg36_1
        del arg37_1
        del buf38
        # Source Nodes: [x_100, x_103, x_104], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf40 = extern_kernels.convolution(buf39, arg123_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=120, bias=None)
        assert_size_stride(buf40, (8, 120, 28, 28), (94080, 784, 28, 1))
        del arg123_1
        buf41 = buf39; del buf39  # reuse
        # Source Nodes: [x_105, x_108], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_11.run(buf40, arg196_1, arg197_1, arg38_1, arg39_1, buf41, 960, 784, grid=grid(960, 784), stream=stream0)
        del arg196_1
        del arg197_1
        del arg38_1
        del arg39_1
        del buf40
        # Source Nodes: [x_105, x_108, x_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf42 = extern_kernels.convolution(buf41, arg124_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (8, 40, 28, 28), (31360, 784, 28, 1))
        del arg124_1
        buf43 = buf37; del buf37  # reuse
        # Source Nodes: [shortcut_7, x_111], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_12.run(buf43, buf42, arg198_1, arg199_1, arg40_1, arg41_1, 6272, 40, grid=grid(6272, 40), stream=stream0)
        del arg198_1
        del arg199_1
        del arg40_1
        del arg41_1
        del buf42
        # Source Nodes: [shortcut_7, x_111, x_116], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf44 = extern_kernels.convolution(buf43, arg125_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf44, (8, 240, 28, 28), (188160, 784, 28, 1))
        del arg125_1
        del buf43
        buf45 = empty_strided((8, 240, 28, 28), (188160, 1, 6720, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_117, x_120], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_13.run(buf44, arg200_1, arg201_1, arg42_1, arg43_1, buf45, 1920, 784, grid=grid(1920, 784), stream=stream0)
        del arg200_1
        del arg201_1
        del arg42_1
        del arg43_1
        del buf44
        # Source Nodes: [x_117, x_120, x_121], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf46 = extern_kernels.convolution(buf45, arg126_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=240, bias=None)
        assert_size_stride(buf46, (8, 240, 14, 14), (47040, 196, 14, 1))
        del arg126_1
        del buf45
        buf47 = empty_strided((8, 240, 14, 14), (47040, 1, 3360, 240), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_122, x_125], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf46, arg202_1, arg203_1, arg44_1, arg45_1, buf47, 1920, 196, grid=grid(1920, 196), stream=stream0)
        del arg202_1
        del arg203_1
        del arg44_1
        del arg45_1
        del buf46
        # Source Nodes: [x_122, x_125, x_127], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf48 = extern_kernels.convolution(buf47, arg127_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (8, 80, 14, 14), (15680, 196, 14, 1))
        del arg127_1
        del buf47
        buf49 = empty_strided((8, 80, 14, 14), (15680, 1, 1120, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_128], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_15.run(buf48, arg204_1, arg205_1, arg46_1, arg47_1, buf49, 640, 196, grid=grid(640, 196), stream=stream0)
        del arg204_1
        del arg205_1
        del arg46_1
        del arg47_1
        del buf48
        # Source Nodes: [x_132], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, arg128_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (8, 480, 14, 14), (94080, 196, 14, 1))
        del arg128_1
        buf51 = reinterpret_tensor(buf41, (8, 480, 14, 14), (94080, 1, 6720, 480), 0); del buf41  # reuse
        # Source Nodes: [x_133, x_136], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf50, arg206_1, arg207_1, arg48_1, arg49_1, buf51, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del arg206_1
        del arg207_1
        del arg48_1
        del arg49_1
        del buf50
        # Source Nodes: [x_133, x_136, x_137], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf52 = extern_kernels.convolution(buf51, arg129_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf52, (8, 480, 14, 14), (94080, 196, 14, 1))
        del arg129_1
        buf53 = buf51; del buf51  # reuse
        # Source Nodes: [x_138, x_141], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf52, arg208_1, arg209_1, arg50_1, arg51_1, buf53, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del arg208_1
        del arg209_1
        del arg50_1
        del arg51_1
        del buf52
        # Source Nodes: [x_138, x_141, x_143], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf54 = extern_kernels.convolution(buf53, arg130_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf54, (8, 80, 14, 14), (15680, 196, 14, 1))
        del arg130_1
        buf55 = buf49; del buf49  # reuse
        # Source Nodes: [shortcut_9, x_144], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_17.run(buf55, buf54, arg210_1, arg211_1, arg52_1, arg53_1, 1568, 80, grid=grid(1568, 80), stream=stream0)
        del arg210_1
        del arg211_1
        del arg52_1
        del arg53_1
        del buf54
        # Source Nodes: [x_149], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf55, arg131_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (8, 480, 14, 14), (94080, 196, 14, 1))
        del arg131_1
        buf57 = buf53; del buf53  # reuse
        # Source Nodes: [x_150, x_153], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf56, arg212_1, arg213_1, arg54_1, arg55_1, buf57, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del arg212_1
        del arg213_1
        del arg54_1
        del arg55_1
        del buf56
        # Source Nodes: [x_150, x_153, x_154], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf58 = extern_kernels.convolution(buf57, arg132_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf58, (8, 480, 14, 14), (94080, 196, 14, 1))
        del arg132_1
        buf59 = buf57; del buf57  # reuse
        # Source Nodes: [x_155, x_158], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf58, arg214_1, arg215_1, arg56_1, arg57_1, buf59, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del arg214_1
        del arg215_1
        del arg56_1
        del arg57_1
        del buf58
        # Source Nodes: [x_155, x_158, x_160], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf60 = extern_kernels.convolution(buf59, arg133_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (8, 80, 14, 14), (15680, 196, 14, 1))
        del arg133_1
        buf61 = buf55; del buf55  # reuse
        # Source Nodes: [shortcut_10, x_161], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_17.run(buf61, buf60, arg216_1, arg217_1, arg58_1, arg59_1, 1568, 80, grid=grid(1568, 80), stream=stream0)
        del arg216_1
        del arg217_1
        del arg58_1
        del arg59_1
        del buf60
        # Source Nodes: [shortcut_10, x_161, x_166], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf62 = extern_kernels.convolution(buf61, arg134_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf62, (8, 480, 14, 14), (94080, 196, 14, 1))
        del arg134_1
        buf63 = buf59; del buf59  # reuse
        # Source Nodes: [x_167, x_170], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf62, arg218_1, arg219_1, arg60_1, arg61_1, buf63, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del arg218_1
        del arg219_1
        del arg60_1
        del arg61_1
        del buf62
        # Source Nodes: [x_167, x_170, x_171], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf64 = extern_kernels.convolution(buf63, arg135_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=480, bias=None)
        assert_size_stride(buf64, (8, 480, 14, 14), (94080, 196, 14, 1))
        del arg135_1
        buf65 = buf63; del buf63  # reuse
        # Source Nodes: [x_172, x_175], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf64, arg220_1, arg221_1, arg62_1, arg63_1, buf65, 3840, 196, grid=grid(3840, 196), stream=stream0)
        del arg220_1
        del arg221_1
        del arg62_1
        del arg63_1
        del buf64
        # Source Nodes: [x_172, x_175, x_177], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf66 = extern_kernels.convolution(buf65, arg136_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf66, (8, 96, 14, 14), (18816, 196, 14, 1))
        del arg136_1
        del buf65
        buf67 = empty_strided((8, 96, 14, 14), (18816, 1, 1344, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_178], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_18.run(buf66, arg222_1, arg223_1, arg64_1, arg65_1, buf67, 768, 196, grid=grid(768, 196), stream=stream0)
        del arg222_1
        del arg223_1
        del arg64_1
        del arg65_1
        del buf66
        # Source Nodes: [x_182], Original ATen: [aten.convolution]
        buf68 = extern_kernels.convolution(buf67, arg137_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 576, 14, 14), (112896, 196, 14, 1))
        del arg137_1
        buf69 = empty_strided((8, 576, 14, 14), (112896, 1, 8064, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_183, x_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf68, arg224_1, arg225_1, arg66_1, arg67_1, buf69, 4608, 196, grid=grid(4608, 196), stream=stream0)
        del arg224_1
        del arg225_1
        del arg66_1
        del arg67_1
        del buf68
        # Source Nodes: [x_183, x_186, x_187], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf70 = extern_kernels.convolution(buf69, arg138_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
        assert_size_stride(buf70, (8, 576, 14, 14), (112896, 196, 14, 1))
        del arg138_1
        buf71 = buf69; del buf69  # reuse
        # Source Nodes: [x_188, x_191], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf70, arg226_1, arg227_1, arg68_1, arg69_1, buf71, 4608, 196, grid=grid(4608, 196), stream=stream0)
        del arg226_1
        del arg227_1
        del arg68_1
        del arg69_1
        del buf70
        # Source Nodes: [x_188, x_191, x_193], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf72 = extern_kernels.convolution(buf71, arg139_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf72, (8, 96, 14, 14), (18816, 196, 14, 1))
        del arg139_1
        buf73 = buf67; del buf67  # reuse
        # Source Nodes: [shortcut_12, x_194], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_20.run(buf73, buf72, arg228_1, arg229_1, arg70_1, arg71_1, 1568, 96, grid=grid(1568, 96), stream=stream0)
        del arg228_1
        del arg229_1
        del arg70_1
        del arg71_1
        del buf72
        # Source Nodes: [shortcut_12, x_194, x_199], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf74 = extern_kernels.convolution(buf73, arg140_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf74, (8, 576, 14, 14), (112896, 196, 14, 1))
        del arg140_1
        del buf73
        buf75 = buf71; del buf71  # reuse
        # Source Nodes: [x_200, x_203], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_19.run(buf74, arg230_1, arg231_1, arg72_1, arg73_1, buf75, 4608, 196, grid=grid(4608, 196), stream=stream0)
        del arg230_1
        del arg231_1
        del arg72_1
        del arg73_1
        del buf74
        # Source Nodes: [x_200, x_203, x_204], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf76 = extern_kernels.convolution(buf75, arg141_1, stride=(2, 2), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=576, bias=None)
        assert_size_stride(buf76, (8, 576, 7, 7), (28224, 49, 7, 1))
        del arg141_1
        del buf75
        buf77 = empty_strided((8, 576, 7, 7), (28224, 1, 4032, 576), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_205, x_208], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_21.run(buf76, arg232_1, arg233_1, arg74_1, arg75_1, buf77, 4608, 49, grid=grid(4608, 49), stream=stream0)
        del arg232_1
        del arg233_1
        del arg74_1
        del arg75_1
        del buf76
        # Source Nodes: [x_205, x_208, x_210], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf78 = extern_kernels.convolution(buf77, arg142_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (8, 192, 7, 7), (9408, 49, 7, 1))
        del arg142_1
        del buf77
        buf79 = empty_strided((8, 192, 7, 7), (9408, 1, 1344, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_211], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_22.run(buf78, arg234_1, arg235_1, arg76_1, arg77_1, buf79, 1536, 49, grid=grid(1536, 49), stream=stream0)
        del arg234_1
        del arg235_1
        del arg76_1
        del arg77_1
        del buf78
        # Source Nodes: [x_215], Original ATen: [aten.convolution]
        buf80 = extern_kernels.convolution(buf79, arg143_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf80, (8, 1152, 7, 7), (56448, 49, 7, 1))
        del arg143_1
        buf81 = reinterpret_tensor(buf29, (8, 1152, 7, 7), (56448, 1, 8064, 1152), 0); del buf29  # reuse
        # Source Nodes: [x_216, x_219], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf80, arg236_1, arg237_1, arg78_1, arg79_1, buf81, 9216, 49, grid=grid(9216, 49), stream=stream0)
        del arg236_1
        del arg237_1
        del arg78_1
        del arg79_1
        del buf80
        # Source Nodes: [x_216, x_219, x_220], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf82 = extern_kernels.convolution(buf81, arg144_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf82, (8, 1152, 7, 7), (56448, 49, 7, 1))
        del arg144_1
        buf83 = buf81; del buf81  # reuse
        # Source Nodes: [x_221, x_224], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf82, arg238_1, arg239_1, arg80_1, arg81_1, buf83, 9216, 49, grid=grid(9216, 49), stream=stream0)
        del arg238_1
        del arg239_1
        del arg80_1
        del arg81_1
        del buf82
        # Source Nodes: [x_221, x_224, x_226], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf84 = extern_kernels.convolution(buf83, arg145_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 192, 7, 7), (9408, 49, 7, 1))
        del arg145_1
        buf85 = buf79; del buf79  # reuse
        # Source Nodes: [shortcut_14, x_227], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_24.run(buf85, buf84, arg240_1, arg241_1, arg82_1, arg83_1, 392, 192, grid=grid(392, 192), stream=stream0)
        del arg240_1
        del arg241_1
        del arg82_1
        del arg83_1
        del buf84
        # Source Nodes: [x_232], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, arg146_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (8, 1152, 7, 7), (56448, 49, 7, 1))
        del arg146_1
        buf87 = buf83; del buf83  # reuse
        # Source Nodes: [x_233, x_236], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf86, arg242_1, arg243_1, arg84_1, arg85_1, buf87, 9216, 49, grid=grid(9216, 49), stream=stream0)
        del arg242_1
        del arg243_1
        del arg84_1
        del arg85_1
        del buf86
        # Source Nodes: [x_233, x_236, x_237], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf88 = extern_kernels.convolution(buf87, arg147_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf88, (8, 1152, 7, 7), (56448, 49, 7, 1))
        del arg147_1
        buf89 = buf87; del buf87  # reuse
        # Source Nodes: [x_238, x_241], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf88, arg244_1, arg245_1, arg86_1, arg87_1, buf89, 9216, 49, grid=grid(9216, 49), stream=stream0)
        del arg244_1
        del arg245_1
        del arg86_1
        del arg87_1
        del buf88
        # Source Nodes: [x_238, x_241, x_243], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf90 = extern_kernels.convolution(buf89, arg148_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf90, (8, 192, 7, 7), (9408, 49, 7, 1))
        del arg148_1
        buf91 = buf85; del buf85  # reuse
        # Source Nodes: [shortcut_15, x_244], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_24.run(buf91, buf90, arg246_1, arg247_1, arg88_1, arg89_1, 392, 192, grid=grid(392, 192), stream=stream0)
        del arg246_1
        del arg247_1
        del arg88_1
        del arg89_1
        del buf90
        # Source Nodes: [x_249], Original ATen: [aten.convolution]
        buf92 = extern_kernels.convolution(buf91, arg149_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (8, 1152, 7, 7), (56448, 49, 7, 1))
        del arg149_1
        buf93 = buf89; del buf89  # reuse
        # Source Nodes: [x_250, x_253], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf92, arg248_1, arg249_1, arg90_1, arg91_1, buf93, 9216, 49, grid=grid(9216, 49), stream=stream0)
        del arg248_1
        del arg249_1
        del arg90_1
        del arg91_1
        del buf92
        # Source Nodes: [x_250, x_253, x_254], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf94 = extern_kernels.convolution(buf93, arg150_1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf94, (8, 1152, 7, 7), (56448, 49, 7, 1))
        del arg150_1
        buf95 = buf93; del buf93  # reuse
        # Source Nodes: [x_255, x_258], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf94, arg250_1, arg251_1, arg92_1, arg93_1, buf95, 9216, 49, grid=grid(9216, 49), stream=stream0)
        del arg250_1
        del arg251_1
        del arg92_1
        del arg93_1
        del buf94
        # Source Nodes: [x_255, x_258, x_260], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf96 = extern_kernels.convolution(buf95, arg151_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf96, (8, 192, 7, 7), (9408, 49, 7, 1))
        del arg151_1
        buf97 = buf91; del buf91  # reuse
        # Source Nodes: [shortcut_16, x_261], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add]
        triton_poi_fused__native_batch_norm_legit_no_training_add_24.run(buf97, buf96, arg252_1, arg253_1, arg94_1, arg95_1, 392, 192, grid=grid(392, 192), stream=stream0)
        del arg252_1
        del arg253_1
        del arg94_1
        del arg95_1
        del buf96
        # Source Nodes: [shortcut_16, x_261, x_266], Original ATen: [aten._native_batch_norm_legit_no_training, aten.add, aten.convolution]
        buf98 = extern_kernels.convolution(buf97, arg152_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (8, 1152, 7, 7), (56448, 49, 7, 1))
        del arg152_1
        del buf97
        buf99 = buf95; del buf95  # reuse
        # Source Nodes: [x_267, x_270], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf98, arg254_1, arg255_1, arg96_1, arg97_1, buf99, 9216, 49, grid=grid(9216, 49), stream=stream0)
        del arg254_1
        del arg255_1
        del arg96_1
        del arg97_1
        del buf98
        # Source Nodes: [x_267, x_270, x_271], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf100 = extern_kernels.convolution(buf99, arg153_1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1152, bias=None)
        assert_size_stride(buf100, (8, 1152, 7, 7), (56448, 49, 7, 1))
        del arg153_1
        buf101 = buf99; del buf99  # reuse
        # Source Nodes: [x_272, x_275], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_23.run(buf100, arg256_1, arg257_1, arg98_1, arg99_1, buf101, 9216, 49, grid=grid(9216, 49), stream=stream0)
        del arg256_1
        del arg257_1
        del arg98_1
        del arg99_1
        del buf100
        # Source Nodes: [x_272, x_275, x_277], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf102 = extern_kernels.convolution(buf101, arg154_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 320, 7, 7), (15680, 49, 7, 1))
        del arg154_1
        del buf101
        buf103 = reinterpret_tensor(buf61, (8, 320, 7, 7), (15680, 1, 2240, 320), 0); del buf61  # reuse
        # Source Nodes: [x_278], Original ATen: [aten._native_batch_norm_legit_no_training]
        triton_poi_fused__native_batch_norm_legit_no_training_25.run(buf102, arg258_1, arg259_1, arg100_1, arg101_1, buf103, 2560, 49, grid=grid(2560, 49), stream=stream0)
        del arg100_1
        del arg101_1
        del arg258_1
        del arg259_1
        del buf102
        # Source Nodes: [x_278, x_283], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution]
        buf104 = extern_kernels.convolution(buf103, arg155_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf104, (8, 1280, 7, 7), (62720, 49, 7, 1))
        del arg155_1
        del buf103
        buf105 = empty_strided((8, 1280, 1, 1), (1280, 1, 10240, 10240), device='cuda', dtype=torch.float32)
        buf106 = reinterpret_tensor(buf105, (8, 1280, 1, 1), (1280, 1, 1, 1), 0); del buf105  # reuse
        # Source Nodes: [x_284, x_288, x_289], Original ATen: [aten._native_batch_norm_legit_no_training, aten.mean, aten.relu]
        triton_per_fused__native_batch_norm_legit_no_training_mean_relu_26.run(buf106, buf104, arg260_1, arg261_1, arg102_1, arg103_1, 10240, 49, grid=grid(10240), stream=stream0)
        del arg102_1
        del arg103_1
        del arg260_1
        del arg261_1
        del buf104
        buf107 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_292], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg157_1, reinterpret_tensor(buf106, (8, 1280), (1280, 1), 0), reinterpret_tensor(arg156_1, (1280, 1000), (1, 1280), 0), alpha=1, beta=1, out=buf107)
        del arg156_1
        del arg157_1
        return (buf107, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
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
    arg22_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((32, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((16, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((48, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((48, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((24, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((72, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((24, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((72, 24, 1, 1), (24, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((72, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((40, 72, 1, 1), (72, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((120, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((120, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((40, 120, 1, 1), (120, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((240, 40, 1, 1), (40, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((240, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((80, 240, 1, 1), (240, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((480, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((80, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((480, 80, 1, 1), (80, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((480, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((96, 480, 1, 1), (480, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((576, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((96, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((576, 96, 1, 1), (96, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((576, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((192, 576, 1, 1), (576, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((1152, 1, 5, 5), (25, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((192, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((1152, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((1152, 1, 3, 3), (9, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((320, 1152, 1, 1), (1152, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((1280, 320, 1, 1), (320, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((1000, 1280), (1280, 1), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((72, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((120, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((40, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((240, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((480, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((576, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((1152, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((1280, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((8, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('mnasnet_100', benchmark_compiled_module)
