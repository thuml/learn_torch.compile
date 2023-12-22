
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


# kernel path: /tmp/torchinductor_youkaichao/j6/cj65mpkg6dfok5f5ikatb25dc45kxdixjq2kndnmk5y3wvw73nqa.py
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
    size_hints=[32, 131072], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24
    xnumel = 89401
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
    tmp0 = tl.load(in_ptr0 + (x2 + (89401*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (268203*y1)), tmp0, xmask & ymask)
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


# kernel path: /tmp/torchinductor_youkaichao/sf/csfwjn2kcv53lxxmhx3j2yylue4onnk24yk3gfllb2ir7kd3carx.py
# Source Nodes: [x_1, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_1 => add_1, mul_1, mul_2, sub
# x_5 => relu
triton_poi_fused__native_batch_norm_legit_no_training_relu_2 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 32768], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 22201
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
    tmp0 = tl.load(in_ptr0 + (x2 + (22201*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (32*x2) + (710432*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yw/cyw3jw2w5pkb7jv4rd5hamf3vo7fgnaw2f7bpdfnmvyzxelu4svp.py
# Source Nodes: [x_1, x_5, x_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_1 => add_1, mul_1, mul_2, sub
# x_5 => relu
# x_6 => convolution_1
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (32*x2) + (288*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yd/cydhqv6tjd5iuc6ftlzmzwa33p6xtudfhysktsmj5hxygiqzobtb.py
# Source Nodes: [x_11, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_11 => relu_1
# x_7 => add_3, mul_4, mul_5, sub_1
triton_poi_fused__native_batch_norm_legit_no_training_relu_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 32768], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 21609
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
    tmp0 = tl.load(in_ptr0 + (x2 + (21609*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (32*x2) + (691488*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rw/crwq6mswpxmqaqzduo6d7x6obffmvziuu6yuuujtvantmhdixw2f.py
# Source Nodes: [x_11, x_12, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_11 => relu_1
# x_12 => convolution_2
# x_7 => add_3, mul_4, mul_5, sub_1
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (32*x2) + (288*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/by/cbylpc4qo3pp5rea3mwoiwitqqbtwvpaenlafau77fiuwvyhwmk5.py
# Source Nodes: [x_13, x_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_13 => add_5, mul_7, mul_8, sub_2
# x_17 => relu_2
triton_poi_fused__native_batch_norm_legit_no_training_relu_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_6', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11063808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 21609) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xr/cxrf5f2k5xjzf2xibuvjdrgh6fvyu77d6p74rcksfjgcqmgjk5wl.py
# Source Nodes: [x_13, x_17, x_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.max_pool2d_with_indices, aten.relu]
# x_13 => add_5, mul_7, mul_8, sub_2
# x_17 => relu_2
# x_18 => max_pool2d_with_indices
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_7 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 8192], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 5329
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 73
    x3 = (xindex // 73)
    y4 = yindex
    x5 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + ((2*x2) + (294*x3) + (21609*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x2) + (294*x3) + (21609*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + (2*x2) + (294*x3) + (21609*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (147 + (2*x2) + (294*x3) + (21609*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (148 + (2*x2) + (294*x3) + (21609*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (149 + (2*x2) + (294*x3) + (21609*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (294 + (2*x2) + (294*x3) + (21609*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (295 + (2*x2) + (294*x3) + (21609*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (296 + (2*x2) + (294*x3) + (21609*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (y0 + (64*x5) + (341056*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/n6/cn66eaw57au34yhb5ch5cdzbfxrrk7f2wmsykyvzxcozbp6yz3v4.py
# Source Nodes: [x_20, x_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_20 => add_7, mul_10, mul_11, sub_3
# x_24 => relu_3
triton_poi_fused__native_batch_norm_legit_no_training_relu_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 8192], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 640
    xnumel = 5329
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
    tmp0 = tl.load(in_ptr0 + (x2 + (5329*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (80*x2) + (426320*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dd/cddympgp324z62jsngmimjkz4mugtnz5tbxvui54hrfcaqxxxfpt.py
# Source Nodes: [x_20, x_24, x_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# x_20 => add_7, mul_10, mul_11, sub_3
# x_24 => relu_3
# x_25 => convolution_4
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 15360
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (80*x2) + (720*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fc/cfcafuvad533362heo4hiasqpibn4eqvs4qzpdsrc2t5l3obces5.py
# Source Nodes: [x_26, x_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# x_26 => add_9, mul_13, mul_14, sub_4
# x_30 => relu_4
triton_poi_fused__native_batch_norm_legit_no_training_relu_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_10', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7742976
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 5041) % 192
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(in_out_ptr0 + (x3), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2c/c2czzsbt4qtpdddjyz7js7ctmw7f4imm3b5cwlaaij2o7vg7zblv.py
# Source Nodes: [x_26, x_30, x_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.max_pool2d_with_indices, aten.relu]
# x_26 => add_9, mul_13, mul_14, sub_4
# x_30 => relu_4
# x_31 => max_pool2d_with_indices_1
triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 2048], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 1225
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 35
    x3 = (xindex // 35)
    y4 = yindex
    x5 = xindex
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + ((2*x2) + (142*x3) + (5041*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x2) + (142*x3) + (5041*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + (2*x2) + (142*x3) + (5041*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (71 + (2*x2) + (142*x3) + (5041*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (72 + (2*x2) + (142*x3) + (5041*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (73 + (2*x2) + (142*x3) + (5041*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (142 + (2*x2) + (142*x3) + (5041*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (143 + (2*x2) + (142*x3) + (5041*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (144 + (2*x2) + (142*x3) + (5041*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (y0 + (192*x5) + (235200*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u4/cu4di554qg5xneigf3xo7ovh3d6g47rpvhbwg6l72z7bcsulvvdz.py
# Source Nodes: [branch5x5, x_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# branch5x5 => relu_6
# x_38 => add_13, mul_19, mul_20, sub_6
triton_poi_fused__native_batch_norm_legit_no_training_relu_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 384
    xnumel = 1225
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
    tmp0 = tl.load(in_ptr0 + (x2 + (1225*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (48*x2) + (58800*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lb/clbpkqst2xtksruxk2ommqhww44yrtvvxdbv3ktrnugacatkqa7e.py
# Source Nodes: [branch5x5, x_38, x_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# branch5x5 => relu_6
# x_38 => add_13, mul_19, mul_20, sub_6
# x_42 => convolution_7
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 32], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 25
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
    tmp0 = tl.load(in_ptr0 + (x2 + (25*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (48*x2) + (1200*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5s/c5sekfj2thm3dlfmzrxkebhbamf6fbu6zfeoyvg46sh733wl3stu.py
# Source Nodes: [branch3x3dbl, x_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# branch3x3dbl => relu_8
# x_48 => add_17, mul_25, mul_26, sub_8
triton_poi_fused__native_batch_norm_legit_no_training_relu_14 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 1225
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
    tmp0 = tl.load(in_ptr0 + (x2 + (1225*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (64*x2) + (78400*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j6/cj6y5obebjrrlbevbsn6jy5k7ukyz4whdgiqmax7hjpjimsslml6.py
# Source Nodes: [branch3x3dbl, x_48, x_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# branch3x3dbl => relu_8
# x_48 => add_17, mul_25, mul_26, sub_8
# x_52 => convolution_9
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (576*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/oj/cojbcdgbg33xv6u7pzxielcdufomcnt7akotilqqc3rcx4ijkfsa.py
# Source Nodes: [branch3x3dbl_1, x_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# branch3x3dbl_1 => relu_9
# x_53 => add_19, mul_28, mul_29, sub_9
triton_poi_fused__native_batch_norm_legit_no_training_relu_16 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 1225
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
    tmp0 = tl.load(in_ptr0 + (x2 + (1225*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (96*x2) + (117600*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/3v/c3vbkkqzf2fkdxlv7svrh7swurkqutcptpcnp2my2afrneciygna.py
# Source Nodes: [branch3x3dbl_1, x_53, x_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# branch3x3dbl_1 => relu_9
# x_53 => add_19, mul_28, mul_29, sub_9
# x_57 => convolution_10
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 9216
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (96*x2) + (864*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jq/cjqyoyhbs4z2a2h5pl6lxx25iwsu37ocrkm3jqctw3y5ieq6f6pj.py
# Source Nodes: [branch_pool], Original ATen: [aten.avg_pool2d]
# branch_pool => avg_pool2d
triton_poi_fused_avg_pool2d_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1881600
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 6720) % 35
    x1 = (xindex // 192) % 35
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 35, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-6912) + x6), tmp10 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x1
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-6720) + x6), tmp18 & xmask, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x1
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-6528) + x6), tmp27 & xmask, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x2
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-192) + x6), tmp36 & xmask, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x6), tmp41 & xmask, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (192 + x6), tmp46 & xmask, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x2
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (6528 + x6), tmp55 & xmask, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (6720 + x6), tmp60 & xmask, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (6912 + x6), tmp65 & xmask, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 36, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x6), tmp145, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fn/cfn2vzb7xl342e3rkotoqpunicbwxptyepxgdcmmhhvzfaidrpz7.py
# Source Nodes: [cat_29], Original ATen: [aten.cat]
# cat_29 => cat
triton_poi_fused_cat_19 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: 'i32', 22: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(21,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 1225
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 256
    x2 = xindex
    y1 = (yindex // 256)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (1225*y0) + (78400*y1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = tl.sqrt(tmp10)
    tmp12 = 1 / tmp11
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp7 * tmp14
    tmp16 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 * tmp16
    tmp18 = tl.load(in_ptr4 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = triton_helpers.maximum(0, tmp19)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp4, tmp20, tmp21)
    tmp23 = tmp0 >= tmp3
    tmp24 = tl.full([1, 1], 128, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tmp23 & tmp25
    tmp27 = tl.load(in_ptr5 + ((-78400) + x2 + (1225*y0) + (78400*y1)), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr6 + (tl.broadcast_to((-64) + y0, [XBLOCK, YBLOCK])), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 - tmp28
    tmp30 = tl.load(in_ptr7 + (tl.broadcast_to((-64) + y0, [XBLOCK, YBLOCK])), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp30 + tmp9
    tmp32 = tl.sqrt(tmp31)
    tmp33 = 1 / tmp32
    tmp34 = tmp33 * tmp13
    tmp35 = tmp29 * tmp34
    tmp36 = tl.load(in_ptr8 + (tl.broadcast_to((-64) + y0, [XBLOCK, YBLOCK])), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp37 = tmp35 * tmp36
    tmp38 = tl.load(in_ptr9 + (tl.broadcast_to((-64) + y0, [XBLOCK, YBLOCK])), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 + tmp38
    tmp40 = triton_helpers.maximum(0, tmp39)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp26, tmp40, tmp41)
    tmp43 = tmp0 >= tmp24
    tmp44 = tl.full([1, 1], 224, tl.int64)
    tmp45 = tmp0 < tmp44
    tmp46 = tmp43 & tmp45
    tmp47 = tl.load(in_ptr10 + ((-156800) + x2 + (1225*y0) + (117600*y1)), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr11 + (tl.broadcast_to((-128) + y0, [XBLOCK, YBLOCK])), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tmp47 - tmp48
    tmp50 = tl.load(in_ptr12 + (tl.broadcast_to((-128) + y0, [XBLOCK, YBLOCK])), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp9
    tmp52 = tl.sqrt(tmp51)
    tmp53 = 1 / tmp52
    tmp54 = tmp53 * tmp13
    tmp55 = tmp49 * tmp54
    tmp56 = tl.load(in_ptr13 + (tl.broadcast_to((-128) + y0, [XBLOCK, YBLOCK])), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp55 * tmp56
    tmp58 = tl.load(in_ptr14 + (tl.broadcast_to((-128) + y0, [XBLOCK, YBLOCK])), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp57 + tmp58
    tmp60 = triton_helpers.maximum(0, tmp59)
    tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
    tmp62 = tl.where(tmp46, tmp60, tmp61)
    tmp63 = tmp0 >= tmp44
    tmp64 = tl.full([1, 1], 256, tl.int64)
    tmp65 = tmp0 < tmp64
    tmp66 = tl.load(in_ptr15 + ((-274400) + x2 + (1225*y0) + (39200*y1)), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.load(in_ptr16 + (tl.broadcast_to((-224) + y0, [XBLOCK, YBLOCK])), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp66 - tmp67
    tmp69 = tl.load(in_ptr17 + (tl.broadcast_to((-224) + y0, [XBLOCK, YBLOCK])), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp70 = tmp69 + tmp9
    tmp71 = tl.sqrt(tmp70)
    tmp72 = 1 / tmp71
    tmp73 = tmp72 * tmp13
    tmp74 = tmp68 * tmp73
    tmp75 = tl.load(in_ptr18 + (tl.broadcast_to((-224) + y0, [XBLOCK, YBLOCK])), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp76 = tmp74 * tmp75
    tmp77 = tl.load(in_ptr19 + (tl.broadcast_to((-224) + y0, [XBLOCK, YBLOCK])), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp78 = tmp76 + tmp77
    tmp79 = triton_helpers.maximum(0, tmp78)
    tmp80 = tl.full(tmp79.shape, 0.0, tmp79.dtype)
    tmp81 = tl.where(tmp63, tmp79, tmp80)
    tmp82 = tl.where(tmp46, tmp62, tmp81)
    tmp83 = tl.where(tmp26, tmp42, tmp82)
    tmp84 = tl.where(tmp4, tmp22, tmp83)
    tl.store(out_ptr0 + (y0 + (256*x2) + (313600*y1)), tmp84, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2l/c2l7r6fg5bsbzlyo45xlhmnlltgd35wvy5yu3enjfnm3dhu3ib76.py
# Source Nodes: [branch_pool_2], Original ATen: [aten.avg_pool2d]
# branch_pool_2 => avg_pool2d_1
triton_poi_fused_avg_pool2d_20 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2508800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 8960) % 35
    x1 = (xindex // 256) % 35
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 35, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-9216) + x6), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x1
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-8960) + x6), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x1
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-8704) + x6), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x2
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-256) + x6), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x6), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (256 + x6), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x2
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (8704 + x6), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (8960 + x6), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (9216 + x6), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 36, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x6), tmp145, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/en/ceneucmbenkgujtxzy55k6y5r2dvk3x66scgdvje5nfeidcouywj.py
# Source Nodes: [cat_28], Original ATen: [aten.cat]
# cat_28 => cat_1
triton_poi_fused_cat_21 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 2048], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: 'i32', 22: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(21,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2304
    xnumel = 1225
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 288
    x2 = xindex
    y1 = (yindex // 288)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (1225*y0) + (78400*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = tl.sqrt(tmp10)
    tmp12 = 1 / tmp11
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp7 * tmp14
    tmp16 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 * tmp16
    tmp18 = tl.load(in_ptr4 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = triton_helpers.maximum(0, tmp19)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp4, tmp20, tmp21)
    tmp23 = tmp0 >= tmp3
    tmp24 = tl.full([1, 1], 128, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tmp23 & tmp25
    tmp27 = tl.load(in_ptr5 + ((-78400) + x2 + (1225*y0) + (78400*y1)), tmp26 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr6 + (tl.broadcast_to((-64) + y0, [XBLOCK, YBLOCK])), tmp26 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 - tmp28
    tmp30 = tl.load(in_ptr7 + (tl.broadcast_to((-64) + y0, [XBLOCK, YBLOCK])), tmp26 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp30 + tmp9
    tmp32 = tl.sqrt(tmp31)
    tmp33 = 1 / tmp32
    tmp34 = tmp33 * tmp13
    tmp35 = tmp29 * tmp34
    tmp36 = tl.load(in_ptr8 + (tl.broadcast_to((-64) + y0, [XBLOCK, YBLOCK])), tmp26 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp37 = tmp35 * tmp36
    tmp38 = tl.load(in_ptr9 + (tl.broadcast_to((-64) + y0, [XBLOCK, YBLOCK])), tmp26 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 + tmp38
    tmp40 = triton_helpers.maximum(0, tmp39)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp26, tmp40, tmp41)
    tmp43 = tmp0 >= tmp24
    tmp44 = tl.full([1, 1], 224, tl.int64)
    tmp45 = tmp0 < tmp44
    tmp46 = tmp43 & tmp45
    tmp47 = tl.load(in_ptr10 + ((-156800) + x2 + (1225*y0) + (117600*y1)), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr11 + (tl.broadcast_to((-128) + y0, [XBLOCK, YBLOCK])), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp49 = tmp47 - tmp48
    tmp50 = tl.load(in_ptr12 + (tl.broadcast_to((-128) + y0, [XBLOCK, YBLOCK])), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp9
    tmp52 = tl.sqrt(tmp51)
    tmp53 = 1 / tmp52
    tmp54 = tmp53 * tmp13
    tmp55 = tmp49 * tmp54
    tmp56 = tl.load(in_ptr13 + (tl.broadcast_to((-128) + y0, [XBLOCK, YBLOCK])), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp55 * tmp56
    tmp58 = tl.load(in_ptr14 + (tl.broadcast_to((-128) + y0, [XBLOCK, YBLOCK])), tmp46 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp57 + tmp58
    tmp60 = triton_helpers.maximum(0, tmp59)
    tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
    tmp62 = tl.where(tmp46, tmp60, tmp61)
    tmp63 = tmp0 >= tmp44
    tmp64 = tl.full([1, 1], 288, tl.int64)
    tmp65 = tmp0 < tmp64
    tmp66 = tl.load(in_ptr15 + ((-274400) + x2 + (1225*y0) + (78400*y1)), tmp63 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.load(in_ptr16 + (tl.broadcast_to((-224) + y0, [XBLOCK, YBLOCK])), tmp63 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp66 - tmp67
    tmp69 = tl.load(in_ptr17 + (tl.broadcast_to((-224) + y0, [XBLOCK, YBLOCK])), tmp63 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp70 = tmp69 + tmp9
    tmp71 = tl.sqrt(tmp70)
    tmp72 = 1 / tmp71
    tmp73 = tmp72 * tmp13
    tmp74 = tmp68 * tmp73
    tmp75 = tl.load(in_ptr18 + (tl.broadcast_to((-224) + y0, [XBLOCK, YBLOCK])), tmp63 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp76 = tmp74 * tmp75
    tmp77 = tl.load(in_ptr19 + (tl.broadcast_to((-224) + y0, [XBLOCK, YBLOCK])), tmp63 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp78 = tmp76 + tmp77
    tmp79 = triton_helpers.maximum(0, tmp78)
    tmp80 = tl.full(tmp79.shape, 0.0, tmp79.dtype)
    tmp81 = tl.where(tmp63, tmp79, tmp80)
    tmp82 = tl.where(tmp46, tmp62, tmp81)
    tmp83 = tl.where(tmp26, tmp42, tmp82)
    tmp84 = tl.where(tmp4, tmp22, tmp83)
    tl.store(out_ptr0 + (y0 + (288*x2) + (352800*y1)), tmp84, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dt/cdtek6x4gmhwjyhxpst2spubgzgh3vlzowrexsnxy6mehq4je3i5.py
# Source Nodes: [branch_pool_4], Original ATen: [aten.avg_pool2d]
# branch_pool_4 => avg_pool2d_2
triton_poi_fused_avg_pool2d_22 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2822400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 10080) % 35
    x1 = (xindex // 288) % 35
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 35, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-10368) + x6), tmp10 & xmask, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x1
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-10080) + x6), tmp18 & xmask, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x1
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-9792) + x6), tmp27 & xmask, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x2
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-288) + x6), tmp36 & xmask, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x6), tmp41 & xmask, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (288 + x6), tmp46 & xmask, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x2
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (9792 + x6), tmp55 & xmask, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (10080 + x6), tmp60 & xmask, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (10368 + x6), tmp65 & xmask, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 36, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x6), tmp145, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/54/c54or37edaieiupegznkqjoblofstcluvziiaxqr7g33eb2oha5a.py
# Source Nodes: [branch_pool_6], Original ATen: [aten.max_pool2d_with_indices]
# branch_pool_6 => max_pool2d_with_indices_2
triton_poi_fused_max_pool2d_with_indices_23 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 512], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_23', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2304
    xnumel = 289
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 17
    x3 = (xindex // 17)
    y0 = yindex % 288
    y1 = (yindex // 288)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (y0 + (576*x2) + (20160*x3) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (288 + y0 + (576*x2) + (20160*x3) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (576 + y0 + (576*x2) + (20160*x3) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (10080 + y0 + (576*x2) + (20160*x3) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (10368 + y0 + (576*x2) + (20160*x3) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (10656 + y0 + (576*x2) + (20160*x3) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (20160 + y0 + (576*x2) + (20160*x3) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (20448 + y0 + (576*x2) + (20160*x3) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (20736 + y0 + (576*x2) + (20160*x3) + (352800*y1)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (x4 + (289*y0) + (221952*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/p5/cp5mf6lfntdupvqik3azojswc72hfu4l6q5dkgvjb3kaqrlrxy6n.py
# Source Nodes: [x_140], Original ATen: [aten.convolution]
# x_140 => convolution_26
triton_poi_fused_convolution_24 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[131072, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_24', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 110592
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 288
    y1 = (yindex // 288)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (288*x2) + (2592*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/7d/c7djxa5flc4im3mxlkgn3umqi2copeuwj75nk2qbet3tpzpc46n2.py
# Source Nodes: [branch3x3, x_141], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# branch3x3 => relu_26
# x_141 => add_53, mul_79, mul_80, sub_26
triton_poi_fused__native_batch_norm_legit_no_training_relu_25 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_25', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 887808
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 289) % 384
    x2 = (xindex // 110976)
    x4 = xindex % 110976
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4 + (221952*x2)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5v/c5vusg6slkctxjlr7odu7irp7cyrbr3t7eyn5z6uae3b6zdc45db.py
# Source Nodes: [branch3x3dbl_11, x_156], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# branch3x3dbl_11 => relu_29
# x_156 => add_59, mul_88, mul_89, sub_29
triton_poi_fused__native_batch_norm_legit_no_training_relu_26 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_26', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 221952
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 289) % 96
    x2 = (xindex // 27744)
    x4 = xindex % 27744
    tmp0 = tl.load(in_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr1 + (x1), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), xmask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4 + (221952*x2)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gl/cgl7ja7oly3ylhk7nyzm4czy5frja33ujbykekmd3y7erbrojnru.py
# Source Nodes: [branch_pool_7, x_161, x_166, x_181], Original ATen: [aten.avg_pool2d, aten.convolution]
# branch_pool_7 => avg_pool2d_3
# x_161 => convolution_30
# x_166 => convolution_31
# x_181 => convolution_34
triton_poi_fused_avg_pool2d_convolution_27 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_convolution_27', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 289
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 768
    y1 = (yindex // 768)
    x5 = (xindex // 17)
    x4 = xindex % 17
    tmp0 = tl.load(in_ptr0 + (x2 + (289*y3)), xmask, eviction_policy='evict_last')
    tmp1 = (-1) + x5
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 17, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tmp3 & tmp5
    tmp7 = (-1) + x4
    tmp8 = tmp7 >= tmp2
    tmp9 = tmp7 < tmp4
    tmp10 = tmp8 & tmp9
    tmp11 = tmp6 & tmp10
    tmp12 = tl.load(in_ptr0 + ((-18) + x2 + (289*y3)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = x4
    tmp16 = tmp15 >= tmp2
    tmp17 = tmp15 < tmp4
    tmp18 = tmp16 & tmp17
    tmp19 = tmp6 & tmp18
    tmp20 = tl.load(in_ptr0 + ((-17) + x2 + (289*y3)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp22 + tmp14
    tmp24 = 1 + x4
    tmp25 = tmp24 >= tmp2
    tmp26 = tmp24 < tmp4
    tmp27 = tmp25 & tmp26
    tmp28 = tmp6 & tmp27
    tmp29 = tl.load(in_ptr0 + ((-16) + x2 + (289*y3)), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp28, tmp29, tmp30)
    tmp32 = tmp31 + tmp23
    tmp33 = x5
    tmp34 = tmp33 >= tmp2
    tmp35 = tmp33 < tmp4
    tmp36 = tmp34 & tmp35
    tmp37 = tmp36 & tmp10
    tmp38 = tl.load(in_ptr0 + ((-1) + x2 + (289*y3)), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = tmp40 + tmp32
    tmp42 = tmp36 & tmp18
    tmp43 = tl.load(in_ptr0 + (x2 + (289*y3)), tmp42 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp42, tmp43, tmp44)
    tmp46 = tmp45 + tmp41
    tmp47 = tmp36 & tmp27
    tmp48 = tl.load(in_ptr0 + (1 + x2 + (289*y3)), tmp47 & xmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp47, tmp48, tmp49)
    tmp51 = tmp50 + tmp46
    tmp52 = 1 + x5
    tmp53 = tmp52 >= tmp2
    tmp54 = tmp52 < tmp4
    tmp55 = tmp53 & tmp54
    tmp56 = tmp55 & tmp10
    tmp57 = tl.load(in_ptr0 + (16 + x2 + (289*y3)), tmp56 & xmask, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.full(tmp57.shape, 0.0, tmp57.dtype)
    tmp59 = tl.where(tmp56, tmp57, tmp58)
    tmp60 = tmp59 + tmp51
    tmp61 = tmp55 & tmp18
    tmp62 = tl.load(in_ptr0 + (17 + x2 + (289*y3)), tmp61 & xmask, eviction_policy='evict_last', other=0.0)
    tmp63 = tl.full(tmp62.shape, 0.0, tmp62.dtype)
    tmp64 = tl.where(tmp61, tmp62, tmp63)
    tmp65 = tmp64 + tmp60
    tmp66 = tmp55 & tmp27
    tmp67 = tl.load(in_ptr0 + (18 + x2 + (289*y3)), tmp66 & xmask, eviction_policy='evict_last', other=0.0)
    tmp68 = tl.full(tmp67.shape, 0.0, tmp67.dtype)
    tmp69 = tl.where(tmp66, tmp67, tmp68)
    tmp70 = tmp69 + tmp65
    tmp71 = tl.full([1, 1], -1, tl.int64)
    tmp72 = tmp1 >= tmp71
    tmp73 = tl.full([1, 1], 18, tl.int64)
    tmp74 = tmp1 < tmp73
    tmp75 = tmp72 & tmp74
    tmp76 = tmp7 >= tmp71
    tmp77 = tmp7 < tmp73
    tmp78 = tmp76 & tmp77
    tmp79 = tmp75 & tmp78
    tmp80 = tl.broadcast_to((-1) + x5, [XBLOCK, YBLOCK])
    tmp81 = tmp80 >= tmp2
    tmp82 = tmp80 < tmp4
    tmp83 = tmp81 & tmp82
    tmp84 = tl.broadcast_to((-1) + x4, [XBLOCK, YBLOCK])
    tmp85 = tmp84 >= tmp2
    tmp86 = tmp84 < tmp4
    tmp87 = tmp85 & tmp86
    tmp88 = tmp83 & tmp87
    tmp89 = tmp88 & tmp79
    tmp90 = 1.0
    tmp91 = tl.full(tmp90.shape, 1.0, tmp90.dtype)
    tmp92 = tl.where(tmp89, tmp90, tmp91)
    tmp93 = tl.full(tmp92.shape, 0.0, tmp92.dtype)
    tmp94 = tl.where(tmp79, tmp92, tmp93)
    tmp95 = tmp15 >= tmp71
    tmp96 = tmp15 < tmp73
    tmp97 = tmp95 & tmp96
    tmp98 = tmp75 & tmp97
    tmp99 = tl.broadcast_to(x4, [XBLOCK, YBLOCK])
    tmp100 = tmp99 >= tmp2
    tmp101 = tmp99 < tmp4
    tmp102 = tmp100 & tmp101
    tmp103 = tmp83 & tmp102
    tmp104 = tmp103 & tmp98
    tmp105 = tl.where(tmp104, tmp90, tmp91)
    tmp106 = tl.full(tmp105.shape, 0.0, tmp105.dtype)
    tmp107 = tl.where(tmp98, tmp105, tmp106)
    tmp108 = tmp107 + tmp94
    tmp109 = tmp24 >= tmp71
    tmp110 = tmp24 < tmp73
    tmp111 = tmp109 & tmp110
    tmp112 = tmp75 & tmp111
    tmp113 = tl.broadcast_to(1 + x4, [XBLOCK, YBLOCK])
    tmp114 = tmp113 >= tmp2
    tmp115 = tmp113 < tmp4
    tmp116 = tmp114 & tmp115
    tmp117 = tmp83 & tmp116
    tmp118 = tmp117 & tmp112
    tmp119 = tl.where(tmp118, tmp90, tmp91)
    tmp120 = tl.full(tmp119.shape, 0.0, tmp119.dtype)
    tmp121 = tl.where(tmp112, tmp119, tmp120)
    tmp122 = tmp121 + tmp108
    tmp123 = tmp33 >= tmp71
    tmp124 = tmp33 < tmp73
    tmp125 = tmp123 & tmp124
    tmp126 = tmp125 & tmp78
    tmp127 = tl.broadcast_to(x5, [XBLOCK, YBLOCK])
    tmp128 = tmp127 >= tmp2
    tmp129 = tmp127 < tmp4
    tmp130 = tmp128 & tmp129
    tmp131 = tmp130 & tmp87
    tmp132 = tmp131 & tmp126
    tmp133 = tl.where(tmp132, tmp90, tmp91)
    tmp134 = tl.full(tmp133.shape, 0.0, tmp133.dtype)
    tmp135 = tl.where(tmp126, tmp133, tmp134)
    tmp136 = tmp135 + tmp122
    tmp137 = tmp125 & tmp97
    tmp138 = tmp130 & tmp102
    tmp139 = tmp138 & tmp137
    tmp140 = tl.where(tmp139, tmp90, tmp91)
    tmp141 = tl.full(tmp140.shape, 0.0, tmp140.dtype)
    tmp142 = tl.where(tmp137, tmp140, tmp141)
    tmp143 = tmp142 + tmp136
    tmp144 = tmp125 & tmp111
    tmp145 = tmp130 & tmp116
    tmp146 = tmp145 & tmp144
    tmp147 = tl.where(tmp146, tmp90, tmp91)
    tmp148 = tl.full(tmp147.shape, 0.0, tmp147.dtype)
    tmp149 = tl.where(tmp144, tmp147, tmp148)
    tmp150 = tmp149 + tmp143
    tmp151 = tmp52 >= tmp71
    tmp152 = tmp52 < tmp73
    tmp153 = tmp151 & tmp152
    tmp154 = tmp153 & tmp78
    tmp155 = tl.broadcast_to(1 + x5, [XBLOCK, YBLOCK])
    tmp156 = tmp155 >= tmp2
    tmp157 = tmp155 < tmp4
    tmp158 = tmp156 & tmp157
    tmp159 = tmp158 & tmp87
    tmp160 = tmp159 & tmp154
    tmp161 = tl.where(tmp160, tmp90, tmp91)
    tmp162 = tl.full(tmp161.shape, 0.0, tmp161.dtype)
    tmp163 = tl.where(tmp154, tmp161, tmp162)
    tmp164 = tmp163 + tmp150
    tmp165 = tmp153 & tmp97
    tmp166 = tmp158 & tmp102
    tmp167 = tmp166 & tmp165
    tmp168 = tl.where(tmp167, tmp90, tmp91)
    tmp169 = tl.full(tmp168.shape, 0.0, tmp168.dtype)
    tmp170 = tl.where(tmp165, tmp168, tmp169)
    tmp171 = tmp170 + tmp164
    tmp172 = tmp153 & tmp111
    tmp173 = tmp158 & tmp116
    tmp174 = tmp173 & tmp172
    tmp175 = tl.where(tmp174, tmp90, tmp91)
    tmp176 = tl.full(tmp175.shape, 0.0, tmp175.dtype)
    tmp177 = tl.where(tmp172, tmp175, tmp176)
    tmp178 = tmp177 + tmp171
    tmp179 = tmp70 / tmp178
    tl.store(out_ptr0 + (y0 + (768*x2) + (221952*y1)), tmp0, xmask)
    tl.store(out_ptr1 + (y0 + (768*x2) + (221952*y1)), tmp0, xmask)
    tl.store(out_ptr2 + (y0 + (768*x2) + (221952*y1)), tmp0, xmask)
    tl.store(out_ptr3 + (y0 + (768*x2) + (221952*y1)), tmp179, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v3/cv3bv2kcintavdcfrjasllotjasnelwigasqranukxwi3lwwxm5n.py
# Source Nodes: [branch7x7, x_167], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# branch7x7 => relu_31
# x_167 => add_63, mul_94, mul_95, sub_31
triton_poi_fused__native_batch_norm_legit_no_training_relu_28 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_28', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 289
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
    tmp0 = tl.load(in_ptr0 + (x2 + (289*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (128*x2) + (36992*y1)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6x/c6xfikxxalmy7mjaqqqlzytqffyd6ud7ebi4mgw7avkuagmykexv.py
# Source Nodes: [branch7x7, x_167, x_171], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# branch7x7 => relu_31
# x_167 => add_63, mul_94, mul_95, sub_31
# x_171 => convolution_32
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_29 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[16384, 8], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_29', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 7
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
    tmp0 = tl.load(in_ptr0 + (x2 + (7*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (896*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/f5/cf5wrk3vaeup7lo2v75vpu44vdxaegtl6d4lqndibbwkmmhn7jen.py
# Source Nodes: [branch7x7_1, x_172, x_176], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# branch7x7_1 => relu_32
# x_172 => add_65, mul_97, mul_98, sub_32
# x_176 => convolution_33
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 8], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 24576
    xnumel = 7
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
    tmp0 = tl.load(in_ptr0 + (x2 + (7*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (896*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h5/ch5jwk7jkhm7ixt2duu6smozzautj2szdvzsgxwdz5t45awhmqgo.py
# Source Nodes: [cat_25], Original ATen: [aten.cat]
# cat_25 => cat_4
triton_poi_fused_cat_31 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp32', 17: '*fp32', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: 'i32', 22: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(21,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_31', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 289
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 768
    x2 = xindex
    y1 = (yindex // 768)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 192, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (289*y0) + (55488*y1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = tl.sqrt(tmp10)
    tmp12 = 1 / tmp11
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp7 * tmp14
    tmp16 = tl.load(in_ptr3 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 * tmp16
    tmp18 = tl.load(in_ptr4 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = triton_helpers.maximum(0, tmp19)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp4, tmp20, tmp21)
    tmp23 = tmp0 >= tmp3
    tmp24 = tl.full([1, 1], 384, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tmp23 & tmp25
    tmp27 = tl.load(in_ptr5 + ((-55488) + x2 + (289*y0) + (55488*y1)), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp28 = tl.load(in_ptr6 + (tl.broadcast_to((-192) + y0, [XBLOCK, YBLOCK])), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp29 = tmp27 - tmp28
    tmp30 = tl.load(in_ptr7 + (tl.broadcast_to((-192) + y0, [XBLOCK, YBLOCK])), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp31 = tmp30 + tmp9
    tmp32 = tl.sqrt(tmp31)
    tmp33 = 1 / tmp32
    tmp34 = tmp33 * tmp13
    tmp35 = tmp29 * tmp34
    tmp36 = tl.load(in_ptr8 + (tl.broadcast_to((-192) + y0, [XBLOCK, YBLOCK])), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp37 = tmp35 * tmp36
    tmp38 = tl.load(in_ptr9 + (tl.broadcast_to((-192) + y0, [XBLOCK, YBLOCK])), tmp26 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tmp37 + tmp38
    tmp40 = triton_helpers.maximum(0, tmp39)
    tmp41 = tl.full(tmp40.shape, 0.0, tmp40.dtype)
    tmp42 = tl.where(tmp26, tmp40, tmp41)
    tmp43 = tmp0 >= tmp24
    tmp44 = tl.full([1, 1], 576, tl.int64)
    tmp45 = tmp0 < tmp44
    tmp46 = tmp43 & tmp45
    tmp47 = tl.load(in_ptr10 + ((-110976) + x2 + (289*y0) + (55488*y1)), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp48 = tl.load(in_ptr11 + (tl.broadcast_to((-384) + y0, [XBLOCK, YBLOCK])), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tmp47 - tmp48
    tmp50 = tl.load(in_ptr12 + (tl.broadcast_to((-384) + y0, [XBLOCK, YBLOCK])), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp51 = tmp50 + tmp9
    tmp52 = tl.sqrt(tmp51)
    tmp53 = 1 / tmp52
    tmp54 = tmp53 * tmp13
    tmp55 = tmp49 * tmp54
    tmp56 = tl.load(in_ptr13 + (tl.broadcast_to((-384) + y0, [XBLOCK, YBLOCK])), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp57 = tmp55 * tmp56
    tmp58 = tl.load(in_ptr14 + (tl.broadcast_to((-384) + y0, [XBLOCK, YBLOCK])), tmp46 & xmask, eviction_policy='evict_last', other=0.0)
    tmp59 = tmp57 + tmp58
    tmp60 = triton_helpers.maximum(0, tmp59)
    tmp61 = tl.full(tmp60.shape, 0.0, tmp60.dtype)
    tmp62 = tl.where(tmp46, tmp60, tmp61)
    tmp63 = tmp0 >= tmp44
    tmp64 = tl.full([1, 1], 768, tl.int64)
    tmp65 = tmp0 < tmp64
    tmp66 = tl.load(in_ptr15 + ((-166464) + x2 + (289*y0) + (55488*y1)), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp67 = tl.load(in_ptr16 + (tl.broadcast_to((-576) + y0, [XBLOCK, YBLOCK])), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp68 = tmp66 - tmp67
    tmp69 = tl.load(in_ptr17 + (tl.broadcast_to((-576) + y0, [XBLOCK, YBLOCK])), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp70 = tmp69 + tmp9
    tmp71 = tl.sqrt(tmp70)
    tmp72 = 1 / tmp71
    tmp73 = tmp72 * tmp13
    tmp74 = tmp68 * tmp73
    tmp75 = tl.load(in_ptr18 + (tl.broadcast_to((-576) + y0, [XBLOCK, YBLOCK])), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp76 = tmp74 * tmp75
    tmp77 = tl.load(in_ptr19 + (tl.broadcast_to((-576) + y0, [XBLOCK, YBLOCK])), tmp63 & xmask, eviction_policy='evict_last', other=0.0)
    tmp78 = tmp76 + tmp77
    tmp79 = triton_helpers.maximum(0, tmp78)
    tmp80 = tl.full(tmp79.shape, 0.0, tmp79.dtype)
    tmp81 = tl.where(tmp63, tmp79, tmp80)
    tmp82 = tl.where(tmp46, tmp62, tmp81)
    tmp83 = tl.where(tmp26, tmp42, tmp82)
    tmp84 = tl.where(tmp4, tmp22, tmp83)
    tl.store(out_ptr0 + (y0 + (768*x2) + (221952*y1)), tmp84, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gv/cgvpry4nfnvhayeqvaqxilcrle56whn7i7ij6zwzj7stbo3g4tll.py
# Source Nodes: [branch7x7_3, x_218], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# branch7x7_3 => relu_41
# x_218 => add_83, mul_124, mul_125, sub_41
triton_poi_fused__native_batch_norm_legit_no_training_relu_32 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_32', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1280
    xnumel = 289
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
    tmp0 = tl.load(in_ptr0 + (x2 + (289*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (160*x2) + (46240*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/33/c33ko5nbth4h7srvfkoubbaen3hszkmlhlqdrsj3di24lcnpsjkk.py
# Source Nodes: [branch7x7_3, x_218, x_222], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# branch7x7_3 => relu_41
# x_218 => add_83, mul_124, mul_125, sub_41
# x_222 => convolution_42
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 8], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 25600
    xnumel = 7
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
    tmp0 = tl.load(in_ptr0 + (x2 + (7*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (160*x2) + (1120*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/qa/cqaxy724tbfmdry7wuaslimtrdg6j5l73adho4d22i73w3lpb34q.py
# Source Nodes: [branch7x7_4, x_223, x_227], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# branch7x7_4 => relu_42
# x_223 => add_85, mul_127, mul_128, sub_42
# x_227 => convolution_43
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 8], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 30720
    xnumel = 7
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
    tmp0 = tl.load(in_ptr0 + (x2 + (7*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (160*x2) + (1120*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/d4/cd4qqfdihhgv2ilueb5ffvbyljk5jt2dqqdicnhu7eo7tqvrkvsw.py
# Source Nodes: [branch_pool_9], Original ATen: [aten.avg_pool2d]
# branch_pool_9 => avg_pool2d_4
triton_poi_fused_avg_pool2d_35 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_35', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1775616
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 13056) % 17
    x1 = (xindex // 768) % 17
    x6 = xindex
    tmp0 = (-1) + x2
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 17, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tmp2 & tmp4
    tmp6 = (-1) + x1
    tmp7 = tmp6 >= tmp1
    tmp8 = tmp6 < tmp3
    tmp9 = tmp7 & tmp8
    tmp10 = tmp5 & tmp9
    tmp11 = tl.load(in_ptr0 + ((-13824) + x6), tmp10, other=0.0)
    tmp12 = tl.full(tmp11.shape, 0.0, tmp11.dtype)
    tmp13 = tl.where(tmp10, tmp11, tmp12)
    tmp14 = x1
    tmp15 = tmp14 >= tmp1
    tmp16 = tmp14 < tmp3
    tmp17 = tmp15 & tmp16
    tmp18 = tmp5 & tmp17
    tmp19 = tl.load(in_ptr0 + ((-13056) + x6), tmp18, other=0.0)
    tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
    tmp21 = tl.where(tmp18, tmp19, tmp20)
    tmp22 = tmp21 + tmp13
    tmp23 = 1 + x1
    tmp24 = tmp23 >= tmp1
    tmp25 = tmp23 < tmp3
    tmp26 = tmp24 & tmp25
    tmp27 = tmp5 & tmp26
    tmp28 = tl.load(in_ptr0 + ((-12288) + x6), tmp27, other=0.0)
    tmp29 = tl.full(tmp28.shape, 0.0, tmp28.dtype)
    tmp30 = tl.where(tmp27, tmp28, tmp29)
    tmp31 = tmp30 + tmp22
    tmp32 = x2
    tmp33 = tmp32 >= tmp1
    tmp34 = tmp32 < tmp3
    tmp35 = tmp33 & tmp34
    tmp36 = tmp35 & tmp9
    tmp37 = tl.load(in_ptr0 + ((-768) + x6), tmp36, other=0.0)
    tmp38 = tl.full(tmp37.shape, 0.0, tmp37.dtype)
    tmp39 = tl.where(tmp36, tmp37, tmp38)
    tmp40 = tmp39 + tmp31
    tmp41 = tmp35 & tmp17
    tmp42 = tl.load(in_ptr0 + (x6), tmp41, other=0.0)
    tmp43 = tl.full(tmp42.shape, 0.0, tmp42.dtype)
    tmp44 = tl.where(tmp41, tmp42, tmp43)
    tmp45 = tmp44 + tmp40
    tmp46 = tmp35 & tmp26
    tmp47 = tl.load(in_ptr0 + (768 + x6), tmp46, other=0.0)
    tmp48 = tl.full(tmp47.shape, 0.0, tmp47.dtype)
    tmp49 = tl.where(tmp46, tmp47, tmp48)
    tmp50 = tmp49 + tmp45
    tmp51 = 1 + x2
    tmp52 = tmp51 >= tmp1
    tmp53 = tmp51 < tmp3
    tmp54 = tmp52 & tmp53
    tmp55 = tmp54 & tmp9
    tmp56 = tl.load(in_ptr0 + (12288 + x6), tmp55, other=0.0)
    tmp57 = tl.full(tmp56.shape, 0.0, tmp56.dtype)
    tmp58 = tl.where(tmp55, tmp56, tmp57)
    tmp59 = tmp58 + tmp50
    tmp60 = tmp54 & tmp17
    tmp61 = tl.load(in_ptr0 + (13056 + x6), tmp60, other=0.0)
    tmp62 = tl.full(tmp61.shape, 0.0, tmp61.dtype)
    tmp63 = tl.where(tmp60, tmp61, tmp62)
    tmp64 = tmp63 + tmp59
    tmp65 = tmp54 & tmp26
    tmp66 = tl.load(in_ptr0 + (13824 + x6), tmp65, other=0.0)
    tmp67 = tl.full(tmp66.shape, 0.0, tmp66.dtype)
    tmp68 = tl.where(tmp65, tmp66, tmp67)
    tmp69 = tmp68 + tmp64
    tmp70 = tl.full([1], -1, tl.int64)
    tmp71 = tmp0 >= tmp70
    tmp72 = tl.full([1], 18, tl.int64)
    tmp73 = tmp0 < tmp72
    tmp74 = tmp71 & tmp73
    tmp75 = tmp6 >= tmp70
    tmp76 = tmp6 < tmp72
    tmp77 = tmp75 & tmp76
    tmp78 = tmp74 & tmp77
    tmp79 = tmp10 & tmp78
    tmp80 = 1.0
    tmp81 = tl.full(tmp80.shape, 1.0, tmp80.dtype)
    tmp82 = tl.where(tmp79, tmp80, tmp81)
    tmp83 = tl.full(tmp82.shape, 0.0, tmp82.dtype)
    tmp84 = tl.where(tmp78, tmp82, tmp83)
    tmp85 = tmp14 >= tmp70
    tmp86 = tmp14 < tmp72
    tmp87 = tmp85 & tmp86
    tmp88 = tmp74 & tmp87
    tmp89 = tmp18 & tmp88
    tmp90 = tl.where(tmp89, tmp80, tmp81)
    tmp91 = tl.full(tmp90.shape, 0.0, tmp90.dtype)
    tmp92 = tl.where(tmp88, tmp90, tmp91)
    tmp93 = tmp92 + tmp84
    tmp94 = tmp23 >= tmp70
    tmp95 = tmp23 < tmp72
    tmp96 = tmp94 & tmp95
    tmp97 = tmp74 & tmp96
    tmp98 = tmp27 & tmp97
    tmp99 = tl.where(tmp98, tmp80, tmp81)
    tmp100 = tl.full(tmp99.shape, 0.0, tmp99.dtype)
    tmp101 = tl.where(tmp97, tmp99, tmp100)
    tmp102 = tmp101 + tmp93
    tmp103 = tmp32 >= tmp70
    tmp104 = tmp32 < tmp72
    tmp105 = tmp103 & tmp104
    tmp106 = tmp105 & tmp77
    tmp107 = tmp36 & tmp106
    tmp108 = tl.where(tmp107, tmp80, tmp81)
    tmp109 = tl.full(tmp108.shape, 0.0, tmp108.dtype)
    tmp110 = tl.where(tmp106, tmp108, tmp109)
    tmp111 = tmp110 + tmp102
    tmp112 = tmp105 & tmp87
    tmp113 = tmp41 & tmp112
    tmp114 = tl.where(tmp113, tmp80, tmp81)
    tmp115 = tl.full(tmp114.shape, 0.0, tmp114.dtype)
    tmp116 = tl.where(tmp112, tmp114, tmp115)
    tmp117 = tmp116 + tmp111
    tmp118 = tmp105 & tmp96
    tmp119 = tmp46 & tmp118
    tmp120 = tl.where(tmp119, tmp80, tmp81)
    tmp121 = tl.full(tmp120.shape, 0.0, tmp120.dtype)
    tmp122 = tl.where(tmp118, tmp120, tmp121)
    tmp123 = tmp122 + tmp117
    tmp124 = tmp51 >= tmp70
    tmp125 = tmp51 < tmp72
    tmp126 = tmp124 & tmp125
    tmp127 = tmp126 & tmp77
    tmp128 = tmp55 & tmp127
    tmp129 = tl.where(tmp128, tmp80, tmp81)
    tmp130 = tl.full(tmp129.shape, 0.0, tmp129.dtype)
    tmp131 = tl.where(tmp127, tmp129, tmp130)
    tmp132 = tmp131 + tmp123
    tmp133 = tmp126 & tmp87
    tmp134 = tmp60 & tmp133
    tmp135 = tl.where(tmp134, tmp80, tmp81)
    tmp136 = tl.full(tmp135.shape, 0.0, tmp135.dtype)
    tmp137 = tl.where(tmp133, tmp135, tmp136)
    tmp138 = tmp137 + tmp132
    tmp139 = tmp126 & tmp96
    tmp140 = tmp65 & tmp139
    tmp141 = tl.where(tmp140, tmp80, tmp81)
    tmp142 = tl.full(tmp141.shape, 0.0, tmp141.dtype)
    tmp143 = tl.where(tmp139, tmp141, tmp142)
    tmp144 = tmp143 + tmp138
    tmp145 = tmp69 / tmp144
    tl.store(out_ptr0 + (x6), tmp145, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/s5/cs5ytbgctxsqa5airiz3jpbkafzzejvibjlajj345hrokvmc7oyi.py
# Source Nodes: [branch7x7_9, x_320], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# branch7x7_9 => relu_61
# x_320 => add_123, mul_184, mul_185, sub_61
triton_poi_fused__native_batch_norm_legit_no_training_relu_36 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 512], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_36', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 289
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
    tmp0 = tl.load(in_ptr0 + (x2 + (289*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (192*x2) + (55488*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dn/cdn35a6nwcqyjlij2zlikcaatzsqg4x3u7bhrlbbvz7shabautpq.py
# Source Nodes: [branch7x7_9, x_320, x_324], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# branch7x7_9 => relu_61
# x_320 => add_123, mul_184, mul_185, sub_61
# x_324 => convolution_62
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 8], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 36864
    xnumel = 7
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
    tmp0 = tl.load(in_ptr0 + (x2 + (7*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (1344*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4d/c4dte5e5fhziql4634puiopwvir3mjqdyo2yhc4ozdevplk5isxc.py
# Source Nodes: [branch_pool_15], Original ATen: [aten.max_pool2d_with_indices]
# branch_pool_15 => max_pool2d_with_indices_3
triton_poi_fused_max_pool2d_with_indices_38 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[8192, 64], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_38', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 8
    x3 = (xindex // 8)
    y0 = yindex % 768
    y1 = (yindex // 768)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (768 + y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (1536 + y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (13056 + y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (13824 + y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (14592 + y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (26112 + y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (26880 + y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (27648 + y0 + (1536*x2) + (26112*x3) + (221952*y1)), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (x4 + (64*y0) + (81920*y1)), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/72/c72jqqyqfk2q4xy4o7eqpt66k3svh3mjj6wfv7p6lxx2it6qetkw.py
# Source Nodes: [branch3x3_1, x_367, x_371], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# branch3x3_1 => relu_70
# x_367 => add_141, mul_211, mul_212, sub_70
# x_371 => convolution_71
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_39 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_39', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 61440
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (1728*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/jl/cjlca225amzfltacrdom25ubmwfo7ynvwki7bjtd6aen4t3veml4.py
# Source Nodes: [branch7x7x3_2, x_387, x_391], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# branch7x7x3_2 => relu_74
# x_387 => add_149, mul_223, mul_224, sub_74
# x_391 => convolution_75
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_40 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_40', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 36864
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (192*x2) + (1728*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/q6/cq666hpl4rlz3trezjororfyx5xylbgpfzgxdkwjculyamcq2dgm.py
# Source Nodes: [branch3x3_2, x_372], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# branch3x3_2 => relu_71
# x_372 => add_143, mul_214, mul_215, sub_71
triton_poi_fused__native_batch_norm_legit_no_training_relu_41 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_41', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 320
    x2 = (xindex // 20480)
    x4 = xindex % 20480
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4 + (81920*x2)), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xv/cxvubwxf5vsx7zxl22wiguyx5kial53punplw7tlwz7e6racb4gb.py
# Source Nodes: [branch7x7x3_3, x_392], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# branch7x7x3_3 => relu_75
# x_392 => add_151, mul_226, mul_227, sub_75
triton_poi_fused__native_batch_norm_legit_no_training_relu_42 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_42', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 192
    x2 = (xindex // 12288)
    x4 = xindex % 12288
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4 + (81920*x2)), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xr/cxrbhhd5ntl5polipaj74hy4kwtgvakklcefhaaq77b3ab47ckl6.py
# Source Nodes: [branch_pool_16, x_397, x_402, x_417], Original ATen: [aten.avg_pool2d, aten.convolution]
# branch_pool_16 => avg_pool2d_7
# x_397 => convolution_76
# x_402 => convolution_77
# x_417 => convolution_80
triton_poi_fused_avg_pool2d_convolution_43 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_convolution_43', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 10240
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 1280
    y1 = (yindex // 1280)
    x5 = (xindex // 8)
    x4 = xindex % 8
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp1 = (-1) + x5
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 8, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tmp3 & tmp5
    tmp7 = (-1) + x4
    tmp8 = tmp7 >= tmp2
    tmp9 = tmp7 < tmp4
    tmp10 = tmp8 & tmp9
    tmp11 = tmp6 & tmp10
    tmp12 = tl.load(in_ptr0 + ((-9) + x2 + (64*y3)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = x4
    tmp16 = tmp15 >= tmp2
    tmp17 = tmp15 < tmp4
    tmp18 = tmp16 & tmp17
    tmp19 = tmp6 & tmp18
    tmp20 = tl.load(in_ptr0 + ((-8) + x2 + (64*y3)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp22 + tmp14
    tmp24 = 1 + x4
    tmp25 = tmp24 >= tmp2
    tmp26 = tmp24 < tmp4
    tmp27 = tmp25 & tmp26
    tmp28 = tmp6 & tmp27
    tmp29 = tl.load(in_ptr0 + ((-7) + x2 + (64*y3)), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp28, tmp29, tmp30)
    tmp32 = tmp31 + tmp23
    tmp33 = x5
    tmp34 = tmp33 >= tmp2
    tmp35 = tmp33 < tmp4
    tmp36 = tmp34 & tmp35
    tmp37 = tmp36 & tmp10
    tmp38 = tl.load(in_ptr0 + ((-1) + x2 + (64*y3)), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = tmp40 + tmp32
    tmp42 = tmp36 & tmp18
    tmp43 = tl.load(in_ptr0 + (x2 + (64*y3)), tmp42 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp42, tmp43, tmp44)
    tmp46 = tmp45 + tmp41
    tmp47 = tmp36 & tmp27
    tmp48 = tl.load(in_ptr0 + (1 + x2 + (64*y3)), tmp47 & xmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp47, tmp48, tmp49)
    tmp51 = tmp50 + tmp46
    tmp52 = 1 + x5
    tmp53 = tmp52 >= tmp2
    tmp54 = tmp52 < tmp4
    tmp55 = tmp53 & tmp54
    tmp56 = tmp55 & tmp10
    tmp57 = tl.load(in_ptr0 + (7 + x2 + (64*y3)), tmp56 & xmask, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.full(tmp57.shape, 0.0, tmp57.dtype)
    tmp59 = tl.where(tmp56, tmp57, tmp58)
    tmp60 = tmp59 + tmp51
    tmp61 = tmp55 & tmp18
    tmp62 = tl.load(in_ptr0 + (8 + x2 + (64*y3)), tmp61 & xmask, eviction_policy='evict_last', other=0.0)
    tmp63 = tl.full(tmp62.shape, 0.0, tmp62.dtype)
    tmp64 = tl.where(tmp61, tmp62, tmp63)
    tmp65 = tmp64 + tmp60
    tmp66 = tmp55 & tmp27
    tmp67 = tl.load(in_ptr0 + (9 + x2 + (64*y3)), tmp66 & xmask, eviction_policy='evict_last', other=0.0)
    tmp68 = tl.full(tmp67.shape, 0.0, tmp67.dtype)
    tmp69 = tl.where(tmp66, tmp67, tmp68)
    tmp70 = tmp69 + tmp65
    tmp71 = tl.full([1, 1], -1, tl.int64)
    tmp72 = tmp1 >= tmp71
    tmp73 = tl.full([1, 1], 9, tl.int64)
    tmp74 = tmp1 < tmp73
    tmp75 = tmp72 & tmp74
    tmp76 = tmp7 >= tmp71
    tmp77 = tmp7 < tmp73
    tmp78 = tmp76 & tmp77
    tmp79 = tmp75 & tmp78
    tmp80 = tl.broadcast_to((-1) + x5, [XBLOCK, YBLOCK])
    tmp81 = tmp80 >= tmp2
    tmp82 = tmp80 < tmp4
    tmp83 = tmp81 & tmp82
    tmp84 = tl.broadcast_to((-1) + x4, [XBLOCK, YBLOCK])
    tmp85 = tmp84 >= tmp2
    tmp86 = tmp84 < tmp4
    tmp87 = tmp85 & tmp86
    tmp88 = tmp83 & tmp87
    tmp89 = tmp88 & tmp79
    tmp90 = 1.0
    tmp91 = tl.full(tmp90.shape, 1.0, tmp90.dtype)
    tmp92 = tl.where(tmp89, tmp90, tmp91)
    tmp93 = tl.full(tmp92.shape, 0.0, tmp92.dtype)
    tmp94 = tl.where(tmp79, tmp92, tmp93)
    tmp95 = tmp15 >= tmp71
    tmp96 = tmp15 < tmp73
    tmp97 = tmp95 & tmp96
    tmp98 = tmp75 & tmp97
    tmp99 = tl.broadcast_to(x4, [XBLOCK, YBLOCK])
    tmp100 = tmp99 >= tmp2
    tmp101 = tmp99 < tmp4
    tmp102 = tmp100 & tmp101
    tmp103 = tmp83 & tmp102
    tmp104 = tmp103 & tmp98
    tmp105 = tl.where(tmp104, tmp90, tmp91)
    tmp106 = tl.full(tmp105.shape, 0.0, tmp105.dtype)
    tmp107 = tl.where(tmp98, tmp105, tmp106)
    tmp108 = tmp107 + tmp94
    tmp109 = tmp24 >= tmp71
    tmp110 = tmp24 < tmp73
    tmp111 = tmp109 & tmp110
    tmp112 = tmp75 & tmp111
    tmp113 = tl.broadcast_to(1 + x4, [XBLOCK, YBLOCK])
    tmp114 = tmp113 >= tmp2
    tmp115 = tmp113 < tmp4
    tmp116 = tmp114 & tmp115
    tmp117 = tmp83 & tmp116
    tmp118 = tmp117 & tmp112
    tmp119 = tl.where(tmp118, tmp90, tmp91)
    tmp120 = tl.full(tmp119.shape, 0.0, tmp119.dtype)
    tmp121 = tl.where(tmp112, tmp119, tmp120)
    tmp122 = tmp121 + tmp108
    tmp123 = tmp33 >= tmp71
    tmp124 = tmp33 < tmp73
    tmp125 = tmp123 & tmp124
    tmp126 = tmp125 & tmp78
    tmp127 = tl.broadcast_to(x5, [XBLOCK, YBLOCK])
    tmp128 = tmp127 >= tmp2
    tmp129 = tmp127 < tmp4
    tmp130 = tmp128 & tmp129
    tmp131 = tmp130 & tmp87
    tmp132 = tmp131 & tmp126
    tmp133 = tl.where(tmp132, tmp90, tmp91)
    tmp134 = tl.full(tmp133.shape, 0.0, tmp133.dtype)
    tmp135 = tl.where(tmp126, tmp133, tmp134)
    tmp136 = tmp135 + tmp122
    tmp137 = tmp125 & tmp97
    tmp138 = tmp130 & tmp102
    tmp139 = tmp138 & tmp137
    tmp140 = tl.where(tmp139, tmp90, tmp91)
    tmp141 = tl.full(tmp140.shape, 0.0, tmp140.dtype)
    tmp142 = tl.where(tmp137, tmp140, tmp141)
    tmp143 = tmp142 + tmp136
    tmp144 = tmp125 & tmp111
    tmp145 = tmp130 & tmp116
    tmp146 = tmp145 & tmp144
    tmp147 = tl.where(tmp146, tmp90, tmp91)
    tmp148 = tl.full(tmp147.shape, 0.0, tmp147.dtype)
    tmp149 = tl.where(tmp144, tmp147, tmp148)
    tmp150 = tmp149 + tmp143
    tmp151 = tmp52 >= tmp71
    tmp152 = tmp52 < tmp73
    tmp153 = tmp151 & tmp152
    tmp154 = tmp153 & tmp78
    tmp155 = tl.broadcast_to(1 + x5, [XBLOCK, YBLOCK])
    tmp156 = tmp155 >= tmp2
    tmp157 = tmp155 < tmp4
    tmp158 = tmp156 & tmp157
    tmp159 = tmp158 & tmp87
    tmp160 = tmp159 & tmp154
    tmp161 = tl.where(tmp160, tmp90, tmp91)
    tmp162 = tl.full(tmp161.shape, 0.0, tmp161.dtype)
    tmp163 = tl.where(tmp154, tmp161, tmp162)
    tmp164 = tmp163 + tmp150
    tmp165 = tmp153 & tmp97
    tmp166 = tmp158 & tmp102
    tmp167 = tmp166 & tmp165
    tmp168 = tl.where(tmp167, tmp90, tmp91)
    tmp169 = tl.full(tmp168.shape, 0.0, tmp168.dtype)
    tmp170 = tl.where(tmp165, tmp168, tmp169)
    tmp171 = tmp170 + tmp164
    tmp172 = tmp153 & tmp111
    tmp173 = tmp158 & tmp116
    tmp174 = tmp173 & tmp172
    tmp175 = tl.where(tmp174, tmp90, tmp91)
    tmp176 = tl.full(tmp175.shape, 0.0, tmp175.dtype)
    tmp177 = tl.where(tmp172, tmp175, tmp176)
    tmp178 = tmp177 + tmp171
    tmp179 = tmp70 / tmp178
    tl.store(out_ptr0 + (y0 + (1280*x2) + (81920*y1)), tmp0, xmask)
    tl.store(out_ptr1 + (y0 + (1280*x2) + (81920*y1)), tmp0, xmask)
    tl.store(out_ptr2 + (y0 + (1280*x2) + (81920*y1)), tmp0, xmask)
    tl.store(out_ptr3 + (y0 + (1280*x2) + (81920*y1)), tmp179, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/dp/cdpu7tn5zoozcjlgfglvp6id5escjgkavdlzgq3hwmrc5gjx4l5e.py
# Source Nodes: [branch3x3_3, x_403], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# branch3x3_3 => relu_77
# x_403 => add_155, mul_232, mul_233, sub_77
triton_poi_fused__native_batch_norm_legit_no_training_relu_44 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_44', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (y0), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (y0), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (y0), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (384*x2) + (24576*y1)), tmp15, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yc/cycc2q7zui5mwayr2ozmt6uycb645go44k4fplrm7ikijohbonpp.py
# Source Nodes: [x_407], Original ATen: [aten.convolution]
# x_407 => convolution_78
triton_poi_fused_convolution_45 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144, 4], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_45', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 147456
    xnumel = 3
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 384
    y1 = (yindex // 384)
    tmp0 = tl.load(in_ptr0 + (x2 + (3*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (384*x2) + (1152*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/2v/c2vrh4xj7ufn4xdk6mc4w7zr7rjywimateg3ojbtfodxuab2gnwm.py
# Source Nodes: [cat_20], Original ATen: [aten.cat]
# cat_20 => cat_9
triton_poi_fused_cat_46 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_46', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 393216
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 64) % 768
    x2 = (xindex // 49152)
    x3 = xindex % 49152
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 384, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (24576*x2)), tmp4, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 - tmp6
    tmp8 = tl.load(in_ptr2 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp9 = 0.001
    tmp10 = tmp8 + tmp9
    tmp11 = tl.sqrt(tmp10)
    tmp12 = 1 / tmp11
    tmp13 = 1.0
    tmp14 = tmp12 * tmp13
    tmp15 = tmp7 * tmp14
    tmp16 = tl.load(in_ptr3 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp17 = tmp15 * tmp16
    tmp18 = tl.load(in_ptr4 + (x1), tmp4, eviction_policy='evict_last', other=0.0)
    tmp19 = tmp17 + tmp18
    tmp20 = triton_helpers.maximum(0, tmp19)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp4, tmp20, tmp21)
    tmp23 = tmp0 >= tmp3
    tmp24 = tl.full([1], 768, tl.int64)
    tmp25 = tmp0 < tmp24
    tmp26 = tl.load(in_ptr5 + ((-24576) + x3 + (24576*x2)), tmp23, other=0.0)
    tmp27 = tl.load(in_ptr6 + ((-384) + x1), tmp23, eviction_policy='evict_last', other=0.0)
    tmp28 = tmp26 - tmp27
    tmp29 = tl.load(in_ptr7 + ((-384) + x1), tmp23, eviction_policy='evict_last', other=0.0)
    tmp30 = tmp29 + tmp9
    tmp31 = tl.sqrt(tmp30)
    tmp32 = 1 / tmp31
    tmp33 = tmp32 * tmp13
    tmp34 = tmp28 * tmp33
    tmp35 = tl.load(in_ptr8 + ((-384) + x1), tmp23, eviction_policy='evict_last', other=0.0)
    tmp36 = tmp34 * tmp35
    tmp37 = tl.load(in_ptr9 + ((-384) + x1), tmp23, eviction_policy='evict_last', other=0.0)
    tmp38 = tmp36 + tmp37
    tmp39 = triton_helpers.maximum(0, tmp38)
    tmp40 = tl.full(tmp39.shape, 0.0, tmp39.dtype)
    tmp41 = tl.where(tmp23, tmp39, tmp40)
    tmp42 = tl.where(tmp4, tmp22, tmp41)
    tl.store(out_ptr0 + (x3 + (131072*x2)), tmp42, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/5d/c5dgnzzp3m4tf7j7ytk4gjm6yjg6vdwk3aucb22peekeil2csq43.py
# Source Nodes: [branch3x3dbl_12, x_418], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# branch3x3dbl_12 => relu_80
# x_418 => add_161, mul_241, mul_242, sub_80
triton_poi_fused__native_batch_norm_legit_no_training_relu_47 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_47', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3584
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 448
    y1 = (yindex // 448)
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask & ymask, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (y0 + (448*x2) + (28672*y1)), tmp15, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ro/croly27oveaalx6vwagt3dfvpe65nxrk2sp7kvagpgdo7tz5h5tb.py
# Source Nodes: [branch3x3dbl_12, x_418, x_422], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
# branch3x3dbl_12 => relu_80
# x_418 => add_161, mul_241, mul_242, sub_80
# x_422 => convolution_81
triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_48 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[262144, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_48', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 172032
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 448
    y1 = (yindex // 448)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (448*x2) + (4032*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vf/cvftgh2a3li4ler3xwerhowqdal76pnrjznlb26m7h5xzrliytjz.py
# Source Nodes: [branch1x1_7, x_398], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# branch1x1_7 => relu_76
# x_398 => add_153, mul_229, mul_230, sub_76
triton_poi_fused__native_batch_norm_legit_no_training_relu_49 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_49', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 163840
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 320
    x2 = (xindex // 20480)
    x4 = xindex % 20480
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4 + (131072*x2)), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/dx/cdxrrxgjtc5auhv2bgvcmqxiv5lawpi2dos7ek3l4mzrq66g3zh4.py
# Source Nodes: [branch_pool_17, x_438], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
# branch_pool_17 => relu_84
# x_438 => add_169, mul_253, mul_254, sub_84
triton_poi_fused__native_batch_norm_legit_no_training_relu_50 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__native_batch_norm_legit_no_training_relu_50', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 64) % 192
    x2 = (xindex // 12288)
    x4 = xindex % 12288
    tmp0 = tl.load(in_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last')
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
    tl.store(out_ptr0 + (x4 + (131072*x2)), tmp15, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/h4/ch4tlk74oqrwiw2pht6sxnzwolo2kbvrt5ugimdc3ufaasyi4fdj.py
# Source Nodes: [branch_pool_18, x_443, x_448, x_463], Original ATen: [aten.avg_pool2d, aten.convolution]
# branch_pool_18 => avg_pool2d_8
# x_443 => convolution_85
# x_448 => convolution_86
# x_463 => convolution_89
triton_poi_fused_avg_pool2d_convolution_51 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_avg_pool2d_convolution_51', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, out_ptr2, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
    xnumel = 64
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 2048
    y1 = (yindex // 2048)
    x5 = (xindex // 8)
    x4 = xindex % 8
    tmp0 = tl.load(in_ptr0 + (x2 + (64*y3)), xmask, eviction_policy='evict_last')
    tmp1 = (-1) + x5
    tmp2 = tl.full([1, 1], 0, tl.int64)
    tmp3 = tmp1 >= tmp2
    tmp4 = tl.full([1, 1], 8, tl.int64)
    tmp5 = tmp1 < tmp4
    tmp6 = tmp3 & tmp5
    tmp7 = (-1) + x4
    tmp8 = tmp7 >= tmp2
    tmp9 = tmp7 < tmp4
    tmp10 = tmp8 & tmp9
    tmp11 = tmp6 & tmp10
    tmp12 = tl.load(in_ptr0 + ((-9) + x2 + (64*y3)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp13 = tl.full(tmp12.shape, 0.0, tmp12.dtype)
    tmp14 = tl.where(tmp11, tmp12, tmp13)
    tmp15 = x4
    tmp16 = tmp15 >= tmp2
    tmp17 = tmp15 < tmp4
    tmp18 = tmp16 & tmp17
    tmp19 = tmp6 & tmp18
    tmp20 = tl.load(in_ptr0 + ((-8) + x2 + (64*y3)), tmp19 & xmask, eviction_policy='evict_last', other=0.0)
    tmp21 = tl.full(tmp20.shape, 0.0, tmp20.dtype)
    tmp22 = tl.where(tmp19, tmp20, tmp21)
    tmp23 = tmp22 + tmp14
    tmp24 = 1 + x4
    tmp25 = tmp24 >= tmp2
    tmp26 = tmp24 < tmp4
    tmp27 = tmp25 & tmp26
    tmp28 = tmp6 & tmp27
    tmp29 = tl.load(in_ptr0 + ((-7) + x2 + (64*y3)), tmp28 & xmask, eviction_policy='evict_last', other=0.0)
    tmp30 = tl.full(tmp29.shape, 0.0, tmp29.dtype)
    tmp31 = tl.where(tmp28, tmp29, tmp30)
    tmp32 = tmp31 + tmp23
    tmp33 = x5
    tmp34 = tmp33 >= tmp2
    tmp35 = tmp33 < tmp4
    tmp36 = tmp34 & tmp35
    tmp37 = tmp36 & tmp10
    tmp38 = tl.load(in_ptr0 + ((-1) + x2 + (64*y3)), tmp37 & xmask, eviction_policy='evict_last', other=0.0)
    tmp39 = tl.full(tmp38.shape, 0.0, tmp38.dtype)
    tmp40 = tl.where(tmp37, tmp38, tmp39)
    tmp41 = tmp40 + tmp32
    tmp42 = tmp36 & tmp18
    tmp43 = tl.load(in_ptr0 + (x2 + (64*y3)), tmp42 & xmask, eviction_policy='evict_last', other=0.0)
    tmp44 = tl.full(tmp43.shape, 0.0, tmp43.dtype)
    tmp45 = tl.where(tmp42, tmp43, tmp44)
    tmp46 = tmp45 + tmp41
    tmp47 = tmp36 & tmp27
    tmp48 = tl.load(in_ptr0 + (1 + x2 + (64*y3)), tmp47 & xmask, eviction_policy='evict_last', other=0.0)
    tmp49 = tl.full(tmp48.shape, 0.0, tmp48.dtype)
    tmp50 = tl.where(tmp47, tmp48, tmp49)
    tmp51 = tmp50 + tmp46
    tmp52 = 1 + x5
    tmp53 = tmp52 >= tmp2
    tmp54 = tmp52 < tmp4
    tmp55 = tmp53 & tmp54
    tmp56 = tmp55 & tmp10
    tmp57 = tl.load(in_ptr0 + (7 + x2 + (64*y3)), tmp56 & xmask, eviction_policy='evict_last', other=0.0)
    tmp58 = tl.full(tmp57.shape, 0.0, tmp57.dtype)
    tmp59 = tl.where(tmp56, tmp57, tmp58)
    tmp60 = tmp59 + tmp51
    tmp61 = tmp55 & tmp18
    tmp62 = tl.load(in_ptr0 + (8 + x2 + (64*y3)), tmp61 & xmask, eviction_policy='evict_last', other=0.0)
    tmp63 = tl.full(tmp62.shape, 0.0, tmp62.dtype)
    tmp64 = tl.where(tmp61, tmp62, tmp63)
    tmp65 = tmp64 + tmp60
    tmp66 = tmp55 & tmp27
    tmp67 = tl.load(in_ptr0 + (9 + x2 + (64*y3)), tmp66 & xmask, eviction_policy='evict_last', other=0.0)
    tmp68 = tl.full(tmp67.shape, 0.0, tmp67.dtype)
    tmp69 = tl.where(tmp66, tmp67, tmp68)
    tmp70 = tmp69 + tmp65
    tmp71 = tl.full([1, 1], -1, tl.int64)
    tmp72 = tmp1 >= tmp71
    tmp73 = tl.full([1, 1], 9, tl.int64)
    tmp74 = tmp1 < tmp73
    tmp75 = tmp72 & tmp74
    tmp76 = tmp7 >= tmp71
    tmp77 = tmp7 < tmp73
    tmp78 = tmp76 & tmp77
    tmp79 = tmp75 & tmp78
    tmp80 = tl.broadcast_to((-1) + x5, [XBLOCK, YBLOCK])
    tmp81 = tmp80 >= tmp2
    tmp82 = tmp80 < tmp4
    tmp83 = tmp81 & tmp82
    tmp84 = tl.broadcast_to((-1) + x4, [XBLOCK, YBLOCK])
    tmp85 = tmp84 >= tmp2
    tmp86 = tmp84 < tmp4
    tmp87 = tmp85 & tmp86
    tmp88 = tmp83 & tmp87
    tmp89 = tmp88 & tmp79
    tmp90 = 1.0
    tmp91 = tl.full(tmp90.shape, 1.0, tmp90.dtype)
    tmp92 = tl.where(tmp89, tmp90, tmp91)
    tmp93 = tl.full(tmp92.shape, 0.0, tmp92.dtype)
    tmp94 = tl.where(tmp79, tmp92, tmp93)
    tmp95 = tmp15 >= tmp71
    tmp96 = tmp15 < tmp73
    tmp97 = tmp95 & tmp96
    tmp98 = tmp75 & tmp97
    tmp99 = tl.broadcast_to(x4, [XBLOCK, YBLOCK])
    tmp100 = tmp99 >= tmp2
    tmp101 = tmp99 < tmp4
    tmp102 = tmp100 & tmp101
    tmp103 = tmp83 & tmp102
    tmp104 = tmp103 & tmp98
    tmp105 = tl.where(tmp104, tmp90, tmp91)
    tmp106 = tl.full(tmp105.shape, 0.0, tmp105.dtype)
    tmp107 = tl.where(tmp98, tmp105, tmp106)
    tmp108 = tmp107 + tmp94
    tmp109 = tmp24 >= tmp71
    tmp110 = tmp24 < tmp73
    tmp111 = tmp109 & tmp110
    tmp112 = tmp75 & tmp111
    tmp113 = tl.broadcast_to(1 + x4, [XBLOCK, YBLOCK])
    tmp114 = tmp113 >= tmp2
    tmp115 = tmp113 < tmp4
    tmp116 = tmp114 & tmp115
    tmp117 = tmp83 & tmp116
    tmp118 = tmp117 & tmp112
    tmp119 = tl.where(tmp118, tmp90, tmp91)
    tmp120 = tl.full(tmp119.shape, 0.0, tmp119.dtype)
    tmp121 = tl.where(tmp112, tmp119, tmp120)
    tmp122 = tmp121 + tmp108
    tmp123 = tmp33 >= tmp71
    tmp124 = tmp33 < tmp73
    tmp125 = tmp123 & tmp124
    tmp126 = tmp125 & tmp78
    tmp127 = tl.broadcast_to(x5, [XBLOCK, YBLOCK])
    tmp128 = tmp127 >= tmp2
    tmp129 = tmp127 < tmp4
    tmp130 = tmp128 & tmp129
    tmp131 = tmp130 & tmp87
    tmp132 = tmp131 & tmp126
    tmp133 = tl.where(tmp132, tmp90, tmp91)
    tmp134 = tl.full(tmp133.shape, 0.0, tmp133.dtype)
    tmp135 = tl.where(tmp126, tmp133, tmp134)
    tmp136 = tmp135 + tmp122
    tmp137 = tmp125 & tmp97
    tmp138 = tmp130 & tmp102
    tmp139 = tmp138 & tmp137
    tmp140 = tl.where(tmp139, tmp90, tmp91)
    tmp141 = tl.full(tmp140.shape, 0.0, tmp140.dtype)
    tmp142 = tl.where(tmp137, tmp140, tmp141)
    tmp143 = tmp142 + tmp136
    tmp144 = tmp125 & tmp111
    tmp145 = tmp130 & tmp116
    tmp146 = tmp145 & tmp144
    tmp147 = tl.where(tmp146, tmp90, tmp91)
    tmp148 = tl.full(tmp147.shape, 0.0, tmp147.dtype)
    tmp149 = tl.where(tmp144, tmp147, tmp148)
    tmp150 = tmp149 + tmp143
    tmp151 = tmp52 >= tmp71
    tmp152 = tmp52 < tmp73
    tmp153 = tmp151 & tmp152
    tmp154 = tmp153 & tmp78
    tmp155 = tl.broadcast_to(1 + x5, [XBLOCK, YBLOCK])
    tmp156 = tmp155 >= tmp2
    tmp157 = tmp155 < tmp4
    tmp158 = tmp156 & tmp157
    tmp159 = tmp158 & tmp87
    tmp160 = tmp159 & tmp154
    tmp161 = tl.where(tmp160, tmp90, tmp91)
    tmp162 = tl.full(tmp161.shape, 0.0, tmp161.dtype)
    tmp163 = tl.where(tmp154, tmp161, tmp162)
    tmp164 = tmp163 + tmp150
    tmp165 = tmp153 & tmp97
    tmp166 = tmp158 & tmp102
    tmp167 = tmp166 & tmp165
    tmp168 = tl.where(tmp167, tmp90, tmp91)
    tmp169 = tl.full(tmp168.shape, 0.0, tmp168.dtype)
    tmp170 = tl.where(tmp165, tmp168, tmp169)
    tmp171 = tmp170 + tmp164
    tmp172 = tmp153 & tmp111
    tmp173 = tmp158 & tmp116
    tmp174 = tmp173 & tmp172
    tmp175 = tl.where(tmp174, tmp90, tmp91)
    tmp176 = tl.full(tmp175.shape, 0.0, tmp175.dtype)
    tmp177 = tl.where(tmp172, tmp175, tmp176)
    tmp178 = tmp177 + tmp171
    tmp179 = tmp70 / tmp178
    tl.store(out_ptr0 + (y0 + (2048*x2) + (131072*y1)), tmp0, xmask)
    tl.store(out_ptr1 + (y0 + (2048*x2) + (131072*y1)), tmp0, xmask)
    tl.store(out_ptr2 + (y0 + (2048*x2) + (131072*y1)), tmp0, xmask)
    tl.store(out_ptr3 + (y0 + (2048*x2) + (131072*y1)), tmp179, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xg/cxg3uaa5qrna7kbl2nmj6jzn43dfpwdtm2rsjvdkcxcff7c2k2cm.py
# Source Nodes: [x_491], Original ATen: [aten.mean]
# x_491 => mean
triton_per_fused_mean_52 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_mean_52', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    rnumel = 64
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (64*x0)), rmask, other=0.0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tmp5 = 64.0
    tmp6 = tmp4 / tmp5
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp6, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1 = args
    args.clear()
    assert_size_stride(arg0_1, (32, ), (1, ))
    assert_size_stride(arg1_1, (32, ), (1, ))
    assert_size_stride(arg2_1, (32, ), (1, ))
    assert_size_stride(arg3_1, (32, ), (1, ))
    assert_size_stride(arg4_1, (64, ), (1, ))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (80, ), (1, ))
    assert_size_stride(arg7_1, (80, ), (1, ))
    assert_size_stride(arg8_1, (192, ), (1, ))
    assert_size_stride(arg9_1, (192, ), (1, ))
    assert_size_stride(arg10_1, (64, ), (1, ))
    assert_size_stride(arg11_1, (64, ), (1, ))
    assert_size_stride(arg12_1, (48, ), (1, ))
    assert_size_stride(arg13_1, (48, ), (1, ))
    assert_size_stride(arg14_1, (64, ), (1, ))
    assert_size_stride(arg15_1, (64, ), (1, ))
    assert_size_stride(arg16_1, (64, ), (1, ))
    assert_size_stride(arg17_1, (64, ), (1, ))
    assert_size_stride(arg18_1, (96, ), (1, ))
    assert_size_stride(arg19_1, (96, ), (1, ))
    assert_size_stride(arg20_1, (96, ), (1, ))
    assert_size_stride(arg21_1, (96, ), (1, ))
    assert_size_stride(arg22_1, (32, ), (1, ))
    assert_size_stride(arg23_1, (32, ), (1, ))
    assert_size_stride(arg24_1, (64, ), (1, ))
    assert_size_stride(arg25_1, (64, ), (1, ))
    assert_size_stride(arg26_1, (48, ), (1, ))
    assert_size_stride(arg27_1, (48, ), (1, ))
    assert_size_stride(arg28_1, (64, ), (1, ))
    assert_size_stride(arg29_1, (64, ), (1, ))
    assert_size_stride(arg30_1, (64, ), (1, ))
    assert_size_stride(arg31_1, (64, ), (1, ))
    assert_size_stride(arg32_1, (96, ), (1, ))
    assert_size_stride(arg33_1, (96, ), (1, ))
    assert_size_stride(arg34_1, (96, ), (1, ))
    assert_size_stride(arg35_1, (96, ), (1, ))
    assert_size_stride(arg36_1, (64, ), (1, ))
    assert_size_stride(arg37_1, (64, ), (1, ))
    assert_size_stride(arg38_1, (64, ), (1, ))
    assert_size_stride(arg39_1, (64, ), (1, ))
    assert_size_stride(arg40_1, (48, ), (1, ))
    assert_size_stride(arg41_1, (48, ), (1, ))
    assert_size_stride(arg42_1, (64, ), (1, ))
    assert_size_stride(arg43_1, (64, ), (1, ))
    assert_size_stride(arg44_1, (64, ), (1, ))
    assert_size_stride(arg45_1, (64, ), (1, ))
    assert_size_stride(arg46_1, (96, ), (1, ))
    assert_size_stride(arg47_1, (96, ), (1, ))
    assert_size_stride(arg48_1, (96, ), (1, ))
    assert_size_stride(arg49_1, (96, ), (1, ))
    assert_size_stride(arg50_1, (64, ), (1, ))
    assert_size_stride(arg51_1, (64, ), (1, ))
    assert_size_stride(arg52_1, (384, ), (1, ))
    assert_size_stride(arg53_1, (384, ), (1, ))
    assert_size_stride(arg54_1, (64, ), (1, ))
    assert_size_stride(arg55_1, (64, ), (1, ))
    assert_size_stride(arg56_1, (96, ), (1, ))
    assert_size_stride(arg57_1, (96, ), (1, ))
    assert_size_stride(arg58_1, (96, ), (1, ))
    assert_size_stride(arg59_1, (96, ), (1, ))
    assert_size_stride(arg60_1, (192, ), (1, ))
    assert_size_stride(arg61_1, (192, ), (1, ))
    assert_size_stride(arg62_1, (128, ), (1, ))
    assert_size_stride(arg63_1, (128, ), (1, ))
    assert_size_stride(arg64_1, (128, ), (1, ))
    assert_size_stride(arg65_1, (128, ), (1, ))
    assert_size_stride(arg66_1, (192, ), (1, ))
    assert_size_stride(arg67_1, (192, ), (1, ))
    assert_size_stride(arg68_1, (128, ), (1, ))
    assert_size_stride(arg69_1, (128, ), (1, ))
    assert_size_stride(arg70_1, (128, ), (1, ))
    assert_size_stride(arg71_1, (128, ), (1, ))
    assert_size_stride(arg72_1, (128, ), (1, ))
    assert_size_stride(arg73_1, (128, ), (1, ))
    assert_size_stride(arg74_1, (128, ), (1, ))
    assert_size_stride(arg75_1, (128, ), (1, ))
    assert_size_stride(arg76_1, (192, ), (1, ))
    assert_size_stride(arg77_1, (192, ), (1, ))
    assert_size_stride(arg78_1, (192, ), (1, ))
    assert_size_stride(arg79_1, (192, ), (1, ))
    assert_size_stride(arg80_1, (192, ), (1, ))
    assert_size_stride(arg81_1, (192, ), (1, ))
    assert_size_stride(arg82_1, (160, ), (1, ))
    assert_size_stride(arg83_1, (160, ), (1, ))
    assert_size_stride(arg84_1, (160, ), (1, ))
    assert_size_stride(arg85_1, (160, ), (1, ))
    assert_size_stride(arg86_1, (192, ), (1, ))
    assert_size_stride(arg87_1, (192, ), (1, ))
    assert_size_stride(arg88_1, (160, ), (1, ))
    assert_size_stride(arg89_1, (160, ), (1, ))
    assert_size_stride(arg90_1, (160, ), (1, ))
    assert_size_stride(arg91_1, (160, ), (1, ))
    assert_size_stride(arg92_1, (160, ), (1, ))
    assert_size_stride(arg93_1, (160, ), (1, ))
    assert_size_stride(arg94_1, (160, ), (1, ))
    assert_size_stride(arg95_1, (160, ), (1, ))
    assert_size_stride(arg96_1, (192, ), (1, ))
    assert_size_stride(arg97_1, (192, ), (1, ))
    assert_size_stride(arg98_1, (192, ), (1, ))
    assert_size_stride(arg99_1, (192, ), (1, ))
    assert_size_stride(arg100_1, (192, ), (1, ))
    assert_size_stride(arg101_1, (192, ), (1, ))
    assert_size_stride(arg102_1, (160, ), (1, ))
    assert_size_stride(arg103_1, (160, ), (1, ))
    assert_size_stride(arg104_1, (160, ), (1, ))
    assert_size_stride(arg105_1, (160, ), (1, ))
    assert_size_stride(arg106_1, (192, ), (1, ))
    assert_size_stride(arg107_1, (192, ), (1, ))
    assert_size_stride(arg108_1, (160, ), (1, ))
    assert_size_stride(arg109_1, (160, ), (1, ))
    assert_size_stride(arg110_1, (160, ), (1, ))
    assert_size_stride(arg111_1, (160, ), (1, ))
    assert_size_stride(arg112_1, (160, ), (1, ))
    assert_size_stride(arg113_1, (160, ), (1, ))
    assert_size_stride(arg114_1, (160, ), (1, ))
    assert_size_stride(arg115_1, (160, ), (1, ))
    assert_size_stride(arg116_1, (192, ), (1, ))
    assert_size_stride(arg117_1, (192, ), (1, ))
    assert_size_stride(arg118_1, (192, ), (1, ))
    assert_size_stride(arg119_1, (192, ), (1, ))
    assert_size_stride(arg120_1, (192, ), (1, ))
    assert_size_stride(arg121_1, (192, ), (1, ))
    assert_size_stride(arg122_1, (192, ), (1, ))
    assert_size_stride(arg123_1, (192, ), (1, ))
    assert_size_stride(arg124_1, (192, ), (1, ))
    assert_size_stride(arg125_1, (192, ), (1, ))
    assert_size_stride(arg126_1, (192, ), (1, ))
    assert_size_stride(arg127_1, (192, ), (1, ))
    assert_size_stride(arg128_1, (192, ), (1, ))
    assert_size_stride(arg129_1, (192, ), (1, ))
    assert_size_stride(arg130_1, (192, ), (1, ))
    assert_size_stride(arg131_1, (192, ), (1, ))
    assert_size_stride(arg132_1, (192, ), (1, ))
    assert_size_stride(arg133_1, (192, ), (1, ))
    assert_size_stride(arg134_1, (192, ), (1, ))
    assert_size_stride(arg135_1, (192, ), (1, ))
    assert_size_stride(arg136_1, (192, ), (1, ))
    assert_size_stride(arg137_1, (192, ), (1, ))
    assert_size_stride(arg138_1, (192, ), (1, ))
    assert_size_stride(arg139_1, (192, ), (1, ))
    assert_size_stride(arg140_1, (192, ), (1, ))
    assert_size_stride(arg141_1, (192, ), (1, ))
    assert_size_stride(arg142_1, (320, ), (1, ))
    assert_size_stride(arg143_1, (320, ), (1, ))
    assert_size_stride(arg144_1, (192, ), (1, ))
    assert_size_stride(arg145_1, (192, ), (1, ))
    assert_size_stride(arg146_1, (192, ), (1, ))
    assert_size_stride(arg147_1, (192, ), (1, ))
    assert_size_stride(arg148_1, (192, ), (1, ))
    assert_size_stride(arg149_1, (192, ), (1, ))
    assert_size_stride(arg150_1, (192, ), (1, ))
    assert_size_stride(arg151_1, (192, ), (1, ))
    assert_size_stride(arg152_1, (320, ), (1, ))
    assert_size_stride(arg153_1, (320, ), (1, ))
    assert_size_stride(arg154_1, (384, ), (1, ))
    assert_size_stride(arg155_1, (384, ), (1, ))
    assert_size_stride(arg156_1, (384, ), (1, ))
    assert_size_stride(arg157_1, (384, ), (1, ))
    assert_size_stride(arg158_1, (384, ), (1, ))
    assert_size_stride(arg159_1, (384, ), (1, ))
    assert_size_stride(arg160_1, (448, ), (1, ))
    assert_size_stride(arg161_1, (448, ), (1, ))
    assert_size_stride(arg162_1, (384, ), (1, ))
    assert_size_stride(arg163_1, (384, ), (1, ))
    assert_size_stride(arg164_1, (384, ), (1, ))
    assert_size_stride(arg165_1, (384, ), (1, ))
    assert_size_stride(arg166_1, (384, ), (1, ))
    assert_size_stride(arg167_1, (384, ), (1, ))
    assert_size_stride(arg168_1, (192, ), (1, ))
    assert_size_stride(arg169_1, (192, ), (1, ))
    assert_size_stride(arg170_1, (320, ), (1, ))
    assert_size_stride(arg171_1, (320, ), (1, ))
    assert_size_stride(arg172_1, (384, ), (1, ))
    assert_size_stride(arg173_1, (384, ), (1, ))
    assert_size_stride(arg174_1, (384, ), (1, ))
    assert_size_stride(arg175_1, (384, ), (1, ))
    assert_size_stride(arg176_1, (384, ), (1, ))
    assert_size_stride(arg177_1, (384, ), (1, ))
    assert_size_stride(arg178_1, (448, ), (1, ))
    assert_size_stride(arg179_1, (448, ), (1, ))
    assert_size_stride(arg180_1, (384, ), (1, ))
    assert_size_stride(arg181_1, (384, ), (1, ))
    assert_size_stride(arg182_1, (384, ), (1, ))
    assert_size_stride(arg183_1, (384, ), (1, ))
    assert_size_stride(arg184_1, (384, ), (1, ))
    assert_size_stride(arg185_1, (384, ), (1, ))
    assert_size_stride(arg186_1, (192, ), (1, ))
    assert_size_stride(arg187_1, (192, ), (1, ))
    assert_size_stride(arg188_1, (32, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg189_1, (32, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg190_1, (64, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg191_1, (80, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg192_1, (192, 80, 3, 3), (720, 9, 3, 1))
    assert_size_stride(arg193_1, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg194_1, (48, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg195_1, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(arg196_1, (64, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg197_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg198_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg199_1, (32, 192, 1, 1), (192, 1, 1, 1))
    assert_size_stride(arg200_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg201_1, (48, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg202_1, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(arg203_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg204_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg205_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg206_1, (64, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg207_1, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg208_1, (48, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg209_1, (64, 48, 5, 5), (1200, 25, 5, 1))
    assert_size_stride(arg210_1, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg211_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg212_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg213_1, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg214_1, (384, 288, 3, 3), (2592, 9, 3, 1))
    assert_size_stride(arg215_1, (64, 288, 1, 1), (288, 1, 1, 1))
    assert_size_stride(arg216_1, (96, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg217_1, (96, 96, 3, 3), (864, 9, 3, 1))
    assert_size_stride(arg218_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg219_1, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg220_1, (128, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(arg221_1, (192, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(arg222_1, (128, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg223_1, (128, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(arg224_1, (128, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(arg225_1, (128, 128, 7, 1), (896, 7, 1, 1))
    assert_size_stride(arg226_1, (192, 128, 1, 7), (896, 7, 7, 1))
    assert_size_stride(arg227_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg228_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg229_1, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg230_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg231_1, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg232_1, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg233_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg234_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg235_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg236_1, (192, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg237_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg238_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg239_1, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg240_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg241_1, (192, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg242_1, (160, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg243_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg244_1, (160, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg245_1, (160, 160, 7, 1), (1120, 7, 1, 1))
    assert_size_stride(arg246_1, (192, 160, 1, 7), (1120, 7, 7, 1))
    assert_size_stride(arg247_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg248_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg249_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg250_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg251_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg252_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg253_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg254_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg255_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg256_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg257_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg258_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg259_1, (320, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg260_1, (192, 768, 1, 1), (768, 1, 1, 1))
    assert_size_stride(arg261_1, (192, 192, 1, 7), (1344, 7, 7, 1))
    assert_size_stride(arg262_1, (192, 192, 7, 1), (1344, 7, 1, 1))
    assert_size_stride(arg263_1, (192, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg264_1, (320, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(arg265_1, (384, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(arg266_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg267_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg268_1, (448, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(arg269_1, (384, 448, 3, 3), (4032, 9, 3, 1))
    assert_size_stride(arg270_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg271_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg272_1, (192, 1280, 1, 1), (1280, 1, 1, 1))
    assert_size_stride(arg273_1, (320, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg274_1, (384, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg275_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg276_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg277_1, (448, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg278_1, (384, 448, 3, 3), (4032, 9, 3, 1))
    assert_size_stride(arg279_1, (384, 384, 1, 3), (1152, 3, 3, 1))
    assert_size_stride(arg280_1, (384, 384, 3, 1), (1152, 3, 1, 1))
    assert_size_stride(arg281_1, (192, 2048, 1, 1), (2048, 1, 1, 1))
    assert_size_stride(arg282_1, (1000, 2048), (2048, 1))
    assert_size_stride(arg283_1, (1000, ), (1, ))
    assert_size_stride(arg284_1, (32, ), (1, ))
    assert_size_stride(arg285_1, (32, ), (1, ))
    assert_size_stride(arg286_1, (32, ), (1, ))
    assert_size_stride(arg287_1, (32, ), (1, ))
    assert_size_stride(arg288_1, (64, ), (1, ))
    assert_size_stride(arg289_1, (64, ), (1, ))
    assert_size_stride(arg290_1, (80, ), (1, ))
    assert_size_stride(arg291_1, (80, ), (1, ))
    assert_size_stride(arg292_1, (192, ), (1, ))
    assert_size_stride(arg293_1, (192, ), (1, ))
    assert_size_stride(arg294_1, (64, ), (1, ))
    assert_size_stride(arg295_1, (64, ), (1, ))
    assert_size_stride(arg296_1, (48, ), (1, ))
    assert_size_stride(arg297_1, (48, ), (1, ))
    assert_size_stride(arg298_1, (64, ), (1, ))
    assert_size_stride(arg299_1, (64, ), (1, ))
    assert_size_stride(arg300_1, (64, ), (1, ))
    assert_size_stride(arg301_1, (64, ), (1, ))
    assert_size_stride(arg302_1, (96, ), (1, ))
    assert_size_stride(arg303_1, (96, ), (1, ))
    assert_size_stride(arg304_1, (96, ), (1, ))
    assert_size_stride(arg305_1, (96, ), (1, ))
    assert_size_stride(arg306_1, (32, ), (1, ))
    assert_size_stride(arg307_1, (32, ), (1, ))
    assert_size_stride(arg308_1, (64, ), (1, ))
    assert_size_stride(arg309_1, (64, ), (1, ))
    assert_size_stride(arg310_1, (48, ), (1, ))
    assert_size_stride(arg311_1, (48, ), (1, ))
    assert_size_stride(arg312_1, (64, ), (1, ))
    assert_size_stride(arg313_1, (64, ), (1, ))
    assert_size_stride(arg314_1, (64, ), (1, ))
    assert_size_stride(arg315_1, (64, ), (1, ))
    assert_size_stride(arg316_1, (96, ), (1, ))
    assert_size_stride(arg317_1, (96, ), (1, ))
    assert_size_stride(arg318_1, (96, ), (1, ))
    assert_size_stride(arg319_1, (96, ), (1, ))
    assert_size_stride(arg320_1, (64, ), (1, ))
    assert_size_stride(arg321_1, (64, ), (1, ))
    assert_size_stride(arg322_1, (64, ), (1, ))
    assert_size_stride(arg323_1, (64, ), (1, ))
    assert_size_stride(arg324_1, (48, ), (1, ))
    assert_size_stride(arg325_1, (48, ), (1, ))
    assert_size_stride(arg326_1, (64, ), (1, ))
    assert_size_stride(arg327_1, (64, ), (1, ))
    assert_size_stride(arg328_1, (64, ), (1, ))
    assert_size_stride(arg329_1, (64, ), (1, ))
    assert_size_stride(arg330_1, (96, ), (1, ))
    assert_size_stride(arg331_1, (96, ), (1, ))
    assert_size_stride(arg332_1, (96, ), (1, ))
    assert_size_stride(arg333_1, (96, ), (1, ))
    assert_size_stride(arg334_1, (64, ), (1, ))
    assert_size_stride(arg335_1, (64, ), (1, ))
    assert_size_stride(arg336_1, (384, ), (1, ))
    assert_size_stride(arg337_1, (384, ), (1, ))
    assert_size_stride(arg338_1, (64, ), (1, ))
    assert_size_stride(arg339_1, (64, ), (1, ))
    assert_size_stride(arg340_1, (96, ), (1, ))
    assert_size_stride(arg341_1, (96, ), (1, ))
    assert_size_stride(arg342_1, (96, ), (1, ))
    assert_size_stride(arg343_1, (96, ), (1, ))
    assert_size_stride(arg344_1, (192, ), (1, ))
    assert_size_stride(arg345_1, (192, ), (1, ))
    assert_size_stride(arg346_1, (128, ), (1, ))
    assert_size_stride(arg347_1, (128, ), (1, ))
    assert_size_stride(arg348_1, (128, ), (1, ))
    assert_size_stride(arg349_1, (128, ), (1, ))
    assert_size_stride(arg350_1, (192, ), (1, ))
    assert_size_stride(arg351_1, (192, ), (1, ))
    assert_size_stride(arg352_1, (128, ), (1, ))
    assert_size_stride(arg353_1, (128, ), (1, ))
    assert_size_stride(arg354_1, (128, ), (1, ))
    assert_size_stride(arg355_1, (128, ), (1, ))
    assert_size_stride(arg356_1, (128, ), (1, ))
    assert_size_stride(arg357_1, (128, ), (1, ))
    assert_size_stride(arg358_1, (128, ), (1, ))
    assert_size_stride(arg359_1, (128, ), (1, ))
    assert_size_stride(arg360_1, (192, ), (1, ))
    assert_size_stride(arg361_1, (192, ), (1, ))
    assert_size_stride(arg362_1, (192, ), (1, ))
    assert_size_stride(arg363_1, (192, ), (1, ))
    assert_size_stride(arg364_1, (192, ), (1, ))
    assert_size_stride(arg365_1, (192, ), (1, ))
    assert_size_stride(arg366_1, (160, ), (1, ))
    assert_size_stride(arg367_1, (160, ), (1, ))
    assert_size_stride(arg368_1, (160, ), (1, ))
    assert_size_stride(arg369_1, (160, ), (1, ))
    assert_size_stride(arg370_1, (192, ), (1, ))
    assert_size_stride(arg371_1, (192, ), (1, ))
    assert_size_stride(arg372_1, (160, ), (1, ))
    assert_size_stride(arg373_1, (160, ), (1, ))
    assert_size_stride(arg374_1, (160, ), (1, ))
    assert_size_stride(arg375_1, (160, ), (1, ))
    assert_size_stride(arg376_1, (160, ), (1, ))
    assert_size_stride(arg377_1, (160, ), (1, ))
    assert_size_stride(arg378_1, (160, ), (1, ))
    assert_size_stride(arg379_1, (160, ), (1, ))
    assert_size_stride(arg380_1, (192, ), (1, ))
    assert_size_stride(arg381_1, (192, ), (1, ))
    assert_size_stride(arg382_1, (192, ), (1, ))
    assert_size_stride(arg383_1, (192, ), (1, ))
    assert_size_stride(arg384_1, (192, ), (1, ))
    assert_size_stride(arg385_1, (192, ), (1, ))
    assert_size_stride(arg386_1, (160, ), (1, ))
    assert_size_stride(arg387_1, (160, ), (1, ))
    assert_size_stride(arg388_1, (160, ), (1, ))
    assert_size_stride(arg389_1, (160, ), (1, ))
    assert_size_stride(arg390_1, (192, ), (1, ))
    assert_size_stride(arg391_1, (192, ), (1, ))
    assert_size_stride(arg392_1, (160, ), (1, ))
    assert_size_stride(arg393_1, (160, ), (1, ))
    assert_size_stride(arg394_1, (160, ), (1, ))
    assert_size_stride(arg395_1, (160, ), (1, ))
    assert_size_stride(arg396_1, (160, ), (1, ))
    assert_size_stride(arg397_1, (160, ), (1, ))
    assert_size_stride(arg398_1, (160, ), (1, ))
    assert_size_stride(arg399_1, (160, ), (1, ))
    assert_size_stride(arg400_1, (192, ), (1, ))
    assert_size_stride(arg401_1, (192, ), (1, ))
    assert_size_stride(arg402_1, (192, ), (1, ))
    assert_size_stride(arg403_1, (192, ), (1, ))
    assert_size_stride(arg404_1, (192, ), (1, ))
    assert_size_stride(arg405_1, (192, ), (1, ))
    assert_size_stride(arg406_1, (192, ), (1, ))
    assert_size_stride(arg407_1, (192, ), (1, ))
    assert_size_stride(arg408_1, (192, ), (1, ))
    assert_size_stride(arg409_1, (192, ), (1, ))
    assert_size_stride(arg410_1, (192, ), (1, ))
    assert_size_stride(arg411_1, (192, ), (1, ))
    assert_size_stride(arg412_1, (192, ), (1, ))
    assert_size_stride(arg413_1, (192, ), (1, ))
    assert_size_stride(arg414_1, (192, ), (1, ))
    assert_size_stride(arg415_1, (192, ), (1, ))
    assert_size_stride(arg416_1, (192, ), (1, ))
    assert_size_stride(arg417_1, (192, ), (1, ))
    assert_size_stride(arg418_1, (192, ), (1, ))
    assert_size_stride(arg419_1, (192, ), (1, ))
    assert_size_stride(arg420_1, (192, ), (1, ))
    assert_size_stride(arg421_1, (192, ), (1, ))
    assert_size_stride(arg422_1, (192, ), (1, ))
    assert_size_stride(arg423_1, (192, ), (1, ))
    assert_size_stride(arg424_1, (192, ), (1, ))
    assert_size_stride(arg425_1, (192, ), (1, ))
    assert_size_stride(arg426_1, (320, ), (1, ))
    assert_size_stride(arg427_1, (320, ), (1, ))
    assert_size_stride(arg428_1, (192, ), (1, ))
    assert_size_stride(arg429_1, (192, ), (1, ))
    assert_size_stride(arg430_1, (192, ), (1, ))
    assert_size_stride(arg431_1, (192, ), (1, ))
    assert_size_stride(arg432_1, (192, ), (1, ))
    assert_size_stride(arg433_1, (192, ), (1, ))
    assert_size_stride(arg434_1, (192, ), (1, ))
    assert_size_stride(arg435_1, (192, ), (1, ))
    assert_size_stride(arg436_1, (320, ), (1, ))
    assert_size_stride(arg437_1, (320, ), (1, ))
    assert_size_stride(arg438_1, (384, ), (1, ))
    assert_size_stride(arg439_1, (384, ), (1, ))
    assert_size_stride(arg440_1, (384, ), (1, ))
    assert_size_stride(arg441_1, (384, ), (1, ))
    assert_size_stride(arg442_1, (384, ), (1, ))
    assert_size_stride(arg443_1, (384, ), (1, ))
    assert_size_stride(arg444_1, (448, ), (1, ))
    assert_size_stride(arg445_1, (448, ), (1, ))
    assert_size_stride(arg446_1, (384, ), (1, ))
    assert_size_stride(arg447_1, (384, ), (1, ))
    assert_size_stride(arg448_1, (384, ), (1, ))
    assert_size_stride(arg449_1, (384, ), (1, ))
    assert_size_stride(arg450_1, (384, ), (1, ))
    assert_size_stride(arg451_1, (384, ), (1, ))
    assert_size_stride(arg452_1, (192, ), (1, ))
    assert_size_stride(arg453_1, (192, ), (1, ))
    assert_size_stride(arg454_1, (320, ), (1, ))
    assert_size_stride(arg455_1, (320, ), (1, ))
    assert_size_stride(arg456_1, (384, ), (1, ))
    assert_size_stride(arg457_1, (384, ), (1, ))
    assert_size_stride(arg458_1, (384, ), (1, ))
    assert_size_stride(arg459_1, (384, ), (1, ))
    assert_size_stride(arg460_1, (384, ), (1, ))
    assert_size_stride(arg461_1, (384, ), (1, ))
    assert_size_stride(arg462_1, (448, ), (1, ))
    assert_size_stride(arg463_1, (448, ), (1, ))
    assert_size_stride(arg464_1, (384, ), (1, ))
    assert_size_stride(arg465_1, (384, ), (1, ))
    assert_size_stride(arg466_1, (384, ), (1, ))
    assert_size_stride(arg467_1, (384, ), (1, ))
    assert_size_stride(arg468_1, (384, ), (1, ))
    assert_size_stride(arg469_1, (384, ), (1, ))
    assert_size_stride(arg470_1, (192, ), (1, ))
    assert_size_stride(arg471_1, (192, ), (1, ))
    assert_size_stride(arg472_1, (8, 3, 299, 299), (268203, 89401, 299, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((8, 3, 299, 299), (268203, 1, 897, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg472_1, buf0, 24, 89401, grid=grid(24, 89401), stream=stream0)
        del arg472_1
        buf1 = empty_strided((32, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [x], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg188_1, buf1, 96, 9, grid=grid(96, 9), stream=stream0)
        del arg188_1
        # Source Nodes: [x], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (8, 32, 149, 149), (710432, 22201, 149, 1))
        del buf0
        del buf1
        buf3 = empty_strided((8, 32, 149, 149), (710432, 1, 4768, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1, x_5], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_2.run(buf2, arg284_1, arg285_1, arg0_1, arg1_1, buf3, 256, 22201, grid=grid(256, 22201), stream=stream0)
        del arg0_1
        del arg1_1
        del arg284_1
        del arg285_1
        del buf2
        buf4 = empty_strided((32, 32, 3, 3), (288, 1, 96, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1, x_5, x_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_3.run(arg189_1, buf4, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del arg189_1
        # Source Nodes: [x_1, x_5, x_6], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf5 = extern_kernels.convolution(buf3, buf4, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (8, 32, 147, 147), (691488, 21609, 147, 1))
        del buf3
        del buf4
        buf6 = empty_strided((8, 32, 147, 147), (691488, 1, 4704, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_4.run(buf5, arg286_1, arg287_1, arg2_1, arg3_1, buf6, 256, 21609, grid=grid(256, 21609), stream=stream0)
        del arg286_1
        del arg287_1
        del arg2_1
        del arg3_1
        del buf5
        buf7 = empty_strided((64, 32, 3, 3), (288, 1, 96, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_11, x_12, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_5.run(arg190_1, buf7, 2048, 9, grid=grid(2048, 9), stream=stream0)
        del arg190_1
        # Source Nodes: [x_11, x_12, x_7], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf8 = extern_kernels.convolution(buf6, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf8, (8, 64, 147, 147), (1382976, 21609, 147, 1))
        del buf6
        del buf7
        buf9 = buf8; del buf8  # reuse
        # Source Nodes: [x_13, x_17], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_6.run(buf9, arg288_1, arg289_1, arg4_1, arg5_1, 11063808, grid=grid(11063808), stream=stream0)
        del arg288_1
        del arg289_1
        del arg4_1
        del arg5_1
        buf10 = empty_strided((8, 64, 73, 73), (341056, 1, 4672, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_13, x_17, x_18], Original ATen: [aten._native_batch_norm_legit_no_training, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_7.run(buf9, buf10, 512, 5329, grid=grid(512, 5329), stream=stream0)
        del buf9
        # Source Nodes: [x_19], Original ATen: [aten.convolution]
        buf11 = extern_kernels.convolution(buf10, arg191_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (8, 80, 73, 73), (426320, 5329, 73, 1))
        del arg191_1
        del buf10
        buf12 = empty_strided((8, 80, 73, 73), (426320, 1, 5840, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_20, x_24], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_8.run(buf11, arg290_1, arg291_1, arg6_1, arg7_1, buf12, 640, 5329, grid=grid(640, 5329), stream=stream0)
        del arg290_1
        del arg291_1
        del arg6_1
        del arg7_1
        del buf11
        buf13 = empty_strided((192, 80, 3, 3), (720, 1, 240, 80), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_20, x_24, x_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_9.run(arg192_1, buf13, 15360, 9, grid=grid(15360, 9), stream=stream0)
        del arg192_1
        # Source Nodes: [x_20, x_24, x_25], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf14 = extern_kernels.convolution(buf12, buf13, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (8, 192, 71, 71), (967872, 5041, 71, 1))
        del buf12
        del buf13
        buf15 = buf14; del buf14  # reuse
        # Source Nodes: [x_26, x_30], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_10.run(buf15, arg292_1, arg293_1, arg8_1, arg9_1, 7742976, grid=grid(7742976), stream=stream0)
        del arg292_1
        del arg293_1
        del arg8_1
        del arg9_1
        buf16 = empty_strided((8, 192, 35, 35), (235200, 1, 6720, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_26, x_30, x_31], Original ATen: [aten._native_batch_norm_legit_no_training, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_max_pool2d_with_indices_relu_11.run(buf15, buf16, 1536, 1225, grid=grid(1536, 1225), stream=stream0)
        del buf15
        # Source Nodes: [x_32], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf16, arg193_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (8, 64, 35, 35), (78400, 1225, 35, 1))
        del arg193_1
        # Source Nodes: [x_37], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf16, arg194_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (8, 48, 35, 35), (58800, 1225, 35, 1))
        del arg194_1
        buf19 = empty_strided((8, 48, 35, 35), (58800, 1, 1680, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch5x5, x_38], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf18, arg296_1, arg297_1, arg12_1, arg13_1, buf19, 384, 1225, grid=grid(384, 1225), stream=stream0)
        del arg12_1
        del arg13_1
        del arg296_1
        del arg297_1
        del buf18
        buf20 = empty_strided((64, 48, 5, 5), (1200, 1, 240, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch5x5, x_38, x_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(arg195_1, buf20, 3072, 25, grid=grid(3072, 25), stream=stream0)
        del arg195_1
        # Source Nodes: [branch5x5, x_38, x_42], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf21 = extern_kernels.convolution(buf19, buf20, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (8, 64, 35, 35), (78400, 1225, 35, 1))
        # Source Nodes: [x_47], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf16, arg196_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (8, 64, 35, 35), (78400, 1225, 35, 1))
        del arg196_1
        buf23 = empty_strided((8, 64, 35, 35), (78400, 1, 2240, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch3x3dbl, x_48], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf22, arg300_1, arg301_1, arg16_1, arg17_1, buf23, 512, 1225, grid=grid(512, 1225), stream=stream0)
        del arg16_1
        del arg17_1
        del arg300_1
        del arg301_1
        del buf22
        buf24 = empty_strided((96, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch3x3dbl, x_48, x_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(arg197_1, buf24, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del arg197_1
        # Source Nodes: [branch3x3dbl, x_48, x_52], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf25 = extern_kernels.convolution(buf23, buf24, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf25, (8, 96, 35, 35), (117600, 1225, 35, 1))
        del buf23
        buf26 = empty_strided((8, 96, 35, 35), (117600, 1, 3360, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch3x3dbl_1, x_53], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf25, arg302_1, arg303_1, arg18_1, arg19_1, buf26, 768, 1225, grid=grid(768, 1225), stream=stream0)
        del arg18_1
        del arg19_1
        del arg302_1
        del arg303_1
        del buf25
        buf27 = empty_strided((96, 96, 3, 3), (864, 1, 288, 96), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch3x3dbl_1, x_53, x_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17.run(arg198_1, buf27, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del arg198_1
        # Source Nodes: [branch3x3dbl_1, x_53, x_57], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf28 = extern_kernels.convolution(buf26, buf27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (8, 96, 35, 35), (117600, 1225, 35, 1))
        del buf26
        buf29 = empty_strided((8, 192, 35, 35), (235200, 1, 6720, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch_pool], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_18.run(buf16, buf29, 1881600, grid=grid(1881600), stream=stream0)
        del buf16
        # Source Nodes: [x_62], Original ATen: [aten.convolution]
        buf30 = extern_kernels.convolution(buf29, arg199_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf30, (8, 32, 35, 35), (39200, 1225, 35, 1))
        del arg199_1
        del buf29
        buf31 = empty_strided((8, 256, 35, 35), (313600, 1, 8960, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_29], Original ATen: [aten.cat]
        triton_poi_fused_cat_19.run(buf17, arg294_1, arg295_1, arg10_1, arg11_1, buf21, arg298_1, arg299_1, arg14_1, arg15_1, buf28, arg304_1, arg305_1, arg20_1, arg21_1, buf30, arg306_1, arg307_1, arg22_1, arg23_1, buf31, 2048, 1225, grid=grid(2048, 1225), stream=stream0)
        del arg10_1
        del arg11_1
        del arg14_1
        del arg15_1
        del arg20_1
        del arg21_1
        del arg22_1
        del arg23_1
        del arg294_1
        del arg295_1
        del arg298_1
        del arg299_1
        del arg304_1
        del arg305_1
        del arg306_1
        del arg307_1
        del buf17
        del buf30
        # Source Nodes: [x_68], Original ATen: [aten.convolution]
        buf32 = extern_kernels.convolution(buf31, arg200_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (8, 64, 35, 35), (78400, 1225, 35, 1))
        del arg200_1
        # Source Nodes: [x_73], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf31, arg201_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (8, 48, 35, 35), (58800, 1225, 35, 1))
        del arg201_1
        buf34 = buf19; del buf19  # reuse
        # Source Nodes: [branch5x5_2, x_74], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf33, arg310_1, arg311_1, arg26_1, arg27_1, buf34, 384, 1225, grid=grid(384, 1225), stream=stream0)
        del arg26_1
        del arg27_1
        del arg310_1
        del arg311_1
        del buf33
        buf35 = buf20; del buf20  # reuse
        # Source Nodes: [branch5x5_2, x_74, x_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(arg202_1, buf35, 3072, 25, grid=grid(3072, 25), stream=stream0)
        del arg202_1
        # Source Nodes: [branch5x5_2, x_74, x_78], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf36 = extern_kernels.convolution(buf34, buf35, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (8, 64, 35, 35), (78400, 1225, 35, 1))
        # Source Nodes: [x_83], Original ATen: [aten.convolution]
        buf37 = extern_kernels.convolution(buf31, arg203_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (8, 64, 35, 35), (78400, 1225, 35, 1))
        del arg203_1
        buf38 = reinterpret_tensor(buf21, (8, 64, 35, 35), (78400, 1, 2240, 64), 0); del buf21  # reuse
        # Source Nodes: [branch3x3dbl_3, x_84], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf37, arg314_1, arg315_1, arg30_1, arg31_1, buf38, 512, 1225, grid=grid(512, 1225), stream=stream0)
        del arg30_1
        del arg314_1
        del arg315_1
        del arg31_1
        del buf37
        buf39 = buf24; del buf24  # reuse
        # Source Nodes: [branch3x3dbl_3, x_84, x_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(arg204_1, buf39, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del arg204_1
        # Source Nodes: [branch3x3dbl_3, x_84, x_88], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf40 = extern_kernels.convolution(buf38, buf39, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (8, 96, 35, 35), (117600, 1225, 35, 1))
        del buf38
        buf41 = reinterpret_tensor(buf28, (8, 96, 35, 35), (117600, 1, 3360, 96), 0); del buf28  # reuse
        # Source Nodes: [branch3x3dbl_4, x_89], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf40, arg316_1, arg317_1, arg32_1, arg33_1, buf41, 768, 1225, grid=grid(768, 1225), stream=stream0)
        del arg316_1
        del arg317_1
        del arg32_1
        del arg33_1
        del buf40
        buf42 = buf27; del buf27  # reuse
        # Source Nodes: [branch3x3dbl_4, x_89, x_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17.run(arg205_1, buf42, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del arg205_1
        # Source Nodes: [branch3x3dbl_4, x_89, x_93], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf43 = extern_kernels.convolution(buf41, buf42, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (8, 96, 35, 35), (117600, 1225, 35, 1))
        del buf41
        buf44 = empty_strided((8, 256, 35, 35), (313600, 1, 8960, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch_pool_2], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_20.run(buf31, buf44, 2508800, grid=grid(2508800), stream=stream0)
        del buf31
        # Source Nodes: [x_98], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, arg206_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (8, 64, 35, 35), (78400, 1225, 35, 1))
        del arg206_1
        del buf44
        buf46 = empty_strided((8, 288, 35, 35), (352800, 1, 10080, 288), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_28], Original ATen: [aten.cat]
        triton_poi_fused_cat_21.run(buf32, arg308_1, arg309_1, arg24_1, arg25_1, buf36, arg312_1, arg313_1, arg28_1, arg29_1, buf43, arg318_1, arg319_1, arg34_1, arg35_1, buf45, arg320_1, arg321_1, arg36_1, arg37_1, buf46, 2304, 1225, grid=grid(2304, 1225), stream=stream0)
        del arg24_1
        del arg25_1
        del arg28_1
        del arg29_1
        del arg308_1
        del arg309_1
        del arg312_1
        del arg313_1
        del arg318_1
        del arg319_1
        del arg320_1
        del arg321_1
        del arg34_1
        del arg35_1
        del arg36_1
        del arg37_1
        del buf32
        del buf36
        # Source Nodes: [x_104], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf46, arg207_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (8, 64, 35, 35), (78400, 1225, 35, 1))
        del arg207_1
        # Source Nodes: [x_109], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf46, arg208_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (8, 48, 35, 35), (58800, 1225, 35, 1))
        del arg208_1
        buf49 = buf34; del buf34  # reuse
        # Source Nodes: [branch5x5_4, x_110], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_12.run(buf48, arg324_1, arg325_1, arg40_1, arg41_1, buf49, 384, 1225, grid=grid(384, 1225), stream=stream0)
        del arg324_1
        del arg325_1
        del arg40_1
        del arg41_1
        del buf48
        buf50 = buf35; del buf35  # reuse
        # Source Nodes: [branch5x5_4, x_110, x_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_13.run(arg209_1, buf50, 3072, 25, grid=grid(3072, 25), stream=stream0)
        del arg209_1
        # Source Nodes: [branch5x5_4, x_110, x_114], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf51 = extern_kernels.convolution(buf49, buf50, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (8, 64, 35, 35), (78400, 1225, 35, 1))
        del buf49
        del buf50
        # Source Nodes: [x_119], Original ATen: [aten.convolution]
        buf52 = extern_kernels.convolution(buf46, arg210_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf52, (8, 64, 35, 35), (78400, 1225, 35, 1))
        del arg210_1
        buf53 = reinterpret_tensor(buf45, (8, 64, 35, 35), (78400, 1, 2240, 64), 0); del buf45  # reuse
        # Source Nodes: [branch3x3dbl_6, x_120], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf52, arg328_1, arg329_1, arg44_1, arg45_1, buf53, 512, 1225, grid=grid(512, 1225), stream=stream0)
        del arg328_1
        del arg329_1
        del arg44_1
        del arg45_1
        del buf52
        buf54 = buf39; del buf39  # reuse
        # Source Nodes: [branch3x3dbl_6, x_120, x_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(arg211_1, buf54, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del arg211_1
        # Source Nodes: [branch3x3dbl_6, x_120, x_124], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf55 = extern_kernels.convolution(buf53, buf54, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (8, 96, 35, 35), (117600, 1225, 35, 1))
        del buf53
        buf56 = reinterpret_tensor(buf43, (8, 96, 35, 35), (117600, 1, 3360, 96), 0); del buf43  # reuse
        # Source Nodes: [branch3x3dbl_7, x_125], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf55, arg330_1, arg331_1, arg46_1, arg47_1, buf56, 768, 1225, grid=grid(768, 1225), stream=stream0)
        del arg330_1
        del arg331_1
        del arg46_1
        del arg47_1
        del buf55
        buf57 = buf42; del buf42  # reuse
        # Source Nodes: [branch3x3dbl_7, x_125, x_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17.run(arg212_1, buf57, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del arg212_1
        # Source Nodes: [branch3x3dbl_7, x_125, x_129], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf58 = extern_kernels.convolution(buf56, buf57, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (8, 96, 35, 35), (117600, 1225, 35, 1))
        del buf56
        buf59 = empty_strided((8, 288, 35, 35), (352800, 1, 10080, 288), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch_pool_4], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_22.run(buf46, buf59, 2822400, grid=grid(2822400), stream=stream0)
        del buf46
        # Source Nodes: [x_134], Original ATen: [aten.convolution]
        buf60 = extern_kernels.convolution(buf59, arg213_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf60, (8, 64, 35, 35), (78400, 1225, 35, 1))
        del arg213_1
        buf61 = buf59; del buf59  # reuse
        # Source Nodes: [cat_27], Original ATen: [aten.cat]
        triton_poi_fused_cat_21.run(buf47, arg322_1, arg323_1, arg38_1, arg39_1, buf51, arg326_1, arg327_1, arg42_1, arg43_1, buf58, arg332_1, arg333_1, arg48_1, arg49_1, buf60, arg334_1, arg335_1, arg50_1, arg51_1, buf61, 2304, 1225, grid=grid(2304, 1225), stream=stream0)
        del arg322_1
        del arg323_1
        del arg326_1
        del arg327_1
        del arg332_1
        del arg333_1
        del arg334_1
        del arg335_1
        del arg38_1
        del arg39_1
        del arg42_1
        del arg43_1
        del arg48_1
        del arg49_1
        del arg50_1
        del arg51_1
        del buf47
        del buf51
        buf74 = empty((8, 768, 17, 17), device='cuda', dtype=torch.float32)
        buf62 = reinterpret_tensor(buf74, (8, 288, 17, 17), (221952, 289, 17, 1), 138720)  # alias
        # Source Nodes: [branch_pool_6], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_23.run(buf61, buf62, 2304, 289, grid=grid(2304, 289), stream=stream0)
        buf63 = empty_strided((384, 288, 3, 3), (2592, 1, 864, 288), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_140], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_24.run(arg214_1, buf63, 110592, 9, grid=grid(110592, 9), stream=stream0)
        del arg214_1
        # Source Nodes: [x_140], Original ATen: [aten.convolution]
        buf64 = extern_kernels.convolution(buf61, buf63, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf64, (8, 384, 17, 17), (110976, 289, 17, 1))
        del buf63
        # Source Nodes: [x_145], Original ATen: [aten.convolution]
        buf65 = extern_kernels.convolution(buf61, arg215_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf65, (8, 64, 35, 35), (78400, 1225, 35, 1))
        del arg215_1
        del buf61
        buf66 = reinterpret_tensor(buf60, (8, 64, 35, 35), (78400, 1, 2240, 64), 0); del buf60  # reuse
        # Source Nodes: [branch3x3dbl_9, x_146], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_14.run(buf65, arg338_1, arg339_1, arg54_1, arg55_1, buf66, 512, 1225, grid=grid(512, 1225), stream=stream0)
        del arg338_1
        del arg339_1
        del arg54_1
        del arg55_1
        del buf65
        buf67 = buf54; del buf54  # reuse
        # Source Nodes: [branch3x3dbl_9, x_146, x_150], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_15.run(arg216_1, buf67, 6144, 9, grid=grid(6144, 9), stream=stream0)
        del arg216_1
        # Source Nodes: [branch3x3dbl_9, x_146, x_150], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf68 = extern_kernels.convolution(buf66, buf67, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf68, (8, 96, 35, 35), (117600, 1225, 35, 1))
        del buf66
        del buf67
        buf69 = reinterpret_tensor(buf58, (8, 96, 35, 35), (117600, 1, 3360, 96), 0); del buf58  # reuse
        # Source Nodes: [branch3x3dbl_10, x_151], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_16.run(buf68, arg340_1, arg341_1, arg56_1, arg57_1, buf69, 768, 1225, grid=grid(768, 1225), stream=stream0)
        del arg340_1
        del arg341_1
        del arg56_1
        del arg57_1
        del buf68
        buf70 = buf57; del buf57  # reuse
        # Source Nodes: [branch3x3dbl_10, x_151, x_155], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_17.run(arg217_1, buf70, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del arg217_1
        # Source Nodes: [branch3x3dbl_10, x_151, x_155], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf71 = extern_kernels.convolution(buf69, buf70, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf71, (8, 96, 17, 17), (27744, 289, 17, 1))
        del buf69
        del buf70
        buf72 = reinterpret_tensor(buf74, (8, 384, 17, 17), (221952, 289, 17, 1), 0)  # alias
        # Source Nodes: [branch3x3, x_141], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_25.run(buf64, arg336_1, arg337_1, arg52_1, arg53_1, buf72, 887808, grid=grid(887808), stream=stream0)
        del arg336_1
        del arg337_1
        del arg52_1
        del arg53_1
        del buf64
        buf73 = reinterpret_tensor(buf74, (8, 96, 17, 17), (221952, 289, 17, 1), 110976)  # alias
        # Source Nodes: [branch3x3dbl_11, x_156], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_26.run(buf71, arg342_1, arg343_1, arg58_1, arg59_1, buf73, 221952, grid=grid(221952), stream=stream0)
        del arg342_1
        del arg343_1
        del arg58_1
        del arg59_1
        del buf71
        buf75 = empty_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cuda', dtype=torch.float32)
        buf77 = empty_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cuda', dtype=torch.float32)
        buf85 = empty_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cuda', dtype=torch.float32)
        buf99 = empty_strided((8, 768, 17, 17), (221952, 1, 13056, 768), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch_pool_7, x_161, x_166, x_181], Original ATen: [aten.avg_pool2d, aten.convolution]
        triton_poi_fused_avg_pool2d_convolution_27.run(buf74, buf75, buf77, buf85, buf99, 6144, 289, grid=grid(6144, 289), stream=stream0)
        del buf62
        del buf72
        del buf73
        del buf74
        # Source Nodes: [x_161], Original ATen: [aten.convolution]
        buf76 = extern_kernels.convolution(buf75, arg218_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf76, (8, 192, 17, 17), (55488, 289, 17, 1))
        del arg218_1
        del buf75
        # Source Nodes: [x_166], Original ATen: [aten.convolution]
        buf78 = extern_kernels.convolution(buf77, arg219_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf78, (8, 128, 17, 17), (36992, 289, 17, 1))
        del arg219_1
        del buf77
        buf79 = empty_strided((8, 128, 17, 17), (36992, 1, 2176, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch7x7, x_167], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_28.run(buf78, arg346_1, arg347_1, arg62_1, arg63_1, buf79, 1024, 289, grid=grid(1024, 289), stream=stream0)
        del arg346_1
        del arg347_1
        del arg62_1
        del arg63_1
        del buf78
        buf80 = empty_strided((128, 128, 1, 7), (896, 1, 896, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch7x7, x_167, x_171], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_29.run(arg220_1, buf80, 16384, 7, grid=grid(16384, 7), stream=stream0)
        del arg220_1
        # Source Nodes: [branch7x7, x_167, x_171], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf81 = extern_kernels.convolution(buf79, buf80, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf81, (8, 128, 17, 17), (36992, 289, 17, 1))
        buf82 = buf79; del buf79  # reuse
        # Source Nodes: [branch7x7_1, x_172], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_28.run(buf81, arg348_1, arg349_1, arg64_1, arg65_1, buf82, 1024, 289, grid=grid(1024, 289), stream=stream0)
        del arg348_1
        del arg349_1
        del arg64_1
        del arg65_1
        del buf81
        buf83 = empty_strided((192, 128, 7, 1), (896, 1, 128, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch7x7_1, x_172, x_176], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30.run(arg221_1, buf83, 24576, 7, grid=grid(24576, 7), stream=stream0)
        del arg221_1
        # Source Nodes: [branch7x7_1, x_172, x_176], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf84 = extern_kernels.convolution(buf82, buf83, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf84, (8, 192, 17, 17), (55488, 289, 17, 1))
        # Source Nodes: [x_181], Original ATen: [aten.convolution]
        buf86 = extern_kernels.convolution(buf85, arg222_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf86, (8, 128, 17, 17), (36992, 289, 17, 1))
        del arg222_1
        buf87 = buf82; del buf82  # reuse
        # Source Nodes: [branch7x7dbl, x_182], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_28.run(buf86, arg352_1, arg353_1, arg68_1, arg69_1, buf87, 1024, 289, grid=grid(1024, 289), stream=stream0)
        del arg352_1
        del arg353_1
        del arg68_1
        del arg69_1
        del buf86
        buf88 = reinterpret_tensor(buf80, (128, 128, 7, 1), (896, 1, 128, 128), 0); del buf80  # reuse
        # Source Nodes: [branch7x7dbl, x_182, x_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_29.run(arg223_1, buf88, 16384, 7, grid=grid(16384, 7), stream=stream0)
        del arg223_1
        # Source Nodes: [branch7x7dbl, x_182, x_186], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf89 = extern_kernels.convolution(buf87, buf88, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf89, (8, 128, 17, 17), (36992, 289, 17, 1))
        buf90 = buf87; del buf87  # reuse
        # Source Nodes: [branch7x7dbl_1, x_187], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_28.run(buf89, arg354_1, arg355_1, arg70_1, arg71_1, buf90, 1024, 289, grid=grid(1024, 289), stream=stream0)
        del arg354_1
        del arg355_1
        del arg70_1
        del arg71_1
        del buf89
        buf91 = reinterpret_tensor(buf88, (128, 128, 1, 7), (896, 1, 896, 128), 0); del buf88  # reuse
        # Source Nodes: [branch7x7dbl_1, x_187, x_191], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_29.run(arg224_1, buf91, 16384, 7, grid=grid(16384, 7), stream=stream0)
        del arg224_1
        # Source Nodes: [branch7x7dbl_1, x_187, x_191], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf92 = extern_kernels.convolution(buf90, buf91, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf92, (8, 128, 17, 17), (36992, 289, 17, 1))
        buf93 = buf90; del buf90  # reuse
        # Source Nodes: [branch7x7dbl_2, x_192], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_28.run(buf92, arg356_1, arg357_1, arg72_1, arg73_1, buf93, 1024, 289, grid=grid(1024, 289), stream=stream0)
        del arg356_1
        del arg357_1
        del arg72_1
        del arg73_1
        del buf92
        buf94 = reinterpret_tensor(buf91, (128, 128, 7, 1), (896, 1, 128, 128), 0); del buf91  # reuse
        # Source Nodes: [branch7x7dbl_2, x_192, x_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_29.run(arg225_1, buf94, 16384, 7, grid=grid(16384, 7), stream=stream0)
        del arg225_1
        # Source Nodes: [branch7x7dbl_2, x_192, x_196], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf95 = extern_kernels.convolution(buf93, buf94, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf95, (8, 128, 17, 17), (36992, 289, 17, 1))
        del buf94
        buf96 = buf93; del buf93  # reuse
        # Source Nodes: [branch7x7dbl_3, x_197], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_28.run(buf95, arg358_1, arg359_1, arg74_1, arg75_1, buf96, 1024, 289, grid=grid(1024, 289), stream=stream0)
        del arg358_1
        del arg359_1
        del arg74_1
        del arg75_1
        del buf95
        buf97 = reinterpret_tensor(buf83, (192, 128, 1, 7), (896, 1, 896, 128), 0); del buf83  # reuse
        # Source Nodes: [branch7x7dbl_3, x_197, x_201], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_30.run(arg226_1, buf97, 24576, 7, grid=grid(24576, 7), stream=stream0)
        del arg226_1
        # Source Nodes: [branch7x7dbl_3, x_197, x_201], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf98 = extern_kernels.convolution(buf96, buf97, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf98, (8, 192, 17, 17), (55488, 289, 17, 1))
        del buf96
        del buf97
        # Source Nodes: [x_206], Original ATen: [aten.convolution]
        buf100 = extern_kernels.convolution(buf99, arg227_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf100, (8, 192, 17, 17), (55488, 289, 17, 1))
        del arg227_1
        buf101 = buf99; del buf99  # reuse
        # Source Nodes: [cat_25], Original ATen: [aten.cat]
        triton_poi_fused_cat_31.run(buf76, arg344_1, arg345_1, arg60_1, arg61_1, buf84, arg350_1, arg351_1, arg66_1, arg67_1, buf98, arg360_1, arg361_1, arg76_1, arg77_1, buf100, arg362_1, arg363_1, arg78_1, arg79_1, buf101, 6144, 289, grid=grid(6144, 289), stream=stream0)
        del arg344_1
        del arg345_1
        del arg350_1
        del arg351_1
        del arg360_1
        del arg361_1
        del arg362_1
        del arg363_1
        del arg60_1
        del arg61_1
        del arg66_1
        del arg67_1
        del arg76_1
        del arg77_1
        del arg78_1
        del arg79_1
        del buf100
        del buf76
        del buf84
        del buf98
        # Source Nodes: [x_212], Original ATen: [aten.convolution]
        buf102 = extern_kernels.convolution(buf101, arg228_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf102, (8, 192, 17, 17), (55488, 289, 17, 1))
        del arg228_1
        # Source Nodes: [x_217], Original ATen: [aten.convolution]
        buf103 = extern_kernels.convolution(buf101, arg229_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf103, (8, 160, 17, 17), (46240, 289, 17, 1))
        del arg229_1
        buf104 = empty_strided((8, 160, 17, 17), (46240, 1, 2720, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch7x7_3, x_218], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf103, arg366_1, arg367_1, arg82_1, arg83_1, buf104, 1280, 289, grid=grid(1280, 289), stream=stream0)
        del arg366_1
        del arg367_1
        del arg82_1
        del arg83_1
        del buf103
        buf105 = empty_strided((160, 160, 1, 7), (1120, 1, 1120, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch7x7_3, x_218, x_222], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(arg230_1, buf105, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del arg230_1
        # Source Nodes: [branch7x7_3, x_218, x_222], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf106 = extern_kernels.convolution(buf104, buf105, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf106, (8, 160, 17, 17), (46240, 289, 17, 1))
        buf107 = buf104; del buf104  # reuse
        # Source Nodes: [branch7x7_4, x_223], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf106, arg368_1, arg369_1, arg84_1, arg85_1, buf107, 1280, 289, grid=grid(1280, 289), stream=stream0)
        del arg368_1
        del arg369_1
        del arg84_1
        del arg85_1
        del buf106
        buf108 = empty_strided((192, 160, 7, 1), (1120, 1, 160, 160), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch7x7_4, x_223, x_227], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34.run(arg231_1, buf108, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del arg231_1
        # Source Nodes: [branch7x7_4, x_223, x_227], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf109 = extern_kernels.convolution(buf107, buf108, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf109, (8, 192, 17, 17), (55488, 289, 17, 1))
        # Source Nodes: [x_232], Original ATen: [aten.convolution]
        buf110 = extern_kernels.convolution(buf101, arg232_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf110, (8, 160, 17, 17), (46240, 289, 17, 1))
        del arg232_1
        buf111 = buf107; del buf107  # reuse
        # Source Nodes: [branch7x7dbl_5, x_233], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf110, arg372_1, arg373_1, arg88_1, arg89_1, buf111, 1280, 289, grid=grid(1280, 289), stream=stream0)
        del arg372_1
        del arg373_1
        del arg88_1
        del arg89_1
        del buf110
        buf112 = reinterpret_tensor(buf105, (160, 160, 7, 1), (1120, 1, 160, 160), 0); del buf105  # reuse
        # Source Nodes: [branch7x7dbl_5, x_233, x_237], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(arg233_1, buf112, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del arg233_1
        # Source Nodes: [branch7x7dbl_5, x_233, x_237], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf113 = extern_kernels.convolution(buf111, buf112, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf113, (8, 160, 17, 17), (46240, 289, 17, 1))
        buf114 = buf111; del buf111  # reuse
        # Source Nodes: [branch7x7dbl_6, x_238], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf113, arg374_1, arg375_1, arg90_1, arg91_1, buf114, 1280, 289, grid=grid(1280, 289), stream=stream0)
        del arg374_1
        del arg375_1
        del arg90_1
        del arg91_1
        del buf113
        buf115 = reinterpret_tensor(buf112, (160, 160, 1, 7), (1120, 1, 1120, 160), 0); del buf112  # reuse
        # Source Nodes: [branch7x7dbl_6, x_238, x_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(arg234_1, buf115, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del arg234_1
        # Source Nodes: [branch7x7dbl_6, x_238, x_242], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf116 = extern_kernels.convolution(buf114, buf115, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf116, (8, 160, 17, 17), (46240, 289, 17, 1))
        buf117 = buf114; del buf114  # reuse
        # Source Nodes: [branch7x7dbl_7, x_243], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf116, arg376_1, arg377_1, arg92_1, arg93_1, buf117, 1280, 289, grid=grid(1280, 289), stream=stream0)
        del arg376_1
        del arg377_1
        del arg92_1
        del arg93_1
        del buf116
        buf118 = reinterpret_tensor(buf115, (160, 160, 7, 1), (1120, 1, 160, 160), 0); del buf115  # reuse
        # Source Nodes: [branch7x7dbl_7, x_243, x_247], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(arg235_1, buf118, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del arg235_1
        # Source Nodes: [branch7x7dbl_7, x_243, x_247], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf119 = extern_kernels.convolution(buf117, buf118, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf119, (8, 160, 17, 17), (46240, 289, 17, 1))
        buf120 = buf117; del buf117  # reuse
        # Source Nodes: [branch7x7dbl_8, x_248], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf119, arg378_1, arg379_1, arg94_1, arg95_1, buf120, 1280, 289, grid=grid(1280, 289), stream=stream0)
        del arg378_1
        del arg379_1
        del arg94_1
        del arg95_1
        del buf119
        buf121 = reinterpret_tensor(buf108, (192, 160, 1, 7), (1120, 1, 1120, 160), 0); del buf108  # reuse
        # Source Nodes: [branch7x7dbl_8, x_248, x_252], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34.run(arg236_1, buf121, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del arg236_1
        # Source Nodes: [branch7x7dbl_8, x_248, x_252], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf122 = extern_kernels.convolution(buf120, buf121, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf122, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf123 = buf85; del buf85  # reuse
        # Source Nodes: [branch_pool_9], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_35.run(buf101, buf123, 1775616, grid=grid(1775616), stream=stream0)
        # Source Nodes: [x_257], Original ATen: [aten.convolution]
        buf124 = extern_kernels.convolution(buf123, arg237_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf124, (8, 192, 17, 17), (55488, 289, 17, 1))
        del arg237_1
        buf125 = buf123; del buf123  # reuse
        # Source Nodes: [cat_24], Original ATen: [aten.cat]
        triton_poi_fused_cat_31.run(buf102, arg364_1, arg365_1, arg80_1, arg81_1, buf109, arg370_1, arg371_1, arg86_1, arg87_1, buf122, arg380_1, arg381_1, arg96_1, arg97_1, buf124, arg382_1, arg383_1, arg98_1, arg99_1, buf125, 6144, 289, grid=grid(6144, 289), stream=stream0)
        del arg364_1
        del arg365_1
        del arg370_1
        del arg371_1
        del arg380_1
        del arg381_1
        del arg382_1
        del arg383_1
        del arg80_1
        del arg81_1
        del arg86_1
        del arg87_1
        del arg96_1
        del arg97_1
        del arg98_1
        del arg99_1
        del buf102
        del buf109
        del buf122
        del buf124
        # Source Nodes: [x_263], Original ATen: [aten.convolution]
        buf126 = extern_kernels.convolution(buf125, arg238_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf126, (8, 192, 17, 17), (55488, 289, 17, 1))
        del arg238_1
        # Source Nodes: [x_268], Original ATen: [aten.convolution]
        buf127 = extern_kernels.convolution(buf125, arg239_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf127, (8, 160, 17, 17), (46240, 289, 17, 1))
        del arg239_1
        buf128 = buf120; del buf120  # reuse
        # Source Nodes: [branch7x7_6, x_269], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf127, arg386_1, arg387_1, arg102_1, arg103_1, buf128, 1280, 289, grid=grid(1280, 289), stream=stream0)
        del arg102_1
        del arg103_1
        del arg386_1
        del arg387_1
        del buf127
        buf129 = reinterpret_tensor(buf118, (160, 160, 1, 7), (1120, 1, 1120, 160), 0); del buf118  # reuse
        # Source Nodes: [branch7x7_6, x_269, x_273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(arg240_1, buf129, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del arg240_1
        # Source Nodes: [branch7x7_6, x_269, x_273], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf130 = extern_kernels.convolution(buf128, buf129, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf130, (8, 160, 17, 17), (46240, 289, 17, 1))
        buf131 = buf128; del buf128  # reuse
        # Source Nodes: [branch7x7_7, x_274], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf130, arg388_1, arg389_1, arg104_1, arg105_1, buf131, 1280, 289, grid=grid(1280, 289), stream=stream0)
        del arg104_1
        del arg105_1
        del arg388_1
        del arg389_1
        del buf130
        buf132 = reinterpret_tensor(buf121, (192, 160, 7, 1), (1120, 1, 160, 160), 0); del buf121  # reuse
        # Source Nodes: [branch7x7_7, x_274, x_278], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34.run(arg241_1, buf132, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del arg241_1
        # Source Nodes: [branch7x7_7, x_274, x_278], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf133 = extern_kernels.convolution(buf131, buf132, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf133, (8, 192, 17, 17), (55488, 289, 17, 1))
        # Source Nodes: [x_283], Original ATen: [aten.convolution]
        buf134 = extern_kernels.convolution(buf125, arg242_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf134, (8, 160, 17, 17), (46240, 289, 17, 1))
        del arg242_1
        buf135 = buf131; del buf131  # reuse
        # Source Nodes: [branch7x7dbl_10, x_284], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf134, arg392_1, arg393_1, arg108_1, arg109_1, buf135, 1280, 289, grid=grid(1280, 289), stream=stream0)
        del arg108_1
        del arg109_1
        del arg392_1
        del arg393_1
        del buf134
        buf136 = reinterpret_tensor(buf129, (160, 160, 7, 1), (1120, 1, 160, 160), 0); del buf129  # reuse
        # Source Nodes: [branch7x7dbl_10, x_284, x_288], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(arg243_1, buf136, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del arg243_1
        # Source Nodes: [branch7x7dbl_10, x_284, x_288], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf137 = extern_kernels.convolution(buf135, buf136, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf137, (8, 160, 17, 17), (46240, 289, 17, 1))
        buf138 = buf135; del buf135  # reuse
        # Source Nodes: [branch7x7dbl_11, x_289], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf137, arg394_1, arg395_1, arg110_1, arg111_1, buf138, 1280, 289, grid=grid(1280, 289), stream=stream0)
        del arg110_1
        del arg111_1
        del arg394_1
        del arg395_1
        del buf137
        buf139 = reinterpret_tensor(buf136, (160, 160, 1, 7), (1120, 1, 1120, 160), 0); del buf136  # reuse
        # Source Nodes: [branch7x7dbl_11, x_289, x_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(arg244_1, buf139, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del arg244_1
        # Source Nodes: [branch7x7dbl_11, x_289, x_293], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf140 = extern_kernels.convolution(buf138, buf139, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf140, (8, 160, 17, 17), (46240, 289, 17, 1))
        buf141 = buf138; del buf138  # reuse
        # Source Nodes: [branch7x7dbl_12, x_294], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf140, arg396_1, arg397_1, arg112_1, arg113_1, buf141, 1280, 289, grid=grid(1280, 289), stream=stream0)
        del arg112_1
        del arg113_1
        del arg396_1
        del arg397_1
        del buf140
        buf142 = reinterpret_tensor(buf139, (160, 160, 7, 1), (1120, 1, 160, 160), 0); del buf139  # reuse
        # Source Nodes: [branch7x7dbl_12, x_294, x_298], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_33.run(arg245_1, buf142, 25600, 7, grid=grid(25600, 7), stream=stream0)
        del arg245_1
        # Source Nodes: [branch7x7dbl_12, x_294, x_298], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf143 = extern_kernels.convolution(buf141, buf142, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf143, (8, 160, 17, 17), (46240, 289, 17, 1))
        del buf142
        buf144 = buf141; del buf141  # reuse
        # Source Nodes: [branch7x7dbl_13, x_299], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_32.run(buf143, arg398_1, arg399_1, arg114_1, arg115_1, buf144, 1280, 289, grid=grid(1280, 289), stream=stream0)
        del arg114_1
        del arg115_1
        del arg398_1
        del arg399_1
        del buf143
        buf145 = reinterpret_tensor(buf132, (192, 160, 1, 7), (1120, 1, 1120, 160), 0); del buf132  # reuse
        # Source Nodes: [branch7x7dbl_13, x_299, x_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_34.run(arg246_1, buf145, 30720, 7, grid=grid(30720, 7), stream=stream0)
        del arg246_1
        # Source Nodes: [branch7x7dbl_13, x_299, x_303], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf146 = extern_kernels.convolution(buf144, buf145, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf146, (8, 192, 17, 17), (55488, 289, 17, 1))
        del buf144
        del buf145
        buf147 = buf101; del buf101  # reuse
        # Source Nodes: [branch_pool_11], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_35.run(buf125, buf147, 1775616, grid=grid(1775616), stream=stream0)
        # Source Nodes: [x_308], Original ATen: [aten.convolution]
        buf148 = extern_kernels.convolution(buf147, arg247_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf148, (8, 192, 17, 17), (55488, 289, 17, 1))
        del arg247_1
        buf149 = buf147; del buf147  # reuse
        # Source Nodes: [cat_23], Original ATen: [aten.cat]
        triton_poi_fused_cat_31.run(buf126, arg384_1, arg385_1, arg100_1, arg101_1, buf133, arg390_1, arg391_1, arg106_1, arg107_1, buf146, arg400_1, arg401_1, arg116_1, arg117_1, buf148, arg402_1, arg403_1, arg118_1, arg119_1, buf149, 6144, 289, grid=grid(6144, 289), stream=stream0)
        del arg100_1
        del arg101_1
        del arg106_1
        del arg107_1
        del arg116_1
        del arg117_1
        del arg118_1
        del arg119_1
        del arg384_1
        del arg385_1
        del arg390_1
        del arg391_1
        del arg400_1
        del arg401_1
        del arg402_1
        del arg403_1
        del buf126
        del buf133
        del buf146
        # Source Nodes: [x_314], Original ATen: [aten.convolution]
        buf150 = extern_kernels.convolution(buf149, arg248_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf150, (8, 192, 17, 17), (55488, 289, 17, 1))
        del arg248_1
        # Source Nodes: [x_319], Original ATen: [aten.convolution]
        buf151 = extern_kernels.convolution(buf149, arg249_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf151, (8, 192, 17, 17), (55488, 289, 17, 1))
        del arg249_1
        buf152 = reinterpret_tensor(buf148, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf148  # reuse
        # Source Nodes: [branch7x7_9, x_320], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf151, arg406_1, arg407_1, arg122_1, arg123_1, buf152, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del arg122_1
        del arg123_1
        del arg406_1
        del arg407_1
        del buf151
        buf153 = empty_strided((192, 192, 1, 7), (1344, 1, 1344, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch7x7_9, x_320, x_324], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37.run(arg250_1, buf153, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del arg250_1
        # Source Nodes: [branch7x7_9, x_320, x_324], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf154 = extern_kernels.convolution(buf152, buf153, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf154, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf155 = buf152; del buf152  # reuse
        # Source Nodes: [branch7x7_10, x_325], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf154, arg408_1, arg409_1, arg124_1, arg125_1, buf155, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del arg124_1
        del arg125_1
        del arg408_1
        del arg409_1
        del buf154
        buf156 = reinterpret_tensor(buf153, (192, 192, 7, 1), (1344, 1, 192, 192), 0); del buf153  # reuse
        # Source Nodes: [branch7x7_10, x_325, x_329], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37.run(arg251_1, buf156, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del arg251_1
        # Source Nodes: [branch7x7_10, x_325, x_329], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf157 = extern_kernels.convolution(buf155, buf156, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf157, (8, 192, 17, 17), (55488, 289, 17, 1))
        # Source Nodes: [x_334], Original ATen: [aten.convolution]
        buf158 = extern_kernels.convolution(buf149, arg252_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf158, (8, 192, 17, 17), (55488, 289, 17, 1))
        del arg252_1
        buf159 = buf155; del buf155  # reuse
        # Source Nodes: [branch7x7dbl_15, x_335], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf158, arg412_1, arg413_1, arg128_1, arg129_1, buf159, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del arg128_1
        del arg129_1
        del arg412_1
        del arg413_1
        del buf158
        buf160 = buf156; del buf156  # reuse
        # Source Nodes: [branch7x7dbl_15, x_335, x_339], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37.run(arg253_1, buf160, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del arg253_1
        # Source Nodes: [branch7x7dbl_15, x_335, x_339], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf161 = extern_kernels.convolution(buf159, buf160, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf161, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf162 = buf159; del buf159  # reuse
        # Source Nodes: [branch7x7dbl_16, x_340], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf161, arg414_1, arg415_1, arg130_1, arg131_1, buf162, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del arg130_1
        del arg131_1
        del arg414_1
        del arg415_1
        del buf161
        buf163 = reinterpret_tensor(buf160, (192, 192, 1, 7), (1344, 1, 1344, 192), 0); del buf160  # reuse
        # Source Nodes: [branch7x7dbl_16, x_340, x_344], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37.run(arg254_1, buf163, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del arg254_1
        # Source Nodes: [branch7x7dbl_16, x_340, x_344], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf164 = extern_kernels.convolution(buf162, buf163, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf164, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf165 = buf162; del buf162  # reuse
        # Source Nodes: [branch7x7dbl_17, x_345], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf164, arg416_1, arg417_1, arg132_1, arg133_1, buf165, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del arg132_1
        del arg133_1
        del arg416_1
        del arg417_1
        del buf164
        buf166 = reinterpret_tensor(buf163, (192, 192, 7, 1), (1344, 1, 192, 192), 0); del buf163  # reuse
        # Source Nodes: [branch7x7dbl_17, x_345, x_349], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37.run(arg255_1, buf166, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del arg255_1
        # Source Nodes: [branch7x7dbl_17, x_345, x_349], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf167 = extern_kernels.convolution(buf165, buf166, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf167, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf168 = buf165; del buf165  # reuse
        # Source Nodes: [branch7x7dbl_18, x_350], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf167, arg418_1, arg419_1, arg134_1, arg135_1, buf168, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del arg134_1
        del arg135_1
        del arg418_1
        del arg419_1
        del buf167
        buf169 = reinterpret_tensor(buf166, (192, 192, 1, 7), (1344, 1, 1344, 192), 0); del buf166  # reuse
        # Source Nodes: [branch7x7dbl_18, x_350, x_354], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37.run(arg256_1, buf169, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del arg256_1
        # Source Nodes: [branch7x7dbl_18, x_350, x_354], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf170 = extern_kernels.convolution(buf168, buf169, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf170, (8, 192, 17, 17), (55488, 289, 17, 1))
        del buf168
        buf171 = buf125; del buf125  # reuse
        # Source Nodes: [branch_pool_13], Original ATen: [aten.avg_pool2d]
        triton_poi_fused_avg_pool2d_35.run(buf149, buf171, 1775616, grid=grid(1775616), stream=stream0)
        del buf149
        # Source Nodes: [x_359], Original ATen: [aten.convolution]
        buf172 = extern_kernels.convolution(buf171, arg257_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf172, (8, 192, 17, 17), (55488, 289, 17, 1))
        del arg257_1
        buf173 = buf171; del buf171  # reuse
        # Source Nodes: [cat_22], Original ATen: [aten.cat]
        triton_poi_fused_cat_31.run(buf150, arg404_1, arg405_1, arg120_1, arg121_1, buf157, arg410_1, arg411_1, arg126_1, arg127_1, buf170, arg420_1, arg421_1, arg136_1, arg137_1, buf172, arg422_1, arg423_1, arg138_1, arg139_1, buf173, 6144, 289, grid=grid(6144, 289), stream=stream0)
        del arg120_1
        del arg121_1
        del arg126_1
        del arg127_1
        del arg136_1
        del arg137_1
        del arg138_1
        del arg139_1
        del arg404_1
        del arg405_1
        del arg410_1
        del arg411_1
        del arg420_1
        del arg421_1
        del arg422_1
        del arg423_1
        del buf150
        del buf157
        del buf170
        buf191 = empty((8, 1280, 8, 8), device='cuda', dtype=torch.float32)
        buf174 = reinterpret_tensor(buf191, (8, 768, 8, 8), (81920, 64, 8, 1), 32768)  # alias
        # Source Nodes: [branch_pool_15], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_38.run(buf173, buf174, 6144, 64, grid=grid(6144, 64), stream=stream0)
        # Source Nodes: [x_366], Original ATen: [aten.convolution]
        buf175 = extern_kernels.convolution(buf173, arg258_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf175, (8, 192, 17, 17), (55488, 289, 17, 1))
        del arg258_1
        buf176 = reinterpret_tensor(buf172, (8, 192, 17, 17), (55488, 1, 3264, 192), 0); del buf172  # reuse
        # Source Nodes: [branch3x3_1, x_367], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf175, arg424_1, arg425_1, arg140_1, arg141_1, buf176, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del arg140_1
        del arg141_1
        del arg424_1
        del arg425_1
        del buf175
        buf177 = empty_strided((320, 192, 3, 3), (1728, 1, 576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch3x3_1, x_367, x_371], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_39.run(arg259_1, buf177, 61440, 9, grid=grid(61440, 9), stream=stream0)
        del arg259_1
        # Source Nodes: [branch3x3_1, x_367, x_371], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf178 = extern_kernels.convolution(buf176, buf177, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf178, (8, 320, 8, 8), (20480, 64, 8, 1))
        del buf177
        # Source Nodes: [x_376], Original ATen: [aten.convolution]
        buf179 = extern_kernels.convolution(buf173, arg260_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf179, (8, 192, 17, 17), (55488, 289, 17, 1))
        del arg260_1
        del buf173
        buf180 = buf176; del buf176  # reuse
        # Source Nodes: [branch7x7x3, x_377], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf179, arg428_1, arg429_1, arg144_1, arg145_1, buf180, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del arg144_1
        del arg145_1
        del arg428_1
        del arg429_1
        del buf179
        buf181 = buf169; del buf169  # reuse
        # Source Nodes: [branch7x7x3, x_377, x_381], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37.run(arg261_1, buf181, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del arg261_1
        # Source Nodes: [branch7x7x3, x_377, x_381], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf182 = extern_kernels.convolution(buf180, buf181, stride=(1, 1), padding=(0, 3), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf182, (8, 192, 17, 17), (55488, 289, 17, 1))
        buf183 = buf180; del buf180  # reuse
        # Source Nodes: [branch7x7x3_1, x_382], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf182, arg430_1, arg431_1, arg146_1, arg147_1, buf183, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del arg146_1
        del arg147_1
        del arg430_1
        del arg431_1
        del buf182
        buf184 = reinterpret_tensor(buf181, (192, 192, 7, 1), (1344, 1, 192, 192), 0); del buf181  # reuse
        # Source Nodes: [branch7x7x3_1, x_382, x_386], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_37.run(arg262_1, buf184, 36864, 7, grid=grid(36864, 7), stream=stream0)
        del arg262_1
        # Source Nodes: [branch7x7x3_1, x_382, x_386], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf185 = extern_kernels.convolution(buf183, buf184, stride=(1, 1), padding=(3, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf185, (8, 192, 17, 17), (55488, 289, 17, 1))
        del buf184
        buf186 = buf183; del buf183  # reuse
        # Source Nodes: [branch7x7x3_2, x_387], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_36.run(buf185, arg432_1, arg433_1, arg148_1, arg149_1, buf186, 1536, 289, grid=grid(1536, 289), stream=stream0)
        del arg148_1
        del arg149_1
        del arg432_1
        del arg433_1
        del buf185
        buf187 = empty_strided((192, 192, 3, 3), (1728, 1, 576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch7x7x3_2, x_387, x_391], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_40.run(arg263_1, buf187, 36864, 9, grid=grid(36864, 9), stream=stream0)
        del arg263_1
        # Source Nodes: [branch7x7x3_2, x_387, x_391], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf188 = extern_kernels.convolution(buf186, buf187, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf188, (8, 192, 8, 8), (12288, 64, 8, 1))
        del buf186
        del buf187
        buf189 = reinterpret_tensor(buf191, (8, 320, 8, 8), (81920, 64, 8, 1), 0)  # alias
        # Source Nodes: [branch3x3_2, x_372], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_41.run(buf178, arg426_1, arg427_1, arg142_1, arg143_1, buf189, 163840, grid=grid(163840), stream=stream0)
        del arg142_1
        del arg143_1
        del arg426_1
        del arg427_1
        del buf178
        buf190 = reinterpret_tensor(buf191, (8, 192, 8, 8), (81920, 64, 8, 1), 20480)  # alias
        # Source Nodes: [branch7x7x3_3, x_392], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_42.run(buf188, arg434_1, arg435_1, arg150_1, arg151_1, buf190, 98304, grid=grid(98304), stream=stream0)
        del arg150_1
        del arg151_1
        del arg434_1
        del arg435_1
        del buf188
        buf192 = empty_strided((8, 1280, 8, 8), (81920, 1, 10240, 1280), device='cuda', dtype=torch.float32)
        buf194 = empty_strided((8, 1280, 8, 8), (81920, 1, 10240, 1280), device='cuda', dtype=torch.float32)
        buf202 = empty_strided((8, 1280, 8, 8), (81920, 1, 10240, 1280), device='cuda', dtype=torch.float32)
        buf213 = empty_strided((8, 1280, 8, 8), (81920, 1, 10240, 1280), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch_pool_16, x_397, x_402, x_417], Original ATen: [aten.avg_pool2d, aten.convolution]
        triton_poi_fused_avg_pool2d_convolution_43.run(buf191, buf192, buf194, buf202, buf213, 10240, 64, grid=grid(10240, 64), stream=stream0)
        del buf174
        del buf189
        del buf190
        del buf191
        # Source Nodes: [x_397], Original ATen: [aten.convolution]
        buf193 = extern_kernels.convolution(buf192, arg264_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf193, (8, 320, 8, 8), (20480, 64, 8, 1))
        del arg264_1
        del buf192
        # Source Nodes: [x_402], Original ATen: [aten.convolution]
        buf195 = extern_kernels.convolution(buf194, arg265_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf195, (8, 384, 8, 8), (24576, 64, 8, 1))
        del arg265_1
        del buf194
        buf196 = empty_strided((8, 384, 8, 8), (24576, 1, 3072, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch3x3_3, x_403], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_44.run(buf195, arg438_1, arg439_1, arg154_1, arg155_1, buf196, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del arg154_1
        del arg155_1
        del arg438_1
        del arg439_1
        del buf195
        buf197 = empty_strided((384, 384, 1, 3), (1152, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_407], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(arg266_1, buf197, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del arg266_1
        # Source Nodes: [x_407], Original ATen: [aten.convolution]
        buf198 = extern_kernels.convolution(buf196, buf197, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf198, (8, 384, 8, 8), (24576, 64, 8, 1))
        buf199 = reinterpret_tensor(buf197, (384, 384, 3, 1), (1152, 1, 384, 384), 0); del buf197  # reuse
        # Source Nodes: [x_412], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(arg267_1, buf199, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del arg267_1
        # Source Nodes: [x_412], Original ATen: [aten.convolution]
        buf200 = extern_kernels.convolution(buf196, buf199, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf200, (8, 384, 8, 8), (24576, 64, 8, 1))
        del buf196
        buf217 = empty((8, 2048, 8, 8), device='cuda', dtype=torch.float32)
        buf201 = reinterpret_tensor(buf217, (8, 768, 8, 8), (131072, 64, 8, 1), 20480)  # alias
        # Source Nodes: [cat_20], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf198, arg440_1, arg441_1, arg156_1, arg157_1, buf200, arg442_1, arg443_1, arg158_1, arg159_1, buf201, 393216, grid=grid(393216), stream=stream0)
        del arg156_1
        del arg157_1
        del arg158_1
        del arg159_1
        del arg440_1
        del arg441_1
        del arg442_1
        del arg443_1
        del buf198
        # Source Nodes: [x_417], Original ATen: [aten.convolution]
        buf203 = extern_kernels.convolution(buf202, arg268_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf203, (8, 448, 8, 8), (28672, 64, 8, 1))
        del arg268_1
        del buf202
        buf204 = empty_strided((8, 448, 8, 8), (28672, 1, 3584, 448), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch3x3dbl_12, x_418], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_47.run(buf203, arg444_1, arg445_1, arg160_1, arg161_1, buf204, 3584, 64, grid=grid(3584, 64), stream=stream0)
        del arg160_1
        del arg161_1
        del arg444_1
        del arg445_1
        del buf203
        buf205 = empty_strided((384, 448, 3, 3), (4032, 1, 1344, 448), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch3x3dbl_12, x_418, x_422], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_48.run(arg269_1, buf205, 172032, 9, grid=grid(172032, 9), stream=stream0)
        del arg269_1
        # Source Nodes: [branch3x3dbl_12, x_418, x_422], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf206 = extern_kernels.convolution(buf204, buf205, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf206, (8, 384, 8, 8), (24576, 64, 8, 1))
        buf207 = reinterpret_tensor(buf200, (8, 384, 8, 8), (24576, 1, 3072, 384), 0); del buf200  # reuse
        # Source Nodes: [branch3x3dbl_13, x_423], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_44.run(buf206, arg446_1, arg447_1, arg162_1, arg163_1, buf207, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del arg162_1
        del arg163_1
        del arg446_1
        del arg447_1
        del buf206
        buf208 = reinterpret_tensor(buf199, (384, 384, 1, 3), (1152, 1, 1152, 384), 0); del buf199  # reuse
        # Source Nodes: [x_427], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(arg270_1, buf208, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del arg270_1
        # Source Nodes: [x_427], Original ATen: [aten.convolution]
        buf209 = extern_kernels.convolution(buf207, buf208, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf209, (8, 384, 8, 8), (24576, 64, 8, 1))
        buf210 = reinterpret_tensor(buf208, (384, 384, 3, 1), (1152, 1, 384, 384), 0); del buf208  # reuse
        # Source Nodes: [x_432], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(arg271_1, buf210, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del arg271_1
        # Source Nodes: [x_432], Original ATen: [aten.convolution]
        buf211 = extern_kernels.convolution(buf207, buf210, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf211, (8, 384, 8, 8), (24576, 64, 8, 1))
        del buf207
        buf212 = reinterpret_tensor(buf217, (8, 768, 8, 8), (131072, 64, 8, 1), 69632)  # alias
        # Source Nodes: [cat_19], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf209, arg448_1, arg449_1, arg164_1, arg165_1, buf211, arg450_1, arg451_1, arg166_1, arg167_1, buf212, 393216, grid=grid(393216), stream=stream0)
        del arg164_1
        del arg165_1
        del arg166_1
        del arg167_1
        del arg448_1
        del arg449_1
        del arg450_1
        del arg451_1
        del buf209
        # Source Nodes: [x_437], Original ATen: [aten.convolution]
        buf214 = extern_kernels.convolution(buf213, arg272_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf214, (8, 192, 8, 8), (12288, 64, 8, 1))
        del arg272_1
        del buf213
        buf215 = reinterpret_tensor(buf217, (8, 320, 8, 8), (131072, 64, 8, 1), 0)  # alias
        # Source Nodes: [branch1x1_7, x_398], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_49.run(buf193, arg436_1, arg437_1, arg152_1, arg153_1, buf215, 163840, grid=grid(163840), stream=stream0)
        del arg152_1
        del arg153_1
        del arg436_1
        del arg437_1
        del buf193
        buf216 = reinterpret_tensor(buf217, (8, 192, 8, 8), (131072, 64, 8, 1), 118784)  # alias
        # Source Nodes: [branch_pool_17, x_438], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf214, arg452_1, arg453_1, arg168_1, arg169_1, buf216, 98304, grid=grid(98304), stream=stream0)
        del arg168_1
        del arg169_1
        del arg452_1
        del arg453_1
        del buf214
        buf218 = empty_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cuda', dtype=torch.float32)
        buf220 = empty_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cuda', dtype=torch.float32)
        buf228 = empty_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cuda', dtype=torch.float32)
        buf239 = empty_strided((8, 2048, 8, 8), (131072, 1, 16384, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [branch_pool_18, x_443, x_448, x_463], Original ATen: [aten.avg_pool2d, aten.convolution]
        triton_poi_fused_avg_pool2d_convolution_51.run(buf217, buf218, buf220, buf228, buf239, 16384, 64, grid=grid(16384, 64), stream=stream0)
        del buf201
        del buf212
        del buf215
        del buf216
        del buf217
        # Source Nodes: [x_443], Original ATen: [aten.convolution]
        buf219 = extern_kernels.convolution(buf218, arg273_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf219, (8, 320, 8, 8), (20480, 64, 8, 1))
        del arg273_1
        del buf218
        # Source Nodes: [x_448], Original ATen: [aten.convolution]
        buf221 = extern_kernels.convolution(buf220, arg274_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf221, (8, 384, 8, 8), (24576, 64, 8, 1))
        del arg274_1
        buf222 = reinterpret_tensor(buf211, (8, 384, 8, 8), (24576, 1, 3072, 384), 0); del buf211  # reuse
        # Source Nodes: [branch3x3_5, x_449], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_44.run(buf221, arg456_1, arg457_1, arg172_1, arg173_1, buf222, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del arg172_1
        del arg173_1
        del arg456_1
        del arg457_1
        del buf221
        buf223 = reinterpret_tensor(buf210, (384, 384, 1, 3), (1152, 1, 1152, 384), 0); del buf210  # reuse
        # Source Nodes: [x_453], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(arg275_1, buf223, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del arg275_1
        # Source Nodes: [x_453], Original ATen: [aten.convolution]
        buf224 = extern_kernels.convolution(buf222, buf223, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf224, (8, 384, 8, 8), (24576, 64, 8, 1))
        buf225 = reinterpret_tensor(buf223, (384, 384, 3, 1), (1152, 1, 384, 384), 0); del buf223  # reuse
        # Source Nodes: [x_458], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(arg276_1, buf225, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del arg276_1
        # Source Nodes: [x_458], Original ATen: [aten.convolution]
        buf226 = extern_kernels.convolution(buf222, buf225, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf226, (8, 384, 8, 8), (24576, 64, 8, 1))
        del buf222
        buf243 = reinterpret_tensor(buf220, (8, 2048, 8, 8), (131072, 64, 8, 1), 0); del buf220  # reuse
        buf227 = reinterpret_tensor(buf243, (8, 768, 8, 8), (131072, 64, 8, 1), 20480)  # alias
        # Source Nodes: [cat_17], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf224, arg458_1, arg459_1, arg174_1, arg175_1, buf226, arg460_1, arg461_1, arg176_1, arg177_1, buf227, 393216, grid=grid(393216), stream=stream0)
        del arg174_1
        del arg175_1
        del arg176_1
        del arg177_1
        del arg458_1
        del arg459_1
        del arg460_1
        del arg461_1
        del buf224
        # Source Nodes: [x_463], Original ATen: [aten.convolution]
        buf229 = extern_kernels.convolution(buf228, arg277_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf229, (8, 448, 8, 8), (28672, 64, 8, 1))
        del arg277_1
        del buf228
        buf230 = buf204; del buf204  # reuse
        # Source Nodes: [branch3x3dbl_15, x_464], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_47.run(buf229, arg462_1, arg463_1, arg178_1, arg179_1, buf230, 3584, 64, grid=grid(3584, 64), stream=stream0)
        del arg178_1
        del arg179_1
        del arg462_1
        del arg463_1
        del buf229
        buf231 = buf205; del buf205  # reuse
        # Source Nodes: [branch3x3dbl_15, x_464, x_468], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_convolution_relu_48.run(arg278_1, buf231, 172032, 9, grid=grid(172032, 9), stream=stream0)
        del arg278_1
        # Source Nodes: [branch3x3dbl_15, x_464, x_468], Original ATen: [aten._native_batch_norm_legit_no_training, aten.convolution, aten.relu]
        buf232 = extern_kernels.convolution(buf230, buf231, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf232, (8, 384, 8, 8), (24576, 64, 8, 1))
        del buf230
        del buf231
        buf233 = reinterpret_tensor(buf226, (8, 384, 8, 8), (24576, 1, 3072, 384), 0); del buf226  # reuse
        # Source Nodes: [branch3x3dbl_16, x_469], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_44.run(buf232, arg464_1, arg465_1, arg180_1, arg181_1, buf233, 3072, 64, grid=grid(3072, 64), stream=stream0)
        del arg180_1
        del arg181_1
        del arg464_1
        del arg465_1
        del buf232
        buf234 = reinterpret_tensor(buf225, (384, 384, 1, 3), (1152, 1, 1152, 384), 0); del buf225  # reuse
        # Source Nodes: [x_473], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(arg279_1, buf234, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del arg279_1
        # Source Nodes: [x_473], Original ATen: [aten.convolution]
        buf235 = extern_kernels.convolution(buf233, buf234, stride=(1, 1), padding=(0, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf235, (8, 384, 8, 8), (24576, 64, 8, 1))
        buf236 = reinterpret_tensor(buf234, (384, 384, 3, 1), (1152, 1, 384, 384), 0); del buf234  # reuse
        # Source Nodes: [x_478], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_45.run(arg280_1, buf236, 147456, 3, grid=grid(147456, 3), stream=stream0)
        del arg280_1
        # Source Nodes: [x_478], Original ATen: [aten.convolution]
        buf237 = extern_kernels.convolution(buf233, buf236, stride=(1, 1), padding=(1, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf237, (8, 384, 8, 8), (24576, 64, 8, 1))
        del buf233
        del buf236
        buf238 = reinterpret_tensor(buf243, (8, 768, 8, 8), (131072, 64, 8, 1), 69632)  # alias
        # Source Nodes: [cat_16], Original ATen: [aten.cat]
        triton_poi_fused_cat_46.run(buf235, arg466_1, arg467_1, arg182_1, arg183_1, buf237, arg468_1, arg469_1, arg184_1, arg185_1, buf238, 393216, grid=grid(393216), stream=stream0)
        del arg182_1
        del arg183_1
        del arg184_1
        del arg185_1
        del arg466_1
        del arg467_1
        del arg468_1
        del arg469_1
        del buf235
        del buf237
        # Source Nodes: [x_483], Original ATen: [aten.convolution]
        buf240 = extern_kernels.convolution(buf239, arg281_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf240, (8, 192, 8, 8), (12288, 64, 8, 1))
        del arg281_1
        del buf239
        buf241 = reinterpret_tensor(buf243, (8, 320, 8, 8), (131072, 64, 8, 1), 0)  # alias
        # Source Nodes: [branch1x1_8, x_444], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_49.run(buf219, arg454_1, arg455_1, arg170_1, arg171_1, buf241, 163840, grid=grid(163840), stream=stream0)
        del arg170_1
        del arg171_1
        del arg454_1
        del arg455_1
        del buf219
        buf242 = reinterpret_tensor(buf243, (8, 192, 8, 8), (131072, 64, 8, 1), 118784)  # alias
        # Source Nodes: [branch_pool_19, x_484], Original ATen: [aten._native_batch_norm_legit_no_training, aten.relu]
        triton_poi_fused__native_batch_norm_legit_no_training_relu_50.run(buf240, arg470_1, arg471_1, arg186_1, arg187_1, buf242, 98304, grid=grid(98304), stream=stream0)
        del arg186_1
        del arg187_1
        del arg470_1
        del arg471_1
        del buf240
        buf244 = empty_strided((8, 2048, 1, 1), (2048, 1, 16384, 16384), device='cuda', dtype=torch.float32)
        buf245 = reinterpret_tensor(buf244, (8, 2048, 1, 1), (2048, 1, 1, 1), 0); del buf244  # reuse
        # Source Nodes: [x_491], Original ATen: [aten.mean]
        triton_per_fused_mean_52.run(buf245, buf243, 16384, 64, grid=grid(16384), stream=stream0)
        del buf227
        del buf238
        del buf241
        del buf242
        del buf243
        buf246 = empty((8, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_496], Original ATen: [aten.addmm]
        extern_kernels.addmm(arg283_1, reinterpret_tensor(buf245, (8, 2048), (2048, 1), 0), reinterpret_tensor(arg282_1, (2048, 1000), (1, 2048), 0), alpha=1, beta=1, out=buf246)
        del arg282_1
        del arg283_1
        return (buf246, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg53_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg54_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg55_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg56_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg57_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg58_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg59_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg60_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg61_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg62_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg63_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg64_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg65_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg66_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg67_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg68_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg69_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg70_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg71_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg72_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg73_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg74_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg75_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg76_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg77_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg78_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg79_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg80_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg81_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg82_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg83_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg84_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg85_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg86_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg87_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg88_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg89_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg90_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg91_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg92_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg93_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg94_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg95_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg96_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg97_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg98_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg99_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg100_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg101_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg102_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg103_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg104_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg105_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg106_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg107_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg108_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg109_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg110_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg111_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg112_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg113_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg114_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg115_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg116_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg117_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg118_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg119_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg120_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg121_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg122_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg123_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg124_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg125_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg126_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg127_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg128_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg129_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg130_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg131_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg132_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg133_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg134_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg135_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg136_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg137_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg138_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg139_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg140_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg141_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg142_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg143_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg144_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg145_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg146_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg147_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg148_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg149_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg150_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg151_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg152_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg153_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg154_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg155_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg156_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg157_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg158_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg159_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg160_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg161_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg162_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg163_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg164_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg165_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg166_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg167_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg168_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg169_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg170_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg171_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg172_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg173_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg174_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg175_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg176_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg177_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg178_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg179_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg180_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg181_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg182_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg183_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg184_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg185_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg186_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg187_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg188_1 = rand_strided((32, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg189_1 = rand_strided((32, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg190_1 = rand_strided((64, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg191_1 = rand_strided((80, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg192_1 = rand_strided((192, 80, 3, 3), (720, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg193_1 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg194_1 = rand_strided((48, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg195_1 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg196_1 = rand_strided((64, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg197_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg198_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg199_1 = rand_strided((32, 192, 1, 1), (192, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg200_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg201_1 = rand_strided((48, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg202_1 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg203_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg204_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg205_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg206_1 = rand_strided((64, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg207_1 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg208_1 = rand_strided((48, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg209_1 = rand_strided((64, 48, 5, 5), (1200, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg210_1 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg211_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg212_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg213_1 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg214_1 = rand_strided((384, 288, 3, 3), (2592, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg215_1 = rand_strided((64, 288, 1, 1), (288, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg216_1 = rand_strided((96, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg217_1 = rand_strided((96, 96, 3, 3), (864, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg218_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg219_1 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg220_1 = rand_strided((128, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg221_1 = rand_strided((192, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg222_1 = rand_strided((128, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg223_1 = rand_strided((128, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg224_1 = rand_strided((128, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg225_1 = rand_strided((128, 128, 7, 1), (896, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg226_1 = rand_strided((192, 128, 1, 7), (896, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg227_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg228_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg229_1 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg230_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg231_1 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg232_1 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg233_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg234_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg235_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg236_1 = rand_strided((192, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg237_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg238_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg239_1 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg240_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg241_1 = rand_strided((192, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg242_1 = rand_strided((160, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg243_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg244_1 = rand_strided((160, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg245_1 = rand_strided((160, 160, 7, 1), (1120, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg246_1 = rand_strided((192, 160, 1, 7), (1120, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg247_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg248_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg249_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg250_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg251_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg252_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg253_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg254_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg255_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg256_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg257_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg258_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg259_1 = rand_strided((320, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg260_1 = rand_strided((192, 768, 1, 1), (768, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg261_1 = rand_strided((192, 192, 1, 7), (1344, 7, 7, 1), device='cuda:0', dtype=torch.float32)
    arg262_1 = rand_strided((192, 192, 7, 1), (1344, 7, 1, 1), device='cuda:0', dtype=torch.float32)
    arg263_1 = rand_strided((192, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg264_1 = rand_strided((320, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg265_1 = rand_strided((384, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg266_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg267_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    arg268_1 = rand_strided((448, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg269_1 = rand_strided((384, 448, 3, 3), (4032, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg270_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg271_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    arg272_1 = rand_strided((192, 1280, 1, 1), (1280, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg273_1 = rand_strided((320, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg274_1 = rand_strided((384, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg275_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg276_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    arg277_1 = rand_strided((448, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg278_1 = rand_strided((384, 448, 3, 3), (4032, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg279_1 = rand_strided((384, 384, 1, 3), (1152, 3, 3, 1), device='cuda:0', dtype=torch.float32)
    arg280_1 = rand_strided((384, 384, 3, 1), (1152, 3, 1, 1), device='cuda:0', dtype=torch.float32)
    arg281_1 = rand_strided((192, 2048, 1, 1), (2048, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg282_1 = rand_strided((1000, 2048), (2048, 1), device='cuda:0', dtype=torch.float32)
    arg283_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg284_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg285_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg286_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg287_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg288_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg289_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg290_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg291_1 = rand_strided((80, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg292_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg293_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg294_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg295_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg296_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg297_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg298_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg299_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg300_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg301_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg302_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg303_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg304_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg305_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg306_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg307_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg308_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg309_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg310_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg311_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg312_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg313_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg314_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg315_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg316_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg317_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg318_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg319_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg320_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg321_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg322_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg323_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg324_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg325_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg326_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg327_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg328_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg329_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg330_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg331_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg332_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg333_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg334_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg335_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg336_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg337_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg338_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg339_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg340_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg341_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg342_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg343_1 = rand_strided((96, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg344_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg345_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg346_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg347_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg348_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg349_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg350_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg351_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg352_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg353_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg354_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg355_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg356_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg357_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg358_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg359_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg360_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg361_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg362_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg363_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg364_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg365_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg366_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg367_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg368_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg369_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg370_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg371_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg372_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg373_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg374_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg375_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg376_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg377_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg378_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg379_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg380_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg381_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg382_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg383_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg384_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg385_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg386_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg387_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg388_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg389_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg390_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg391_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg392_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg393_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg394_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg395_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg396_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg397_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg398_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg399_1 = rand_strided((160, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg400_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg401_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg402_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg403_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg404_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg405_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg406_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg407_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg408_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg409_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg410_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg411_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg412_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg413_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg414_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg415_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg416_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg417_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg418_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg419_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg420_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg421_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg422_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg423_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg424_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg425_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg426_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg427_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg428_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg429_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg430_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg431_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg432_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg433_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg434_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg435_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg436_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg437_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg438_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg439_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg440_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg441_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg442_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg443_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg444_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg445_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg446_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg447_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg448_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg449_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg450_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg451_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg452_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg453_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg454_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg455_1 = rand_strided((320, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg456_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg457_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg458_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg459_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg460_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg461_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg462_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg463_1 = rand_strided((448, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg464_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg465_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg466_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg467_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg468_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg469_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg470_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg471_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg472_1 = rand_strided((8, 3, 299, 299), (268203, 89401, 299, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1, arg327_1, arg328_1, arg329_1, arg330_1, arg331_1, arg332_1, arg333_1, arg334_1, arg335_1, arg336_1, arg337_1, arg338_1, arg339_1, arg340_1, arg341_1, arg342_1, arg343_1, arg344_1, arg345_1, arg346_1, arg347_1, arg348_1, arg349_1, arg350_1, arg351_1, arg352_1, arg353_1, arg354_1, arg355_1, arg356_1, arg357_1, arg358_1, arg359_1, arg360_1, arg361_1, arg362_1, arg363_1, arg364_1, arg365_1, arg366_1, arg367_1, arg368_1, arg369_1, arg370_1, arg371_1, arg372_1, arg373_1, arg374_1, arg375_1, arg376_1, arg377_1, arg378_1, arg379_1, arg380_1, arg381_1, arg382_1, arg383_1, arg384_1, arg385_1, arg386_1, arg387_1, arg388_1, arg389_1, arg390_1, arg391_1, arg392_1, arg393_1, arg394_1, arg395_1, arg396_1, arg397_1, arg398_1, arg399_1, arg400_1, arg401_1, arg402_1, arg403_1, arg404_1, arg405_1, arg406_1, arg407_1, arg408_1, arg409_1, arg410_1, arg411_1, arg412_1, arg413_1, arg414_1, arg415_1, arg416_1, arg417_1, arg418_1, arg419_1, arg420_1, arg421_1, arg422_1, arg423_1, arg424_1, arg425_1, arg426_1, arg427_1, arg428_1, arg429_1, arg430_1, arg431_1, arg432_1, arg433_1, arg434_1, arg435_1, arg436_1, arg437_1, arg438_1, arg439_1, arg440_1, arg441_1, arg442_1, arg443_1, arg444_1, arg445_1, arg446_1, arg447_1, arg448_1, arg449_1, arg450_1, arg451_1, arg452_1, arg453_1, arg454_1, arg455_1, arg456_1, arg457_1, arg458_1, arg459_1, arg460_1, arg461_1, arg462_1, arg463_1, arg464_1, arg465_1, arg466_1, arg467_1, arg468_1, arg469_1, arg470_1, arg471_1, arg472_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('gluon_inception_v3', benchmark_compiled_module)
