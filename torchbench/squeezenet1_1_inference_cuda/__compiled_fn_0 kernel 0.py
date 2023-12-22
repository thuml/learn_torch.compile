
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
# Source Nodes: [l__mod___features_0], Original ATen: [aten.convolution]
# l__mod___features_0 => convolution
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


# kernel path: /tmp/torchinductor_youkaichao/k5/ck5b76sdupxyn7nezhz226yrtzhzrbav3ccwxevg2lotiu6crky7.py
# Source Nodes: [l__mod___features_0], Original ATen: [aten.convolution]
# l__mod___features_0 => convolution
triton_poi_fused_convolution_1 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
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


# kernel path: /tmp/torchinductor_youkaichao/hj/chj3gyxxq3hgd3mhgss73ngcly5jkpoa6elidbyeyjyfrzmbdbqs.py
# Source Nodes: [l__mod___features_0, l__mod___features_1], Original ATen: [aten.convolution, aten.relu]
# l__mod___features_0 => convolution
# l__mod___features_1 => relu
triton_poi_fused_convolution_relu_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3154176
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12321) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fe/cfeg62kaguyapg45aisj47mkunoyyvc6jjkt6iaf6fbi4wdjckzc.py
# Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_2], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# l__mod___features_0 => convolution
# l__mod___features_1 => relu
# l__mod___features_2 => max_pool2d_with_indices
triton_poi_fused_convolution_max_pool2d_with_indices_relu_3 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 3025
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 55
    x3 = (xindex // 55)
    y4 = yindex
    x5 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + ((2*x2) + (222*x3) + (12321*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x2) + (222*x3) + (12321*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + (2*x2) + (222*x3) + (12321*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (111 + (2*x2) + (222*x3) + (12321*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (112 + (2*x2) + (222*x3) + (12321*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (113 + (2*x2) + (222*x3) + (12321*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (222 + (2*x2) + (222*x3) + (12321*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (223 + (2*x2) + (222*x3) + (12321*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (224 + (2*x2) + (222*x3) + (12321*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (y0 + (64*x5) + (193600*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/36/c36p3tlf76b4xtfzvr4dyglsi5x7xly4thnn6re5pyzfrukgsxul.py
# Source Nodes: [getattr_l__mod___features___3___squeeze, x], Original ATen: [aten.convolution, aten.relu]
# getattr_l__mod___features___3___squeeze => convolution_1
# x => relu_1
triton_poi_fused_convolution_relu_4 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[64, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 64
    xnumel = 3025
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
    tmp0 = tl.load(in_ptr0 + (x2 + (3025*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(out_ptr0 + (y0 + (16*x2) + (48400*y1)), tmp3, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/t2/ct2t5kbj24syqq7huqtmrfabtviktkce7bq6uwc7cfh6htnbte4y.py
# Source Nodes: [getattr_l__mod___features___3___expand3x3], Original ATen: [aten.convolution]
# getattr_l__mod___features___3___expand3x3 => convolution_3
triton_poi_fused_convolution_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_5', 'mutated_arg_names': []},
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
    y0 = yindex % 16
    y1 = (yindex // 16)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (16*x2) + (144*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/v5/cv5vu3apxc7e45f5lwajbwlajcsh4jrq4olv2llwyacvw4ssmoaq.py
# Source Nodes: [cat_15], Original ATen: [aten.cat]
# cat_15 => cat
triton_poi_fused_cat_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 3025
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 128
    x2 = xindex
    y1 = (yindex // 128)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (3025*y0) + (193600*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = triton_helpers.maximum(0, tmp7)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tmp0 >= tmp3
    tmp12 = tl.full([1, 1], 128, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + ((-193600) + x2 + (3025*y0) + (193600*y1)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr3 + (tl.broadcast_to((-64) + y0, [XBLOCK, YBLOCK])), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp11, tmp17, tmp18)
    tmp20 = tl.where(tmp4, tmp10, tmp19)
    tl.store(out_ptr0 + (y0 + (128*x2) + (387200*y1)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/yv/cyveab7l377zzqb2digbo4irr6rjm274ydlxbprsnogakcv7nhft.py
# Source Nodes: [cat_14], Original ATen: [aten.cat]
# cat_14 => cat_1
triton_poi_fused_cat_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1548800
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 3025) % 128
    x2 = (xindex // 387200)
    x3 = xindex % 387200
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 64, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (193600*x2)), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = triton_helpers.maximum(0, tmp7)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tmp0 >= tmp3
    tmp12 = tl.full([1], 128, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + ((-193600) + x3 + (193600*x2)), tmp11 & xmask, other=0.0)
    tmp15 = tl.load(in_ptr3 + ((-64) + x1), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp11, tmp17, tmp18)
    tmp20 = tl.where(tmp4, tmp10, tmp19)
    tl.store(out_ptr0 + (x4), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/vp/cvppbry6jek5wleurvb4m652bfq2elrrx7kyauvl64tkymorpzed.py
# Source Nodes: [cat_14, l__mod___features_5], Original ATen: [aten.cat, aten.max_pool2d_with_indices]
# cat_14 => cat_1
# l__mod___features_5 => max_pool2d_with_indices_1
triton_poi_fused_cat_max_pool2d_with_indices_8 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_max_pool2d_with_indices_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 729
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 27
    x3 = (xindex // 27)
    y4 = yindex
    x5 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + ((2*x2) + (110*x3) + (3025*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x2) + (110*x3) + (3025*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + (2*x2) + (110*x3) + (3025*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (55 + (2*x2) + (110*x3) + (3025*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (56 + (2*x2) + (110*x3) + (3025*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (57 + (2*x2) + (110*x3) + (3025*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (110 + (2*x2) + (110*x3) + (3025*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (111 + (2*x2) + (110*x3) + (3025*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (112 + (2*x2) + (110*x3) + (3025*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (y0 + (128*x5) + (93312*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ue/cuehii6eeyvz4n6oihttw6jne76a5neif6a47asvgr4yuviinqyu.py
# Source Nodes: [getattr_l__mod___features___6___squeeze, x_2], Original ATen: [aten.convolution, aten.relu]
# getattr_l__mod___features___6___squeeze => convolution_7
# x_2 => relu_7
triton_poi_fused_convolution_relu_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 128
    xnumel = 729
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
    tmp0 = tl.load(in_ptr0 + (x2 + (729*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(out_ptr0 + (y0 + (32*x2) + (23328*y1)), tmp3, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/5s/c5st7ve4ysbtbcceapq3c7y7e7bp3dylyn6l3pbj5vjuvx5f7efm.py
# Source Nodes: [getattr_l__mod___features___6___expand3x3], Original ATen: [aten.convolution]
# getattr_l__mod___features___6___expand3x3 => convolution_9
triton_poi_fused_convolution_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[4096, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 4096
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


# kernel path: /tmp/torchinductor_youkaichao/hx/chxosxgixhjqhogdwtgjqqlvjbdzonyb5uhnc5udcpsbieokw6ep.py
# Source Nodes: [cat_13], Original ATen: [aten.cat]
# cat_13 => cat_2
triton_poi_fused_cat_11 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 729
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
    tmp3 = tl.full([1, 1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (729*y0) + (93312*y1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = triton_helpers.maximum(0, tmp7)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tmp0 >= tmp3
    tmp12 = tl.full([1, 1], 256, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + ((-93312) + x2 + (729*y0) + (93312*y1)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr3 + (tl.broadcast_to((-128) + y0, [XBLOCK, YBLOCK])), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp11, tmp17, tmp18)
    tmp20 = tl.where(tmp4, tmp10, tmp19)
    tl.store(out_ptr0 + (y0 + (256*x2) + (186624*y1)), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/rv/crvvzftlme2hxizykvhcpsie6qrhwqevwbr4wg4iodvucelqkryt.py
# Source Nodes: [cat_12], Original ATen: [aten.cat]
# cat_12 => cat_3
triton_poi_fused_cat_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 746496
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 729) % 256
    x2 = (xindex // 186624)
    x3 = xindex % 186624
    x4 = xindex
    tmp0 = x1
    tmp1 = tl.full([1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1], 128, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x3 + (93312*x2)), tmp4 & xmask, other=0.0)
    tmp6 = tl.load(in_ptr1 + (x1), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = triton_helpers.maximum(0, tmp7)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tmp0 >= tmp3
    tmp12 = tl.full([1], 256, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + ((-93312) + x3 + (93312*x2)), tmp11 & xmask, other=0.0)
    tmp15 = tl.load(in_ptr3 + ((-128) + x1), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp11, tmp17, tmp18)
    tmp20 = tl.where(tmp4, tmp10, tmp19)
    tl.store(out_ptr0 + (x4), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/gy/cgytqaqoaxhjldabxnu6vequyy4scy3z2swhkmr5vvmg4mubp4f4.py
# Source Nodes: [cat_12, l__mod___features_8], Original ATen: [aten.cat, aten.max_pool2d_with_indices]
# cat_12 => cat_3
# l__mod___features_8 => max_pool2d_with_indices_2
triton_poi_fused_cat_max_pool2d_with_indices_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_max_pool2d_with_indices_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 169
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 13
    x3 = (xindex // 13)
    y4 = yindex
    x5 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + ((2*x2) + (54*x3) + (729*y4)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x2) + (54*x3) + (729*y4)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + (2*x2) + (54*x3) + (729*y4)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (27 + (2*x2) + (54*x3) + (729*y4)), xmask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (28 + (2*x2) + (54*x3) + (729*y4)), xmask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (29 + (2*x2) + (54*x3) + (729*y4)), xmask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (54 + (2*x2) + (54*x3) + (729*y4)), xmask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (55 + (2*x2) + (54*x3) + (729*y4)), xmask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (56 + (2*x2) + (54*x3) + (729*y4)), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (y0 + (256*x5) + (43264*y1)), tmp16, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ca/ccas5ahid3gbxd3ue6dpoxws467drrwtlc5fhfxgikgdplztkmok.py
# Source Nodes: [getattr_l__mod___features___9___squeeze, x_4], Original ATen: [aten.convolution, aten.relu]
# getattr_l__mod___features___9___squeeze => convolution_13
# x_4 => relu_13
triton_poi_fused_convolution_relu_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_14', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 169
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
    tmp0 = tl.load(in_ptr0 + (x2 + (169*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(out_ptr0 + (y0 + (48*x2) + (8112*y1)), tmp3, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sr/csrpxrjhc7w4s6qljplj2gz5a4af4wplf3qw7bkuzvcrrwrvtem4.py
# Source Nodes: [getattr_l__mod___features___9___expand3x3], Original ATen: [aten.convolution]
# getattr_l__mod___features___9___expand3x3 => convolution_15
triton_poi_fused_convolution_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_15', 'mutated_arg_names': []},
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
    y0 = yindex % 48
    y1 = (yindex // 48)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (48*x2) + (432*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h3/ch3vostrsvrw4lhiy3mawmwnqhpgpczhucmvjus24qxlpglvpeev.py
# Source Nodes: [cat_11], Original ATen: [aten.cat]
# cat_11 => cat_4
triton_poi_fused_cat_16 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 169
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 384
    x2 = xindex
    y1 = (yindex // 384)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 192, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (169*y0) + (32448*y1)), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = triton_helpers.maximum(0, tmp7)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tmp0 >= tmp3
    tmp12 = tl.full([1, 1], 384, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + ((-32448) + x2 + (169*y0) + (32448*y1)), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr3 + (tl.broadcast_to((-192) + y0, [XBLOCK, YBLOCK])), tmp11 & xmask & ymask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp11, tmp17, tmp18)
    tmp20 = tl.where(tmp4, tmp10, tmp19)
    tl.store(out_ptr0 + (y0 + (384*x2) + (64896*y1)), tmp20, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ia/ciavj2qifots7zkj7fxffoppzhn64aaarm5rgwcagdjlj6g672ey.py
# Source Nodes: [cat_10, getattr_l__mod___features___11___squeeze, x_6], Original ATen: [aten.cat, aten.convolution, aten.relu]
# cat_10 => cat_5
# getattr_l__mod___features___11___squeeze => convolution_19
# x_6 => relu_19
triton_poi_fused_cat_convolution_relu_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_convolution_relu_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 169
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
    tmp0 = tl.load(in_ptr0 + (x2 + (169*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(out_ptr0 + (y0 + (64*x2) + (10816*y1)), tmp3, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/sh/cshibw45bliheobr4qptpkkba5sjvumkusbzxet3zdu4mahupm3q.py
# Source Nodes: [getattr_l__mod___features___11___expand3x3], Original ATen: [aten.convolution]
# getattr_l__mod___features___11___expand3x3 => convolution_21
triton_poi_fused_convolution_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 16384
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


# kernel path: /tmp/torchinductor_youkaichao/gs/cgslk7zrn2k7pbkfvvyzr2ugapusittmayzvp7furz2p3pswk2zf.py
# Source Nodes: [cat_9], Original ATen: [aten.cat]
# cat_9 => cat_6
triton_poi_fused_cat_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 169
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    y0 = yindex % 512
    x2 = xindex
    y1 = (yindex // 512)
    tmp0 = y0
    tmp1 = tl.full([1, 1], 0, tl.int64)
    tmp2 = tmp0 >= tmp1
    tmp3 = tl.full([1, 1], 256, tl.int64)
    tmp4 = tmp0 < tmp3
    tmp5 = tl.load(in_ptr0 + (x2 + (169*y0) + (43264*y1)), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp6 = tl.load(in_ptr1 + (tl.broadcast_to(y0, [XBLOCK, YBLOCK])), tmp4 & xmask, eviction_policy='evict_last', other=0.0)
    tmp7 = tmp5 + tmp6
    tmp8 = triton_helpers.maximum(0, tmp7)
    tmp9 = tl.full(tmp8.shape, 0.0, tmp8.dtype)
    tmp10 = tl.where(tmp4, tmp8, tmp9)
    tmp11 = tmp0 >= tmp3
    tmp12 = tl.full([1, 1], 512, tl.int64)
    tmp13 = tmp0 < tmp12
    tmp14 = tl.load(in_ptr2 + ((-43264) + x2 + (169*y0) + (43264*y1)), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp15 = tl.load(in_ptr3 + (tl.broadcast_to((-256) + y0, [XBLOCK, YBLOCK])), tmp11 & xmask, eviction_policy='evict_last', other=0.0)
    tmp16 = tmp14 + tmp15
    tmp17 = triton_helpers.maximum(0, tmp16)
    tmp18 = tl.full(tmp17.shape, 0.0, tmp17.dtype)
    tmp19 = tl.where(tmp11, tmp17, tmp18)
    tmp20 = tl.where(tmp4, tmp10, tmp19)
    tl.store(out_ptr0 + (y0 + (512*x2) + (86528*y1)), tmp20, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/g2/cg2u2uylkfmpkkdp27vqcfewa2ggiilnerpgq6iy7a5vlrar6u4e.py
# Source Nodes: [cat_8, flatten, l__mod___classifier_1, l__mod___classifier_2, x_9], Original ATen: [aten.cat, aten.convolution, aten.mean, aten.relu, aten.view]
# cat_8 => cat_7
# flatten => view
# l__mod___classifier_1 => convolution_25
# l__mod___classifier_2 => relu_25
# x_9 => mean
triton_per_fused_cat_convolution_mean_relu_view_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_cat_convolution_mean_relu_view_20', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 4000
    rnumel = 169
    RBLOCK: tl.constexpr = 256
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x3 = xindex
    x0 = xindex % 1000
    tmp0 = tl.load(in_ptr0 + (r2 + (169*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = 169.0
    tmp9 = tmp7 / tmp8
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp9, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg3_1, (16, ), (1, ))
    assert_size_stride(arg4_1, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg5_1, (64, ), (1, ))
    assert_size_stride(arg6_1, (64, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg7_1, (64, ), (1, ))
    assert_size_stride(arg8_1, (16, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg9_1, (16, ), (1, ))
    assert_size_stride(arg10_1, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(arg11_1, (64, ), (1, ))
    assert_size_stride(arg12_1, (64, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(arg13_1, (64, ), (1, ))
    assert_size_stride(arg14_1, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(arg15_1, (32, ), (1, ))
    assert_size_stride(arg16_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg17_1, (128, ), (1, ))
    assert_size_stride(arg18_1, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg19_1, (128, ), (1, ))
    assert_size_stride(arg20_1, (32, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg21_1, (32, ), (1, ))
    assert_size_stride(arg22_1, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(arg23_1, (128, ), (1, ))
    assert_size_stride(arg24_1, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(arg25_1, (128, ), (1, ))
    assert_size_stride(arg26_1, (48, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(arg27_1, (48, ), (1, ))
    assert_size_stride(arg28_1, (192, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg29_1, (192, ), (1, ))
    assert_size_stride(arg30_1, (192, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(arg31_1, (192, ), (1, ))
    assert_size_stride(arg32_1, (48, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg33_1, (48, ), (1, ))
    assert_size_stride(arg34_1, (192, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(arg35_1, (192, ), (1, ))
    assert_size_stride(arg36_1, (192, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(arg37_1, (192, ), (1, ))
    assert_size_stride(arg38_1, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(arg39_1, (64, ), (1, ))
    assert_size_stride(arg40_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg41_1, (256, ), (1, ))
    assert_size_stride(arg42_1, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg43_1, (256, ), (1, ))
    assert_size_stride(arg44_1, (64, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg45_1, (64, ), (1, ))
    assert_size_stride(arg46_1, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(arg47_1, (256, ), (1, ))
    assert_size_stride(arg48_1, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg49_1, (256, ), (1, ))
    assert_size_stride(arg50_1, (1000, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(arg51_1, (1000, ), (1, ))
    assert_size_stride(arg52_1, (4, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg52_1, buf0, 12, 50176, grid=grid(12, 50176), stream=stream0)
        del arg52_1
        buf1 = empty_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 192, 9, grid=grid(192, 9), stream=stream0)
        del arg0_1
        # Source Nodes: [l__mod___features_0], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 64, 111, 111), (788544, 12321, 111, 1))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Source Nodes: [l__mod___features_0, l__mod___features_1], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_2.run(buf3, arg1_1, 3154176, grid=grid(3154176), stream=stream0)
        del arg1_1
        buf4 = empty_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_2], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_3.run(buf3, buf4, 256, 3025, grid=grid(256, 3025), stream=stream0)
        del buf3
        # Source Nodes: [getattr_l__mod___features___3___squeeze], Original ATen: [aten.convolution]
        buf5 = extern_kernels.convolution(buf4, arg2_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 16, 55, 55), (48400, 3025, 55, 1))
        del arg2_1
        del buf4
        buf6 = empty_strided((4, 16, 55, 55), (48400, 1, 880, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___3___squeeze, x], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_4.run(buf5, arg3_1, buf6, 64, 3025, grid=grid(64, 3025), stream=stream0)
        del arg3_1
        del buf5
        # Source Nodes: [getattr_l__mod___features___3___expand1x1], Original ATen: [aten.convolution]
        buf7 = extern_kernels.convolution(buf6, arg4_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf7, (4, 64, 55, 55), (193600, 3025, 55, 1))
        del arg4_1
        buf8 = empty_strided((64, 16, 3, 3), (144, 1, 48, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___3___expand3x3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(arg6_1, buf8, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del arg6_1
        # Source Nodes: [getattr_l__mod___features___3___expand3x3], Original ATen: [aten.convolution]
        buf9 = extern_kernels.convolution(buf6, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 64, 55, 55), (193600, 3025, 55, 1))
        buf10 = empty_strided((4, 128, 55, 55), (387200, 1, 7040, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_15], Original ATen: [aten.cat]
        triton_poi_fused_cat_6.run(buf7, arg5_1, buf9, arg7_1, buf10, 512, 3025, grid=grid(512, 3025), stream=stream0)
        del arg5_1
        del arg7_1
        del buf7
        del buf9
        # Source Nodes: [cat_15, getattr_l__mod___features___4___squeeze], Original ATen: [aten.cat, aten.convolution]
        buf11 = extern_kernels.convolution(buf10, arg8_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf11, (4, 16, 55, 55), (48400, 3025, 55, 1))
        del arg8_1
        buf12 = buf6; del buf6  # reuse
        # Source Nodes: [cat_15, getattr_l__mod___features___4___squeeze, x_1], Original ATen: [aten.cat, aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_4.run(buf11, arg9_1, buf12, 64, 3025, grid=grid(64, 3025), stream=stream0)
        del arg9_1
        del buf11
        # Source Nodes: [getattr_l__mod___features___4___expand1x1], Original ATen: [aten.convolution]
        buf13 = extern_kernels.convolution(buf12, arg10_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 64, 55, 55), (193600, 3025, 55, 1))
        del arg10_1
        buf14 = buf8; del buf8  # reuse
        # Source Nodes: [getattr_l__mod___features___4___expand3x3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_5.run(arg12_1, buf14, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del arg12_1
        # Source Nodes: [getattr_l__mod___features___4___expand3x3], Original ATen: [aten.convolution]
        buf15 = extern_kernels.convolution(buf12, buf14, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf15, (4, 64, 55, 55), (193600, 3025, 55, 1))
        del buf12
        del buf14
        buf16 = reinterpret_tensor(buf10, (4, 128, 55, 55), (387200, 3025, 55, 1), 0); del buf10  # reuse
        # Source Nodes: [cat_14], Original ATen: [aten.cat]
        triton_poi_fused_cat_7.run(buf13, arg11_1, buf15, arg13_1, buf16, 1548800, grid=grid(1548800), stream=stream0)
        del arg11_1
        del arg13_1
        del buf13
        del buf15
        buf17 = empty_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_14, l__mod___features_5], Original ATen: [aten.cat, aten.max_pool2d_with_indices]
        triton_poi_fused_cat_max_pool2d_with_indices_8.run(buf16, buf17, 512, 729, grid=grid(512, 729), stream=stream0)
        del buf16
        # Source Nodes: [getattr_l__mod___features___6___squeeze], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, arg14_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 32, 27, 27), (23328, 729, 27, 1))
        del arg14_1
        del buf17
        buf19 = empty_strided((4, 32, 27, 27), (23328, 1, 864, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___6___squeeze, x_2], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_9.run(buf18, arg15_1, buf19, 128, 729, grid=grid(128, 729), stream=stream0)
        del arg15_1
        del buf18
        # Source Nodes: [getattr_l__mod___features___6___expand1x1], Original ATen: [aten.convolution]
        buf20 = extern_kernels.convolution(buf19, arg16_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf20, (4, 128, 27, 27), (93312, 729, 27, 1))
        del arg16_1
        buf21 = empty_strided((128, 32, 3, 3), (288, 1, 96, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___6___expand3x3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(arg18_1, buf21, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg18_1
        # Source Nodes: [getattr_l__mod___features___6___expand3x3], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf19, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 128, 27, 27), (93312, 729, 27, 1))
        buf23 = empty_strided((4, 256, 27, 27), (186624, 1, 6912, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_13], Original ATen: [aten.cat]
        triton_poi_fused_cat_11.run(buf20, arg17_1, buf22, arg19_1, buf23, 1024, 729, grid=grid(1024, 729), stream=stream0)
        del arg17_1
        del arg19_1
        del buf20
        del buf22
        # Source Nodes: [cat_13, getattr_l__mod___features___7___squeeze], Original ATen: [aten.cat, aten.convolution]
        buf24 = extern_kernels.convolution(buf23, arg20_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf24, (4, 32, 27, 27), (23328, 729, 27, 1))
        del arg20_1
        buf25 = buf19; del buf19  # reuse
        # Source Nodes: [cat_13, getattr_l__mod___features___7___squeeze, x_3], Original ATen: [aten.cat, aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_9.run(buf24, arg21_1, buf25, 128, 729, grid=grid(128, 729), stream=stream0)
        del arg21_1
        del buf24
        # Source Nodes: [getattr_l__mod___features___7___expand1x1], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf25, arg22_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 128, 27, 27), (93312, 729, 27, 1))
        del arg22_1
        buf27 = buf21; del buf21  # reuse
        # Source Nodes: [getattr_l__mod___features___7___expand3x3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_10.run(arg24_1, buf27, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg24_1
        # Source Nodes: [getattr_l__mod___features___7___expand3x3], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf25, buf27, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 128, 27, 27), (93312, 729, 27, 1))
        del buf25
        del buf27
        buf29 = reinterpret_tensor(buf23, (4, 256, 27, 27), (186624, 729, 27, 1), 0); del buf23  # reuse
        # Source Nodes: [cat_12], Original ATen: [aten.cat]
        triton_poi_fused_cat_12.run(buf26, arg23_1, buf28, arg25_1, buf29, 746496, grid=grid(746496), stream=stream0)
        del arg23_1
        del arg25_1
        del buf26
        del buf28
        buf30 = empty_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_12, l__mod___features_8], Original ATen: [aten.cat, aten.max_pool2d_with_indices]
        triton_poi_fused_cat_max_pool2d_with_indices_13.run(buf29, buf30, 1024, 169, grid=grid(1024, 169), stream=stream0)
        del buf29
        # Source Nodes: [getattr_l__mod___features___9___squeeze], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, arg26_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 48, 13, 13), (8112, 169, 13, 1))
        del arg26_1
        del buf30
        buf32 = empty_strided((4, 48, 13, 13), (8112, 1, 624, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___9___squeeze, x_4], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_14.run(buf31, arg27_1, buf32, 192, 169, grid=grid(192, 169), stream=stream0)
        del arg27_1
        del buf31
        # Source Nodes: [getattr_l__mod___features___9___expand1x1], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, arg28_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 192, 13, 13), (32448, 169, 13, 1))
        del arg28_1
        buf34 = empty_strided((192, 48, 3, 3), (432, 1, 144, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___9___expand3x3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg30_1, buf34, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del arg30_1
        # Source Nodes: [getattr_l__mod___features___9___expand3x3], Original ATen: [aten.convolution]
        buf35 = extern_kernels.convolution(buf32, buf34, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf35, (4, 192, 13, 13), (32448, 169, 13, 1))
        buf36 = empty_strided((4, 384, 13, 13), (64896, 1, 4992, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_11], Original ATen: [aten.cat]
        triton_poi_fused_cat_16.run(buf33, arg29_1, buf35, arg31_1, buf36, 1536, 169, grid=grid(1536, 169), stream=stream0)
        del arg29_1
        del arg31_1
        del buf33
        del buf35
        # Source Nodes: [cat_11, getattr_l__mod___features___10___squeeze], Original ATen: [aten.cat, aten.convolution]
        buf37 = extern_kernels.convolution(buf36, arg32_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf37, (4, 48, 13, 13), (8112, 169, 13, 1))
        del arg32_1
        buf38 = buf32; del buf32  # reuse
        # Source Nodes: [cat_11, getattr_l__mod___features___10___squeeze, x_5], Original ATen: [aten.cat, aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_14.run(buf37, arg33_1, buf38, 192, 169, grid=grid(192, 169), stream=stream0)
        del arg33_1
        del buf37
        # Source Nodes: [getattr_l__mod___features___10___expand1x1], Original ATen: [aten.convolution]
        buf39 = extern_kernels.convolution(buf38, arg34_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 192, 13, 13), (32448, 169, 13, 1))
        del arg34_1
        buf40 = buf34; del buf34  # reuse
        # Source Nodes: [getattr_l__mod___features___10___expand3x3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_15.run(arg36_1, buf40, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del arg36_1
        # Source Nodes: [getattr_l__mod___features___10___expand3x3], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf38, buf40, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 192, 13, 13), (32448, 169, 13, 1))
        del buf38
        del buf40
        buf42 = buf36; del buf36  # reuse
        # Source Nodes: [cat_10], Original ATen: [aten.cat]
        triton_poi_fused_cat_16.run(buf39, arg35_1, buf41, arg37_1, buf42, 1536, 169, grid=grid(1536, 169), stream=stream0)
        del arg35_1
        del arg37_1
        del buf39
        del buf41
        # Source Nodes: [cat_10, getattr_l__mod___features___11___squeeze], Original ATen: [aten.cat, aten.convolution]
        buf43 = extern_kernels.convolution(buf42, arg38_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 64, 13, 13), (10816, 169, 13, 1))
        del arg38_1
        del buf42
        buf44 = empty_strided((4, 64, 13, 13), (10816, 1, 832, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_10, getattr_l__mod___features___11___squeeze, x_6], Original ATen: [aten.cat, aten.convolution, aten.relu]
        triton_poi_fused_cat_convolution_relu_17.run(buf43, arg39_1, buf44, 256, 169, grid=grid(256, 169), stream=stream0)
        del arg39_1
        del buf43
        # Source Nodes: [getattr_l__mod___features___11___expand1x1], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, arg40_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (4, 256, 13, 13), (43264, 169, 13, 1))
        del arg40_1
        buf46 = empty_strided((256, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___11___expand3x3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(arg42_1, buf46, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg42_1
        # Source Nodes: [getattr_l__mod___features___11___expand3x3], Original ATen: [aten.convolution]
        buf47 = extern_kernels.convolution(buf44, buf46, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf47, (4, 256, 13, 13), (43264, 169, 13, 1))
        buf48 = empty_strided((4, 512, 13, 13), (86528, 1, 6656, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_9], Original ATen: [aten.cat]
        triton_poi_fused_cat_19.run(buf45, arg41_1, buf47, arg43_1, buf48, 2048, 169, grid=grid(2048, 169), stream=stream0)
        del arg41_1
        del arg43_1
        del buf45
        del buf47
        # Source Nodes: [cat_9, getattr_l__mod___features___12___squeeze], Original ATen: [aten.cat, aten.convolution]
        buf49 = extern_kernels.convolution(buf48, arg44_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf49, (4, 64, 13, 13), (10816, 169, 13, 1))
        del arg44_1
        buf50 = buf44; del buf44  # reuse
        # Source Nodes: [cat_9, getattr_l__mod___features___12___squeeze, x_7], Original ATen: [aten.cat, aten.convolution, aten.relu]
        triton_poi_fused_cat_convolution_relu_17.run(buf49, arg45_1, buf50, 256, 169, grid=grid(256, 169), stream=stream0)
        del arg45_1
        del buf49
        # Source Nodes: [getattr_l__mod___features___12___expand1x1], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf50, arg46_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 256, 13, 13), (43264, 169, 13, 1))
        del arg46_1
        buf52 = buf46; del buf46  # reuse
        # Source Nodes: [getattr_l__mod___features___12___expand3x3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_18.run(arg48_1, buf52, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg48_1
        # Source Nodes: [getattr_l__mod___features___12___expand3x3], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf50, buf52, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 256, 13, 13), (43264, 169, 13, 1))
        del buf50
        del buf52
        buf54 = buf48; del buf48  # reuse
        # Source Nodes: [cat_8], Original ATen: [aten.cat]
        triton_poi_fused_cat_19.run(buf51, arg47_1, buf53, arg49_1, buf54, 2048, 169, grid=grid(2048, 169), stream=stream0)
        del arg47_1
        del arg49_1
        del buf51
        del buf53
        # Source Nodes: [cat_8, l__mod___classifier_1], Original ATen: [aten.cat, aten.convolution]
        buf55 = extern_kernels.convolution(buf54, arg50_1, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 1000, 13, 13), (169000, 169, 13, 1))
        del arg50_1
        del buf54
        buf56 = empty_strided((4, 1000, 1, 1), (1000, 1, 4000, 4000), device='cuda', dtype=torch.float32)
        buf57 = reinterpret_tensor(buf56, (4, 1000), (1000, 1), 0); del buf56  # reuse
        # Source Nodes: [cat_8, flatten, l__mod___classifier_1, l__mod___classifier_2, x_9], Original ATen: [aten.cat, aten.convolution, aten.mean, aten.relu, aten.view]
        triton_per_fused_cat_convolution_mean_relu_view_20.run(buf57, buf55, arg51_1, 4000, 169, grid=grid(4000), stream=stream0)
        del arg51_1
        return (buf57, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((64, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((16, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((64, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((32, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((48, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((192, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((192, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((48, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg33_1 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg34_1 = rand_strided((192, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg35_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg36_1 = rand_strided((192, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg37_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg38_1 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg39_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg40_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg41_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg42_1 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg43_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg44_1 = rand_strided((64, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg45_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg46_1 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg47_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg48_1 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg49_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg50_1 = rand_strided((1000, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    arg51_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg52_1 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('squeezenet1_1', benchmark_compiled_module)
