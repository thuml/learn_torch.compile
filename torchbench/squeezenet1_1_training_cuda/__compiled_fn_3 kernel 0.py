
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


# kernel path: /tmp/torchinductor_youkaichao/ss/cssi6obvbluhzm5ehlld5fw4iqwah2dltb5glr27sm4g7urwdaoj.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_0 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': []},
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

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/bl/cblk6svfk6pfeht5rnk7ahtn22ceqodtyevbsr2eo5bqsfyjos7o.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_1 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/l5/cl5jonohzzpwzarwyibsi7bk3ld2olpsttjnkqjrc5fyzeahf3s4.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_2 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/f5/cf5lnuwbhyotrv4loz3defle4d4fxeigpuotmqm4a3x5ejc2ines.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/sn/csn6njzezltomn3l4ajc7vzvzcev4gmx3lwk7mrblbvktajw7bmv.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/2o/c2oqi4asse3gubdfplggkjrpbk6dfzdnmypau7dcjdiwujo6oixd.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_5 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/yi/cyiejmlwevkekffif3pidj2xl5jjuxhpalau5ytqt7ek3cjj5pdl.py
# Source Nodes: [l__mod___features_0, l__mod___features_1], Original ATen: [aten.convolution, aten.relu]
# l__mod___features_0 => convolution
# l__mod___features_1 => relu
triton_poi_fused_convolution_relu_6 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 12321
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
    tmp0 = tl.load(in_ptr0 + (x2 + (12321*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(out_ptr0 + (y0 + (64*x2) + (788544*y1)), tmp3, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fz/cfzw2p6j3izvwq2bz3jclakdzyjzi36mqh3x4sni6sugs4vk5xcg.py
# Source Nodes: [l__mod___features_2], Original ATen: [aten.max_pool2d_with_indices]
# l__mod___features_2 => getitem, getitem_1
triton_poi_fused_max_pool2d_with_indices_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 774400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 55
    x2 = (xindex // 3520) % 55
    x3 = (xindex // 193600)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*x1) + (14208*x2) + (788544*x3)), xmask)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + (128*x1) + (14208*x2) + (788544*x3)), xmask)
    tmp3 = tl.load(in_ptr0 + (128 + x0 + (128*x1) + (14208*x2) + (788544*x3)), xmask)
    tmp5 = tl.load(in_ptr0 + (7104 + x0 + (128*x1) + (14208*x2) + (788544*x3)), xmask)
    tmp7 = tl.load(in_ptr0 + (7168 + x0 + (128*x1) + (14208*x2) + (788544*x3)), xmask)
    tmp9 = tl.load(in_ptr0 + (7232 + x0 + (128*x1) + (14208*x2) + (788544*x3)), xmask)
    tmp11 = tl.load(in_ptr0 + (14208 + x0 + (128*x1) + (14208*x2) + (788544*x3)), xmask)
    tmp13 = tl.load(in_ptr0 + (14272 + x0 + (128*x1) + (14208*x2) + (788544*x3)), xmask)
    tmp15 = tl.load(in_ptr0 + (14336 + x0 + (128*x1) + (14208*x2) + (788544*x3)), xmask)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp1 > tmp0
    tmp18 = 1 + (2*x1) + (222*x2)
    tmp19 = (2*x1) + (222*x2)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp3 > tmp2
    tmp22 = 2 + (2*x1) + (222*x2)
    tmp23 = tl.where(tmp21, tmp22, tmp20)
    tmp24 = tmp5 > tmp4
    tmp25 = 111 + (2*x1) + (222*x2)
    tmp26 = tl.where(tmp24, tmp25, tmp23)
    tmp27 = tmp7 > tmp6
    tmp28 = 112 + (2*x1) + (222*x2)
    tmp29 = tl.where(tmp27, tmp28, tmp26)
    tmp30 = tmp9 > tmp8
    tmp31 = 113 + (2*x1) + (222*x2)
    tmp32 = tl.where(tmp30, tmp31, tmp29)
    tmp33 = tmp11 > tmp10
    tmp34 = 222 + (2*x1) + (222*x2)
    tmp35 = tl.where(tmp33, tmp34, tmp32)
    tmp36 = tmp13 > tmp12
    tmp37 = 223 + (2*x1) + (222*x2)
    tmp38 = tl.where(tmp36, tmp37, tmp35)
    tmp39 = tmp15 > tmp14
    tmp40 = 224 + (2*x1) + (222*x2)
    tmp41 = tl.where(tmp39, tmp40, tmp38)
    tl.store(out_ptr0 + (x4), tmp16, xmask)
    tl.store(out_ptr1 + (x4), tmp41, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ns/cnsbmny7twsddtrl4pqiqra5hu7uhblywax5fnihizzxkwloasgl.py
# Source Nodes: [getattr_l__mod___features___3___squeeze, x], Original ATen: [aten.convolution, aten.relu]
# getattr_l__mod___features___3___squeeze => convolution_1
# x => relu_1
triton_poi_fused_convolution_relu_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_8', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/bn/cbnshgh54vfswndhnjn5kb6m4ku5f4zbo4ukpuneul6j6bwzx4sy.py
# Source Nodes: [cat_15], Original ATen: [aten.cat]
# cat_15 => cat
triton_poi_fused_cat_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_9', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/kd/ckdqnoaqjqdpgmbkzscllxbrpjv5zy3zhunuxqzitpkjcf2etfeo.py
# Source Nodes: [l__mod___features_5], Original ATen: [aten.max_pool2d_with_indices]
# l__mod___features_5 => getitem_2, getitem_3
triton_poi_fused_max_pool2d_with_indices_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 373248
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 27
    x2 = (xindex // 3456) % 27
    x3 = (xindex // 93312)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (256*x1) + (14080*x2) + (387200*x3)), xmask)
    tmp1 = tl.load(in_ptr0 + (128 + x0 + (256*x1) + (14080*x2) + (387200*x3)), xmask)
    tmp3 = tl.load(in_ptr0 + (256 + x0 + (256*x1) + (14080*x2) + (387200*x3)), xmask)
    tmp5 = tl.load(in_ptr0 + (7040 + x0 + (256*x1) + (14080*x2) + (387200*x3)), xmask)
    tmp7 = tl.load(in_ptr0 + (7168 + x0 + (256*x1) + (14080*x2) + (387200*x3)), xmask)
    tmp9 = tl.load(in_ptr0 + (7296 + x0 + (256*x1) + (14080*x2) + (387200*x3)), xmask)
    tmp11 = tl.load(in_ptr0 + (14080 + x0 + (256*x1) + (14080*x2) + (387200*x3)), xmask)
    tmp13 = tl.load(in_ptr0 + (14208 + x0 + (256*x1) + (14080*x2) + (387200*x3)), xmask)
    tmp15 = tl.load(in_ptr0 + (14336 + x0 + (256*x1) + (14080*x2) + (387200*x3)), xmask)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp1 > tmp0
    tmp18 = 1 + (2*x1) + (110*x2)
    tmp19 = (2*x1) + (110*x2)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp3 > tmp2
    tmp22 = 2 + (2*x1) + (110*x2)
    tmp23 = tl.where(tmp21, tmp22, tmp20)
    tmp24 = tmp5 > tmp4
    tmp25 = 55 + (2*x1) + (110*x2)
    tmp26 = tl.where(tmp24, tmp25, tmp23)
    tmp27 = tmp7 > tmp6
    tmp28 = 56 + (2*x1) + (110*x2)
    tmp29 = tl.where(tmp27, tmp28, tmp26)
    tmp30 = tmp9 > tmp8
    tmp31 = 57 + (2*x1) + (110*x2)
    tmp32 = tl.where(tmp30, tmp31, tmp29)
    tmp33 = tmp11 > tmp10
    tmp34 = 110 + (2*x1) + (110*x2)
    tmp35 = tl.where(tmp33, tmp34, tmp32)
    tmp36 = tmp13 > tmp12
    tmp37 = 111 + (2*x1) + (110*x2)
    tmp38 = tl.where(tmp36, tmp37, tmp35)
    tmp39 = tmp15 > tmp14
    tmp40 = 112 + (2*x1) + (110*x2)
    tmp41 = tl.where(tmp39, tmp40, tmp38)
    tl.store(out_ptr0 + (x4), tmp16, xmask)
    tl.store(out_ptr1 + (x4), tmp41, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xf/cxfrywfuwiqfgpaux6vmdgkrqddu5ypb3uhjrb34oyphc7ktbee3.py
# Source Nodes: [getattr_l__mod___features___6___squeeze, x_2], Original ATen: [aten.convolution, aten.relu]
# getattr_l__mod___features___6___squeeze => convolution_7
# x_2 => relu_7
triton_poi_fused_convolution_relu_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_11', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/zk/czklnplxmww5zw6f4icdqtqyzujdpe3olxs42bl2t3bgeonlwlsw.py
# Source Nodes: [cat_13], Original ATen: [aten.cat]
# cat_13 => cat_2
triton_poi_fused_cat_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_12', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/dl/cdlubvc7u2rqedfxagvusnolg36gtr7n6y7tvma55aa4hsfqy46h.py
# Source Nodes: [l__mod___features_8], Original ATen: [aten.max_pool2d_with_indices]
# l__mod___features_8 => getitem_4, getitem_5
triton_poi_fused_max_pool2d_with_indices_13 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 173056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256) % 13
    x2 = (xindex // 3328) % 13
    x3 = (xindex // 43264)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*x1) + (13824*x2) + (186624*x3)), xmask)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + (512*x1) + (13824*x2) + (186624*x3)), xmask)
    tmp3 = tl.load(in_ptr0 + (512 + x0 + (512*x1) + (13824*x2) + (186624*x3)), xmask)
    tmp5 = tl.load(in_ptr0 + (6912 + x0 + (512*x1) + (13824*x2) + (186624*x3)), xmask)
    tmp7 = tl.load(in_ptr0 + (7168 + x0 + (512*x1) + (13824*x2) + (186624*x3)), xmask)
    tmp9 = tl.load(in_ptr0 + (7424 + x0 + (512*x1) + (13824*x2) + (186624*x3)), xmask)
    tmp11 = tl.load(in_ptr0 + (13824 + x0 + (512*x1) + (13824*x2) + (186624*x3)), xmask)
    tmp13 = tl.load(in_ptr0 + (14080 + x0 + (512*x1) + (13824*x2) + (186624*x3)), xmask)
    tmp15 = tl.load(in_ptr0 + (14336 + x0 + (512*x1) + (13824*x2) + (186624*x3)), xmask)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp1 > tmp0
    tmp18 = 1 + (2*x1) + (54*x2)
    tmp19 = (2*x1) + (54*x2)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp3 > tmp2
    tmp22 = 2 + (2*x1) + (54*x2)
    tmp23 = tl.where(tmp21, tmp22, tmp20)
    tmp24 = tmp5 > tmp4
    tmp25 = 27 + (2*x1) + (54*x2)
    tmp26 = tl.where(tmp24, tmp25, tmp23)
    tmp27 = tmp7 > tmp6
    tmp28 = 28 + (2*x1) + (54*x2)
    tmp29 = tl.where(tmp27, tmp28, tmp26)
    tmp30 = tmp9 > tmp8
    tmp31 = 29 + (2*x1) + (54*x2)
    tmp32 = tl.where(tmp30, tmp31, tmp29)
    tmp33 = tmp11 > tmp10
    tmp34 = 54 + (2*x1) + (54*x2)
    tmp35 = tl.where(tmp33, tmp34, tmp32)
    tmp36 = tmp13 > tmp12
    tmp37 = 55 + (2*x1) + (54*x2)
    tmp38 = tl.where(tmp36, tmp37, tmp35)
    tmp39 = tmp15 > tmp14
    tmp40 = 56 + (2*x1) + (54*x2)
    tmp41 = tl.where(tmp39, tmp40, tmp38)
    tl.store(out_ptr0 + (x4), tmp16, xmask)
    tl.store(out_ptr1 + (x4), tmp41, xmask)
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


# kernel path: /tmp/torchinductor_youkaichao/b7/cb7jblc4ru7savoxds6y73ru7pqdvubilpve4lpnj4d6t7yd6qlc.py
# Source Nodes: [cat_11], Original ATen: [aten.cat]
# cat_11 => cat_4
triton_poi_fused_cat_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_15', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ph/cphqc2vlzh2mmmekq5cbhia2w5n5mxzgxadq2jh6zy3hitclcthe.py
# Source Nodes: [getattr_l__mod___features___11___squeeze, x_6], Original ATen: [aten.convolution, aten.relu]
# getattr_l__mod___features___11___squeeze => convolution_19
# x_6 => relu_19
triton_poi_fused_convolution_relu_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_16', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/al/calmweplh2clgfask72iva56aijgn3axotgj4ztrdhlmtaphdygf.py
# Source Nodes: [cat_9], Original ATen: [aten.cat]
# cat_9 => cat_6
triton_poi_fused_cat_17 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_17', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/um/cum57zatdfnonepo6lijamkhglgikzwhsnzx7fwxlblhrk6a7b3y.py
# Source Nodes: [l__mod___classifier_1, l__mod___classifier_2, pred, x_9], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.threshold_backward, aten.view]
# l__mod___classifier_1 => convolution_25
# l__mod___classifier_2 => relu_25
# pred => view
# x_9 => mean
triton_per_fused_convolution_mean_relu_threshold_backward_view_18 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: '*i1', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_per_fused_convolution_mean_relu_threshold_backward_view_18', 'mutated_arg_names': ['in_out_ptr0']}
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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
    x1 = (xindex // 1000)
    tmp0 = tl.load(in_ptr0 + (r2 + (169*x3)), rmask & xmask, other=0.0)
    tmp1 = tl.load(in_ptr1 + (x0), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
    tmp6 = tl.where(rmask & xmask, tmp4, 0)
    tmp7 = tl.sum(tmp6, 1)[:, None]
    tmp8 = 0.0
    tmp9 = tmp3 <= tmp8
    tmp10 = 169.0
    tmp11 = tmp7 / tmp10
    tl.store(out_ptr0 + (x0 + (1000*r2) + (169000*x1)), tmp9, rmask & xmask)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp11, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xs/cxsavwfqjg7jl23hfbtbc37oozkhxsxufckjebg6dyga24t3jqjc.py
# Source Nodes: [getattr_l__mod___features___12___expand3x3, getattr_l__mod___features___12___expand3x3_activation], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
# getattr_l__mod___features___12___expand3x3 => convolution_24
# getattr_l__mod___features___12___expand3x3_activation => relu_24
triton_poi_fused_convolution_relu_threshold_backward_19 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_threshold_backward_19', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 169
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
    tmp0 = tl.load(in_ptr0 + (x2 + (169*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tl.store(out_ptr0 + (y0 + (256*x2) + (43264*y1)), tmp5, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/u2/cu2uza6gatdifqwxhkvnumlxwenlbmbbmjvq73otqrwxjo63irh7.py
# Source Nodes: [getattr_l__mod___features___10___expand3x3, getattr_l__mod___features___10___expand3x3_activation], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
# getattr_l__mod___features___10___expand3x3 => convolution_18
# getattr_l__mod___features___10___expand3x3_activation => relu_18
triton_poi_fused_convolution_relu_threshold_backward_20 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_threshold_backward_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 169
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
    tmp0 = tl.load(in_ptr0 + (x2 + (169*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tl.store(out_ptr0 + (y0 + (192*x2) + (32448*y1)), tmp5, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lp/clpsqh63p6zs3rqsmws4eauiql7es6izvgff4brtxgcoxofdosbo.py
# Source Nodes: [getattr_l__mod___features___7___expand3x3, getattr_l__mod___features___7___expand3x3_activation], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
# getattr_l__mod___features___7___expand3x3 => convolution_12
# getattr_l__mod___features___7___expand3x3_activation => relu_12
triton_poi_fused_convolution_relu_threshold_backward_21 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_threshold_backward_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 729
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
    tmp0 = tl.load(in_ptr0 + (x2 + (729*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tl.store(out_ptr0 + (y0 + (128*x2) + (93312*y1)), tmp5, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/uu/cuuaxore7fnya7uzo2dcch6pwhoi35rzahjoilxr5jzaaeiwwq77.py
# Source Nodes: [getattr_l__mod___features___4___expand3x3, getattr_l__mod___features___4___expand3x3_activation], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
# getattr_l__mod___features___4___expand3x3 => convolution_6
# getattr_l__mod___features___4___expand3x3_activation => relu_6
triton_poi_fused_convolution_relu_threshold_backward_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_threshold_backward_22', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 3025
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
    tmp0 = tl.load(in_ptr0 + (x2 + (3025*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tl.store(out_ptr0 + (y0 + (64*x2) + (193600*y1)), tmp5, xmask & ymask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (16, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_4, (16, ), (1, ))
    assert_size_stride(primals_5, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_6, (64, ), (1, ))
    assert_size_stride(primals_7, (64, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_8, (64, ), (1, ))
    assert_size_stride(primals_9, (16, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_10, (16, ), (1, ))
    assert_size_stride(primals_11, (64, 16, 1, 1), (16, 1, 1, 1))
    assert_size_stride(primals_12, (64, ), (1, ))
    assert_size_stride(primals_13, (64, 16, 3, 3), (144, 9, 3, 1))
    assert_size_stride(primals_14, (64, ), (1, ))
    assert_size_stride(primals_15, (32, 128, 1, 1), (128, 1, 1, 1))
    assert_size_stride(primals_16, (32, ), (1, ))
    assert_size_stride(primals_17, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_18, (128, ), (1, ))
    assert_size_stride(primals_19, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_20, (128, ), (1, ))
    assert_size_stride(primals_21, (32, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_22, (32, ), (1, ))
    assert_size_stride(primals_23, (128, 32, 1, 1), (32, 1, 1, 1))
    assert_size_stride(primals_24, (128, ), (1, ))
    assert_size_stride(primals_25, (128, 32, 3, 3), (288, 9, 3, 1))
    assert_size_stride(primals_26, (128, ), (1, ))
    assert_size_stride(primals_27, (48, 256, 1, 1), (256, 1, 1, 1))
    assert_size_stride(primals_28, (48, ), (1, ))
    assert_size_stride(primals_29, (192, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_30, (192, ), (1, ))
    assert_size_stride(primals_31, (192, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_32, (192, ), (1, ))
    assert_size_stride(primals_33, (48, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_34, (48, ), (1, ))
    assert_size_stride(primals_35, (192, 48, 1, 1), (48, 1, 1, 1))
    assert_size_stride(primals_36, (192, ), (1, ))
    assert_size_stride(primals_37, (192, 48, 3, 3), (432, 9, 3, 1))
    assert_size_stride(primals_38, (192, ), (1, ))
    assert_size_stride(primals_39, (64, 384, 1, 1), (384, 1, 1, 1))
    assert_size_stride(primals_40, (64, ), (1, ))
    assert_size_stride(primals_41, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_42, (256, ), (1, ))
    assert_size_stride(primals_43, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_44, (256, ), (1, ))
    assert_size_stride(primals_45, (64, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_46, (64, ), (1, ))
    assert_size_stride(primals_47, (256, 64, 1, 1), (64, 1, 1, 1))
    assert_size_stride(primals_48, (256, ), (1, ))
    assert_size_stride(primals_49, (256, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(primals_50, (256, ), (1, ))
    assert_size_stride(primals_51, (1000, 512, 1, 1), (512, 1, 1, 1))
    assert_size_stride(primals_52, (1000, ), (1, ))
    assert_size_stride(primals_53, (4, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 192, 9, grid=grid(192, 9), stream=stream0)
        del primals_1
        buf1 = empty_strided((64, 16, 3, 3), (144, 1, 48, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_7, buf1, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_7
        buf2 = empty_strided((64, 16, 3, 3), (144, 1, 48, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_13, buf2, 1024, 9, grid=grid(1024, 9), stream=stream0)
        del primals_13
        buf3 = empty_strided((128, 32, 3, 3), (288, 1, 96, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_19, buf3, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_19
        buf4 = empty_strided((128, 32, 3, 3), (288, 1, 96, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_25, buf4, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del primals_25
        buf5 = empty_strided((192, 48, 3, 3), (432, 1, 144, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_31, buf5, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_31
        buf6 = empty_strided((192, 48, 3, 3), (432, 1, 144, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_37, buf6, 9216, 9, grid=grid(9216, 9), stream=stream0)
        del primals_37
        buf7 = empty_strided((256, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_43, buf7, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_43
        buf8 = empty_strided((256, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_49, buf8, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del primals_49
        buf9 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_53, buf9, 12, 50176, grid=grid(12, 50176), stream=stream0)
        del primals_53
        # Source Nodes: [l__mod___features_0], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf9, buf0, stride=(2, 2), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 64, 111, 111), (788544, 12321, 111, 1))
        buf11 = empty_strided((4, 64, 111, 111), (788544, 1, 7104, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0, l__mod___features_1], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_6.run(buf10, primals_2, buf11, 256, 12321, grid=grid(256, 12321), stream=stream0)
        del buf10
        del primals_2
        buf12 = empty_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cuda', dtype=torch.float32)
        buf13 = empty_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cuda', dtype=torch.int64)
        # Source Nodes: [l__mod___features_2], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_7.run(buf11, buf12, buf13, 774400, grid=grid(774400), stream=stream0)
        # Source Nodes: [getattr_l__mod___features___3___squeeze], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf12, primals_3, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 16, 55, 55), (48400, 3025, 55, 1))
        buf15 = empty_strided((4, 16, 55, 55), (48400, 1, 880, 16), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___3___squeeze, x], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_8.run(buf14, primals_4, buf15, 64, 3025, grid=grid(64, 3025), stream=stream0)
        del primals_4
        # Source Nodes: [getattr_l__mod___features___3___expand1x1], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, primals_5, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 64, 55, 55), (193600, 3025, 55, 1))
        # Source Nodes: [getattr_l__mod___features___3___expand3x3], Original ATen: [aten.convolution]
        buf17 = extern_kernels.convolution(buf15, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf17, (4, 64, 55, 55), (193600, 3025, 55, 1))
        buf18 = empty_strided((4, 128, 55, 55), (387200, 1, 7040, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_15], Original ATen: [aten.cat]
        triton_poi_fused_cat_9.run(buf16, primals_6, buf17, primals_8, buf18, 512, 3025, grid=grid(512, 3025), stream=stream0)
        # Source Nodes: [getattr_l__mod___features___4___squeeze], Original ATen: [aten.convolution]
        buf19 = extern_kernels.convolution(buf18, primals_9, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 16, 55, 55), (48400, 3025, 55, 1))
        buf20 = reinterpret_tensor(buf14, (4, 16, 55, 55), (48400, 1, 880, 16), 0); del buf14  # reuse
        # Source Nodes: [getattr_l__mod___features___4___squeeze, x_1], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_8.run(buf19, primals_10, buf20, 64, 3025, grid=grid(64, 3025), stream=stream0)
        del buf19
        del primals_10
        # Source Nodes: [getattr_l__mod___features___4___expand1x1], Original ATen: [aten.convolution]
        buf21 = extern_kernels.convolution(buf20, primals_11, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf21, (4, 64, 55, 55), (193600, 3025, 55, 1))
        # Source Nodes: [getattr_l__mod___features___4___expand3x3], Original ATen: [aten.convolution]
        buf22 = extern_kernels.convolution(buf20, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 64, 55, 55), (193600, 3025, 55, 1))
        buf23 = empty_strided((4, 128, 55, 55), (387200, 1, 7040, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_14], Original ATen: [aten.cat]
        triton_poi_fused_cat_9.run(buf21, primals_12, buf22, primals_14, buf23, 512, 3025, grid=grid(512, 3025), stream=stream0)
        buf24 = empty_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cuda', dtype=torch.float32)
        buf25 = empty_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cuda', dtype=torch.int64)
        # Source Nodes: [l__mod___features_5], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_10.run(buf23, buf24, buf25, 373248, grid=grid(373248), stream=stream0)
        # Source Nodes: [getattr_l__mod___features___6___squeeze], Original ATen: [aten.convolution]
        buf26 = extern_kernels.convolution(buf24, primals_15, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 32, 27, 27), (23328, 729, 27, 1))
        buf27 = empty_strided((4, 32, 27, 27), (23328, 1, 864, 32), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___6___squeeze, x_2], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_11.run(buf26, primals_16, buf27, 128, 729, grid=grid(128, 729), stream=stream0)
        del primals_16
        # Source Nodes: [getattr_l__mod___features___6___expand1x1], Original ATen: [aten.convolution]
        buf28 = extern_kernels.convolution(buf27, primals_17, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf28, (4, 128, 27, 27), (93312, 729, 27, 1))
        # Source Nodes: [getattr_l__mod___features___6___expand3x3], Original ATen: [aten.convolution]
        buf29 = extern_kernels.convolution(buf27, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 128, 27, 27), (93312, 729, 27, 1))
        buf30 = empty_strided((4, 256, 27, 27), (186624, 1, 6912, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_13], Original ATen: [aten.cat]
        triton_poi_fused_cat_12.run(buf28, primals_18, buf29, primals_20, buf30, 1024, 729, grid=grid(1024, 729), stream=stream0)
        # Source Nodes: [getattr_l__mod___features___7___squeeze], Original ATen: [aten.convolution]
        buf31 = extern_kernels.convolution(buf30, primals_21, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf31, (4, 32, 27, 27), (23328, 729, 27, 1))
        buf32 = reinterpret_tensor(buf26, (4, 32, 27, 27), (23328, 1, 864, 32), 0); del buf26  # reuse
        # Source Nodes: [getattr_l__mod___features___7___squeeze, x_3], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_11.run(buf31, primals_22, buf32, 128, 729, grid=grid(128, 729), stream=stream0)
        del buf31
        del primals_22
        # Source Nodes: [getattr_l__mod___features___7___expand1x1], Original ATen: [aten.convolution]
        buf33 = extern_kernels.convolution(buf32, primals_23, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf33, (4, 128, 27, 27), (93312, 729, 27, 1))
        # Source Nodes: [getattr_l__mod___features___7___expand3x3], Original ATen: [aten.convolution]
        buf34 = extern_kernels.convolution(buf32, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf34, (4, 128, 27, 27), (93312, 729, 27, 1))
        buf35 = empty_strided((4, 256, 27, 27), (186624, 1, 6912, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_12], Original ATen: [aten.cat]
        triton_poi_fused_cat_12.run(buf33, primals_24, buf34, primals_26, buf35, 1024, 729, grid=grid(1024, 729), stream=stream0)
        buf36 = empty_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cuda', dtype=torch.float32)
        buf37 = empty_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cuda', dtype=torch.int64)
        # Source Nodes: [l__mod___features_8], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_13.run(buf35, buf36, buf37, 173056, grid=grid(173056), stream=stream0)
        # Source Nodes: [getattr_l__mod___features___9___squeeze], Original ATen: [aten.convolution]
        buf38 = extern_kernels.convolution(buf36, primals_27, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf38, (4, 48, 13, 13), (8112, 169, 13, 1))
        buf39 = empty_strided((4, 48, 13, 13), (8112, 1, 624, 48), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___9___squeeze, x_4], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_14.run(buf38, primals_28, buf39, 192, 169, grid=grid(192, 169), stream=stream0)
        del primals_28
        # Source Nodes: [getattr_l__mod___features___9___expand1x1], Original ATen: [aten.convolution]
        buf40 = extern_kernels.convolution(buf39, primals_29, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf40, (4, 192, 13, 13), (32448, 169, 13, 1))
        # Source Nodes: [getattr_l__mod___features___9___expand3x3], Original ATen: [aten.convolution]
        buf41 = extern_kernels.convolution(buf39, buf5, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf41, (4, 192, 13, 13), (32448, 169, 13, 1))
        buf42 = empty_strided((4, 384, 13, 13), (64896, 1, 4992, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_11], Original ATen: [aten.cat]
        triton_poi_fused_cat_15.run(buf40, primals_30, buf41, primals_32, buf42, 1536, 169, grid=grid(1536, 169), stream=stream0)
        # Source Nodes: [getattr_l__mod___features___10___squeeze], Original ATen: [aten.convolution]
        buf43 = extern_kernels.convolution(buf42, primals_33, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf43, (4, 48, 13, 13), (8112, 169, 13, 1))
        buf44 = reinterpret_tensor(buf38, (4, 48, 13, 13), (8112, 1, 624, 48), 0); del buf38  # reuse
        # Source Nodes: [getattr_l__mod___features___10___squeeze, x_5], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_14.run(buf43, primals_34, buf44, 192, 169, grid=grid(192, 169), stream=stream0)
        del buf43
        del primals_34
        # Source Nodes: [getattr_l__mod___features___10___expand1x1], Original ATen: [aten.convolution]
        buf45 = extern_kernels.convolution(buf44, primals_35, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf45, (4, 192, 13, 13), (32448, 169, 13, 1))
        # Source Nodes: [getattr_l__mod___features___10___expand3x3], Original ATen: [aten.convolution]
        buf46 = extern_kernels.convolution(buf44, buf6, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf46, (4, 192, 13, 13), (32448, 169, 13, 1))
        buf47 = empty_strided((4, 384, 13, 13), (64896, 1, 4992, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_10], Original ATen: [aten.cat]
        triton_poi_fused_cat_15.run(buf45, primals_36, buf46, primals_38, buf47, 1536, 169, grid=grid(1536, 169), stream=stream0)
        # Source Nodes: [getattr_l__mod___features___11___squeeze], Original ATen: [aten.convolution]
        buf48 = extern_kernels.convolution(buf47, primals_39, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf48, (4, 64, 13, 13), (10816, 169, 13, 1))
        buf49 = empty_strided((4, 64, 13, 13), (10816, 1, 832, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [getattr_l__mod___features___11___squeeze, x_6], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_16.run(buf48, primals_40, buf49, 256, 169, grid=grid(256, 169), stream=stream0)
        del primals_40
        # Source Nodes: [getattr_l__mod___features___11___expand1x1], Original ATen: [aten.convolution]
        buf50 = extern_kernels.convolution(buf49, primals_41, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf50, (4, 256, 13, 13), (43264, 169, 13, 1))
        # Source Nodes: [getattr_l__mod___features___11___expand3x3], Original ATen: [aten.convolution]
        buf51 = extern_kernels.convolution(buf49, buf7, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf51, (4, 256, 13, 13), (43264, 169, 13, 1))
        buf52 = empty_strided((4, 512, 13, 13), (86528, 1, 6656, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_9], Original ATen: [aten.cat]
        triton_poi_fused_cat_17.run(buf50, primals_42, buf51, primals_44, buf52, 2048, 169, grid=grid(2048, 169), stream=stream0)
        # Source Nodes: [getattr_l__mod___features___12___squeeze], Original ATen: [aten.convolution]
        buf53 = extern_kernels.convolution(buf52, primals_45, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf53, (4, 64, 13, 13), (10816, 169, 13, 1))
        buf54 = reinterpret_tensor(buf48, (4, 64, 13, 13), (10816, 1, 832, 64), 0); del buf48  # reuse
        # Source Nodes: [getattr_l__mod___features___12___squeeze, x_7], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_16.run(buf53, primals_46, buf54, 256, 169, grid=grid(256, 169), stream=stream0)
        del buf53
        del primals_46
        # Source Nodes: [getattr_l__mod___features___12___expand1x1], Original ATen: [aten.convolution]
        buf55 = extern_kernels.convolution(buf54, primals_47, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf55, (4, 256, 13, 13), (43264, 169, 13, 1))
        # Source Nodes: [getattr_l__mod___features___12___expand3x3], Original ATen: [aten.convolution]
        buf56 = extern_kernels.convolution(buf54, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf56, (4, 256, 13, 13), (43264, 169, 13, 1))
        buf57 = empty_strided((4, 512, 13, 13), (86528, 1, 6656, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [cat_8], Original ATen: [aten.cat]
        triton_poi_fused_cat_17.run(buf55, primals_48, buf56, primals_50, buf57, 2048, 169, grid=grid(2048, 169), stream=stream0)
        # Source Nodes: [l__mod___classifier_1], Original ATen: [aten.convolution]
        buf58 = extern_kernels.convolution(buf57, primals_51, stride=(1, 1), padding=(0, 0), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf58, (4, 1000, 13, 13), (169000, 169, 13, 1))
        buf59 = empty_strided((4, 1000, 1, 1), (1000, 1, 4000, 4000), device='cuda', dtype=torch.float32)
        buf61 = empty_strided((4, 1000, 13, 13), (169000, 1, 13000, 1000), device='cuda', dtype=torch.bool)
        buf60 = reinterpret_tensor(buf59, (4, 1000), (1000, 1), 0); del buf59  # reuse
        # Source Nodes: [l__mod___classifier_1, l__mod___classifier_2, pred, x_9], Original ATen: [aten.convolution, aten.mean, aten.relu, aten.threshold_backward, aten.view]
        triton_per_fused_convolution_mean_relu_threshold_backward_view_18.run(buf60, buf58, primals_52, buf61, 4000, 169, grid=grid(4000), stream=stream0)
        del buf58
        del primals_52
        buf62 = empty_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___12___expand3x3, getattr_l__mod___features___12___expand3x3_activation], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        triton_poi_fused_convolution_relu_threshold_backward_19.run(buf56, primals_50, buf62, 1024, 169, grid=grid(1024, 169), stream=stream0)
        del buf56
        del primals_50
        buf63 = empty_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___12___expand1x1, getattr_l__mod___features___12___expand1x1_activation], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        triton_poi_fused_convolution_relu_threshold_backward_19.run(buf55, primals_48, buf63, 1024, 169, grid=grid(1024, 169), stream=stream0)
        del buf55
        del primals_48
        buf64 = empty_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___11___expand3x3, getattr_l__mod___features___11___expand3x3_activation], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        triton_poi_fused_convolution_relu_threshold_backward_19.run(buf51, primals_44, buf64, 1024, 169, grid=grid(1024, 169), stream=stream0)
        del buf51
        del primals_44
        buf65 = empty_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___11___expand1x1, getattr_l__mod___features___11___expand1x1_activation], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        triton_poi_fused_convolution_relu_threshold_backward_19.run(buf50, primals_42, buf65, 1024, 169, grid=grid(1024, 169), stream=stream0)
        del buf50
        del primals_42
        buf66 = empty_strided((4, 192, 13, 13), (32448, 1, 2496, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___10___expand3x3, getattr_l__mod___features___10___expand3x3_activation], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        triton_poi_fused_convolution_relu_threshold_backward_20.run(buf46, primals_38, buf66, 768, 169, grid=grid(768, 169), stream=stream0)
        del buf46
        del primals_38
        buf67 = empty_strided((4, 192, 13, 13), (32448, 1, 2496, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___10___expand1x1, getattr_l__mod___features___10___expand1x1_activation], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        triton_poi_fused_convolution_relu_threshold_backward_20.run(buf45, primals_36, buf67, 768, 169, grid=grid(768, 169), stream=stream0)
        del buf45
        del primals_36
        buf68 = empty_strided((4, 192, 13, 13), (32448, 1, 2496, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___9___expand3x3, getattr_l__mod___features___9___expand3x3_activation], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        triton_poi_fused_convolution_relu_threshold_backward_20.run(buf41, primals_32, buf68, 768, 169, grid=grid(768, 169), stream=stream0)
        del buf41
        del primals_32
        buf69 = empty_strided((4, 192, 13, 13), (32448, 1, 2496, 192), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___9___expand1x1, getattr_l__mod___features___9___expand1x1_activation], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        triton_poi_fused_convolution_relu_threshold_backward_20.run(buf40, primals_30, buf69, 768, 169, grid=grid(768, 169), stream=stream0)
        del buf40
        del primals_30
        buf70 = empty_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___7___expand3x3, getattr_l__mod___features___7___expand3x3_activation], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        triton_poi_fused_convolution_relu_threshold_backward_21.run(buf34, primals_26, buf70, 512, 729, grid=grid(512, 729), stream=stream0)
        del buf34
        del primals_26
        buf71 = empty_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___7___expand1x1, getattr_l__mod___features___7___expand1x1_activation], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        triton_poi_fused_convolution_relu_threshold_backward_21.run(buf33, primals_24, buf71, 512, 729, grid=grid(512, 729), stream=stream0)
        del buf33
        del primals_24
        buf72 = empty_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___6___expand3x3, getattr_l__mod___features___6___expand3x3_activation], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        triton_poi_fused_convolution_relu_threshold_backward_21.run(buf29, primals_20, buf72, 512, 729, grid=grid(512, 729), stream=stream0)
        del buf29
        del primals_20
        buf73 = empty_strided((4, 128, 27, 27), (93312, 1, 3456, 128), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___6___expand1x1, getattr_l__mod___features___6___expand1x1_activation], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        triton_poi_fused_convolution_relu_threshold_backward_21.run(buf28, primals_18, buf73, 512, 729, grid=grid(512, 729), stream=stream0)
        del buf28
        del primals_18
        buf74 = empty_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___4___expand3x3, getattr_l__mod___features___4___expand3x3_activation], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        triton_poi_fused_convolution_relu_threshold_backward_22.run(buf22, primals_14, buf74, 256, 3025, grid=grid(256, 3025), stream=stream0)
        del buf22
        del primals_14
        buf75 = empty_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___4___expand1x1, getattr_l__mod___features___4___expand1x1_activation], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        triton_poi_fused_convolution_relu_threshold_backward_22.run(buf21, primals_12, buf75, 256, 3025, grid=grid(256, 3025), stream=stream0)
        del buf21
        del primals_12
        buf76 = empty_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___3___expand3x3, getattr_l__mod___features___3___expand3x3_activation], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        triton_poi_fused_convolution_relu_threshold_backward_22.run(buf17, primals_8, buf76, 256, 3025, grid=grid(256, 3025), stream=stream0)
        del buf17
        del primals_8
        buf77 = empty_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cuda', dtype=torch.bool)
        # Source Nodes: [getattr_l__mod___features___3___expand1x1, getattr_l__mod___features___3___expand1x1_activation], Original ATen: [aten.convolution, aten.relu, aten.threshold_backward]
        triton_poi_fused_convolution_relu_threshold_backward_22.run(buf16, primals_6, buf77, 256, 3025, grid=grid(256, 3025), stream=stream0)
        del buf16
        del primals_6
        return (buf60, buf0, primals_3, primals_5, buf1, primals_9, primals_11, buf2, primals_15, primals_17, buf3, primals_21, primals_23, buf4, primals_27, primals_29, buf5, primals_33, primals_35, buf6, primals_39, primals_41, buf7, primals_45, primals_47, buf8, primals_51, buf9, buf11, buf12, buf13, buf15, buf18, buf20, buf23, buf24, buf25, buf27, buf30, buf32, buf35, buf36, buf37, buf39, buf42, buf44, buf47, buf49, buf52, buf54, buf57, buf61, buf62, buf63, buf64, buf65, buf66, buf67, buf68, buf69, buf70, buf71, buf72, buf73, buf74, buf75, buf76, buf77, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((16, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((64, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((16, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((64, 16, 1, 1), (16, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((64, 16, 3, 3), (144, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((32, 128, 1, 1), (128, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_18 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_19 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_20 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_21 = rand_strided((32, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_22 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_23 = rand_strided((128, 32, 1, 1), (32, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_24 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_25 = rand_strided((128, 32, 3, 3), (288, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_26 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_27 = rand_strided((48, 256, 1, 1), (256, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_28 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_29 = rand_strided((192, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_30 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_31 = rand_strided((192, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_32 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_33 = rand_strided((48, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_34 = rand_strided((48, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_35 = rand_strided((192, 48, 1, 1), (48, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_36 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_37 = rand_strided((192, 48, 3, 3), (432, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_38 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_39 = rand_strided((64, 384, 1, 1), (384, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_40 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_41 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_42 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_43 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_44 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_45 = rand_strided((64, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_46 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_47 = rand_strided((256, 64, 1, 1), (64, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_48 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_49 = rand_strided((256, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_50 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_51 = rand_strided((1000, 512, 1, 1), (512, 1, 1, 1), device='cuda:0', dtype=torch.float32)
    primals_52 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_53 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('squeezenet1_1', benchmark_compiled_module)
