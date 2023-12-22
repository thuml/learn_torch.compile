
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


# kernel path: /tmp/torchinductor_youkaichao/ou/coupvvlzjesrz7nhi4uzingga4t5kvxyuupuiwtf4tkgp3pzfylr.py
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
    size_hints=[256, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_0', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 192
    xnumel = 121
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
    tmp0 = tl.load(in_ptr0 + (x2 + (121*y3)), xmask & ymask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (3*x2) + (363*y1)), tmp0, xmask & ymask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_youkaichao/yz/cyzk3qxvucgzmffnlduylbcbenxu6ovfdapjeyk44az47rqqn2jq.py
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
    size_hints=[16384, 32], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_1', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 12288
    xnumel = 25
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
    tmp0 = tl.load(in_ptr0 + (x2 + (25*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (1600*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/h7/ch7o2vq5qd4tpsiyo25iwl7747ccpbfrrmhdtu3ugqradmme3mcq.py
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
    size_hints=[131072, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 73728
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


# kernel path: /tmp/torchinductor_youkaichao/3z/c3z53rehqn4yrio4uap6pc3ubt4s7rth45pddjjd2vzm54adiwxt.py
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
    size_hints=[131072, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 98304
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (384*x2) + (3456*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/fh/cfh3qoix5fz3h67nil6hnqyjzd7us4o7fueifmx2u7vtbaxtearr.py
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
    size_hints=[65536, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 65536
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (256*x2) + (2304*y1)), tmp0, xmask)
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


# kernel path: /tmp/torchinductor_youkaichao/75/c752xkyx22zuqfw67kb64xtqa5uzfmh3oiyuqrwo2wsqrb3lg4ma.py
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
    size_hints=[256, 4096], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_6', 'mutated_arg_names': []},
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
    tl.store(out_ptr0 + (y0 + (64*x2) + (193600*y1)), tmp3, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6v/c6vcbz47q2jk4dilgfsqsm2yhtdizpiqbtmmuz3zvzeccz4w5xhy.py
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
    size_hints=[262144], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 186624
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 27
    x2 = (xindex // 1728) % 27
    x3 = (xindex // 46656)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*x1) + (7040*x2) + (193600*x3)), xmask)
    tmp1 = tl.load(in_ptr0 + (64 + x0 + (128*x1) + (7040*x2) + (193600*x3)), xmask)
    tmp3 = tl.load(in_ptr0 + (128 + x0 + (128*x1) + (7040*x2) + (193600*x3)), xmask)
    tmp5 = tl.load(in_ptr0 + (3520 + x0 + (128*x1) + (7040*x2) + (193600*x3)), xmask)
    tmp7 = tl.load(in_ptr0 + (3584 + x0 + (128*x1) + (7040*x2) + (193600*x3)), xmask)
    tmp9 = tl.load(in_ptr0 + (3648 + x0 + (128*x1) + (7040*x2) + (193600*x3)), xmask)
    tmp11 = tl.load(in_ptr0 + (7040 + x0 + (128*x1) + (7040*x2) + (193600*x3)), xmask)
    tmp13 = tl.load(in_ptr0 + (7104 + x0 + (128*x1) + (7040*x2) + (193600*x3)), xmask)
    tmp15 = tl.load(in_ptr0 + (7168 + x0 + (128*x1) + (7040*x2) + (193600*x3)), xmask)
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


# kernel path: /tmp/torchinductor_youkaichao/2l/c2l3fdupntfeccfc7efnapnl3fspejyf3zpb363dhhl74jw4luve.py
# Source Nodes: [l__mod___features_3, l__mod___features_4], Original ATen: [aten.convolution, aten.relu]
# l__mod___features_3 => convolution_1
# l__mod___features_4 => relu_1
triton_poi_fused_convolution_relu_8 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_8', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
    xnumel = 729
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
    tmp0 = tl.load(in_ptr0 + (x2 + (729*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(out_ptr0 + (y0 + (192*x2) + (139968*y1)), tmp3, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/xr/cxrnrrm3zji3u33x7edkn35vxmwvtaaga4kuqk25ec5eh6gqj6g2.py
# Source Nodes: [l__mod___features_5], Original ATen: [aten.max_pool2d_with_indices]
# l__mod___features_5 => getitem_2, getitem_3
triton_poi_fused_max_pool2d_with_indices_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_9', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 129792
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 192
    x1 = (xindex // 192) % 13
    x2 = (xindex // 2496) % 13
    x3 = (xindex // 32448)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (384*x1) + (10368*x2) + (139968*x3)), xmask)
    tmp1 = tl.load(in_ptr0 + (192 + x0 + (384*x1) + (10368*x2) + (139968*x3)), xmask)
    tmp3 = tl.load(in_ptr0 + (384 + x0 + (384*x1) + (10368*x2) + (139968*x3)), xmask)
    tmp5 = tl.load(in_ptr0 + (5184 + x0 + (384*x1) + (10368*x2) + (139968*x3)), xmask)
    tmp7 = tl.load(in_ptr0 + (5376 + x0 + (384*x1) + (10368*x2) + (139968*x3)), xmask)
    tmp9 = tl.load(in_ptr0 + (5568 + x0 + (384*x1) + (10368*x2) + (139968*x3)), xmask)
    tmp11 = tl.load(in_ptr0 + (10368 + x0 + (384*x1) + (10368*x2) + (139968*x3)), xmask)
    tmp13 = tl.load(in_ptr0 + (10560 + x0 + (384*x1) + (10368*x2) + (139968*x3)), xmask)
    tmp15 = tl.load(in_ptr0 + (10752 + x0 + (384*x1) + (10368*x2) + (139968*x3)), xmask)
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


# kernel path: /tmp/torchinductor_youkaichao/vx/cvxh44p6whkddqqtawshcgiw4p7igitww44pkc7gyb3smqorhih4.py
# Source Nodes: [l__mod___features_6, l__mod___features_7], Original ATen: [aten.convolution, aten.relu]
# l__mod___features_6 => convolution_2
# l__mod___features_7 => relu_2
triton_poi_fused_convolution_relu_10 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1536
    xnumel = 169
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
    tmp0 = tl.load(in_ptr0 + (x2 + (169*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(out_ptr0 + (y0 + (384*x2) + (64896*y1)), tmp3, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/24/c24tpbm6ucwx6eeams3kxd3enh6hm4ct67rzfh7ryaunrflzv67a.py
# Source Nodes: [l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.relu]
# l__mod___features_8 => convolution_3
# l__mod___features_9 => relu_3
triton_poi_fused_convolution_relu_11 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_11', 'mutated_arg_names': []},
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
    tl.store(out_ptr0 + (y0 + (256*x2) + (43264*y1)), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ow/cowlybvywlnetkbxnu3upeukqz2mobauvobalhult3fxoyykkm57.py
# Source Nodes: [x], Original ATen: [aten.max_pool2d_with_indices]
# x => getitem_4, getitem_5
triton_poi_fused_max_pool2d_with_indices_12 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i64', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_max_pool2d_with_indices_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 256
    x1 = (xindex // 256) % 6
    x2 = (xindex // 1536) % 6
    x3 = (xindex // 9216)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (512*x1) + (6656*x2) + (43264*x3)), None)
    tmp1 = tl.load(in_ptr0 + (256 + x0 + (512*x1) + (6656*x2) + (43264*x3)), None)
    tmp3 = tl.load(in_ptr0 + (512 + x0 + (512*x1) + (6656*x2) + (43264*x3)), None)
    tmp5 = tl.load(in_ptr0 + (3328 + x0 + (512*x1) + (6656*x2) + (43264*x3)), None)
    tmp7 = tl.load(in_ptr0 + (3584 + x0 + (512*x1) + (6656*x2) + (43264*x3)), None)
    tmp9 = tl.load(in_ptr0 + (3840 + x0 + (512*x1) + (6656*x2) + (43264*x3)), None)
    tmp11 = tl.load(in_ptr0 + (6656 + x0 + (512*x1) + (6656*x2) + (43264*x3)), None)
    tmp13 = tl.load(in_ptr0 + (6912 + x0 + (512*x1) + (6656*x2) + (43264*x3)), None)
    tmp15 = tl.load(in_ptr0 + (7168 + x0 + (512*x1) + (6656*x2) + (43264*x3)), None)
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tmp17 = tmp1 > tmp0
    tmp18 = 1 + (2*x1) + (26*x2)
    tmp19 = (2*x1) + (26*x2)
    tmp20 = tl.where(tmp17, tmp18, tmp19)
    tmp21 = tmp3 > tmp2
    tmp22 = 2 + (2*x1) + (26*x2)
    tmp23 = tl.where(tmp21, tmp22, tmp20)
    tmp24 = tmp5 > tmp4
    tmp25 = 13 + (2*x1) + (26*x2)
    tmp26 = tl.where(tmp24, tmp25, tmp23)
    tmp27 = tmp7 > tmp6
    tmp28 = 14 + (2*x1) + (26*x2)
    tmp29 = tl.where(tmp27, tmp28, tmp26)
    tmp30 = tmp9 > tmp8
    tmp31 = 15 + (2*x1) + (26*x2)
    tmp32 = tl.where(tmp30, tmp31, tmp29)
    tmp33 = tmp11 > tmp10
    tmp34 = 26 + (2*x1) + (26*x2)
    tmp35 = tl.where(tmp33, tmp34, tmp32)
    tmp36 = tmp13 > tmp12
    tmp37 = 27 + (2*x1) + (26*x2)
    tmp38 = tl.where(tmp36, tmp37, tmp35)
    tmp39 = tmp15 > tmp14
    tmp40 = 28 + (2*x1) + (26*x2)
    tmp41 = tl.where(tmp39, tmp40, tmp38)
    tl.store(out_ptr0 + (x4), tmp16, None)
    tl.store(out_ptr1 + (x4), tmp41, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/m4/cm47dis3vjekwumjwkeolurcbsljfxfsmrh26bo7fu5torm7xojn.py
# Source Nodes: [x_1, x_2], Original ATen: [aten._adaptive_avg_pool2d, aten.view]
# x_1 => _adaptive_avg_pool2d
# x_2 => view
triton_poi_fused__adaptive_avg_pool2d_view_13 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[65536], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__adaptive_avg_pool2d_view_13', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 9216
    x1 = (xindex // 9216)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((256*(x0 % 36)) + (9216*x1) + (x0 // 36)), None, eviction_policy='evict_last')
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/xl/cxlw4fg4twxfylzsqfnpnkqu5monvhphnvs4syq5fwfrqqlbw6ts.py
# Source Nodes: [l__mod___classifier_2], Original ATen: [aten.relu, aten.threshold_backward]
# l__mod___classifier_2 => relu_5
triton_poi_fused_relu_threshold_backward_14 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*i1', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_threshold_backward_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 4096
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tmp4 = 0.0
    tmp5 = tmp3 <= tmp4
    tl.store(in_out_ptr0 + (x2), tmp3, None)
    tl.store(out_ptr0 + (x2), tmp5, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wr/cwrvirhjvkstissbe2wyi6255azqcy3br56t6tteg2smtlegnsvb.py
# Source Nodes: [l__mod___classifier_5], Original ATen: [aten.relu]
# l__mod___classifier_5 => relu_6
triton_poi_fused_relu_15 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_15', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 16384
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 4096
    tmp0 = tl.load(in_out_ptr0 + (x2), None)
    tmp1 = tl.load(in_ptr0 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x2), tmp3, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17 = args
    args.clear()
    assert_size_stride(primals_1, (64, 3, 11, 11), (363, 121, 11, 1))
    assert_size_stride(primals_2, (64, ), (1, ))
    assert_size_stride(primals_3, (192, 64, 5, 5), (1600, 25, 5, 1))
    assert_size_stride(primals_4, (192, ), (1, ))
    assert_size_stride(primals_5, (384, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(primals_6, (384, ), (1, ))
    assert_size_stride(primals_7, (256, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(primals_8, (256, ), (1, ))
    assert_size_stride(primals_9, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(primals_10, (256, ), (1, ))
    assert_size_stride(primals_11, (4096, 9216), (9216, 1))
    assert_size_stride(primals_12, (4096, ), (1, ))
    assert_size_stride(primals_13, (4096, 4096), (4096, 1))
    assert_size_stride(primals_14, (4096, ), (1, ))
    assert_size_stride(primals_15, (1000, 4096), (4096, 1))
    assert_size_stride(primals_16, (1000, ), (1, ))
    assert_size_stride(primals_17, (4, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((64, 3, 11, 11), (363, 1, 33, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        stream0 = get_cuda_stream(0)
        triton_poi_fused_0.run(primals_1, buf0, 192, 121, grid=grid(192, 121), stream=stream0)
        del primals_1
        buf1 = empty_strided((192, 64, 5, 5), (1600, 1, 320, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_1.run(primals_3, buf1, 12288, 25, grid=grid(12288, 25), stream=stream0)
        del primals_3
        buf2 = empty_strided((384, 192, 3, 3), (1728, 1, 576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_2.run(primals_5, buf2, 73728, 9, grid=grid(73728, 9), stream=stream0)
        del primals_5
        buf3 = empty_strided((256, 384, 3, 3), (3456, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(primals_7, buf3, 98304, 9, grid=grid(98304, 9), stream=stream0)
        del primals_7
        buf4 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(primals_9, buf4, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del primals_9
        buf5 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(primals_17, buf5, 12, 50176, grid=grid(12, 50176), stream=stream0)
        del primals_17
        # Source Nodes: [l__mod___features_0], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf5, buf0, stride=(4, 4), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 64, 55, 55), (193600, 3025, 55, 1))
        buf7 = empty_strided((4, 64, 55, 55), (193600, 1, 3520, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0, l__mod___features_1], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_6.run(buf6, primals_2, buf7, 256, 3025, grid=grid(256, 3025), stream=stream0)
        del buf6
        del primals_2
        buf8 = empty_strided((4, 64, 27, 27), (46656, 1, 1728, 64), device='cuda', dtype=torch.float32)
        buf9 = empty_strided((4, 64, 27, 27), (46656, 1, 1728, 64), device='cuda', dtype=torch.int64)
        # Source Nodes: [l__mod___features_2], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_7.run(buf7, buf8, buf9, 186624, grid=grid(186624), stream=stream0)
        # Source Nodes: [l__mod___features_3], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf8, buf1, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 192, 27, 27), (139968, 729, 27, 1))
        buf11 = empty_strided((4, 192, 27, 27), (139968, 1, 5184, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_3, l__mod___features_4], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_8.run(buf10, primals_4, buf11, 768, 729, grid=grid(768, 729), stream=stream0)
        del buf10
        del primals_4
        buf12 = empty_strided((4, 192, 13, 13), (32448, 1, 2496, 192), device='cuda', dtype=torch.float32)
        buf13 = empty_strided((4, 192, 13, 13), (32448, 1, 2496, 192), device='cuda', dtype=torch.int64)
        # Source Nodes: [l__mod___features_5], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_9.run(buf11, buf12, buf13, 129792, grid=grid(129792), stream=stream0)
        # Source Nodes: [l__mod___features_6], Original ATen: [aten.convolution]
        buf14 = extern_kernels.convolution(buf12, buf2, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf14, (4, 384, 13, 13), (64896, 169, 13, 1))
        buf15 = empty_strided((4, 384, 13, 13), (64896, 1, 4992, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_6, l__mod___features_7], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_10.run(buf14, primals_6, buf15, 1536, 169, grid=grid(1536, 169), stream=stream0)
        del buf14
        del primals_6
        # Source Nodes: [l__mod___features_8], Original ATen: [aten.convolution]
        buf16 = extern_kernels.convolution(buf15, buf3, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 256, 13, 13), (43264, 169, 13, 1))
        buf17 = empty_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_11.run(buf16, primals_8, buf17, 1024, 169, grid=grid(1024, 169), stream=stream0)
        del primals_8
        # Source Nodes: [l__mod___features_10], Original ATen: [aten.convolution]
        buf18 = extern_kernels.convolution(buf17, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf18, (4, 256, 13, 13), (43264, 169, 13, 1))
        buf19 = reinterpret_tensor(buf16, (4, 256, 13, 13), (43264, 1, 3328, 256), 0); del buf16  # reuse
        # Source Nodes: [l__mod___features_10, l__mod___features_11], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_11.run(buf18, primals_10, buf19, 1024, 169, grid=grid(1024, 169), stream=stream0)
        del buf18
        del primals_10
        buf20 = empty_strided((4, 256, 6, 6), (9216, 1, 1536, 256), device='cuda', dtype=torch.float32)
        buf21 = empty_strided((4, 256, 6, 6), (9216, 1, 1536, 256), device='cuda', dtype=torch.int64)
        # Source Nodes: [x], Original ATen: [aten.max_pool2d_with_indices]
        triton_poi_fused_max_pool2d_with_indices_12.run(buf19, buf20, buf21, 36864, grid=grid(36864), stream=stream0)
        buf22 = empty((4, 9216), device='cuda', dtype=torch.float32)
        # Source Nodes: [x_1, x_2], Original ATen: [aten._adaptive_avg_pool2d, aten.view]
        triton_poi_fused__adaptive_avg_pool2d_view_13.run(buf20, buf22, 36864, grid=grid(36864), stream=stream0)
        buf23 = empty((4, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf22, reinterpret_tensor(primals_11, (9216, 4096), (1, 9216), 0), out=buf23)
        buf24 = buf23; del buf23  # reuse
        buf28 = empty((4, 4096), device='cuda', dtype=torch.bool)
        # Source Nodes: [l__mod___classifier_2], Original ATen: [aten.relu, aten.threshold_backward]
        triton_poi_fused_relu_threshold_backward_14.run(buf24, primals_12, buf28, 16384, grid=grid(16384), stream=stream0)
        del primals_12
        buf25 = empty((4, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf24, reinterpret_tensor(primals_13, (4096, 4096), (1, 4096), 0), out=buf25)
        buf26 = buf25; del buf25  # reuse
        # Source Nodes: [l__mod___classifier_5], Original ATen: [aten.relu]
        triton_poi_fused_relu_15.run(buf26, primals_14, 16384, grid=grid(16384), stream=stream0)
        del primals_14
        buf27 = empty((4, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [pred], Original ATen: [aten.addmm]
        extern_kernels.addmm(primals_16, buf26, reinterpret_tensor(primals_15, (4096, 1000), (1, 4096), 0), alpha=1, beta=1, out=buf27)
        del primals_16
        return (buf27, buf0, buf1, buf2, buf3, buf4, buf5, buf7, buf8, buf9, buf11, buf12, buf13, buf15, buf17, buf19, buf20, buf21, buf22, buf24, buf26, reinterpret_tensor(primals_15, (1000, 4096), (4096, 1), 0), reinterpret_tensor(primals_13, (4096, 4096), (4096, 1), 0), buf28, reinterpret_tensor(primals_11, (4096, 9216), (9216, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((64, 3, 11, 11), (363, 121, 11, 1), device='cuda:0', dtype=torch.float32)
    primals_2 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_3 = rand_strided((192, 64, 5, 5), (1600, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    primals_4 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_5 = rand_strided((384, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_6 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_7 = rand_strided((256, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_8 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_9 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    primals_10 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_11 = rand_strided((4096, 9216), (9216, 1), device='cuda:0', dtype=torch.float32)
    primals_12 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_13 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_14 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_15 = rand_strided((1000, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    primals_16 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    primals_17 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('alexnet', benchmark_compiled_module)
