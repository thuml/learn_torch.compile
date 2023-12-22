
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


# kernel path: /tmp/torchinductor_youkaichao/s4/cs4j4a5lrbpmajwchzgztuhscsucsj2m32a3eazfwrzaesclhbsi.py
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
    size_hints=[256, 128], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_1', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/tm/ctmk63kzh6uhfp7voudsj2mdjvqtl5meofkco42cq3dck6lyoi5n.py
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
    size_hints=[1048576], 
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_2', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 774400
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3025) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/e6/ce6dmytmddczh4px425ia3f6ahhbyadzdca5b2jna3mvqkdeq3ad.py
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
    size_hints=[256, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_3', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
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
    y0 = yindex % 64
    y1 = (yindex // 64)
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
    tl.store(out_ptr0 + (y0 + (64*x5) + (46656*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/zh/czhhhqwqzu77ew5zweva3j6rwltfupyy5fkfazhxjgy6227k7bir.py
# Source Nodes: [l__mod___features_3], Original ATen: [aten.convolution]
# l__mod___features_3 => convolution_1
triton_poi_fused_convolution_4 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_4', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/22/c225yzn3yuswqosm3k4rbggvnj5fk2h5wsak7ntwxdcfgg2l3had.py
# Source Nodes: [l__mod___features_3, l__mod___features_4], Original ATen: [aten.convolution, aten.relu]
# l__mod___features_3 => convolution_1
# l__mod___features_4 => relu_1
triton_poi_fused_convolution_relu_5 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_5', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 559872
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 729) % 192
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/o3/co35uslk4lldqvqtp7h4wy347r2s2bpov3366tsfdlcktmsvw2hq.py
# Source Nodes: [l__mod___features_3, l__mod___features_4, l__mod___features_5], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# l__mod___features_3 => convolution_1
# l__mod___features_4 => relu_1
# l__mod___features_5 => max_pool2d_with_indices_1
triton_poi_fused_convolution_max_pool2d_with_indices_relu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 768
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
    y0 = yindex % 192
    y1 = (yindex // 192)
    tmp0 = tl.load(in_ptr0 + ((2*x2) + (54*x3) + (729*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x2) + (54*x3) + (729*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + (2*x2) + (54*x3) + (729*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (27 + (2*x2) + (54*x3) + (729*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (28 + (2*x2) + (54*x3) + (729*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (29 + (2*x2) + (54*x3) + (729*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (54 + (2*x2) + (54*x3) + (729*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (55 + (2*x2) + (54*x3) + (729*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (56 + (2*x2) + (54*x3) + (729*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(out_ptr0 + (y0 + (192*x5) + (32448*y1)), tmp16, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/je/cjedg5pzd75runuy3d3y35ob5huianhdlzjq3xf54po6xmrtlb7o.py
# Source Nodes: [l__mod___features_6], Original ATen: [aten.convolution]
# l__mod___features_6 => convolution_2
triton_poi_fused_convolution_7 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_7', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/zs/czsjqtcufya5el4nhbvcdkb62unzzux5kmvxwxtbp3zhd7ogsnq4.py
# Source Nodes: [l__mod___features_6, l__mod___features_7], Original ATen: [aten.convolution, aten.relu]
# l__mod___features_6 => convolution_2
# l__mod___features_7 => relu_2
triton_poi_fused_convolution_relu_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_8', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/6f/c6fku6p44mbo6ppdmygeblpsrlvlvlbwmzpni6t2lvm53xfx2amz.py
# Source Nodes: [l__mod___features_6, l__mod___features_7, l__mod___features_8], Original ATen: [aten.convolution, aten.relu]
# l__mod___features_6 => convolution_2
# l__mod___features_7 => relu_2
# l__mod___features_8 => convolution_3
triton_poi_fused_convolution_relu_9 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_9', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/3o/c3ov67dtt5ndio7odczsqqohenca3enwcpqv4gotujgzk25nkwgw.py
# Source Nodes: [l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.relu]
# l__mod___features_6 => convolution_2
# l__mod___features_7 => relu_2
# l__mod___features_8 => convolution_3
# l__mod___features_9 => relu_3
triton_poi_fused_convolution_relu_10 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_10', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/sw/cswqrzbbbgijww7g6fa4a37yucooyufnbuhjz25t6ed2sraz2snz.py
# Source Nodes: [l__mod___features_10, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.relu]
# l__mod___features_10 => convolution_4
# l__mod___features_6 => convolution_2
# l__mod___features_7 => relu_2
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
    size_hints=[65536, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_11', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/23/c23yz7ahfkvxxa7lqi6kylpk6kf3pycr3pf3uwz2rsxcvhbl465v.py
# Source Nodes: [l__mod___features_10, l__mod___features_11, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.relu]
# l__mod___features_10 => convolution_4
# l__mod___features_11 => relu_4
# l__mod___features_6 => convolution_2
# l__mod___features_7 => relu_2
# l__mod___features_8 => convolution_3
# l__mod___features_9 => relu_3
triton_poi_fused_convolution_relu_12 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_12', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 173056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 169) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), xmask)
    tmp1 = tl.load(in_ptr0 + (x1), xmask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/iu/ciufpd4rbf7o6zc47hliqk3xqx53qtwsanqymuyttdnl6vowoitb.py
# Source Nodes: [l__mod___features_10, l__mod___features_11, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9, x, x_1], Original ATen: [aten._adaptive_avg_pool2d, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# l__mod___features_10 => convolution_4
# l__mod___features_11 => relu_4
# l__mod___features_6 => convolution_2
# l__mod___features_7 => relu_2
# l__mod___features_8 => convolution_3
# l__mod___features_9 => relu_3
# x => max_pool2d_with_indices_2
# x_1 => _adaptive_avg_pool2d
triton_poi_fused__adaptive_avg_pool2d_convolution_max_pool2d_with_indices_relu_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__adaptive_avg_pool2d_convolution_max_pool2d_with_indices_relu_13', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 36864
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 6
    x1 = (xindex // 6) % 6
    x2 = (xindex // 36)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (26*x1) + (169*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (26*x1) + (169*x2)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (2 + (2*x0) + (26*x1) + (169*x2)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (13 + (2*x0) + (26*x1) + (169*x2)), None, eviction_policy='evict_last')
    tmp7 = tl.load(in_ptr0 + (14 + (2*x0) + (26*x1) + (169*x2)), None, eviction_policy='evict_last')
    tmp9 = tl.load(in_ptr0 + (15 + (2*x0) + (26*x1) + (169*x2)), None, eviction_policy='evict_last')
    tmp11 = tl.load(in_ptr0 + (26 + (2*x0) + (26*x1) + (169*x2)), None, eviction_policy='evict_last')
    tmp13 = tl.load(in_ptr0 + (27 + (2*x0) + (26*x1) + (169*x2)), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr0 + (28 + (2*x0) + (26*x1) + (169*x2)), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tmp8 = triton_helpers.maximum(tmp7, tmp6)
    tmp10 = triton_helpers.maximum(tmp9, tmp8)
    tmp12 = triton_helpers.maximum(tmp11, tmp10)
    tmp14 = triton_helpers.maximum(tmp13, tmp12)
    tmp16 = triton_helpers.maximum(tmp15, tmp14)
    tl.store(in_out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bj/cbjpmkyxmly47b32tpjjvnlyq3rhf5ffeadmswxjzilbfpttmgzz.py
# Source Nodes: [l__mod___classifier_2], Original ATen: [aten.relu]
# l__mod___classifier_2 => relu_5
triton_poi_fused_relu_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_14', 'mutated_arg_names': ['in_out_ptr0']},
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 11, 11), (363, 121, 11, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (192, 64, 5, 5), (1600, 25, 5, 1))
    assert_size_stride(arg3_1, (192, ), (1, ))
    assert_size_stride(arg4_1, (384, 192, 3, 3), (1728, 9, 3, 1))
    assert_size_stride(arg5_1, (384, ), (1, ))
    assert_size_stride(arg6_1, (256, 384, 3, 3), (3456, 9, 3, 1))
    assert_size_stride(arg7_1, (256, ), (1, ))
    assert_size_stride(arg8_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg9_1, (256, ), (1, ))
    assert_size_stride(arg10_1, (4096, 9216), (9216, 1))
    assert_size_stride(arg11_1, (4096, ), (1, ))
    assert_size_stride(arg12_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg13_1, (4096, ), (1, ))
    assert_size_stride(arg14_1, (1000, 4096), (4096, 1))
    assert_size_stride(arg15_1, (1000, ), (1, ))
    assert_size_stride(arg16_1, (4, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg16_1, buf0, 12, 50176, grid=grid(12, 50176), stream=stream0)
        del arg16_1
        buf1 = empty_strided((64, 3, 11, 11), (363, 1, 33, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 192, 121, grid=grid(192, 121), stream=stream0)
        del arg0_1
        # Source Nodes: [l__mod___features_0], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(4, 4), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 64, 55, 55), (193600, 3025, 55, 1))
        del buf0
        del buf1
        buf3 = buf2; del buf2  # reuse
        # Source Nodes: [l__mod___features_0, l__mod___features_1], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_2.run(buf3, arg1_1, 774400, grid=grid(774400), stream=stream0)
        del arg1_1
        buf4 = empty_strided((4, 64, 27, 27), (46656, 1, 1728, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_2], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_3.run(buf3, buf4, 256, 729, grid=grid(256, 729), stream=stream0)
        del buf3
        buf5 = empty_strided((192, 64, 5, 5), (1600, 1, 320, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_3], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_4.run(arg2_1, buf5, 12288, 25, grid=grid(12288, 25), stream=stream0)
        del arg2_1
        # Source Nodes: [l__mod___features_3], Original ATen: [aten.convolution]
        buf6 = extern_kernels.convolution(buf4, buf5, stride=(1, 1), padding=(2, 2), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf6, (4, 192, 27, 27), (139968, 729, 27, 1))
        del buf4
        del buf5
        buf7 = buf6; del buf6  # reuse
        # Source Nodes: [l__mod___features_3, l__mod___features_4], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_5.run(buf7, arg3_1, 559872, grid=grid(559872), stream=stream0)
        del arg3_1
        buf8 = empty_strided((4, 192, 13, 13), (32448, 1, 2496, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_3, l__mod___features_4, l__mod___features_5], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_6.run(buf7, buf8, 768, 169, grid=grid(768, 169), stream=stream0)
        del buf7
        buf9 = empty_strided((384, 192, 3, 3), (1728, 1, 576, 192), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_6], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_7.run(arg4_1, buf9, 73728, 9, grid=grid(73728, 9), stream=stream0)
        del arg4_1
        # Source Nodes: [l__mod___features_6], Original ATen: [aten.convolution]
        buf10 = extern_kernels.convolution(buf8, buf9, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf10, (4, 384, 13, 13), (64896, 169, 13, 1))
        del buf8
        del buf9
        buf11 = empty_strided((4, 384, 13, 13), (64896, 1, 4992, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_6, l__mod___features_7], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_8.run(buf10, arg5_1, buf11, 1536, 169, grid=grid(1536, 169), stream=stream0)
        del arg5_1
        del buf10
        buf12 = empty_strided((256, 384, 3, 3), (3456, 1, 1152, 384), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_6, l__mod___features_7, l__mod___features_8], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_9.run(arg6_1, buf12, 98304, 9, grid=grid(98304, 9), stream=stream0)
        del arg6_1
        # Source Nodes: [l__mod___features_6, l__mod___features_7, l__mod___features_8], Original ATen: [aten.convolution, aten.relu]
        buf13 = extern_kernels.convolution(buf11, buf12, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf13, (4, 256, 13, 13), (43264, 169, 13, 1))
        del buf11
        del buf12
        buf14 = empty_strided((4, 256, 13, 13), (43264, 1, 3328, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_10.run(buf13, arg7_1, buf14, 1024, 169, grid=grid(1024, 169), stream=stream0)
        del arg7_1
        del buf13
        buf15 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_10, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_11.run(arg8_1, buf15, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg8_1
        # Source Nodes: [l__mod___features_10, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.relu]
        buf16 = extern_kernels.convolution(buf14, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 256, 13, 13), (43264, 169, 13, 1))
        del buf14
        del buf15
        buf17 = buf16; del buf16  # reuse
        # Source Nodes: [l__mod___features_10, l__mod___features_11, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_12.run(buf17, arg9_1, 173056, grid=grid(173056), stream=stream0)
        del arg9_1
        buf18 = empty((4, 256, 6, 6), device='cuda', dtype=torch.float32)
        buf19 = buf18; del buf18  # reuse
        # Source Nodes: [l__mod___features_10, l__mod___features_11, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9, x, x_1], Original ATen: [aten._adaptive_avg_pool2d, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__adaptive_avg_pool2d_convolution_max_pool2d_with_indices_relu_13.run(buf19, buf17, 36864, grid=grid(36864), stream=stream0)
        del buf17
        buf20 = empty((4, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf19, (4, 9216), (9216, 1), 0), reinterpret_tensor(arg10_1, (9216, 4096), (1, 9216), 0), out=buf20)
        del arg10_1
        del buf19
        buf21 = buf20; del buf20  # reuse
        # Source Nodes: [l__mod___classifier_2], Original ATen: [aten.relu]
        triton_poi_fused_relu_14.run(buf21, arg11_1, 16384, grid=grid(16384), stream=stream0)
        del arg11_1
        buf22 = empty((4, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___classifier_2], Original ATen: [aten.relu]
        extern_kernels.mm(buf21, reinterpret_tensor(arg12_1, (4096, 4096), (1, 4096), 0), out=buf22)
        del arg12_1
        del buf21
        buf23 = buf22; del buf22  # reuse
        # Source Nodes: [l__mod___classifier_5], Original ATen: [aten.relu]
        triton_poi_fused_relu_14.run(buf23, arg13_1, 16384, grid=grid(16384), stream=stream0)
        del arg13_1
        buf24 = empty((4, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___classifier_5, x_3], Original ATen: [aten.addmm, aten.relu]
        extern_kernels.addmm(arg15_1, buf23, reinterpret_tensor(arg14_1, (4096, 1000), (1, 4096), 0), alpha=1, beta=1, out=buf24)
        del arg14_1
        del arg15_1
        return (buf24, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 11, 11), (363, 121, 11, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((192, 64, 5, 5), (1600, 25, 5, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((192, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((384, 192, 3, 3), (1728, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((384, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((256, 384, 3, 3), (3456, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((4096, 9216), (9216, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((1000, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('alexnet', benchmark_compiled_module)
