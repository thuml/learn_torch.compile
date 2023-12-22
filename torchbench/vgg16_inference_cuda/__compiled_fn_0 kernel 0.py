
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


# kernel path: /tmp/torchinductor_youkaichao/gm/cgmyxb2kkntsn4hrfupxddbzbayzwhuat4yslnbufmw4ed34jmu3.py
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
    size_hints=[256, 65536], tile_hint=TileHint.DEFAULT,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_2', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 50176
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
    tmp0 = tl.load(in_ptr0 + (x2 + (50176*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(out_ptr0 + (y0 + (64*x2) + (3211264*y1)), tmp3, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/lm/clmiclgpmhcvnej3cvyw2u2sh5if4rcirdutvv7u76mmpcgt2xck.py
# Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_2], Original ATen: [aten.convolution, aten.relu]
# l__mod___features_0 => convolution
# l__mod___features_1 => relu
# l__mod___features_2 => convolution_1
triton_poi_fused_convolution_relu_3 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_3', 'mutated_arg_names': []},
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
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (64*x2) + (576*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/wm/cwmng6boyzfi5vovujdhpv2iynjt4jhvazioeny2u36sqgqtyhyz.py
# Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_2, l__mod___features_3], Original ATen: [aten.convolution, aten.relu]
# l__mod___features_0 => convolution
# l__mod___features_1 => relu
# l__mod___features_2 => convolution_1
# l__mod___features_3 => relu_1
triton_poi_fused_convolution_relu_4 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_relu_4', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12845056
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 50176) % 64
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp3, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/wq/cwqjlhqbxnn53kiql5h3ocwc5rglyoqpw56qrpygqplvqdmz6bns.py
# Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_2, l__mod___features_3, l__mod___features_4], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# l__mod___features_0 => convolution
# l__mod___features_1 => relu
# l__mod___features_2 => convolution_1
# l__mod___features_3 => relu_1
# l__mod___features_4 => max_pool2d_with_indices
triton_poi_fused_convolution_max_pool2d_with_indices_relu_5 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[256, 16384], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_5', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 256
    xnumel = 12544
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 112
    x3 = (xindex // 112)
    y4 = yindex
    x5 = xindex
    y0 = yindex % 64
    y1 = (yindex // 64)
    tmp0 = tl.load(in_ptr0 + ((2*x2) + (448*x3) + (50176*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x2) + (448*x3) + (50176*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (224 + (2*x2) + (448*x3) + (50176*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (225 + (2*x2) + (448*x3) + (50176*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (y0 + (64*x5) + (802816*y1)), tmp6, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/j6/cj6z4jy3klm7dg4wmrzmry27hyj5ulua57cr5qoq3q5kvoeggyoc.py
# Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# l__mod___features_0 => convolution
# l__mod___features_1 => relu
# l__mod___features_2 => convolution_1
# l__mod___features_3 => relu_1
# l__mod___features_4 => max_pool2d_with_indices
# l__mod___features_5 => convolution_2
triton_poi_fused_convolution_max_pool2d_with_indices_relu_6 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_6', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 8192
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


# kernel path: /tmp/torchinductor_youkaichao/y5/cy5uqlpdcyedbqfqcbq2zb4lr4nwr6ykozxmmlxrmtkxkwcop3w3.py
# Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# l__mod___features_0 => convolution
# l__mod___features_1 => relu
# l__mod___features_2 => convolution_1
# l__mod___features_3 => relu_1
# l__mod___features_4 => max_pool2d_with_indices
# l__mod___features_5 => convolution_2
# l__mod___features_6 => relu_2
triton_poi_fused_convolution_max_pool2d_with_indices_relu_7 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_7', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2 + (12544*y3)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), ymask, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(out_ptr0 + (y0 + (128*x2) + (1605632*y1)), tmp3, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/y2/cy26nao75upaps4wa5vo7umodsv72qtmzglfcu6wgy2gkhz4rjme.py
# Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# l__mod___features_0 => convolution
# l__mod___features_1 => relu
# l__mod___features_2 => convolution_1
# l__mod___features_3 => relu_1
# l__mod___features_4 => max_pool2d_with_indices
# l__mod___features_5 => convolution_2
# l__mod___features_6 => relu_2
# l__mod___features_7 => convolution_3
triton_poi_fused_convolution_max_pool2d_with_indices_relu_8 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_8', 'mutated_arg_names': []},
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
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (1152*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/my/cmyyw4qd7iudtxdceqhvcrysr33xsosoy36md7os7am4jh34rqrf.py
# Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# l__mod___features_0 => convolution
# l__mod___features_1 => relu
# l__mod___features_2 => convolution_1
# l__mod___features_3 => relu_1
# l__mod___features_4 => max_pool2d_with_indices
# l__mod___features_5 => convolution_2
# l__mod___features_6 => relu_2
# l__mod___features_7 => convolution_3
# l__mod___features_8 => relu_3
triton_poi_fused_convolution_max_pool2d_with_indices_relu_9 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_9', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6422528
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 12544) % 128
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp3, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/ik/cikdaer2wo7ceiotnevbdtjm6j7xjiggordf7pxxaawml7fzkhzh.py
# Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# l__mod___features_0 => convolution
# l__mod___features_1 => relu
# l__mod___features_2 => convolution_1
# l__mod___features_3 => relu_1
# l__mod___features_4 => max_pool2d_with_indices
# l__mod___features_5 => convolution_2
# l__mod___features_6 => relu_2
# l__mod___features_7 => convolution_3
# l__mod___features_8 => relu_3
# l__mod___features_9 => max_pool2d_with_indices_1
triton_poi_fused_convolution_max_pool2d_with_indices_relu_10 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[512, 4096], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_10', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 512
    xnumel = 3136
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 56
    x3 = (xindex // 56)
    y4 = yindex
    x5 = xindex
    y0 = yindex % 128
    y1 = (yindex // 128)
    tmp0 = tl.load(in_ptr0 + ((2*x2) + (224*x3) + (12544*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x2) + (224*x3) + (12544*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (112 + (2*x2) + (224*x3) + (12544*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (113 + (2*x2) + (224*x3) + (12544*y4)), xmask & ymask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (y0 + (128*x5) + (401408*y1)), tmp6, xmask & ymask)
''')


# kernel path: /tmp/torchinductor_youkaichao/ru/cruorssqoamf3qpkho4iwzxjkugf6wvig7uc5j5sk7qspris5pbu.py
# Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# l__mod___features_0 => convolution
# l__mod___features_1 => relu
# l__mod___features_10 => convolution_4
# l__mod___features_2 => convolution_1
# l__mod___features_3 => relu_1
# l__mod___features_4 => max_pool2d_with_indices
# l__mod___features_5 => convolution_2
# l__mod___features_6 => relu_2
# l__mod___features_7 => convolution_3
# l__mod___features_8 => relu_3
# l__mod___features_9 => max_pool2d_with_indices_1
triton_poi_fused_convolution_max_pool2d_with_indices_relu_11 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[32768, 16], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_11', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 32768
    xnumel = 9
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
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (128*x2) + (1152*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/x5/cx5sqekoni3t23niu5mwkansgq3obbpohplqz6esjg7ow5zz4fhj.py
# Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# l__mod___features_0 => convolution
# l__mod___features_1 => relu
# l__mod___features_10 => convolution_4
# l__mod___features_11 => relu_4
# l__mod___features_2 => convolution_1
# l__mod___features_3 => relu_1
# l__mod___features_4 => max_pool2d_with_indices
# l__mod___features_5 => convolution_2
# l__mod___features_6 => relu_2
# l__mod___features_7 => convolution_3
# l__mod___features_8 => relu_3
# l__mod___features_9 => max_pool2d_with_indices_1
triton_poi_fused_convolution_max_pool2d_with_indices_relu_12 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_12', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 3136
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
    tmp0 = tl.load(in_ptr0 + (x2 + (3136*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(out_ptr0 + (y0 + (256*x2) + (802816*y1)), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/w2/cw2ejikdeelad3l6nv3pw5xkt4j33xpyoyg6qzehok5hndoyreiq.py
# Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# l__mod___features_0 => convolution
# l__mod___features_1 => relu
# l__mod___features_10 => convolution_4
# l__mod___features_11 => relu_4
# l__mod___features_12 => convolution_5
# l__mod___features_2 => convolution_1
# l__mod___features_3 => relu_1
# l__mod___features_4 => max_pool2d_with_indices
# l__mod___features_5 => convolution_2
# l__mod___features_6 => relu_2
# l__mod___features_7 => convolution_3
# l__mod___features_8 => relu_3
# l__mod___features_9 => max_pool2d_with_indices_1
triton_poi_fused_convolution_max_pool2d_with_indices_relu_13 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_13', 'mutated_arg_names': []},
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


# kernel path: /tmp/torchinductor_youkaichao/ui/cuim4ano3dlzs2h3gob6rckzuceywwu7ibzwpdgxkixvt6fq2xvx.py
# Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# l__mod___features_0 => convolution
# l__mod___features_1 => relu
# l__mod___features_10 => convolution_4
# l__mod___features_11 => relu_4
# l__mod___features_12 => convolution_5
# l__mod___features_13 => relu_5
# l__mod___features_14 => convolution_6
# l__mod___features_15 => relu_6
# l__mod___features_2 => convolution_1
# l__mod___features_3 => relu_1
# l__mod___features_4 => max_pool2d_with_indices
# l__mod___features_5 => convolution_2
# l__mod___features_6 => relu_2
# l__mod___features_7 => convolution_3
# l__mod___features_8 => relu_3
# l__mod___features_9 => max_pool2d_with_indices_1
triton_poi_fused_convolution_max_pool2d_with_indices_relu_14 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_14', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3211264
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 3136) % 256
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp3, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/qb/cqb5pb6hsid4tdxhy6zxhtunljfhtgg72tthgzdolye4hseytkrx.py
# Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# l__mod___features_0 => convolution
# l__mod___features_1 => relu
# l__mod___features_10 => convolution_4
# l__mod___features_11 => relu_4
# l__mod___features_12 => convolution_5
# l__mod___features_13 => relu_5
# l__mod___features_14 => convolution_6
# l__mod___features_15 => relu_6
# l__mod___features_16 => max_pool2d_with_indices_2
# l__mod___features_2 => convolution_1
# l__mod___features_3 => relu_1
# l__mod___features_4 => max_pool2d_with_indices
# l__mod___features_5 => convolution_2
# l__mod___features_6 => relu_2
# l__mod___features_7 => convolution_3
# l__mod___features_8 => relu_3
# l__mod___features_9 => max_pool2d_with_indices_1
triton_poi_fused_convolution_max_pool2d_with_indices_relu_15 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[1024, 1024], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_15', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 28
    x3 = (xindex // 28)
    y4 = yindex
    x5 = xindex
    y0 = yindex % 256
    y1 = (yindex // 256)
    tmp0 = tl.load(in_ptr0 + ((2*x2) + (112*x3) + (3136*y4)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x2) + (112*x3) + (3136*y4)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (56 + (2*x2) + (112*x3) + (3136*y4)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (57 + (2*x2) + (112*x3) + (3136*y4)), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (y0 + (256*x5) + (200704*y1)), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4k/c4kqokhvi5iwukdqfvjpjjdly67mqtovbldcx7k3hs3biz7cxfyi.py
# Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# l__mod___features_0 => convolution
# l__mod___features_1 => relu
# l__mod___features_10 => convolution_4
# l__mod___features_11 => relu_4
# l__mod___features_12 => convolution_5
# l__mod___features_13 => relu_5
# l__mod___features_14 => convolution_6
# l__mod___features_15 => relu_6
# l__mod___features_16 => max_pool2d_with_indices_2
# l__mod___features_17 => convolution_7
# l__mod___features_2 => convolution_1
# l__mod___features_3 => relu_1
# l__mod___features_4 => max_pool2d_with_indices
# l__mod___features_5 => convolution_2
# l__mod___features_6 => relu_2
# l__mod___features_7 => convolution_3
# l__mod___features_8 => relu_3
# l__mod___features_9 => max_pool2d_with_indices_1
triton_poi_fused_convolution_max_pool2d_with_indices_relu_16 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_16', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 131072
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


# kernel path: /tmp/torchinductor_youkaichao/cn/ccn5ubsxme3e5b3hfzfrwkwfstbul34ubvn4zggbets6kuyyhuxd.py
# Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_18, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# l__mod___features_0 => convolution
# l__mod___features_1 => relu
# l__mod___features_10 => convolution_4
# l__mod___features_11 => relu_4
# l__mod___features_12 => convolution_5
# l__mod___features_13 => relu_5
# l__mod___features_14 => convolution_6
# l__mod___features_15 => relu_6
# l__mod___features_16 => max_pool2d_with_indices_2
# l__mod___features_17 => convolution_7
# l__mod___features_18 => relu_7
# l__mod___features_2 => convolution_1
# l__mod___features_3 => relu_1
# l__mod___features_4 => max_pool2d_with_indices
# l__mod___features_5 => convolution_2
# l__mod___features_6 => relu_2
# l__mod___features_7 => convolution_3
# l__mod___features_8 => relu_3
# l__mod___features_9 => max_pool2d_with_indices_1
triton_poi_fused_convolution_max_pool2d_with_indices_relu_17 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3, 4))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_17', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 784
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2 + (784*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(out_ptr0 + (y0 + (512*x2) + (401408*y1)), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/44/c4455untxni5mux6nnvos6gwyr5h7b35fk7zyyep22nfi5waka3z.py
# Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_18, l__mod___features_19, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# l__mod___features_0 => convolution
# l__mod___features_1 => relu
# l__mod___features_10 => convolution_4
# l__mod___features_11 => relu_4
# l__mod___features_12 => convolution_5
# l__mod___features_13 => relu_5
# l__mod___features_14 => convolution_6
# l__mod___features_15 => relu_6
# l__mod___features_16 => max_pool2d_with_indices_2
# l__mod___features_17 => convolution_7
# l__mod___features_18 => relu_7
# l__mod___features_19 => convolution_8
# l__mod___features_2 => convolution_1
# l__mod___features_3 => relu_1
# l__mod___features_4 => max_pool2d_with_indices
# l__mod___features_5 => convolution_2
# l__mod___features_6 => relu_2
# l__mod___features_7 => convolution_3
# l__mod___features_8 => relu_3
# l__mod___features_9 => max_pool2d_with_indices_1
triton_poi_fused_convolution_max_pool2d_with_indices_relu_18 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_18', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 262144
    xnumel = 9
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2 + (9*y3)), xmask, eviction_policy='evict_last')
    tl.store(out_ptr0 + (y0 + (512*x2) + (4608*y1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/4e/c4e7awinkvzcrw2c7vmhlujeyvvw7josqrzjs4cory3a6nd2sp55.py
# Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_18, l__mod___features_19, l__mod___features_2, l__mod___features_20, l__mod___features_21, l__mod___features_22, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# l__mod___features_0 => convolution
# l__mod___features_1 => relu
# l__mod___features_10 => convolution_4
# l__mod___features_11 => relu_4
# l__mod___features_12 => convolution_5
# l__mod___features_13 => relu_5
# l__mod___features_14 => convolution_6
# l__mod___features_15 => relu_6
# l__mod___features_16 => max_pool2d_with_indices_2
# l__mod___features_17 => convolution_7
# l__mod___features_18 => relu_7
# l__mod___features_19 => convolution_8
# l__mod___features_2 => convolution_1
# l__mod___features_20 => relu_8
# l__mod___features_21 => convolution_9
# l__mod___features_22 => relu_9
# l__mod___features_3 => relu_1
# l__mod___features_4 => max_pool2d_with_indices
# l__mod___features_5 => convolution_2
# l__mod___features_6 => relu_2
# l__mod___features_7 => convolution_3
# l__mod___features_8 => relu_3
# l__mod___features_9 => max_pool2d_with_indices_1
triton_poi_fused_convolution_max_pool2d_with_indices_relu_19 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_19', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1605632
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 784) % 512
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp3, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/bk/cbkya4qnznblwg263ycf4tdc2awjvjqtcyxyr4xrhc7hchnnzo6p.py
# Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_18, l__mod___features_19, l__mod___features_2, l__mod___features_20, l__mod___features_21, l__mod___features_22, l__mod___features_23, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# l__mod___features_0 => convolution
# l__mod___features_1 => relu
# l__mod___features_10 => convolution_4
# l__mod___features_11 => relu_4
# l__mod___features_12 => convolution_5
# l__mod___features_13 => relu_5
# l__mod___features_14 => convolution_6
# l__mod___features_15 => relu_6
# l__mod___features_16 => max_pool2d_with_indices_2
# l__mod___features_17 => convolution_7
# l__mod___features_18 => relu_7
# l__mod___features_19 => convolution_8
# l__mod___features_2 => convolution_1
# l__mod___features_20 => relu_8
# l__mod___features_21 => convolution_9
# l__mod___features_22 => relu_9
# l__mod___features_23 => max_pool2d_with_indices_3
# l__mod___features_3 => relu_1
# l__mod___features_4 => max_pool2d_with_indices
# l__mod___features_5 => convolution_2
# l__mod___features_6 => relu_2
# l__mod___features_7 => convolution_3
# l__mod___features_8 => relu_3
# l__mod___features_9 => max_pool2d_with_indices_1
triton_poi_fused_convolution_max_pool2d_with_indices_relu_20 = async_compile.triton('triton_', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(
    size_hints=[2048, 256], tile_hint=TileHint.SQUARE,
    filename=__file__,
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_20', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 14
    x3 = (xindex // 14)
    y4 = yindex
    x5 = xindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_ptr0 + ((2*x2) + (56*x3) + (784*y4)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x2) + (56*x3) + (784*y4)), xmask, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (28 + (2*x2) + (56*x3) + (784*y4)), xmask, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (29 + (2*x2) + (56*x3) + (784*y4)), xmask, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(out_ptr0 + (y0 + (512*x5) + (100352*y1)), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/6r/c6rietlhdsqmhmei547ajgnw4hwzpaojprpg2lcldwqmzgagkhay.py
# Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_18, l__mod___features_19, l__mod___features_2, l__mod___features_20, l__mod___features_21, l__mod___features_22, l__mod___features_23, l__mod___features_24, l__mod___features_25, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# l__mod___features_0 => convolution
# l__mod___features_1 => relu
# l__mod___features_10 => convolution_4
# l__mod___features_11 => relu_4
# l__mod___features_12 => convolution_5
# l__mod___features_13 => relu_5
# l__mod___features_14 => convolution_6
# l__mod___features_15 => relu_6
# l__mod___features_16 => max_pool2d_with_indices_2
# l__mod___features_17 => convolution_7
# l__mod___features_18 => relu_7
# l__mod___features_19 => convolution_8
# l__mod___features_2 => convolution_1
# l__mod___features_20 => relu_8
# l__mod___features_21 => convolution_9
# l__mod___features_22 => relu_9
# l__mod___features_23 => max_pool2d_with_indices_3
# l__mod___features_24 => convolution_10
# l__mod___features_25 => relu_10
# l__mod___features_3 => relu_1
# l__mod___features_4 => max_pool2d_with_indices
# l__mod___features_5 => convolution_2
# l__mod___features_6 => relu_2
# l__mod___features_7 => convolution_3
# l__mod___features_8 => relu_3
# l__mod___features_9 => max_pool2d_with_indices_1
triton_poi_fused_convolution_max_pool2d_with_indices_relu_21 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_21', 'mutated_arg_names': []},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_ptr0, in_ptr1, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 196
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y3 = yindex
    y0 = yindex % 512
    y1 = (yindex // 512)
    tmp0 = tl.load(in_ptr0 + (x2 + (196*y3)), xmask, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr1 + (y0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(out_ptr0 + (y0 + (512*x2) + (100352*y1)), tmp3, xmask)
''')


# kernel path: /tmp/torchinductor_youkaichao/nw/cnwkdj7ohfr2qsf3w6vg22j32qsugmkjy2jbtoa6tok2pz2kfaxh.py
# Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_18, l__mod___features_19, l__mod___features_2, l__mod___features_20, l__mod___features_21, l__mod___features_22, l__mod___features_23, l__mod___features_24, l__mod___features_25, l__mod___features_26, l__mod___features_27, l__mod___features_28, l__mod___features_29, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# l__mod___features_0 => convolution
# l__mod___features_1 => relu
# l__mod___features_10 => convolution_4
# l__mod___features_11 => relu_4
# l__mod___features_12 => convolution_5
# l__mod___features_13 => relu_5
# l__mod___features_14 => convolution_6
# l__mod___features_15 => relu_6
# l__mod___features_16 => max_pool2d_with_indices_2
# l__mod___features_17 => convolution_7
# l__mod___features_18 => relu_7
# l__mod___features_19 => convolution_8
# l__mod___features_2 => convolution_1
# l__mod___features_20 => relu_8
# l__mod___features_21 => convolution_9
# l__mod___features_22 => relu_9
# l__mod___features_23 => max_pool2d_with_indices_3
# l__mod___features_24 => convolution_10
# l__mod___features_25 => relu_10
# l__mod___features_26 => convolution_11
# l__mod___features_27 => relu_11
# l__mod___features_28 => convolution_12
# l__mod___features_29 => relu_12
# l__mod___features_3 => relu_1
# l__mod___features_4 => max_pool2d_with_indices
# l__mod___features_5 => convolution_2
# l__mod___features_6 => relu_2
# l__mod___features_7 => convolution_3
# l__mod___features_8 => relu_3
# l__mod___features_9 => max_pool2d_with_indices_1
triton_poi_fused_convolution_max_pool2d_with_indices_relu_22 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_convolution_max_pool2d_with_indices_relu_22', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 401408
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x3 = xindex
    x1 = (xindex // 196) % 512
    tmp0 = tl.load(in_out_ptr0 + (x3), None)
    tmp1 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp3 = triton_helpers.maximum(0, tmp2)
    tl.store(in_out_ptr0 + (x3), tmp3, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/zz/czzo4kvq7nigqnosprg52cht4ekdf6b4wwtnigc47ijpei5pb6bp.py
# Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_18, l__mod___features_19, l__mod___features_2, l__mod___features_20, l__mod___features_21, l__mod___features_22, l__mod___features_23, l__mod___features_24, l__mod___features_25, l__mod___features_26, l__mod___features_27, l__mod___features_28, l__mod___features_29, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9, x, x_1], Original ATen: [aten._adaptive_avg_pool2d, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
# l__mod___features_0 => convolution
# l__mod___features_1 => relu
# l__mod___features_10 => convolution_4
# l__mod___features_11 => relu_4
# l__mod___features_12 => convolution_5
# l__mod___features_13 => relu_5
# l__mod___features_14 => convolution_6
# l__mod___features_15 => relu_6
# l__mod___features_16 => max_pool2d_with_indices_2
# l__mod___features_17 => convolution_7
# l__mod___features_18 => relu_7
# l__mod___features_19 => convolution_8
# l__mod___features_2 => convolution_1
# l__mod___features_20 => relu_8
# l__mod___features_21 => convolution_9
# l__mod___features_22 => relu_9
# l__mod___features_23 => max_pool2d_with_indices_3
# l__mod___features_24 => convolution_10
# l__mod___features_25 => relu_10
# l__mod___features_26 => convolution_11
# l__mod___features_27 => relu_11
# l__mod___features_28 => convolution_12
# l__mod___features_29 => relu_12
# l__mod___features_3 => relu_1
# l__mod___features_4 => max_pool2d_with_indices
# l__mod___features_5 => convolution_2
# l__mod___features_6 => relu_2
# l__mod___features_7 => convolution_3
# l__mod___features_8 => relu_3
# l__mod___features_9 => max_pool2d_with_indices_1
# x => max_pool2d_with_indices_4
# x_1 => _adaptive_avg_pool2d
triton_poi_fused__adaptive_avg_pool2d_convolution_max_pool2d_with_indices_relu_23 = async_compile.triton('triton_', '''
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
    triton_meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]},
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__adaptive_avg_pool2d_convolution_max_pool2d_with_indices_relu_23', 'mutated_arg_names': ['in_out_ptr0']},
    min_elem_per_thread=0
)
@triton.jit
def triton_(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 100352
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 7
    x1 = (xindex // 7)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((2*x0) + (28*x1)), None, eviction_policy='evict_last')
    tmp1 = tl.load(in_ptr0 + (1 + (2*x0) + (28*x1)), None, eviction_policy='evict_last')
    tmp3 = tl.load(in_ptr0 + (14 + (2*x0) + (28*x1)), None, eviction_policy='evict_last')
    tmp5 = tl.load(in_ptr0 + (15 + (2*x0) + (28*x1)), None, eviction_policy='evict_last')
    tmp2 = triton_helpers.maximum(tmp1, tmp0)
    tmp4 = triton_helpers.maximum(tmp3, tmp2)
    tmp6 = triton_helpers.maximum(tmp5, tmp4)
    tl.store(in_out_ptr0 + (x2), tmp6, None)
''')


# kernel path: /tmp/torchinductor_youkaichao/si/csiimh4goza4cme4zoolhxf7w7itircfppuzm7o4ymysrz2mlanv.py
# Source Nodes: [l__mod___classifier_1], Original ATen: [aten.relu]
# l__mod___classifier_1 => relu_13
triton_poi_fused_relu_24 = async_compile.triton('triton_', '''
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
    inductor_meta={'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_relu_24', 'mutated_arg_names': ['in_out_ptr0']},
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1 = args
    args.clear()
    assert_size_stride(arg0_1, (64, 3, 3, 3), (27, 9, 3, 1))
    assert_size_stride(arg1_1, (64, ), (1, ))
    assert_size_stride(arg2_1, (64, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg3_1, (64, ), (1, ))
    assert_size_stride(arg4_1, (128, 64, 3, 3), (576, 9, 3, 1))
    assert_size_stride(arg5_1, (128, ), (1, ))
    assert_size_stride(arg6_1, (128, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg7_1, (128, ), (1, ))
    assert_size_stride(arg8_1, (256, 128, 3, 3), (1152, 9, 3, 1))
    assert_size_stride(arg9_1, (256, ), (1, ))
    assert_size_stride(arg10_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg11_1, (256, ), (1, ))
    assert_size_stride(arg12_1, (256, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg13_1, (256, ), (1, ))
    assert_size_stride(arg14_1, (512, 256, 3, 3), (2304, 9, 3, 1))
    assert_size_stride(arg15_1, (512, ), (1, ))
    assert_size_stride(arg16_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg17_1, (512, ), (1, ))
    assert_size_stride(arg18_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg19_1, (512, ), (1, ))
    assert_size_stride(arg20_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg21_1, (512, ), (1, ))
    assert_size_stride(arg22_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg23_1, (512, ), (1, ))
    assert_size_stride(arg24_1, (512, 512, 3, 3), (4608, 9, 3, 1))
    assert_size_stride(arg25_1, (512, ), (1, ))
    assert_size_stride(arg26_1, (4096, 25088), (25088, 1))
    assert_size_stride(arg27_1, (4096, ), (1, ))
    assert_size_stride(arg28_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg29_1, (4096, ), (1, ))
    assert_size_stride(arg30_1, (1000, 4096), (4096, 1))
    assert_size_stride(arg31_1, (1000, ), (1, ))
    assert_size_stride(arg32_1, (4, 3, 224, 224), (150528, 50176, 224, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((4, 3, 224, 224), (150528, 1, 672, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0], Original ATen: [aten.convolution]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_convolution_0.run(arg32_1, buf0, 12, 50176, grid=grid(12, 50176), stream=stream0)
        del arg32_1
        buf1 = empty_strided((64, 3, 3, 3), (27, 1, 9, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0], Original ATen: [aten.convolution]
        triton_poi_fused_convolution_1.run(arg0_1, buf1, 192, 9, grid=grid(192, 9), stream=stream0)
        del arg0_1
        # Source Nodes: [l__mod___features_0], Original ATen: [aten.convolution]
        buf2 = extern_kernels.convolution(buf0, buf1, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf2, (4, 64, 224, 224), (3211264, 50176, 224, 1))
        del buf0
        del buf1
        buf3 = empty_strided((4, 64, 224, 224), (3211264, 1, 14336, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0, l__mod___features_1], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_2.run(buf2, arg1_1, buf3, 256, 50176, grid=grid(256, 50176), stream=stream0)
        del arg1_1
        del buf2
        buf4 = empty_strided((64, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_2], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_3.run(arg2_1, buf4, 4096, 9, grid=grid(4096, 9), stream=stream0)
        del arg2_1
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_2], Original ATen: [aten.convolution, aten.relu]
        buf5 = extern_kernels.convolution(buf3, buf4, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf5, (4, 64, 224, 224), (3211264, 50176, 224, 1))
        del buf3
        del buf4
        buf6 = buf5; del buf5  # reuse
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_2, l__mod___features_3], Original ATen: [aten.convolution, aten.relu]
        triton_poi_fused_convolution_relu_4.run(buf6, arg3_1, 12845056, grid=grid(12845056), stream=stream0)
        del arg3_1
        buf7 = empty_strided((4, 64, 112, 112), (802816, 1, 7168, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_2, l__mod___features_3, l__mod___features_4], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_5.run(buf6, buf7, 256, 12544, grid=grid(256, 12544), stream=stream0)
        del buf6
        buf8 = empty_strided((128, 64, 3, 3), (576, 1, 192, 64), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_6.run(arg4_1, buf8, 8192, 9, grid=grid(8192, 9), stream=stream0)
        del arg4_1
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        buf9 = extern_kernels.convolution(buf7, buf8, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf9, (4, 128, 112, 112), (1605632, 12544, 112, 1))
        del buf8
        buf10 = empty_strided((4, 128, 112, 112), (1605632, 1, 14336, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_7.run(buf9, arg5_1, buf10, 512, 12544, grid=grid(512, 12544), stream=stream0)
        del arg5_1
        del buf9
        buf11 = empty_strided((128, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_8.run(arg6_1, buf11, 16384, 9, grid=grid(16384, 9), stream=stream0)
        del arg6_1
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        buf12 = extern_kernels.convolution(buf10, buf11, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf12, (4, 128, 112, 112), (1605632, 12544, 112, 1))
        del buf10
        del buf11
        buf13 = buf12; del buf12  # reuse
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_9.run(buf13, arg7_1, 6422528, grid=grid(6422528), stream=stream0)
        del arg7_1
        buf14 = empty_strided((4, 128, 56, 56), (401408, 1, 7168, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_10.run(buf13, buf14, 512, 3136, grid=grid(512, 3136), stream=stream0)
        del buf13
        buf15 = empty_strided((256, 128, 3, 3), (1152, 1, 384, 128), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_11.run(arg8_1, buf15, 32768, 9, grid=grid(32768, 9), stream=stream0)
        del arg8_1
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        buf16 = extern_kernels.convolution(buf14, buf15, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf16, (4, 256, 56, 56), (802816, 3136, 56, 1))
        del buf15
        buf17 = reinterpret_tensor(buf7, (4, 256, 56, 56), (802816, 1, 14336, 256), 0); del buf7  # reuse
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_12.run(buf16, arg9_1, buf17, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        del arg9_1
        del buf16
        buf18 = empty_strided((256, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_13.run(arg10_1, buf18, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg10_1
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        buf19 = extern_kernels.convolution(buf17, buf18, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf19, (4, 256, 56, 56), (802816, 3136, 56, 1))
        buf20 = buf17; del buf17  # reuse
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_12.run(buf19, arg11_1, buf20, 1024, 3136, grid=grid(1024, 3136), stream=stream0)
        del arg11_1
        del buf19
        buf21 = buf18; del buf18  # reuse
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_13.run(arg12_1, buf21, 65536, 9, grid=grid(65536, 9), stream=stream0)
        del arg12_1
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        buf22 = extern_kernels.convolution(buf20, buf21, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf22, (4, 256, 56, 56), (802816, 3136, 56, 1))
        del buf20
        del buf21
        buf23 = buf22; del buf22  # reuse
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_14.run(buf23, arg13_1, 3211264, grid=grid(3211264), stream=stream0)
        del arg13_1
        buf24 = empty_strided((4, 256, 28, 28), (200704, 1, 7168, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_15.run(buf23, buf24, 1024, 784, grid=grid(1024, 784), stream=stream0)
        del buf23
        buf25 = empty_strided((512, 256, 3, 3), (2304, 1, 768, 256), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_16.run(arg14_1, buf25, 131072, 9, grid=grid(131072, 9), stream=stream0)
        del arg14_1
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        buf26 = extern_kernels.convolution(buf24, buf25, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf26, (4, 512, 28, 28), (401408, 784, 28, 1))
        del buf24
        del buf25
        buf27 = reinterpret_tensor(buf14, (4, 512, 28, 28), (401408, 1, 14336, 512), 0); del buf14  # reuse
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_18, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_17.run(buf26, arg15_1, buf27, 2048, 784, grid=grid(2048, 784), stream=stream0)
        del arg15_1
        del buf26
        buf28 = empty_strided((512, 512, 3, 3), (4608, 1, 1536, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_18, l__mod___features_19, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_18.run(arg16_1, buf28, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg16_1
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_18, l__mod___features_19, l__mod___features_2, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        buf29 = extern_kernels.convolution(buf27, buf28, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf29, (4, 512, 28, 28), (401408, 784, 28, 1))
        buf30 = buf27; del buf27  # reuse
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_18, l__mod___features_19, l__mod___features_2, l__mod___features_20, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_17.run(buf29, arg17_1, buf30, 2048, 784, grid=grid(2048, 784), stream=stream0)
        del arg17_1
        del buf29
        buf31 = buf28; del buf28  # reuse
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_18, l__mod___features_19, l__mod___features_2, l__mod___features_20, l__mod___features_21, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_18.run(arg18_1, buf31, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg18_1
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_18, l__mod___features_19, l__mod___features_2, l__mod___features_20, l__mod___features_21, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        buf32 = extern_kernels.convolution(buf30, buf31, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf32, (4, 512, 28, 28), (401408, 784, 28, 1))
        del buf30
        buf33 = buf32; del buf32  # reuse
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_18, l__mod___features_19, l__mod___features_2, l__mod___features_20, l__mod___features_21, l__mod___features_22, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_19.run(buf33, arg19_1, 1605632, grid=grid(1605632), stream=stream0)
        del arg19_1
        buf34 = empty_strided((4, 512, 14, 14), (100352, 1, 7168, 512), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_18, l__mod___features_19, l__mod___features_2, l__mod___features_20, l__mod___features_21, l__mod___features_22, l__mod___features_23, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_20.run(buf33, buf34, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del buf33
        buf35 = buf31; del buf31  # reuse
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_18, l__mod___features_19, l__mod___features_2, l__mod___features_20, l__mod___features_21, l__mod___features_22, l__mod___features_23, l__mod___features_24, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_18.run(arg20_1, buf35, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg20_1
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_18, l__mod___features_19, l__mod___features_2, l__mod___features_20, l__mod___features_21, l__mod___features_22, l__mod___features_23, l__mod___features_24, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        buf36 = extern_kernels.convolution(buf34, buf35, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf36, (4, 512, 14, 14), (100352, 196, 14, 1))
        buf37 = buf34; del buf34  # reuse
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_18, l__mod___features_19, l__mod___features_2, l__mod___features_20, l__mod___features_21, l__mod___features_22, l__mod___features_23, l__mod___features_24, l__mod___features_25, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_21.run(buf36, arg21_1, buf37, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg21_1
        del buf36
        buf38 = buf35; del buf35  # reuse
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_18, l__mod___features_19, l__mod___features_2, l__mod___features_20, l__mod___features_21, l__mod___features_22, l__mod___features_23, l__mod___features_24, l__mod___features_25, l__mod___features_26, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_18.run(arg22_1, buf38, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg22_1
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_18, l__mod___features_19, l__mod___features_2, l__mod___features_20, l__mod___features_21, l__mod___features_22, l__mod___features_23, l__mod___features_24, l__mod___features_25, l__mod___features_26, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        buf39 = extern_kernels.convolution(buf37, buf38, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf39, (4, 512, 14, 14), (100352, 196, 14, 1))
        buf40 = buf37; del buf37  # reuse
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_18, l__mod___features_19, l__mod___features_2, l__mod___features_20, l__mod___features_21, l__mod___features_22, l__mod___features_23, l__mod___features_24, l__mod___features_25, l__mod___features_26, l__mod___features_27, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_21.run(buf39, arg23_1, buf40, 2048, 196, grid=grid(2048, 196), stream=stream0)
        del arg23_1
        del buf39
        buf41 = buf38; del buf38  # reuse
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_18, l__mod___features_19, l__mod___features_2, l__mod___features_20, l__mod___features_21, l__mod___features_22, l__mod___features_23, l__mod___features_24, l__mod___features_25, l__mod___features_26, l__mod___features_27, l__mod___features_28, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_18.run(arg24_1, buf41, 262144, 9, grid=grid(262144, 9), stream=stream0)
        del arg24_1
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_18, l__mod___features_19, l__mod___features_2, l__mod___features_20, l__mod___features_21, l__mod___features_22, l__mod___features_23, l__mod___features_24, l__mod___features_25, l__mod___features_26, l__mod___features_27, l__mod___features_28, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        buf42 = extern_kernels.convolution(buf40, buf41, stride=(1, 1), padding=(1, 1), dilation=(1, 1), transposed=False, output_padding=(0, 0), groups=1, bias=None)
        assert_size_stride(buf42, (4, 512, 14, 14), (100352, 196, 14, 1))
        del buf40
        del buf41
        buf43 = buf42; del buf42  # reuse
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_18, l__mod___features_19, l__mod___features_2, l__mod___features_20, l__mod___features_21, l__mod___features_22, l__mod___features_23, l__mod___features_24, l__mod___features_25, l__mod___features_26, l__mod___features_27, l__mod___features_28, l__mod___features_29, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9], Original ATen: [aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused_convolution_max_pool2d_with_indices_relu_22.run(buf43, arg25_1, 401408, grid=grid(401408), stream=stream0)
        del arg25_1
        buf44 = empty((4, 512, 7, 7), device='cuda', dtype=torch.float32)
        buf45 = buf44; del buf44  # reuse
        # Source Nodes: [l__mod___features_0, l__mod___features_1, l__mod___features_10, l__mod___features_11, l__mod___features_12, l__mod___features_13, l__mod___features_14, l__mod___features_15, l__mod___features_16, l__mod___features_17, l__mod___features_18, l__mod___features_19, l__mod___features_2, l__mod___features_20, l__mod___features_21, l__mod___features_22, l__mod___features_23, l__mod___features_24, l__mod___features_25, l__mod___features_26, l__mod___features_27, l__mod___features_28, l__mod___features_29, l__mod___features_3, l__mod___features_4, l__mod___features_5, l__mod___features_6, l__mod___features_7, l__mod___features_8, l__mod___features_9, x, x_1], Original ATen: [aten._adaptive_avg_pool2d, aten.convolution, aten.max_pool2d_with_indices, aten.relu]
        triton_poi_fused__adaptive_avg_pool2d_convolution_max_pool2d_with_indices_relu_23.run(buf45, buf43, 100352, grid=grid(100352), stream=stream0)
        del buf43
        buf46 = empty((4, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(reinterpret_tensor(buf45, (4, 25088), (25088, 1), 0), reinterpret_tensor(arg26_1, (25088, 4096), (1, 25088), 0), out=buf46)
        del arg26_1
        del buf45
        buf47 = buf46; del buf46  # reuse
        # Source Nodes: [l__mod___classifier_1], Original ATen: [aten.relu]
        triton_poi_fused_relu_24.run(buf47, arg27_1, 16384, grid=grid(16384), stream=stream0)
        del arg27_1
        buf48 = empty((4, 4096), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___classifier_1], Original ATen: [aten.relu]
        extern_kernels.mm(buf47, reinterpret_tensor(arg28_1, (4096, 4096), (1, 4096), 0), out=buf48)
        del arg28_1
        del buf47
        buf49 = buf48; del buf48  # reuse
        # Source Nodes: [l__mod___classifier_4], Original ATen: [aten.relu]
        triton_poi_fused_relu_24.run(buf49, arg29_1, 16384, grid=grid(16384), stream=stream0)
        del arg29_1
        buf50 = empty((4, 1000), device='cuda', dtype=torch.float32)
        # Source Nodes: [l__mod___classifier_4, x_3], Original ATen: [aten.addmm, aten.relu]
        extern_kernels.addmm(arg31_1, buf49, reinterpret_tensor(arg30_1, (4096, 1000), (1, 4096), 0), alpha=1, beta=1, out=buf50)
        del arg30_1
        del arg31_1
        return (buf50, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((64, 3, 3, 3), (27, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg1_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg2_1 = rand_strided((64, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg3_1 = rand_strided((64, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg4_1 = rand_strided((128, 64, 3, 3), (576, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg5_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg6_1 = rand_strided((128, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg7_1 = rand_strided((128, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg8_1 = rand_strided((256, 128, 3, 3), (1152, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg9_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg10_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg11_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg12_1 = rand_strided((256, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg13_1 = rand_strided((256, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg14_1 = rand_strided((512, 256, 3, 3), (2304, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg15_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg16_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg17_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg18_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg19_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg20_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg21_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg22_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg23_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg24_1 = rand_strided((512, 512, 3, 3), (4608, 9, 3, 1), device='cuda:0', dtype=torch.float32)
    arg25_1 = rand_strided((512, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg26_1 = rand_strided((4096, 25088), (25088, 1), device='cuda:0', dtype=torch.float32)
    arg27_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg28_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg29_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg30_1 = rand_strided((1000, 4096), (4096, 1), device='cuda:0', dtype=torch.float32)
    arg31_1 = rand_strided((1000, ), (1, ), device='cuda:0', dtype=torch.float32)
    arg32_1 = rand_strided((4, 3, 224, 224), (150528, 50176, 224, 1), device='cuda:0', dtype=torch.float32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('vgg16', benchmark_compiled_module)
